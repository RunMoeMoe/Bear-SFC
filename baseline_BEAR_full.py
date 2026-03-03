# -*- coding: utf-8 -*-
from __future__ import annotations
"""
baseline_BEAR_full.py
=====================
完整 BEAR 基线（可训练版，无外部 DL 依赖）
- Central：在线 bandit 学习 (N, alpha) 的指引
- Edge：束搜索 + 不相交约束，选出 N 条路径（N-1 在役 + 1 热备）
- 资源：bw_each = bw/(N-1)，reserve_equal_split(paths_active, path_backup, ...)
- 运行期：仅对“需要切换”的会话尝试 failover；>=2 同时 down 直接失败；未命中释放
- mode="train" 时持续在线更新；mode="eval" 冻结策略

依赖：
- env.py: enumerate_candidates / check_paths_feasible / reserve_equal_split /
         predict_reliability / try_failover / inject_failures / release_expired /
         get_session_ref / is_path_up (可选) / count_down_active_paths (可选)
- metrics.py: EventLogger / SummaryAggregator / EpisodeSummaryWriter
- quota.py:  QuotaManager
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math
import random
import numpy as np

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter
from quota import QuotaManager


# =========================
# 工具函数：不相交/权重
# =========================
def _edges_of(path: List[int]) -> List[Tuple[int, int]]:
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def _disjoint_ok(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode in ("EDGE", "LINK"):
        e1 = set(_edges_of(p1)); e2 = set(_edges_of(p2))
        return len(e1 & e2) == 0
    s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
    return len(s1 & s2) == 0

def _set_disjoint(paths: List[List[int]], mode: str) -> bool:
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            if not _disjoint_ok(paths[i], paths[j], mode):
                return False
    return True

def _cand_weight(c: Dict[str, Any]) -> float:
    feats = c.get("feats", [])
    if isinstance(feats, list) and len(feats) >= 1:
        try:
            return float(feats[0])
        except Exception:
            pass
    return float(len(c.get("path", [])))

def _std_request(req) -> Dict[str, Any]:
    if isinstance(req, dict):
        return {
            "sid": int(req["sid"]),
            "src": int(req["src"]),
            "dst": int(req["dst"]),
            "L":   int(req["L"]),
            "bw":  float(req["bw"]),
            "ttl": int(req["ttl"]),
            "t_arrive": int(req.get("t_arrive", req.get("t", 0))),
        }
    return {
        "sid": int(getattr(req, "sid")),
        "src": int(getattr(req, "src")),
        "dst": int(getattr(req, "dst")),
        "L":   int(getattr(req, "L")),
        "bw":  float(getattr(req, "bw")),
        "ttl": int(getattr(req, "ttl")),
        "t_arrive": int(getattr(req, "t_arrive", getattr(req, "t", 0))),
    }


# =========================
# Central：Bandit 策略
# =========================
class DiscreteArmBandit:
    """
    离散臂（N 的选择）— Exp3 / Hedge 形式
    - arms: 列表，如 [2,3,4,5]
    - 学习目标：最大化奖励（高奖励->选择概率更大）
    """
    def __init__(self, arms: List[int], gamma: float = 0.07, init_w: float = 1.0):
        assert len(arms) >= 1
        self.arms = list(arms)
        self.K = len(arms)
        self.gamma = float(gamma)
        self.w = np.ones(self.K, dtype=np.float64) * float(init_w)

    def probs(self) -> np.ndarray:
        w_sum = self.w.sum()
        if w_sum <= 0:
            p = np.ones(self.K, dtype=np.float64) / self.K
        else:
            p = (1 - self.gamma) * (self.w / w_sum) + self.gamma / self.K
        return p

    def select(self) -> Tuple[int, int, float]:
        p = self.probs()
        idx = int(np.random.choice(np.arange(self.K), p=p))
        return self.arms[idx], idx, p[idx]

    def update(self, idx: int, reward: float, p_selected: float):
        # Exp3 权重更新
        x_hat = reward / max(p_selected, 1e-9)
        growth = math.exp(self.gamma * x_hat / self.K)
        self.w[idx] *= growth
        # 数值稳定
        if not np.isfinite(self.w).all():
            self.w = np.clip(self.w, 1e-9, 1e9)


class WeightedAlphaBandit:
    """
    候选路径重要性权重 → α via softmax
    - 对每次请求的候选集长度 K 可变，不保存固定维度参数；
    - 用“参考模板”+“在线回传标量”做缩放/温度调节；
    - 训练时返回一个“温度 T”和“重要性缩放 lambda”；
      α = softmax( -weight * lambda / T )
    """
    def __init__(self, init_temp: float = 0.35, init_lmbd: float = 1.0, lr: float = 0.05):
        self.temp = float(init_temp)
        self.lmbd = float(init_lmbd)
        self.lr = float(lr)
        self._clip()

    def _clip(self):
        self.temp = float(np.clip(self.temp, 0.05, 2.0))
        self.lmbd = float(np.clip(self.lmbd, 0.2, 5.0))

    def suggest(self, weights: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        # 输入为候选权重（越小越好），输出 α 概率
        w = weights - weights.min()
        logits = - w * self.lmbd / max(1e-6, self.temp)
        logits -= logits.max()
        exps = np.exp(logits)
        alpha = exps / max(1e-9, exps.sum())
        meta = {"alpha_temp": self.temp, "alpha_lambda": self.lmbd, "alpha_max": float(alpha.max())}
        return alpha, meta

    def update(self, reward: float):
        # 简单一阶：若 reward 高 → 降温/增强区分（temp↓, lmbd↑）；反之相反
        # 奖励归一化后作正负微调
        r = float(np.tanh(reward))  # in (-1,1)
        self.temp -= self.lr * r * 0.5
        self.lmbd += self.lr * r * 0.5
        self._clip()


# =========================
# Edge：束搜索 + 自适应 N
# =========================
def beam_search_disjoint_paths(cands: List[Dict[str, Any]],
                               scores: np.ndarray,
                               N: int,
                               disjoint_mode: str,
                               beam_size: int = 8) -> Optional[List[List[int]]]:
    """
    在候选集中根据分数（越大越优）挑选 N 条两两不相交路径（束搜索）。
    返回：长度为 N 的路径列表，若失败返回 None。
    """
    order = np.argsort(-scores)  # 大到小
    seq = [cands[i] for i in order]
    beams = [([] , 0.0)]  # (paths, total_score)
    for c in seq:
        p = c.get("path", [])
        s = float(scores[order[seq.index(c)]]) if len(scores) == len(seq) else 0.0
        new_beams = []
        for paths, sc in beams:
            # 不选
            new_beams.append((paths, sc))
            # 选
            ok = True
            for q in paths:
                if not _disjoint_ok(p, q, disjoint_mode):
                    ok = False; break
            if ok:
                new_beams.append((paths + [list(p)], sc + s))
        # 保留 top beam_size
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        # 早停
        for paths, _ in beams:
            if len(paths) >= N:
                return paths[:N]
    # 最终检查
    for paths, _ in beams:
        if len(paths) >= N and _set_disjoint(paths[:N], disjoint_mode):
            return paths[:N]
    return None


# =========================
# BEAR-FULL 系统
# =========================
class BEARFullSystem:
    """
    与 runner 对接：
      - run_one_episode(ep_idx, steps, mode, fixed_N=None) -> summary
      - run(mode, epochs, steps, fixed_N=None) -> None

    关键配置：
      - disjoint_mode: EDGE/NODE/DZ
      - K_cand_max: 候选上限
      - N_min..N_max: 路径数量范围（总数 N；在役 N-1，热备 1）
      - beam_size: 束搜索宽度
      - reward weights: (w_place, w_fo, w_rel, w_cost, w_lat)
    """
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 48,
                 N_min: int = 2, N_max: int = 5,
                 beam_size: int = 8,
                 bandit_gamma: float = 0.07,
                 alpha_temp0: float = 0.35,
                 alpha_lmbd0: float = 1.0,
                 alpha_lr: float = 0.05,
                 reward_weights: Tuple[float, float, float, float, float] = (2.0, 1.5, 1.0, -0.8, -0.6)):
        self.env = env
        self.result_dir = Path(result_dir); self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)

        # 覆盖写 CSV
        for p in [self.result_dir / "events_bear.csv", self.result_dir / "episode_summary_bear.csv"]:
            try: p.unlink()
            except FileNotFoundError: pass
        self.ev_logger = EventLogger(self.result_dir / "events_bear.csv")
        self.ep_writer = EpisodeSummaryWriter(self.result_dir / "episode_summary_bear.csv")
        self.agg = SummaryAggregator()

        self.disjoint_mode = (disjoint_mode or "EDGE").upper()
        self.K_cand_max = int(K_cand_max)
        self.N_min = max(2, int(N_min)); self.N_max = max(self.N_min, int(N_max))
        self.beam_size = int(beam_size)

        # 中央 bandit
        self.bandit_N = DiscreteArmBandit(list(range(self.N_min, self.N_max + 1)), gamma=bandit_gamma)
        self.bandit_alpha = WeightedAlphaBandit(init_temp=alpha_temp0, init_lmbd=alpha_lmbd0, lr=alpha_lr)

        # 奖励权重
        self.w_place, self.w_fo, self.w_rel, self.w_cost, self.w_lat = reward_weights

        # 配额
        self.qm = QuotaManager(N_min=self.N_min, N_max=self.N_max, smooth_tau=0.9)
        self._last_summary = None
        self._mode = "eval"

    # ---------- 候选 ----------
    def _enumerate_sorted(self, src: int, dst: int) -> List[Dict[str, Any]]:
        cands = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
        good = [c for c in cands if c.get("ok", True) and isinstance(c.get("path", None), list)]
        good.sort(key=_cand_weight)
        return good

    # ---------- place 核心 ----------
    def place_request(self, req_raw) -> Dict[str, Any]:
        req = _std_request(req_raw)
        sid, src, dst, L, bw, ttl, t_arr = req["sid"], req["src"], req["dst"], req["L"], req["bw"], req["ttl"], req["t_arrive"]

        cands = self._enumerate_sorted(src, dst)
        if not cands:
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-FULL", "sid": sid,
                "success": 0, "reason": "no_candidates",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 0, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "bear_alpha_temp": "", "bear_alpha_lambda": "", "bear_alpha_max": "", "bear_N": 0,
                "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 中央选择 N
        N, idxN, pN = self.bandit_N.select()
        # α（来自候选 weight）
        weights = np.array([_cand_weight(c) for c in cands], dtype=np.float64)
        alpha, meta_alpha = self.bandit_alpha.suggest(weights)

        # 边缘：束搜索基于 α 的分数（也可混入 1/weight）
        scores = alpha * (1.0 / (weights + 1e-6))
        chosen = beam_search_disjoint_paths(cands, scores, N, self.disjoint_mode, beam_size=self.beam_size)

        # 遇到不够不相交时，可尝试自适应增大 N（向下回退：先 N，再 N-1，最后 N=2）
        tried_N = [N]
        if chosen is None:
            for trial_N in list(range(min(N+1, self.N_max), self.N_min-1, -1)):
                if trial_N in tried_N: continue
                chosen = beam_search_disjoint_paths(cands, scores, trial_N, self.disjoint_mode, beam_size=self.beam_size)
                tried_N.append(trial_N)
                if chosen is not None:
                    N = trial_N
                    break

        if chosen is None or len(chosen) < 2:
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-FULL", "sid": sid,
                "success": 0, "reason": "not_enough_disjoint_paths",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "bear_alpha_temp": meta_alpha["alpha_temp"],
                "bear_alpha_lambda": meta_alpha["alpha_lambda"],
                "bear_alpha_max": meta_alpha["alpha_max"],
                "bear_N": N,
                "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            # bandit 对失败也要学习（弱负奖励）
            if self._mode == "train":
                self._bandit_feedback(idxN, pN, place_succ=0, fo_succ=0, rel=0.0, cost=0.0, lat=0.0)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 将 N 条按权重/时延排序，取前 N-1 作为 active，余下 1 条为 backup
        chosen_sorted = sorted(chosen, key=lambda p: len(p))
        active_paths = chosen_sorted[:max(1, N-1)]
        backup_path = chosen_sorted[max(1, N-1)]

        bw_each = bw / float(max(1, len(active_paths)))
        bundle = list(active_paths) + [backup_path]
        if not self.env.check_paths_feasible(bundle, bw_each):
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-FULL", "sid": sid,
                "success": 0, "reason": "insufficient_capacity",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "bear_alpha_temp": meta_alpha["alpha_temp"],
                "bear_alpha_lambda": meta_alpha["alpha_lambda"],
                "bear_alpha_max": meta_alpha["alpha_max"],
                "bear_N": N,
                "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            if self._mode == "train":
                self._bandit_feedback(idxN, pN, place_succ=0, fo_succ=0, rel=0.0, cost=0.0, lat=0.0)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        res = self.env.reserve_equal_split(
            sid=sid,
            paths_active=[list(p) for p in active_paths],
            path_backup=list(backup_path),
            bw_each=bw_each,
            L=L, ttl=ttl
        )
        if not res.get("success", 0):
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-FULL", "sid": sid,
                "success": 0, "reason": "reserve_failed",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "bear_alpha_temp": meta_alpha["alpha_temp"],
                "bear_alpha_lambda": meta_alpha["alpha_lambda"],
                "bear_alpha_max": meta_alpha["alpha_max"],
                "bear_N": N,
                "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            if self._mode == "train":
                self._bandit_feedback(idxN, pN, place_succ=0, fo_succ=0, rel=0.0, cost=0.0, lat=0.0)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        rel_pred = float(self.env.predict_reliability(active_paths, backup_path, L))
        weak_cnt = self._plan_weak_replicas(req, active_paths)

        ev = {
            "t": t_arr, "event": "place", "method": "BEAR-FULL", "sid": sid,
            "success": 1, "reason": "",
            "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
            "latency_ms": float(res.get("latency_ms", 0.0)),
            "cost_total": float(res.get("cost_total", 0.0)),
            "cost_bw": float(res.get("cost_bw", 0.0)),
            "cost_cpu": float(res.get("cost_cpu", 0.0)),
            "emp_reli_pred": rel_pred,
            "fail_idx": "", "new_active": "",
            "bear_alpha_temp": meta_alpha["alpha_temp"],
            "bear_alpha_lambda": meta_alpha["alpha_lambda"],
            "bear_alpha_max": meta_alpha["alpha_max"],
            "bear_N": N,
            "weak_replicas_cnt": int(weak_cnt),
        }
        self.ev_logger.log(ev); self.agg.ingest(ev)
        self.qm.consume_for_request(used_paths=len(active_paths), placed=True)

        # 训练时给中央层一个即时奖励（鼓励成功/低成本/低延迟/高可靠）
        if self._mode == "train":
            rew = self._instant_reward(place_succ=1,
                                       fo_succ=None,  # failover 在运行期再反馈
                                       rel=rel_pred,
                                       cost=float(res.get("cost_total", 0.0)),
                                       lat=float(res.get("latency_ms", 0.0)))
            self._bandit_feedback(idxN, pN, place_succ=1, fo_succ=None, rel=rel_pred,
                                  cost=res.get("cost_total", 0.0), lat=res.get("latency_ms", 0.0),
                                  extra_reward=rew)
        return ev

    # ---------- 弱副本（元信息，不耗资源） ----------
    def _plan_weak_replicas(self, req: Dict[str, Any], active_paths: List[List[int]]) -> int:
        L = int(req["L"])
        if L <= 1: return 0
        if L == 2: return 1
        k1 = int(math.floor(0.25 * L))
        k2 = int(math.ceil(0.75 * L))
        return max(1, k2 - k1)

    # ---------- 运行期 ----------
    def _count_down_active_paths(self, session: Dict[str, Any]) -> int:
        if hasattr(self.env, "count_down_active_paths"):
            try: return int(self.env.count_down_active_paths(session))
            except Exception: pass
        if hasattr(self.env, "get_down_active_paths"):
            try:
                lst = self.env.get_down_active_paths(session)
                return len(lst) if lst is not None else 0
            except Exception: pass
        paths = session.get("active_set") or session.get("paths_active") or session.get("paths") or []
        if not hasattr(self.env, "is_path_up"): return -1
        cnt = 0
        for p in (paths or []):
            try:
                if not self.env.is_path_up(p): cnt += 1
            except Exception:
                continue
        return cnt

    # ---------- 奖励与 bandit 更新 ----------
    def _instant_reward(self,
                        place_succ: int,
                        fo_succ: Optional[int],
                        rel: float,
                        cost: float,
                        lat: float) -> float:
        # 规范化：部分量纲化处理（不精确，但足够引导在线学习）
        c_norm = - self.w_cost * float(cost) / 100.0
        l_norm = - self.w_lat * float(lat) / 10.0
        r = self.w_place * (1.0 if place_succ else 0.0) \
            + (self.w_fo * (1.0 if (fo_succ is True) else 0.0) if fo_succ is not None else 0.0) \
            + self.w_rel * float(rel) \
            + c_norm + l_norm
        return float(r)

    def _bandit_feedback(self,
                         idxN: int, pN: float,
                         place_succ: int,
                         fo_succ: Optional[int],
                         rel: float, cost: float, lat: float,
                         extra_reward: Optional[float] = None):
        rew = self._instant_reward(place_succ, fo_succ, rel, cost, lat)
        if extra_reward is not None:
            rew += float(extra_reward) * 0.2  # 轻量合并
        # N-bandit 更新
        self.bandit_N.update(idxN, reward=rew, p_selected=pN)
        # alpha-bandit 更新：只用标量 reward 进行温度/强度微调
        self.bandit_alpha.update(reward=rew)

    # ---------- 单轮 ----------
    def run_one_episode(self, ep_idx: int, steps: int, mode: str = "eval", fixed_N: Optional[int] = None) -> Dict[str, Any]:
        self._mode = str(mode or "eval").lower()
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset(); self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        N_ref = max(2, int(round(0.5 * (self.N_min + self.N_max))))
        self.qm.set_epoch_quota(N_ref)

        alive: Dict[int, Dict[str, Any]] = {}

        for t in range(0, steps):
            # 到达
            req = self.env.maybe_next_request(t)
            if req is not None:
                if isinstance(req, dict): req.setdefault("t_arrive", t)
                else:
                    try: setattr(req, "t_arrive", t)
                    except Exception: pass
                evp = self.place_request(req)
                if evp.get("success", 0) == 1 and hasattr(self.env, "get_session_ref"):
                    sid = int(evp["sid"])
                    try: alive[sid] = self.env.get_session_ref(sid)
                    except Exception: pass

            # 注入失效
            self.env.inject_failures(t)

            # 仅对需要切换的会话尝试 failover
            for s in list(alive.values()):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0: continue
                sid_i = int(s.get("sid", -1))

                if down_cnt >= 2:
                    evf = {
                        "t": t, "event": "failover", "method": "BEAR-FULL", "sid": sid_i,
                        "success": 0, "reason": "multi_path_down",
                        "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                        "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                        "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                        "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    }
                    self.ev_logger.log(evf); self.agg.ingest(evf)
                    # 对 central 反馈失败
                    if self._mode == "train":
                        self._bandit_feedback(idxN=0, pN=1.0, place_succ=1, fo_succ=0, rel=0.0, cost=0.0, lat=0.0)
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive.pop(sid_i, None)
                    continue

                r = self.env.try_failover(s)
                if r.get("failed", 0) == 0:
                    continue

                hit = bool(r.get("backup_hit", 0))
                evf2 = {
                    "t": t, "event": "failover", "method": "BEAR-FULL", "sid": sid_i,
                    "success": 1 if hit else 0,
                    "reason": "" if hit else "no_usable_backup",
                    "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                    "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": r.get("fail_idx", ""), "new_active": r.get("new_active", ""),
                }
                self.ev_logger.log(evf2); self.agg.ingest(evf2)

                if self._mode == "train":
                    self._bandit_feedback(idxN=0, pN=1.0, place_succ=1, fo_succ=(1 if hit else 0),
                                          rel=0.0, cost=0.0, lat=float(r.get("latency_ms", 0.0)))

                if not hit:
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive.pop(sid_i, None)

            # 释放到期
            released = self.env.release_expired(t)
            if released:
                for sid in released:
                    alive.pop(int(sid), None)

        summary = self.agg.finalize()
        self._last_summary = summary
        self.ep_writer.write(ep_idx, summary)
        return summary

    # ---------- 多轮 ----------
    def run(self, mode: str, epochs: int, steps: int, fixed_N: Optional[int] = None) -> None:
        mode = str(mode or "eval").lower()
        for ep in range(epochs):
            summ = self.run_one_episode(ep_idx=ep, steps=steps, mode=mode, fixed_N=fixed_N)
            print(f"[BEAR-FULL EP {ep:03d}] "
                  f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                  f"lat={summ['avg_latency_ms']:.3f}")
        try:
            self.ev_logger.close()
            self.ep_writer.close()
        except Exception:
            pass