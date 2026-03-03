# -*- coding: utf-8 -*-
from __future__ import annotations
"""
baseline_BEAR.py
================
BEAR-SFC 基线（推理版）
- 中央层（Heuristic Central Policy）给出 (N, alpha) 指引
- 边缘层在候选中选择 N 条两两不相交路径：N-1 为在役，1 条为热备
- 资源预留：bw_each = bw / (N-1)，reserve_equal_split(paths_active=[...], path_backup=backup)
- 运行期 failover：单条在役 down -> 切热备；若 >=2 条同时 down -> 失败释放
- 日志/汇总：兼容现有 events_bear.csv / episode_summary_bear.csv 字段

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
import numpy as np

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter
from quota import QuotaManager


# ----------------------------
# 基础工具
# ----------------------------
def _edges_of(path: List[int]) -> List[Tuple[int, int]]:
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def _disjoint_ok(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode in ("EDGE", "LINK"):
        e1 = set(_edges_of(p1)); e2 = set(_edges_of(p2))
        return len(e1 & e2) == 0
    # NODE / DZ / ZONE：中间节点不交
    s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
    return len(s1 & s2) == 0

def _set_disjoint(paths: List[List[int]], mode: str) -> bool:
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            if not _disjoint_ok(paths[i], paths[j], mode):
                return False
    return True

def _cand_weight(c: Dict[str, Any]) -> float:
    # 候选权重：优先 feats[0]（一般为延迟/代价），否则用 hop 数
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


# ----------------------------
# BEAR 基线系统
# ----------------------------
class BEARSystem:
    """
    与 runner 对接：
      - run_one_episode(ep_idx, steps, mode, fixed_N=None) -> summary
      - run(mode, epochs, steps, fixed_N=None) -> None

    关键参数：
      - disjoint_mode: 不相交判定模式（EDGE/NODE/DZ/ZONE），与拓扑配置一致
      - K_cand_max: 候选上限（与 env.enumerate_candidates 对齐）
      - N_min, N_max: 中央层可用的 N 取值范围（N>=2）
      - alpha_temp: 将候选权重映射为 soft 分配概率时的温度（越小越偏好最优）
    """
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 32,
                 N_min: int = 2, N_max: int = 5,
                 alpha_temp: float = 0.35):
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
        self.N_min = max(2, int(N_min))
        self.N_max = max(self.N_min, int(N_max))
        self.alpha_temp = float(alpha_temp)

        # 配额口径（N 为中央层建议、在役为 N-1）
        self.qm = QuotaManager(N_min=self.N_min, N_max=self.N_max, smooth_tau=0.9)
        self._last_summary = None

    # ---------- 候选生成与排序 ----------
    def _enumerate_sorted(self, src: int, dst: int) -> List[Dict[str, Any]]:
        cands = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
        good = [c for c in cands if c.get("ok", True) and isinstance(c.get("path", None), list)]
        good.sort(key=_cand_weight)  # 权重越小越优
        return good

    # ---------- 中央层策略（启发式） ----------
    def _central_policy(self,
                        req: Dict[str, Any],
                        cand_sorted: List[Dict[str, Any]]) -> Tuple[int, np.ndarray, Dict[str, float]]:
        """
        返回： (N, alpha, meta)
        - N: 选择的路径总数（N-1 工作 + 1 备份）
        - alpha: 对候选的 soft 分配概率（长度 = len(cand_sorted)），用于边缘侧打分
        - meta: 记录用于日志的指引参数
        启发式：
          * N 与请求带宽 bw、候选数量相关，bw 越大、候选越多，N 越偏大
          * alpha 对最小权重候选更偏好（softmax(-w/T)）
        """
        bw = float(req["bw"])
        K = max(1, len(cand_sorted))
        # 以 bw 与 K 映射到 [N_min, N_max]
        # 归一化：bw_n in [0,1]（假设 0~100 为常见范围，必要时由 env 适配）
        bw_n = min(1.0, bw / 100.0)
        k_n = min(1.0, K / float(max(K, self.K_cand_max)))
        s = 0.6 * bw_n + 0.4 * k_n
        N = int(round(self.N_min + s * (self.N_max - self.N_min)))
        N = min(max(N, self.N_min), self.N_max)

        # alpha：softmax(-weight / T)，越低的 weight 概率越高
        weights = np.array([_cand_weight(c) for c in cand_sorted], dtype=np.float64)
        if len(weights) == 0:
            alpha = np.ones((1,), dtype=np.float64)
        else:
            w = weights - weights.min()
            logits = -w / max(1e-6, self.alpha_temp)
            logits -= logits.max()
            exps = np.exp(logits)
            alpha = exps / max(1e-9, exps.sum())

        meta = dict(bear_N=N, bear_alpha_temp=self.alpha_temp, bear_alpha_max=float(alpha.max()))
        return N, alpha, meta

    # ---------- 边缘层：基于 alpha 的不相交选择 ----------
    def _edge_pick_paths(self,
                         cand_sorted: List[Dict[str, Any]],
                         alpha: np.ndarray,
                         N: int) -> Optional[Tuple[List[List[int]], List[int]]]:
        """
        返回：(active_paths, backup_path)
        过程：
          1) 以 alpha 为权重的排序优先级（alpha 大者优先），
          2) 以贪心方式挑出 N 条两两不相交的端到端路径，
          3) 将其中代价最小的 N-1 条作为在役，其余 1 条作为热备。
        """
        if N < 2:
            return None
        order = np.argsort(-alpha)  # 概率从大到小
        picked: List[List[int]] = []
        for idx in order:
            p = cand_sorted[int(idx)].get("path", [])
            if not p:
                continue
            if not picked or all(_disjoint_ok(p, q, self.disjoint_mode) for q in picked):
                picked.append(list(p))
            if len(picked) >= N:
                break
        if len(picked) < N:
            return None
        # 以候选权重/跳数排序，N-1 条最优作在役，剩下 1 条作备份
        picked_sorted = sorted(picked, key=lambda x: len(x))
        active = picked_sorted[:N-1]
        backup = picked_sorted[N-1]
        return active, backup

    # ---------- 单请求放置 ----------
    def place_request(self, req_raw) -> Dict[str, Any]:
        req = _std_request(req_raw)
        sid, src, dst, L, bw, ttl, t_arr = req["sid"], req["src"], req["dst"], req["L"], req["bw"], req["ttl"], req["t_arrive"]

        cand = self._enumerate_sorted(src, dst)
        if not cand:
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-SFC", "sid": sid,
                "success": 0, "reason": "no_candidates",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 0, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "bear_alpha_temp": self.alpha_temp, "bear_alpha_max": "", "bear_N": 0,
                "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 中央策略
        N, alpha, meta = self._central_policy(req, cand)
        # 边缘选择
        choice = self._edge_pick_paths(cand, alpha, N)
        if choice is None:
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-SFC", "sid": sid,
                "success": 0, "reason": "not_enough_disjoint_paths",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                **meta, "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        active_paths, backup_path = choice
        bw_each = bw / float(max(1, N - 1))

        # 容量可行性
        bundle = list(active_paths) + [backup_path]
        if not self.env.check_paths_feasible(bundle, bw_each):
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-SFC", "sid": sid,
                "success": 0, "reason": "insufficient_capacity",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                **meta, "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 资源预留：在役 N-1 条 + 热备 1 条；bw_each= bw/(N-1)
        res = self.env.reserve_equal_split(
            sid=sid,
            paths_active=[list(p) for p in active_paths],
            path_backup=list(backup_path),
            bw_each=bw_each,
            L=L, ttl=ttl
        )
        if not res.get("success", 0):
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-SFC", "sid": sid,
                "success": 0, "reason": "reserve_failed",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                **meta, "weak_replicas_cnt": 0,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 可靠性估计（使用 env 提供的接口）
        rel_pred = float(self.env.predict_reliability(active_paths, backup_path, L))

        # 弱副本（工程占位：仅记录元信息，不消耗实际资源；后续可映射到 env 的 VNF 级 CPU 预留）
        weak_replicas_cnt = self._plan_weak_replicas(req, active_paths)

        ev = {
            "t": t_arr, "event": "place", "method": "BEAR-SFC", "sid": sid,
            "success": 1, "reason": "",
            "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
            "latency_ms": float(res.get("latency_ms", 0.0)),
            "cost_total": float(res.get("cost_total", 0.0)),
            "cost_bw": float(res.get("cost_bw", 0.0)),
            "cost_cpu": float(res.get("cost_cpu", 0.0)),
            "emp_reli_pred": rel_pred,
            "fail_idx": "", "new_active": "",
            **meta, "weak_replicas_cnt": int(weak_replicas_cnt),
        }
        self.ev_logger.log(ev); self.agg.ingest(ev)
        # 记配额：在役路径数 = N-1
        self.qm.consume_for_request(used_paths=N-1, placed=True)
        return ev

    # ---------- 弱副本：工程占位策略 ----------
    def _plan_weak_replicas(self, req: Dict[str, Any], active_paths: List[List[int]]) -> int:
        """
        返回为该会话建议部署的弱副本数量（仅记录在事件里，不占用资源）。
        一个简单策略：对 SFC 长度 L >= 3 的请求，在中间 50% 的层各放置一个弱副本。
        """
        L = int(req["L"])
        if L <= 1:
            return 0
        if L == 2:
            return 1
        k1 = int(math.floor(0.25 * L))
        k2 = int(math.ceil(0.75 * L))
        return max(1, k2 - k1)

    # ---------- failover 辅助 ----------
    def _count_down_active_paths(self, session: Dict[str, Any]) -> int:
        if hasattr(self.env, "count_down_active_paths"):
            try:
                return int(self.env.count_down_active_paths(session))
            except Exception:
                pass
        if hasattr(self.env, "get_down_active_paths"):
            try:
                lst = self.env.get_down_active_paths(session)
                return len(lst) if lst is not None else 0
            except Exception:
                pass
        # 回退：若无专用接口，则逐条检查
        paths = session.get("active_set") or session.get("paths_active") or session.get("paths") or []
        if not hasattr(self.env, "is_path_up"):
            return -1
        cnt = 0
        for p in (paths or []):
            try:
                if not self.env.is_path_up(p):
                    cnt += 1
            except Exception:
                continue
        return cnt

    # ---------- 单轮 ----------
    def run_one_episode(self, ep_idx: int, steps: int, mode: str = "eval", fixed_N: Optional[int] = None) -> Dict[str, Any]:
        # reset & 元信息
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset(); self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        # 名义 N（用于配额统计），选用 (N_min+N_max)/2
        N_ref = max(2, int(round(0.5 * (self.N_min + self.N_max))))
        self.qm.set_epoch_quota(N_ref)

        alive: Dict[int, Dict[str, Any]] = {}

        for t in range(0, steps):
            # 到达
            req = self.env.maybe_next_request(t)
            if req is not None:
                if isinstance(req, dict):
                    req.setdefault("t_arrive", t)
                else:
                    try: setattr(req, "t_arrive", t)
                    except Exception: pass
                evp = self.place_request(req)
                if evp.get("success", 0) == 1 and hasattr(self.env, "get_session_ref"):
                    sid = int(evp["sid"])
                    try:
                        alive[sid] = self.env.get_session_ref(sid)
                    except Exception:
                        pass

            # 注入失效
            self.env.inject_failures(t)

            # 仅对需要切换的会话做 failover
            for s in list(alive.values()):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:
                    continue
                sid_i = int(s.get("sid", -1))

                if down_cnt >= 2:
                    # 多条在役同时 down：直接失败并释放
                    evf = {
                        "t": t, "event": "failover", "method": "BEAR-SFC", "sid": sid_i,
                        "success": 0, "reason": "multi_path_down",
                        "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                        "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                        "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                        "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    }
                    self.ev_logger.log(evf); self.agg.ingest(evf)
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive.pop(sid_i, None)
                    continue

                # 恰好 1 条在役 down：尝试切换到热备
                r = self.env.try_failover(s)
                if r.get("failed", 0) == 0:
                    continue

                evf2 = {
                    "t": t, "event": "failover", "method": "BEAR-SFC", "sid": sid_i,
                    "success": 1 if r.get("backup_hit", 0) else 0,
                    "reason": "" if r.get("backup_hit", 0) else "no_usable_backup",
                    "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                    "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": r.get("fail_idx", ""), "new_active": r.get("new_active", ""),
                }
                self.ev_logger.log(evf2); self.agg.ingest(evf2)

                # 未命中备份：立即释放，避免重复尝试
                if int(evf2["success"]) == 0:
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive.pop(sid_i, None)

            # 释放到期
            released = self.env.release_expired(t)
            if released:
                for sid in released:
                    alive.pop(int(sid), None)

        # 汇总
        summary = self.agg.finalize()
        self._last_summary = summary
        self.ep_writer.write(ep_idx, summary)
        return summary

    # ---------- 多轮 ----------
    def run(self, mode: str, epochs: int, steps: int, fixed_N: Optional[int] = None) -> None:
        mode = str(mode or "eval").lower()
        for ep in range(epochs):
            summ = self.run_one_episode(ep_idx=ep, steps=steps, mode=mode, fixed_N=fixed_N)
            print(f"[BEAR EP {ep:03d}] "
                  f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                  f"lat={summ['avg_latency_ms']:.3f}")
        try:
            self.ev_logger.close()
            self.ep_writer.close()
        except Exception:
            pass