# -*- coding: utf-8 -*-
from __future__ import annotations
"""
baseline_MPD_DCBJOH.py
======================
MP-DCBJOH（Multipath Protection + DCBJOH）基线实现
- 对每个 SFC 请求，从候选路径集中自适应选择 h∈[2, k_r] 条路径
  其中 (h-1) 为工作路径、1 条为共享备份路径；
- 带宽与计算按 Δr/(h-1) 等份在每条路径与其 VNF 上预留；
- 路径间按 disjoint_mode（推荐 "DZ"/"ZONE"）两两不相交；
- 放置成功后写事件与汇总；运行阶段按既有口径尝试 failover：
  * 单条工作路径宕机 → 切换至共享备份（fo_hit）；
  * 同时 ≥2 条工作路径宕机 → 直接判失败并释放（fo_miss）。

依赖：env.py / metrics.py / quota.py 与项目现有实现保持一致。
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import itertools
import time
import numpy as np

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter
from quota import QuotaManager


# ----------------------------
# 基础工具：不相交判定 / 评分等
# ----------------------------
def _paths_disjoint(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode in ("EDGE", "LINK"):
        e1 = {(p1[i], p1[i+1]) for i in range(len(p1)-1)}
        e2 = {(p2[i], p2[i+1]) for i in range(len(p2)-1)}
        return len(e1 & e2) == 0
    # NODE/DZ/ZONE：中间节点不交
    s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
    return len(s1 & s2) == 0


def _set_disjoint_ok(paths: List[List[int]], mode: str = "EDGE") -> bool:
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            if not _paths_disjoint(paths[i], paths[j], mode):
                return False
    return True


def _score_path(cand: Dict[str, Any]) -> float:
    """用于排序候选路径（越小越好）。优先用 feats[0] 视作时延；无则按跳数。"""
    feats = cand.get("feats", [])
    if isinstance(feats, list) and len(feats) >= 1:
        try:
            return float(feats[0])
        except Exception:
            pass
    path = cand.get("path", [])
    return float(len(path))  # 替代：路径长度


def _std_request(req) -> Dict[str, Any]:
    """兼容 dict / 对象，将请求规范化为字典。"""
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
# DCBJOH 主体
# ----------------------------
class MPDCBJOHSystem:
    """
    与 runner 对接的系统封装：
      - run_one_episode(ep_idx, steps, mode, fixed_N=None) -> summary
      - run(mode, epochs, steps, fixed_N=None)

    关键参数：
      - disjoint_mode: 不相交模式，推荐 "DZ"/"ZONE"；也支持 "EDGE"/"NODE"
      - K_cand_max: 枚举候选数量上限（与 env.enumerate_candidates 对齐）
      - k_r: 每个请求最多使用的总路径数（含备份）。h 从 2..k_r 自适应
      - rank_policy: 候选路径排序依据（"latency" 或 "len"）
    """
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "DZ",
                 K_cand_max: int = 32,
                 N_min: int = 2, N_max: int = 5,
                 k_r: int = 4,
                 rank_policy: str = "latency"):
        self.env = env
        self.result_dir = Path(result_dir); self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)

        # 覆盖写 CSV（与既有口径一致）
        for _p in [self.result_dir / "events_bear.csv", self.result_dir / "episode_summary_bear.csv"]:
            try: _p.unlink()
            except FileNotFoundError: pass
        self.ev_logger = EventLogger(self.result_dir / "events_bear.csv")
        self.ep_writer = EpisodeSummaryWriter(self.result_dir / "episode_summary_bear.csv")
        self.agg = SummaryAggregator()

        self.disjoint_mode = (disjoint_mode or "DZ").upper()
        self.K_cand_max = int(K_cand_max)
        self.k_r = max(2, int(k_r))
        self.rank_policy = str(rank_policy).lower()

        # 与现框架一致的配额管理（用于统计未用配额等，可选）
        self.qm = QuotaManager(N_min=N_min, N_max=N_max, smooth_tau=0.9)
        self._last_summary = None

    # ---------- 关键内部：候选集构建 ----------
    def _enumerate_sorted_candidates(self, src: int, dst: int) -> List[Dict[str, Any]]:
        """
        从 env 获取候选路径（已按 disjoint_mode 约束产生或原始候选），
        再按评分排序，过滤 ok==False 的项。
        """
        cand = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
        good = [c for c in cand if c.get("ok", True)]
        if self.rank_policy == "len":
            good.sort(key=lambda c: float(len(c.get("path", []))))
        else:
            good.sort(key=_score_path)  # 默认按估计时延
        return good

    # ---------- 关键内部：组合与校核 ----------
    def _try_place_with_h(self,
                           req: Dict[str, Any],
                           cand_sorted: List[Dict[str, Any]],
                           h: int) -> Optional[Dict[str, Any]]:
        """
        尝试用 h 条路径（h-1 工作 + 1 备份）进行放置。
        成功返回事件字典（并已写日志）；失败返回 None。
        """
        sid, src, dst, L, bw, ttl, t_arr = req["sid"], req["src"], req["dst"], req["L"], req["bw"], req["ttl"], req["t_arrive"]
        if h < 2:
            return None

        # 先从候选中挑选 (h-1) 条两两不相交的工作路径（贪心 + 组合回退）
        # 为控制复杂度，先用贪心生成一批工作集候选，再为每个工作集合挑 1 条与其不相交的备份。
        work_sets: List[List[List[int]]] = []

        # 简单贪心：从头扫描，尽量选不相交
        def greedy_pick(k_needed: int) -> Optional[List[List[int]]]:
            chosen = []
            for c in cand_sorted:
                p = c.get("path", [])
                if not chosen or all(_paths_disjoint(p, q, self.disjoint_mode) for q in chosen):
                    chosen.append(p)
                    if len(chosen) >= k_needed:
                        return chosen
            return None

        g = greedy_pick(h - 1)
        if g is not None:
            work_sets.append(g)

        # 若贪心失败或过少，尝试小规模组合（最多取前 M 条以控爆炸）
        M = min(len(cand_sorted), 12)  # 控制组合复杂度
        base_paths = [cand_sorted[i].get("path", []) for i in range(M)]
        for comb in itertools.combinations(range(M), h - 1):
            paths = [base_paths[i] for i in comb]
            if _set_disjoint_ok(paths, self.disjoint_mode):
                # 去重
                if all(any(paths[j] != x[j] for j in range(len(paths))) for x in work_sets):
                    work_sets.append(paths)
            if len(work_sets) >= 8:  # 限制工作集数量，防止爆炸
                break

        if not work_sets:
            return None

        # 为每个工作集挑 1 条备份，与所有工作路径均不相交
        for act in work_sets:
            # 先行计算每条路径需预留带宽
            bw_each = bw / float(h - 1)
            # 候选备份路径从 cand_sorted 扫描
            for c in cand_sorted:
                bak = c.get("path", [])
                if all(_paths_disjoint(bak, w, self.disjoint_mode) for w in act):
                    bundle = act + [bak]
                    # 容量可行性校核（链路与节点资源）
                    if not self.env.check_paths_feasible(bundle, bw_each):
                        continue
                    # 预留（原子提交）：等份到 (h-1) 条工作 + 1 条备份
                    res = self.env.reserve_equal_split(
                        sid=sid,
                        paths_active=[list(p) for p in act],
                        path_backup=list(bak),
                        bw_each=bw_each,
                        L=L, ttl=ttl
                    )
                    if not res.get("success", 0):
                        continue

                    # 成功：记录事件
                    rel_pred = float(self.env.predict_reliability(act, bak, L))
                    ev = {
                        "t": t_arr, "event": "place", "method": "MP-DCBJOH", "sid": sid,
                        "success": 1, "reason": "",
                        "src": src, "dst": dst, "L": L, "bw": bw, "N": h,
                        "num_paths": len(act) + 1,
                        "latency_ms": float(res.get("latency_ms", 0.0)),
                        "cost_total": float(res.get("cost_total", 0.0)),
                        "cost_bw": float(res.get("cost_bw", 0.0)),
                        "cost_cpu": float(res.get("cost_cpu", 0.0)),
                        "emp_reli_pred": rel_pred,
                        "fail_idx": "", "new_active": "",
                    }
                    self.ev_logger.log(ev)
                    self.agg.ingest(ev)
                    # 记配额（用于统计未用配额比等）——工作路径数量视为“使用的配额”
                    self.qm.consume_for_request(used_paths=len(act), placed=True)
                    return ev

        # 若所有工作集均无可用备份或预留失败
        return None

    # ---------- 单请求放置入口 ----------
    def place_request(self, req_raw) -> Dict[str, Any]:
        req = _std_request(req_raw)
        sid, src, dst, L, bw, ttl, t_arr = req["sid"], req["src"], req["dst"], req["L"], req["bw"], req["ttl"], req["t_arrive"]

        # 枚举并排序候选
        cand_sorted = self._enumerate_sorted_candidates(src, dst)
        if not cand_sorted:
            ev = {
                "t": t_arr, "event": "place", "method": "MP-DCBJOH", "sid": sid,
                "success": 0, "reason": "no_candidates",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 0, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 自适应 h：从 2 .. k_r 逐级尝试，一旦成功即返回
        for h in range(2, self.k_r + 1):
            ev = self._try_place_with_h(req, cand_sorted, h=h)
            if ev is not None:
                return ev

        # 全部失败
        ev = {
            "t": t_arr, "event": "place", "method": "MP-DCBJOH", "sid": sid,
            "success": 0, "reason": "no_feasible_bundle",
            "src": src, "dst": dst, "L": L, "bw": bw, "N": 0, "num_paths": 0,
            "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
            "emp_reli_pred": "", "fail_idx": "", "new_active": "",
        }
        self.ev_logger.log(ev); self.agg.ingest(ev)
        self.qm.consume_for_request(used_paths=0, placed=False)
        return ev

    # ---------- Failover 处理 ----------
    def _count_down_active_paths(self, session: Dict[str, Any]) -> int:
        """
        统计该会话当前“在用”路径中有多少条处于 down。
        优先用 env 的接口；否则回退逐路径检查。
        """
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

    # ---------- 单轮（episode） ----------
    def run_one_episode(self, ep_idx: int, steps: int, mode: str = "eval", fixed_N: Optional[int] = None) -> Dict[str, Any]:
        """
        运行一轮：
          - t=0..steps-1：到达→放置；失效注入→仅对需要切换的会话尝试 failover；
            多路同时 down 直接判失败并释放；未命中备份立即释放；
          - 释放到期；写汇总。
        """
        # 1) reset 与元信息
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset()
        self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        # 2) 设定本轮名义 N（仅用于配额统计/日志显示，真实 h 自适应）
        #    取 min(max(2, N_min), k_r) 做一个参考
        N_ref = min(max(2, self.qm.N_min), self.k_r)
        self.qm.set_epoch_quota(N_ref)

        # 3) 主循环
        alive_sessions: Dict[int, Dict[str, Any]] = {}

        for t in range(0, steps):
            # 3.1 到达并放置
            req = self.env.maybe_next_request(t)
            if req is not None:
                # 标注到达时间（便于事件一致）
                if isinstance(req, dict):
                    req.setdefault("t_arrive", t)
                else:
                    try: setattr(req, "t_arrive", t)
                    except Exception: pass

                ev = self.place_request(req)
                if ev.get("success", 0) == 1 and hasattr(self.env, "get_session_ref"):
                    sid = int(ev["sid"])
                    try:
                        alive_sessions[sid] = self.env.get_session_ref(sid)
                    except Exception:
                        pass

            # 3.2 注入失效
            self.env.inject_failures(t)

            # 3.3 仅对“需要切换”的会话尝试 failover
            sess_list = list(alive_sessions.values())
            for s in (sess_list or []):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:
                    continue
                sid_i = int(s.get("sid", -1))

                # 多条在用同时 down：直接判失败并释放
                if down_cnt >= 2:
                    ev_fail = {
                        "t": t, "event": "failover", "method": "MP-DCBJOH", "sid": sid_i,
                        "success": 0, "reason": "multi_path_down",
                        "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                        "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                        "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                        "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    }
                    self.ev_logger.log(ev_fail); self.agg.ingest(ev_fail)
                    try:
                        if hasattr(self.env, "release_session"):
                            self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)
                    continue

                # 恰好 1 条在用 down：尝试切到共享备份
                r = self.env.try_failover(s)
                if r.get("failed", 0) == 0:
                    # 无失败，不产生事件
                    continue

                ev = {
                    "t": t, "event": "failover", "method": "MP-DCBJOH", "sid": sid_i,
                    "success": 1 if r.get("backup_hit", 0) else 0,
                    "reason": "" if r.get("backup_hit", 0) else "no_usable_backup",
                    "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                    "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": r.get("fail_idx", ""), "new_active": r.get("new_active", ""),
                }
                self.ev_logger.log(ev); self.agg.ingest(ev)

                # 未命中：立即释放，避免后续重复尝试
                if int(ev["success"]) == 0:
                    try:
                        if hasattr(self.env, "release_session"):
                            self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)

            # 3.4 释放到期
            released = self.env.release_expired(t)
            if released:
                for sid in released:
                    alive_sessions.pop(int(sid), None)

        # 4) 汇总
        summary = self.agg.finalize()
        self._last_summary = summary
        self.ep_writer.write(ep_idx, summary)
        return summary

    # ---------- 多轮 ----------
    def run(self, mode: str, epochs: int, steps: int, fixed_N: Optional[int] = None) -> None:
        """
        统一入口：本基线仅评估（可多轮）；如需训练流程，可忽略 mode 或固定为 "eval"。
        """
        mode = str(mode or "eval").lower()
        for ep in range(epochs):
            summ = self.run_one_episode(ep_idx=ep, steps=steps, mode=mode, fixed_N=fixed_N)
            print(f"[MP-DCBJOH EP {ep:03d}] "
                  f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                  f"lat={summ['avg_latency_ms']:.3f}")

        # 关闭日志资源
        try:
            self.ev_logger.close()
            self.ep_writer.close()
        except Exception:
            pass