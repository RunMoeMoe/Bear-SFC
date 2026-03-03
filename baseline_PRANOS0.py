# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter, compute_central_reward
from quota import QuotaManager
from algo_edge import HeuristicSelector  # 仅用作候选打分的兜底
# 注：不依赖 algo_central 的 DQN；baseline 固定 N（来自 fixed_N 或配置）

# =========================
# HEU_Cost 风格启发式编排器
# =========================
class HeuCostOrchestrator:
    """
    简化的 HEU_Cost 风格启发式：
      - 对每个请求，先枚举候选路径集（按 EDGE/NODE/DZ 约束、不做全路径枚举）
      - 对候选做“成本代理 + 时延”的排序，贪心取前 N 条（N-1 在用 + 1 备用）
      - 可行性检查 + 原子资源预留（reserve_equal_split）
    说明：
      - 成本代理优先顺序：候选特征 feats 中若包含 (latency, hop, cost-like) 则加权；否则退化为路径长度与延迟加权。
      - 该启发式仅使用 env 暴露的标准接口：enumerate_candidates / check_paths_feasible / reserve_equal_split / predict_reliability
    """
    def __init__(self, env, disjoint_mode: str = "EDGE", K_cand_max: int = 32):
        self.env = env
        self.disjoint_mode = str(disjoint_mode).upper()
        self.K_cand_max = int(K_cand_max)
        self.edge_heur = HeuristicSelector()  # 仅在缺少特征时兜底

    # ------- 工具：候选打分 ------- #
    def _score_candidate(self, c: Dict[str, Any]) -> float:
        """
        返回越小越好的分数。优先使用 feats，否则退化（len(path)+lat 的线性组合）。
        约定 feats 含义（若存在）：[latency_ms, hop_cnt, cost_hint, ...]
        """
        feats = np.asarray(c.get("feats", []), dtype=np.float32)
        if feats.size >= 3:
            lat, hop, cost = float(feats[0]), float(feats[1]), float(feats[2])
            return 1.0 * lat + 0.5 * hop + 1.0 * cost
        elif feats.size >= 1:
            lat = float(feats[0]); hop = float(len(c.get("path", [])))
            return 1.0 * lat + 0.5 * hop
        else:
            hop = float(len(c.get("path", [])))
            # 如果 env 提供 path_latency，可调用：lat = env.path_latency(c["path"])
            return 0.5 * hop

    def _select_paths(self, src: int, dst: int, N: int) -> Tuple[List[List[int]], Optional[List[int]], Dict[str, Any]]:
        cand = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max)
        if not cand:
            return [], None, {"reason": "no_candidates"}

        # 过滤可用候选（ok==True），并按“打分”从小到大排序
        valid = [c for c in cand if c.get("ok", True)]
        if len(valid) < N:
            return [], None, {"reason": "insufficient_candidates"}
        valid.sort(key=self._score_candidate)

        chosen = [c["path"] for c in valid[:N]]
        paths_active = chosen[: max(1, N - 1)]
        path_backup = chosen[max(1, N - 1)] if N >= 2 else None
        return paths_active, path_backup, {"K": len(cand)}

    # ------- 放置 ------- #
    def place_one(self, request: Dict[str, Any], N: int,
                  logger: Optional[EventLogger] = None) -> Dict[str, Any]:
        # 兼容 dict/SFCRequest
        if isinstance(request, dict):
            sid, src, dst, L, bw, ttl = int(request["sid"]), int(request["src"]), int(request["dst"]), int(request["L"]), float(request["bw"]), int(request["ttl"])
            t_arr = int(request.get("t_arrive", 0))
        else:
            sid = int(getattr(request, "sid")); src = int(getattr(request, "src")); dst = int(getattr(request, "dst"))
            L = int(getattr(request, "L")); bw = float(getattr(request, "bw")); ttl = int(getattr(request, "ttl"))
            t_arr = int(getattr(request, "t_arrive", 0))

        paths_active, path_backup, diag = self._select_paths(src, dst, N)
        if len(paths_active) < max(1, N - 1) or path_backup is None:
            ev = {"t": t_arr, "event": "place", "method": "baseline_heu", "sid": sid,
                  "success": 0, "reason": diag.get("reason", "select_failed"),
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                  "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                  "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
            if logger: logger.log(ev)
            return ev

        bw_each = bw / max(1, N - 1)
        if not self.env.check_paths_feasible(paths_active + [path_backup], bw_each):
            ev = {"t": t_arr, "event": "place", "method": "baseline_heu", "sid": sid,
                  "success": 0, "reason": "insufficient_capacity",
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                  "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                  "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
            if logger: logger.log(ev)
            return ev

        res = self.env.reserve_equal_split(sid=sid, paths_active=paths_active, path_backup=path_backup,
                                           bw_each=bw_each, L=L, ttl=ttl)
        if not res.get("success", 0):
            ev = {"t": t_arr, "event": "place", "method": "baseline_heu", "sid": sid,
                  "success": 0, "reason": "reserve_failed",
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                  "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                  "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
            if logger: logger.log(ev)
            return ev

        rel_pred = float(self.env.predict_reliability(paths_active, path_backup, L))
        ev = {"t": t_arr, "event": "place", "method": "baseline_heu", "sid": sid,
              "success": 1, "reason": "",
              "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(paths_active) + 1,
              "latency_ms": float(res.get("latency_ms", 0.0)),
              "cost_total": float(res.get("cost_total", 0.0)),
              "cost_bw": float(res.get("cost_bw", 0.0)),
              "cost_cpu": float(res.get("cost_cpu", 0.0)),
              "emp_reli_pred": rel_pred, "fail_idx": "", "new_active": ""}
        if logger: logger.log(ev)
        return ev

    # ------- 故障切换（仅调用 env.try_failover；规范化事件） ------- #
    def try_failover(self, session: Dict[str, Any], t_now: int, logger: Optional[EventLogger] = None) -> Dict[str, Any]:
        res = self.env.try_failover(session)
        if res.get("failed", 0) == 0:
            return {"failed": 0}
        ev = {"t": t_now, "event": "failover", "method": "baseline_heu", "sid": int(session.get("sid", -1)),
              "success": 1 if res.get("backup_hit", 0) else 0,
              "reason": "" if res.get("backup_hit", 0) else "no_usable_backup",
              "src": session.get("src", ""), "dst": session.get("dst", ""), "L": session.get("L", ""),
              "bw": session.get("bw", ""), "N": session.get("N", ""), "num_paths": session.get("num_paths", ""),
              "latency_ms": float(res.get("latency_ms", 0.0)),
              "cost_total": "", "cost_bw": "", "cost_cpu": "",
              "emp_reli_pred": "", "fail_idx": res.get("fail_idx", ""), "new_active": res.get("new_active", "")}
        if logger: logger.log(ev)
        return ev


# =========================
# Baseline System（可直接在 runner 调用）
# =========================
class BaselineSystem:
    """
    运行模式与 bear.System 对齐：
      - mode = "baseline_heu"（唯一模式）
      - 每集 steps 时间步：到达→放置→注入失效→（仅对需要切换的会话）failover→释放→汇总
    训练项：不包含 DQN/ppo，保持固定 N（通过 fixed_N 或者 N_min）
    """
    def __init__(self, env,
                 result_dir: str, save_dir: str,
                 disjoint_mode: str = "EDGE", K_cand_max: int = 32,
                 N_min: int = 2, N_max: int = 5):
        self.env = env
        self.result_dir = Path(result_dir); self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)

        # 覆盖写 CSV
        for _p in [self.result_dir / "events_bear.csv", self.result_dir / "episode_summary_bear.csv"]:
            try: _p.unlink()
            except FileNotFoundError: pass

        self.ev_logger = EventLogger(self.result_dir / "events_bear.csv")
        self.ep_writer = EpisodeSummaryWriter(self.result_dir / "episode_summary_bear.csv")
        self.agg = SummaryAggregator()
        self.orch = HeuCostOrchestrator(env, disjoint_mode=disjoint_mode, K_cand_max=K_cand_max)
        self.qm = QuotaManager(N_min=N_min, N_max=N_max, smooth_tau=0.9)
        self._last_summary: Optional[Dict[str, Any]] = None

    # 与 bear.py 相同的“需要 failover 的判断”，保证行为一致
    def _count_down_active_paths(self, session: Dict[str, Any]) -> int:
        if hasattr(self.env, "count_down_active_paths"):
            try: return int(self.env.count_down_active_paths(session))
            except Exception: pass
        if hasattr(self.env, "get_down_active_paths"):
            try:
                lst = self.env.get_down_active_paths(session)
                return len(lst) if lst is not None else 0
            except Exception: pass
        paths = (session.get("active_set") or session.get("paths_active") or session.get("paths") or [])
        if not hasattr(self.env, "is_path_up"): return -1
        cnt = 0
        for p in (paths or []):
            try: up = bool(self.env.is_path_up(p))
            except Exception: continue
            if not up: cnt += 1
        return cnt

    def run_one_episode(self, ep_idx: int, steps: int, mode: str, fixed_N: Optional[int] = None) -> Dict[str, Any]:
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset(); self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        N = int(fixed_N if fixed_N is not None else max(2, self.qm.N_min))
        self.qm.set_epoch_quota(N)
        alive_sessions: Dict[int, Dict[str, Any]] = {}

        for t in range(0, steps):
            # 1) 到达->放置
            req = self.env.maybe_next_request(t)
            if req is not None:
                ev = self.orch.place_one(req, N=N, logger=self.ev_logger)
                self.agg.ingest(ev)
                used_paths = int(ev.get("num_paths", 0))
                self.qm.consume_for_request(used_paths=used_paths, placed=(ev.get("success", 0) == 1))
                if ev.get("success", 0) == 1 and hasattr(self.env, "get_session_ref"):
                    sid = int(ev["sid"]); alive_sessions[sid] = self.env.get_session_ref(sid)

            # 2) 失效->仅对需要切换的会话做 failover；多条宕机直接失败并释放
            self.env.inject_failures(t)
            sess_list = list(alive_sessions.values()) if alive_sessions else getattr(self.env, "get_all_sessions", lambda: [])()
            for s in (sess_list or []):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:  # 无需切换
                    continue
                sid_i = int(s.get("sid", -1))
                if down_cnt >= 2:
                    ev_fail = {"t": t, "event": "failover", "method": "baseline_heu",
                               "sid": sid_i, "success": 0, "reason": "multi_path_down",
                               "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                               "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                               "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                               "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                    if self.ev_logger: self.ev_logger.log(ev_fail)
                    self.agg.ingest(ev_fail)
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)
                    continue

                r = self.orch.try_failover(s, t_now=t, logger=self.ev_logger)
                if isinstance(r, dict) and str(r.get("event", "")).lower() in ("failover","fo","fail-over","switch","switchover","switch_over"):
                    if "success" not in r and "hit" in r:
                        r["success"] = 1 if (r.get("hit") in (1, True, "1", "true", "True")) else 0
                    self.agg.ingest(r)
                    if int(r.get("success", 0) or 0) == 0:  # 未命中备用 -> 立即释放清理
                        try:
                            if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                        finally:
                            alive_sessions.pop(sid_i, None)

            # 3) 释放到期
            released = self.env.release_expired(t)
            if released:
                for sid in released:
                    alive_sessions.pop(int(sid), None)

        # 4) 汇总
        summary = self.agg.finalize(); self._last_summary = summary
        self.ep_writer.write(ep_idx, summary)
        return summary

    def run(self, mode: str, epochs: int, steps: int, fixed_N: Optional[int] = None) -> None:
        mode = "baseline_heu"  # baseline 固定模式
        for ep in range(epochs):
            summ = self.run_one_episode(ep, steps, mode=mode, fixed_N=fixed_N)
            print(f"[BASELINE EP {ep:03d}] place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} lat={summ['avg_latency_ms']:.3f}")
        self.ev_logger.close(); self.ep_writer.close()