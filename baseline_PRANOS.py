# -*- coding: utf-8 -*-
from __future__ import annotations
"""
PRANOS Baseline (F2-LP Relaxation + Greedy Rounding) with Sliding-Window
------------------------------------------------------------------------
- 忠于论文“两阶段”思想：对批量请求先解 LP 放松（F2），再进行贪心四舍五入（不可分嵌入）；
- 集成到现有在线仿真：每个 step 注入失效，并按你现有逻辑执行 failover（增强版触发策略）；
- 与 runner 接口保持一致：
    class BaselineSystem:
        - run_one_episode(ep_idx, steps, mode, fixed_N=None) -> Dict[str, Any]
        - run(mode, epochs, steps, fixed_N=None) -> None
  输出：
    - events_bear.csv（事件流：place/failover，覆盖写）
    - episode_summary_bear.csv（每集汇总，覆盖写）
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter
from quota import QuotaManager

# --------- 统一请求字段提取：兼容 dict 与对象(SFCRequest) ---------
def _norm_req(req) -> Dict[str, Any]:
    """返回标准化字段：sid, src, dst, L, bw, ttl, t_arrive"""
    if isinstance(req, dict):
        return {
            "sid": req.get("sid"),
            "src": req.get("src"),
            "dst": req.get("dst"),
            "L":   req.get("L"),
            "bw":  req.get("bw"),
            "ttl": req.get("ttl"),
            "t_arrive": req.get("t_arrive", req.get("t", 0)),
        }
    # object-like
    return {
        "sid": getattr(req, "sid", None),
        "src": getattr(req, "src", None),
        "dst": getattr(req, "dst", None),
        "L":   getattr(req, "L", None),
        "bw":  getattr(req, "bw", None),
        "ttl": getattr(req, "ttl", None),
        "t_arrive": getattr(req, "t_arrive", getattr(req, "t", 0)),
    }

# ======================
# Failover 相关的辅助函数
# ======================
def _get_active_paths(session: Dict[str, Any]) -> List[List[int]]:
    # 常见字段名兼容
    for k in ("active_set", "paths_active", "paths", "act_paths"):
        v = session.get(k)
        if v:
            return v
    return []

def _get_backup_path(session: Dict[str, Any]):
    # 常见字段名兼容
    for k in ("path_backup", "backup_path", "bak_path", "backup"):
        v = session.get(k)
        if v:
            return v
    return None

def _try_call_env_failover(env, session, sid, backup_path) -> Dict[str, Any]:
    """
    更健壮的 failover 启动：
    - 优先使用 env.try_failover(session)
    - 若无或失败/返回空，再尝试其他可能存在的接口：
        try_failover_by_sid(sid) / switch_to_backup(session/sid) / promote_backup(...) / failover(...)
    - 最后：若都没有而 backup_path 处于 up，则记录一个“逻辑命中”并返回统一格式，
      但不会篡改 env 内部状态（此分支仅作为万不得已的记录；正常应依赖 env 接口完成切换）。
    返回统一字典格式，至少包含：{"backup_hit": 0/1, "latency_ms": float, "fail_idx": ?, "new_active": ?}
    """
    # 1) 标准接口
    if hasattr(env, "try_failover"):
        r = env.try_failover(session)
        if isinstance(r, dict):
            return r

    # 2) 备选接口
    for fn_name in ("try_failover_by_sid", "switch_to_backup", "promote_backup", "failover"):
        if hasattr(env, fn_name):
            fn = getattr(env, fn_name)
            try:
                if fn_name in ("try_failover_by_sid",):
                    r = fn(sid)
                else:
                    # 常见签名：fn(session) 或 fn(sid)
                    try:
                        r = fn(session)
                    except TypeError:
                        r = fn(sid)
                if isinstance(r, dict):
                    return r
            except Exception:
                pass

    # 3) 最后兜底（不更改 env 状态，只做记录）
    hit = False
    if hasattr(env, "is_path_up") and backup_path:
        try:
            hit = bool(env.is_path_up(backup_path))
        except Exception:
            hit = False
    return {
        "backup_hit": 1 if hit else 0,
        "latency_ms": 0.0,
        "fail_idx": "",
        "new_active": backup_path if hit else "",
        "_note": "fallback_record_only"  # 表示没有实际切换，仅做记录
    }

# ------------- LP 求解器（优先 pulp；若不可用，退化为 soft LP 权重） -------------
_SOLVER = None
try:
    import pulp  # type: ignore
    _SOLVER = "pulp"
except Exception:
    _SOLVER = None


# ======================
# 不相交检查工具
# ======================
def _paths_disjoint(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode == "EDGE":
        e1 = {(p1[i], p1[i+1]) for i in range(len(p1)-1)}
        e2 = {(p2[i], p2[i+1]) for i in range(len(p2)-1)}
        return len(e1 & e2) == 0
    if mode == "NODE":
        s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
        return len(s1 & s2) == 0
    # 默认 EDGE
    e1 = {(p1[i], p1[i+1]) for i in range(len(p1)-1)}
    e2 = {(p2[i], p2[i+1]) for i in range(len(p2)-1)}
    return len(e1 & e2) == 0


def _set_disjoint_ok(paths: List[List[int]], mode: str = "EDGE") -> bool:
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            if not _paths_disjoint(paths[i], paths[j], mode):
                return False
    return True


def _score_candidate(feats: List[float], path: List[int]) -> float:
    """仅用于 fallback 的排序/权重：latency/hop/cost 的线性组合；缺省退化为 hop。"""
    if feats and len(feats) >= 3:
        lat, hop, cost = float(feats[0]), float(feats[1]), float(feats[2])  # NOTE: typo fix below
        return 1.0 * lat + 0.5 * hop + 1.0 * cost
    if feats:
        lat = float(feats[0]); hop = float(len(path))
        return 1.0 * lat + 0.5 * hop
    return 0.5 * float(len(path))


# ======================
# PRANOS：单批（窗口）LP + Rounding
# ======================
class _PRANOSBatchPlanner:
    """
    对一批请求执行 LP 放松（F2）+ 贪心四舍五入。
    变量（候选路径级分数）：
      a_{r,c} ∈ [0,1]：请求 r 在候选路径 c 上的主动份额；Σ_c a_{r,c} = K_act (K_act=N-1)
      b_{r,c} ∈ [0,1]：请求 r 的备份份额；Σ_c b_{r,c} = 1
      a_{r,c} + b_{r,c} ≤ 1 保证主动与备份不重合
    容量约束：为避免大规模边约束膨胀，放在 rounding 时通过 env.check_paths_feasible 严格校验。
    目标：min Σ_{r,c}(a_{r,c} + β·b_{r,c})·cost(c)；cost(c) 用 feats[2]/lat/hop 近似。
    """
    def __init__(self, env, disjoint_mode: str, K_cand_max: int, N: int):
        self.env = env
        self.disjoint_mode = (disjoint_mode or "EDGE").upper()
        self.K_cand_max = int(K_cand_max)
        self.N = max(2, int(N))  # 至少 2（N-1 主动 + 1 备份）
        self.K_act = max(1, self.N - 1)

    def build_candidates(self, batch: List[Any]) -> Dict[int, List[Dict[str, Any]]]:
        req2cand: Dict[int, List[Dict[str, Any]]] = {}
        for req in batch:
            _r = _norm_req(req)
            sid = int(_r["sid"])
            src, dst = int(_r["src"]), int(_r["dst"])
            cands = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
            vv = []
            for c in cands:
                if not c.get("ok", True):
                    continue
                path = c.get("path", [])
                feats = c.get("feats", [])
                lat = float(feats[0]) if (isinstance(feats, list) and len(feats) >= 1) else 0.0
                edges = {(path[i], path[i+1]) for i in range(len(path)-1)}
                vv.append({"path": path, "feats": feats, "latency": lat, "edge_set": edges, "ok": True})
            req2cand[sid] = vv
        return req2cand

    # --------- LP（pulp）---------
    def _solve_lp_pulp(
        self, batch: List[Any], req2cand: Dict[int, List[Dict[str, Any]]]
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        prob = pulp.LpProblem("PRANOS_F2", pulp.LpMinimize)
        a_vars, b_vars = {}, {}

        def c_cost(cand):
            feats = cand.get("feats", [])
            if isinstance(feats, list) and len(feats) >= 3:
                return float(feats[2])
            if isinstance(feats, list) and len(feats) >= 1:
                return float(feats[0]) + 0.1 * len(cand.get("path", []))
            return 0.5 * len(cand.get("path", []))

        beta = 1.0
        obj_terms = []
        for req in batch:
            _r = _norm_req(req)
            sid = int(_r["sid"])
            cands = req2cand.get(sid, [])
            for i, cand in enumerate(cands):
                a = pulp.LpVariable(f"a_{sid}_{i}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
                b = pulp.LpVariable(f"b_{sid}_{i}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
                a_vars[(sid, i)] = a; b_vars[(sid, i)] = b
                obj_terms.append((c_cost(cand)) * (a + beta * b))

        # Σ a = K_act；Σ b = 1；a+b ≤ 1
        for req in batch:
            _r = _norm_req(req)
            sid = int(_r["sid"])
            cands = req2cand.get(sid, [])
            prob += pulp.lpSum([a_vars[(sid, i)] for i in range(len(cands))]) == float(self.K_act), f"act_{sid}"
            prob += pulp.lpSum([b_vars[(sid, i)] for i in range(len(cands))]) == 1.0, f"bak_{sid}"
            for i in range(len(cands)):
                prob += a_vars[(sid, i)] + b_vars[(sid, i)] <= 1.0, f"no_overlap_{sid}_{i}"

        prob += pulp.lpSum(obj_terms)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        a_val, b_val = {}, {}
        for k, v in a_vars.items():
            try: a_val[k] = float(v.value())
            except Exception: a_val[k] = 0.0
        for k, v in b_vars.items():
            try: b_val[k] = float(v.value())
            except Exception: b_val[k] = 0.0
        return a_val, b_val

    # --------- fallback：soft LP 权重 ---------
    def _solve_lp_fallback(
        self, batch: List[Any], req2cand: Dict[int, List[Dict[str, Any]]]
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
        a_val, b_val = {}, {}
        for req in batch:
            _r = _norm_req(req)
            sid = int(_r["sid"])
            cands = req2cand.get(sid, [])
            if not cands:
                continue
            scores = np.array([_score_candidate(c.get("feats", []), c.get("path", [])) for c in cands], dtype=np.float32)
            w = np.exp(-scores); w = w / max(1e-12, np.sum(w))
            # 主动 K_act：取最小 score 的前 K_act
            idx_sorted = np.argsort(scores)  # 从小到大（更好）
            a = np.zeros_like(w)
            top = idx_sorted[: min(len(idx_sorted), self.K_act)]
            a[top] = 1.0
            # 备份：取不在主动中的次优
            b = np.zeros_like(w)
            for i in idx_sorted:
                if a[i] == 0:
                    b[i] = 1.0
                    break
            for i in range(len(cands)):
                a_val[(sid, i)] = float(a[i]); b_val[(sid, i)] = float(b[i])
        return a_val, b_val

    def solve_and_round(self, batch: List[Any]) -> Dict[int, Dict[str, Any]]:
        req2cand = self.build_candidates(batch)
        if _SOLVER == "pulp":
            a_val, b_val = self._solve_lp_pulp(batch, req2cand)
        else:
            a_val, b_val = self._solve_lp_fallback(batch, req2cand)

        sel: Dict[int, Dict[str, Any]] = {}
        for req in batch:
            _r = _norm_req(req)
            sid = int(_r["sid"])
            cands = req2cand.get(sid, [])
            if not cands:
                sel[sid] = {"active": [], "backup": None, "diag": "no_candidates"}
                continue

            order_a = sorted(range(len(cands)), key=lambda i: a_val.get((sid, i), 0.0), reverse=True)
            order_b = sorted(range(len(cands)), key=lambda i: b_val.get((sid, i), 0.0), reverse=True)

            active_paths: List[List[int]] = []
            for i in order_a:
                if len(active_paths) >= max(1, self.K_act):
                    break
                p = cands[i]["path"]
                if all(_paths_disjoint(p, q, self.disjoint_mode) for q in active_paths):
                    active_paths.append(p)

            backup_path = None
            for i in order_b:
                p = cands[i]["path"]
                if all(_paths_disjoint(p, q, self.disjoint_mode) for q in active_paths):
                    backup_path = p
                    break

            sel[sid] = {"active": active_paths, "backup": backup_path,
                        "diag": "" if (active_paths and backup_path) else "insufficient_after_rounding"}
        return sel


# ======================
# Baseline System（接口与 runner 对齐）
# ======================
class BaselineSystem:
    """
    - 滑动窗口：收集一批请求后执行 LP + rounding 并 reserve；
    - 每步注入失效；仅对“需要切换”的会话尝试 failover；
      若 >=2 条在用路径宕机：直接 failover 失败并释放；
      若 failover 未命中：立即释放；
    - 覆盖写 events_bear.csv 与 episode_summary_bear.csv。
    """
    def __init__(self, env,
                 result_dir: str, save_dir: str,
                 disjoint_mode: str = "EDGE", K_cand_max: int = 32,
                 N_min: int = 2, N_max: int = 5,
                 window_size: int = 32):
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

        self.disjoint_mode = (disjoint_mode or "EDGE").upper()
        self.K_cand_max = int(K_cand_max)
        self.N_min, self.N_max = int(N_min), int(N_max)
        self.qm = QuotaManager(N_min=N_min, N_max=N_max, smooth_tau=0.9)
        self.window_size = int(window_size)

    def _count_down_active_paths(self, session: Dict[str, Any]) -> int:
        # 返回 -1 表示无法判定
        if hasattr(self.env, "count_down_active_paths"):
            try: return int(self.env.count_down_active_paths(session))
            except Exception: return -1
        if hasattr(self.env, "get_down_active_paths"):
            try:
                lst = self.env.get_down_active_paths(session)
                return len(lst) if lst is not None else 0
            except Exception: return -1
        paths = _get_active_paths(session)
        if not hasattr(self.env, "is_path_up"): return -1
        cnt = 0
        for p in (paths or []):
            try:
                if not self.env.is_path_up(p):
                    cnt += 1
            except Exception:
                return -1
        return cnt

    def _place_batch_from_lp(self, batch: List[Any], N: int, alive_sessions: Dict[int, Dict[str, Any]]):
        if not batch:
            return

        planner = _PRANOSBatchPlanner(self.env, self.disjoint_mode, self.K_cand_max, N=N)
        sel = planner.solve_and_round(batch)

        for req in batch:
            _r = _norm_req(req)
            sid = int(_r["sid"]); src, dst = int(_r["src"]), int(_r["dst"])
            L, bw, ttl = int(_r["L"]), float(_r["bw"]), int(_r["ttl"])
            t_arr = int(_r.get("t_arrive", 0))

            chosen = sel.get(sid, {"active": [], "backup": None})
            act = chosen.get("active") or []
            bak = chosen.get("backup", None)

            if len(act) < max(1, N-1) or bak is None:
                ev = {"t": t_arr, "event": "place", "method": "PRANOS", "sid": sid,
                      "success": 0, "reason": "select_or_rounding_failed",
                      "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                      "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                      "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                self.ev_logger.log(ev); self.agg.ingest(ev)
                self.qm.consume_for_request(used_paths=0, placed=False)
                continue

            bw_each = bw / max(1, N-1)
            paths_all = act + ([bak] if bak else [])
            if (not _set_disjoint_ok(paths_all, self.disjoint_mode)) or \
               (not self.env.check_paths_feasible(paths_all, bw_each)):
                ev = {"t": t_arr, "event": "place", "method": "PRANOS", "sid": sid,
                      "success": 0, "reason": "feasible_check_failed",
                      "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                      "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                      "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                self.ev_logger.log(ev); self.agg.ingest(ev)
                self.qm.consume_for_request(used_paths=0, placed=False)
                continue

            res = self.env.reserve_equal_split(sid=sid, paths_active=act, path_backup=bak,
                                               bw_each=bw_each, L=L, ttl=ttl)
            if not res.get("success", 0):
                ev = {"t": t_arr, "event": "place", "method": "PRANOS", "sid": sid,
                      "success": 0, "reason": "reserve_failed",
                      "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                      "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                      "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                self.ev_logger.log(ev); self.agg.ingest(ev)
                self.qm.consume_for_request(used_paths=0, placed=False)
                continue

            rel_pred = float(self.env.predict_reliability(act, bak, L))
            ev = {"t": t_arr, "event": "place", "method": "PRANOS", "sid": sid,
                  "success": 1, "reason": "",
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(act)+1,
                  "latency_ms": float(res.get("latency_ms", 0.0)),
                  "cost_total": float(res.get("cost_total", 0.0)),
                  "cost_bw": float(res.get("cost_bw", 0.0)),
                  "cost_cpu": float(res.get("cost_cpu", 0.0)),
                  "emp_reli_pred": rel_pred, "fail_idx": "", "new_active": ""}
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=max(1, N-1), placed=True)

            # 登记会话句柄（用于后续 failover）
            if hasattr(self.env, "get_session_ref"):
                try:
                    alive_sessions[sid] = self.env.get_session_ref(sid)
                except Exception:
                    pass

    def run_one_episode(self, ep_idx: int, steps: int, mode: str, fixed_N: Optional[int] = None) -> Dict[str, Any]:
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset(); self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        N = int(fixed_N if fixed_N is not None else max(2, self.qm.N_min))
        self.qm.set_epoch_quota(N)

        alive_sessions: Dict[int, Dict[str, Any]] = {}
        batch_buf: List[Any] = []

        for t in range(0, steps):
            # 1) 收集到达请求到窗口
            req = self.env.maybe_next_request(t)
            if req is not None:
                batch_buf.append(req)

            # 窗口到阈值则做一次 LP+rounding 并落地 reserve
            if len(batch_buf) >= self.window_size:
                self._place_batch_from_lp(batch_buf, N=N, alive_sessions=alive_sessions)
                batch_buf = []

            # 2) 注入失效
            self.env.inject_failures(t)

            # 2bis) 仅对需要切换的会话尝试；>=2 宕机直接失败并释放；未命中也释放
            sess_list = list(alive_sessions.values())
            for s in (sess_list or []):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:
                    continue

                sid_i = int(s.get("sid", -1))
                backup_path = _get_backup_path(s)
                active_paths = _get_active_paths(s)

                if down_cnt >= 2:
                    ev_fail = {"t": t, "event": "failover", "method": "PRANOS",
                               "sid": sid_i, "success": 0, "reason": "multi_path_down",
                               "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                               "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                               "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                               "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                    self.ev_logger.log(ev_fail); self.agg.ingest(ev_fail)
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)
                    continue

                # 单条主动宕机时：先检查是否有备份、备份是否 up
                if backup_path is None:
                    reason = "no_backup_path"
                    ev_fail = {"t": t, "event": "failover", "method": "PRANOS",
                               "sid": sid_i, "success": 0, "reason": reason,
                               "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                               "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                               "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                               "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                    self.ev_logger.log(ev_fail); self.agg.ingest(ev_fail)
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)
                    continue

                # 若 env 暴露 is_path_up，则在尝试前过滤掉明显不可用的备份
                if hasattr(self.env, "is_path_up"):
                    try:
                        if not self.env.is_path_up(backup_path):
                            reason = "backup_not_up"
                            ev_fail = {"t": t, "event": "failover", "method": "PRANOS",
                                       "sid": sid_i, "success": 0, "reason": reason,
                                       "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                                       "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                                       "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                                       "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                            self.ev_logger.log(ev_fail); self.agg.ingest(ev_fail)
                            try:
                                if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                            finally:
                                alive_sessions.pop(sid_i, None)
                            continue
                    except Exception:
                        pass  # 若无法判定，则继续尝试

                # 调用 env 的 failover 接口（多路回退）
                r = _try_call_env_failover(self.env, s, sid_i, backup_path)

                # 统一判断是否命中：兼容返回多种键名
                hit_flag = None
                for key in ("backup_hit", "fo_hit", "hit", "success"):
                    v = r.get(key, None)
                    if isinstance(v, (bool, int)):
                        hit_flag = bool(v)
                        break

                new_active = r.get("new_active") or r.get("switched_to") or r.get("new_path", "")
                fail_idx = r.get("fail_idx", r.get("failed_active_idx", ""))

                ev = {"t": t, "event": "failover", "method": "PRANOS", "sid": sid_i,
                      "success": 1 if hit_flag else 0,
                      "reason": "" if hit_flag else (r.get("reason") or r.get("_note") or "no_usable_backup"),
                      "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                      "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                      "latency_ms": float(r.get("latency_ms", 0.0)),
                      "cost_total": "", "cost_bw": "", "cost_cpu": "",
                      "emp_reli_pred": "", "fail_idx": fail_idx, "new_active": new_active}
                self.ev_logger.log(ev); self.agg.ingest(ev)

                if int(ev["success"]) == 0:
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)

            # 3) 释放到期
            released = self.env.release_expired(t)
            if released:
                for sid in released:
                    alive_sessions.pop(int(sid), None)

        # 集尾处理残余窗口
        if batch_buf:
            self._place_batch_from_lp(batch_buf, N=N, alive_sessions=alive_sessions)

        summary = self.agg.finalize()
        self.ep_writer.write(ep_idx, summary)
        return summary

    def run(self, mode: str, epochs: int, steps: int, fixed_N: Optional[int] = None) -> None:
        for ep in range(epochs):
            summ = self.run_one_episode(ep, steps, mode=mode, fixed_N=fixed_N)
            print(f"[PRANOS-LP EP {ep:03d}] place={summ['place_rate']:.3f} "
                  f"fo_hit={summ['fo_hit_rate']:.3f} emp_av={summ['emp_avail']:.3f} "
                  f"cost={summ['avg_cost_total']:.4f} lat={summ['avg_latency_ms']:.3f}")
        self.ev_logger.close(); self.ep_writer.close()