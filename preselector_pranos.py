# -*- coding: utf-8 -*-
from __future__ import annotations
"""
PRANOS-style Preselector (Sliding-window F2-LP + Greedy Rounding)
-----------------------------------------------------------------
- 在到达侧以“滑动窗口”批量收集请求；
- 对窗口内请求先做 LP 放松（F2 近似）得到 per-request 的路径权重，然后做贪心 rounding；
- 对每个请求导出至多 top_k 组“可行候选组合”（N-1 主动 + 1 备份，互不相交、容量可行）；
- 由上层（BEAR-SFC）在相同观察状态下对候选组合再做二次决策，以提高 place_rate。
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# -----------------------------
# 通用工具
# -----------------------------
def _paths_disjoint(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode == "EDGE":
        e1 = {(p1[i], p1[i+1]) for i in range(len(p1)-1)}
        e2 = {(p2[i], p2[i+1]) for i in range(len(p2)-1)}
        return len(e1 & e2) == 0
    s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
    return len(s1 & s2) == 0

def _set_disjoint_ok(paths: List[List[int]], mode: str = "EDGE") -> bool:
    for i in range(len(paths)):
        for j in range(i+1, len(paths)):
            if not _paths_disjoint(paths[i], paths[j], mode):
                return False
    return True

def _norm_req(req) -> Dict[str, Any]:
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
    return {
        "sid": getattr(req, "sid", None),
        "src": getattr(req, "src", None),
        "dst": getattr(req, "dst", None),
        "L":   getattr(req, "L", None),
        "bw":  getattr(req, "bw", None),
        "ttl": getattr(req, "ttl", None),
        "t_arrive": getattr(req, "t_arrive", getattr(req, "t", 0)),
    }

def _score_candidate_for_fallback(feats: List[float], path: List[int]) -> float:
    if feats and len(feats) >= 3:
        lat, hop, cost = float(feats[0]), float(feats[1]), float(feats[2])
        return 1.0 * lat + 0.5 * hop + 1.0 * cost
    if feats:
        lat = float(feats[0]); hop = float(len(path))
        return 1.0 * lat + 0.5 * hop
    return 0.5 * float(len(path))


# -----------------------------
# LP 依赖：pulp（可选）
# -----------------------------
_SOLVER = None
try:
    import pulp  # type: ignore
    _SOLVER = "pulp"
except Exception:
    _SOLVER = None


# -----------------------------
# 预筛选器主体
# -----------------------------
class PRANOSPreselector:
    """
    使用方式：
        pre = PRANOSPreselector(env, disjoint_mode="EDGE", window_size=32, K_cand_max=32, top_k=4)
        pre.feed(req)              # 每个 step 把 maybe_next_request(t) 的返回喂入
        if pre.ready():
            pool = pre.run(N)      # 返回 {sid: {"request":std_req, "combos":[...]}}
            system.inject_candidate_pool(pool)   # 在 BEAR 内部消费该候选池

    pool 结构：
        { sid: {
            "request": {...标准化请求...},
            "combos": [
               {"active":[path,..], "backup":path, "bw_each":float, "lat_est":float,
                "cost_est":float, "score":float, "ok":1},
               ...
            ]
        }, ... }
    """
    def __init__(self,
                 env,
                 disjoint_mode: str = "EDGE",
                 window_size: int = 32,
                 K_cand_max: int = 32,
                 top_k: int = 4):
        self.env = env
        self.disjoint_mode = (disjoint_mode or "EDGE").upper()
        self.window_size = int(window_size)
        self.K_cand_max = int(K_cand_max)
        self.top_k = int(top_k)
        self._buf: List[Any] = []

    # ---------- 外部接口 ----------
    def feed(self, req) -> None:
        if req is not None:
            self._buf.append(req)

    def ready(self) -> bool:
        return len(self._buf) >= self.window_size

    def size(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf = []

    def run(self, N: int) -> Dict[int, Dict[str, Any]]:
        batch = self._buf
        self._buf = []
        if not batch:
            return {}
        req2cand = self._build_candidates(batch)
        if _SOLVER == "pulp":
            a_val, b_val, sid_order = self._solve_lp_pulp(batch, req2cand, N)
        else:
            a_val, b_val, sid_order = self._solve_soft_weights(batch, req2cand, N)
        pool = self._round_and_make_combos(batch, req2cand, a_val, b_val, N, self.top_k)
        out: Dict[int, Dict[str, Any]] = {}
        for req in batch:
            r = _norm_req(req); sid = int(r["sid"])
            out[sid] = {"request": r, "combos": pool.get(sid, [])}
        return out

    # ---------- 内部：候选枚举 ----------
    def _build_candidates(self, batch: List[Any]) -> Dict[int, List[Dict[str, Any]]]:
        req2cand: Dict[int, List[Dict[str, Any]]] = {}
        for req in batch:
            r = _norm_req(req)
            sid = int(r["sid"]); src, dst = int(r["src"]), int(r["dst"])
            cands = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
            vv = []
            for c in cands:
                if not c.get("ok", True):
                    continue
                path = c.get("path", [])
                feats = c.get("feats", [])
                lat = float(feats[0]) if (isinstance(feats, list) and len(feats) >= 1) else 0.0
                vv.append({"path": path, "feats": feats, "latency": lat})
            req2cand[sid] = vv
        return req2cand

    # ---------- 内部：LP 求解 ----------
    def _solve_lp_pulp(self, batch: List[Any], req2cand: Dict[int, List[Dict[str, Any]]], N: int
                       ) -> Tuple[Dict[Tuple[int,int], float], Dict[Tuple[int,int], float], List[int]]:
        import pulp  # type: ignore
        K_act = max(1, int(N) - 1)
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
        sid_order: List[int] = []
        for req in batch:
            r = _norm_req(req); sid = int(r["sid"])
            sid_order.append(sid)
            cands = req2cand.get(sid, [])
            for i, cand in enumerate(cands):
                a = pulp.LpVariable(f"a_{sid}_{i}", lowBound=0.0, upBound=1.0)
                b = pulp.LpVariable(f"b_{sid}_{i}", lowBound=0.0, upBound=1.0)
                a_vars[(sid, i)] = a; b_vars[(sid, i)] = b
                obj_terms.append((c_cost(cand)) * (a + beta * b))

        for req in batch:
            r = _norm_req(req); sid = int(r["sid"])
            cands = req2cand.get(sid, [])
            prob += pulp.lpSum([a_vars[(sid, i)] for i in range(len(cands))]) == float(K_act), f"act_{sid}"
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
        return a_val, b_val, sid_order

    def _solve_soft_weights(self, batch: List[Any], req2cand: Dict[int, List[Dict[str, Any]]], N: int
                            ) -> Tuple[Dict[Tuple[int,int], float], Dict[Tuple[int,int], float], List[int]]:
        K_act = max(1, int(N) - 1)
        a_val, b_val = {}, {}
        sid_order: List[int] = []
        for req in batch:
            r = _norm_req(req); sid = int(r["sid"])
            sid_order.append(sid)
            cands = req2cand.get(sid, [])
            if not cands: continue
            scores = np.array([_score_candidate_for_fallback(c.get("feats", []), c.get("path", [])) for c in cands], dtype=np.float32)
            idx = np.argsort(scores)  # 小优
            for i, _ in enumerate(cands):
                a_val[(sid, i)] = 0.0
                b_val[(sid, i)] = 0.0
            for j in idx[: min(len(idx), K_act)]:
                a_val[(sid, j)] = 1.0
            for j in idx:
                if a_val[(sid, j)] == 0.0:
                    b_val[(sid, j)] = 1.0
                    break
        return a_val, b_val, sid_order

    # ---------- 内部：rounding + 生成组合 ----------
    def _round_and_make_combos(self,
                               batch: List[Any],
                               req2cand: Dict[int, List[Dict[str, Any]]],
                               a_val: Dict[Tuple[int,int], float],
                               b_val: Dict[Tuple[int,int], float],
                               N: int,
                               top_k: int) -> Dict[int, List[Dict[str, Any]]]:
        K_act = max(1, int(N) - 1)
        out: Dict[int, List[Dict[str, Any]]] = {}

        for req in batch:
            r = _norm_req(req); sid = int(r["sid"]); bw = float(r["bw"])
            cands = req2cand.get(sid, [])
            if not cands:
                out[sid] = []
                continue

            ord_a = sorted(range(len(cands)), key=lambda i: a_val.get((sid, i), 0.0), reverse=True)
            ord_b = sorted(range(len(cands)), key=lambda i: b_val.get((sid, i), 0.0), reverse=True)

            active_sets: List[List[List[int]]] = []
            start_ids = ord_a[: min(len(ord_a), top_k * 2)]
            for _ in start_ids:
                cur: List[List[int]] = []
                used = set()
                for i in ord_a:
                    if len(cur) >= K_act: break
                    p = cands[i]["path"]
                    if not cur or all(_paths_disjoint(p, q, self.disjoint_mode) for q in cur):
                        sig = tuple(p)
                        if sig in used: continue
                        cur.append(p); used.add(sig)
                if len(cur) == K_act:
                    # 去重
                    if all(any(cur[j] != x[j] for j in range(K_act)) for x in active_sets):
                        active_sets.append(cur)

            combos: List[Dict[str, Any]] = []
            bw_each = bw / float(K_act)
            for act in active_sets:
                bak_list = []
                for i in ord_b:
                    p = cands[i]["path"]
                    if all(_paths_disjoint(p, q, self.disjoint_mode) for q in act):
                        bak_list.append(p)
                    if len(bak_list) >= top_k:
                        break
                for bak in bak_list:
                    bundle = act + [bak]
                    if not self.env.check_paths_feasible(bundle, bw_each):
                        continue
                    # 估算指标（可选）
                    lat_est = 0.0
                    try:
                        if hasattr(self.env, "estimate_latency"):
                            lat_est = float(self.env.estimate_latency(act, bak))
                        else:
                            lat_est = float(sum(len(p) for p in act) + len(bak))
                    except Exception:
                        lat_est = float(sum(len(p) for p in act) + len(bak))
                    combos.append({
                        "active": [list(p) for p in act],
                        "backup": list(bak),
                        "bw_each": float(bw_each),
                        "lat_est": float(lat_est),
                        "cost_est": 0.0,
                        "score": float(len(bak) + sum(len(p) for p in act)),
                        "ok": 1
                    })
            out[sid] = combos[: top_k] if len(combos) > top_k else combos
        return out