# -*- coding: utf-8 -*-
from __future__ import annotations
"""
baseline_OptSEP.py
==================
Opt-SEP（Optimal SFC Embedding and Protection）基线实现（工程化版）

核心思想（与论文一致，工程化简化）：
- 在“分层图”LGG上以资源感知权重获得最小权重 SFP（RBL）；
- 若存在两条“最小权重且边不相交”的 SFP（LBC-SEP），直接作为 (pSFP,bSFP)；
- 否则触发 OPSI：枚举所有最小权重的 pSFP（k-shortest，直到出现非最小权重），
  对每条 pSFP 调用 BSI（Backup SFP Identifier）在 LGG 中寻找对应 bSFP；
  在所有 (pSFP,bSFP) 对中选资源消耗最小的一对；
- 以“专用保护（Dedicated Protection）”方式部署：主路径全量带宽、备份路径全量带宽；
- 运行期 failover：主路径失效 → 切换备份；主备同时失效 → 失败并释放。

对接需求：
- 依赖 env 的下列接口（与 PRANOS/BEAR/MP-DCBJOH 一致）：
  enumerate_candidates(src,dst,disjoint_mode,K)
  check_paths_feasible(paths,bw_each)
  reserve_equal_split(sid, paths_active, path_backup, bw_each, L, ttl)
  predict_reliability(active_paths, backup_path, L)
  try_failover(session)
  inject_failures(t)
  release_expired(t)
  get_session_ref(sid)
  is_path_up(...) / count_down_active_paths(...)（若有）
- 依赖 metrics：EventLogger / SummaryAggregator / EpisodeSummaryWriter
- 依赖 quota：QuotaManager（用于统计配额/占用，一致化口径）

备注：
- LGG/RBL/LBC-SEP/BSI 在工程上通过 env.enumerate_candidates(...) 的“特征/权重”近似实现：
  * “最小权重 SFP” == 候选集中权重最小（feats[0] 或 cost）的路径
  * “所有最小权重 SFP” == 取与最小值相同（±eps）的前 k 条（直到出现大于最小值的路径停止）
  * LBC-SEP == 在“最小权重集合”内找一对边不相交
  * BSI == 在“最小权重集合之外的近邻”或“全候选”中，寻找与该 pSFP 边不相交、附加消耗最小的 bSFP
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter
from quota import QuotaManager


# ----------------------------
# 工具：不相交判定与权重获取
# ----------------------------
def _edges_of(path: List[int]) -> List[Tuple[int, int]]:
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def _edge_disjoint(p1: List[int], p2: List[int]) -> bool:
    e1 = set(_edges_of(p1))
    e2 = set(_edges_of(p2))
    return len(e1 & e2) == 0

def _get_weight(c: Dict[str, Any]) -> float:
    """候选的“最小权重”度量：优先 feats[0]，其次 'cost'，最后按跳数。"""
    feats = c.get("feats", [])
    if isinstance(feats, list) and len(feats) >= 1:
        try:
            return float(feats[0])
        except Exception:
            pass
    if "cost" in c:
        try:
            return float(c["cost"])
        except Exception:
            pass
    path = c.get("path", [])
    return float(len(path))

def _sort_by_weight(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cands2 = [c for c in cands if c.get("ok", True) and isinstance(c.get("path", None), list)]
    cands2.sort(key=_get_weight)
    return cands2


# ----------------------------
# 核心：Opt-SEP 系统
# ----------------------------
class OptSEPSystem:
    """
    与 runner 对接的封装：
      - run_one_episode(ep_idx, steps, mode, fixed_N=None) -> summary
      - run(mode, epochs, steps, fixed_N=None)

    关键超参：
      - disjoint_mode：生成候选时的约束（这里对 DP 保护使用“边不相交”判断）
      - K_cand_max：枚举候选上限（含最短与近似最短）
      - k_shortest_cap：用于“最小权重集合”的上限（直到出现非最小为止）
      - eps_equal：判断“与最小权重相等”的容差
    """
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 64,
                 k_shortest_cap: int = 32,
                 eps_equal: float = 1e-6):
        self.env = env
        self.result_dir = Path(result_dir); self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)

        # 覆盖写 CSV（保证每次运行重置）
        for f in [self.result_dir / "events_bear.csv", self.result_dir / "episode_summary_bear.csv"]:
            try: f.unlink()
            except FileNotFoundError: pass

        self.ev_logger = EventLogger(self.result_dir / "events_bear.csv")
        self.ep_writer = EpisodeSummaryWriter(self.result_dir / "episode_summary_bear.csv")
        self.agg = SummaryAggregator()

        self.disjoint_mode = (disjoint_mode or "EDGE").upper()
        self.K_cand_max = int(K_cand_max)
        self.k_shortest_cap = int(k_shortest_cap)
        self.eps_equal = float(eps_equal)

        # 与既有统计口径一致（Opt-SEP 固定 DP：N=2）
        self.qm = QuotaManager(N_min=2, N_max=2, smooth_tau=0.9)
        self._last_summary = None

    # ---------- 生成候选并划分“最小权重集合” ----------
    def _get_candidates(self, src: int, dst: int) -> List[Dict[str, Any]]:
        cands = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
        return _sort_by_weight(cands)

    def _split_least_set(self, cands_sorted: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        返回 (least_set, others)。least_set：与最小权重相等（±eps）的 k 条路径（上限 k_shortest_cap），
        others：其余候选（按权重升序）。
        """
        if not cands_sorted:
            return [], []
        w0 = _get_weight(cands_sorted[0])
        least = []
        others = []
        for c in cands_sorted:
            w = _get_weight(c)
            if len(least) < self.k_shortest_cap and math.isclose(w, w0, rel_tol=0.0, abs_tol=self.eps_equal):
                least.append(c)
            else:
                others.append(c)
        return least, others

    # ---------- LBC-SEP：在最小权重集合内找一对边不相交 ----------
    def _lbc_sep(self, least_set: List[Dict[str, Any]]) -> Optional[Tuple[List[int], List[int]]]:
        n = len(least_set)
        for i in range(n):
            pi = least_set[i].get("path", [])
            for j in range(i+1, n):
                pj = least_set[j].get("path", [])
                if _edge_disjoint(pi, pj):
                    return list(pi), list(pj)
        return None

    # ---------- BSI：给定 pSFP，在（least_set + others）中找“附加消耗最小”的 bSFP ----------
    def _bsi(self,
             pSFP: List[int],
             least_set: List[Dict[str, Any]],
             others: List[Dict[str, Any]]) -> Optional[List[int]]:
        # 优先在 least_set 中找与 pSFP 边不相交的；否则到 others 中找
        for pool in (least_set, others):
            best = None
            best_w = float("inf")
            for c in pool:
                b = c.get("path", [])
                if _edge_disjoint(pSFP, b):
                    w = _get_weight(c)
                    if w < best_w:
                        best = list(b); best_w = w
            if best is not None:
                return best
        # 实在没有 disjoint，允许退化（严格 Opt-SEP 需要 disjoint，这里返回 None）
        return None

    # ---------- 从 (least_set, others) 构造 (pSFP, bSFP) ----------
    def _build_primary_backup_pair(self,
                                   least_set: List[Dict[str, Any]],
                                   others: List[Dict[str, Any]]) -> Optional[Tuple[List[int], List[int], str]]:
        # 1) LBC-SEP：若最小集合内存在两条边不相交 → 直接使用
        lb = self._lbc_sep(least_set)
        if lb is not None:
            pSFP, bSFP = lb
            return pSFP, bSFP, "LBC-SEP"

        # 2) OPSI：对每条最小集合内的 pSFP，调用 BSI 寻找 bSFP，取“主备合计权重最小”的一对
        best_pair = None
        best_cost = float("inf")
        for c in least_set:
            p = c.get("path", [])
            b = self._bsi(p, least_set, others)
            if b is None:
                continue
            cost = _get_weight(c)  # 主路径权重
            # 为简洁起见，将 b 的权重一并计入（与论文一致为“总带宽消耗最小”）
            # 这里将备份路径的权重取自 others/least_set 中对应条目的权重
            # 若 env 的费用模型更复杂（含 CPU），最终部署时会以 env 的 cost_* 为准
            cost += _get_weight({"path": b, "feats": []})
            if cost < best_cost:
                best_cost = cost
                best_pair = (list(p), list(b))
        if best_pair is not None:
            return best_pair[0], best_pair[1], "OPSI+BSI"
        return None

    # ---------- 单请求放置 ----------
    def place_request(self, req_raw) -> Dict[str, Any]:
        # 规范化
        if isinstance(req_raw, dict):
            sid = int(req_raw["sid"]); src = int(req_raw["src"]); dst = int(req_raw["dst"])
            L = int(req_raw["L"]); bw = float(req_raw["bw"]); ttl = int(req_raw["ttl"])
            t_arr = int(req_raw.get("t_arrive", req_raw.get("t", 0)))
        else:
            sid = int(getattr(req_raw, "sid")); src = int(getattr(req_raw, "src")); dst = int(getattr(req_raw, "dst"))
            L = int(getattr(req_raw, "L")); bw = float(getattr(req_raw, "bw")); ttl = int(getattr(req_raw, "ttl"))
            t_arr = int(getattr(req_raw, "t_arrive", getattr(req_raw, "t", 0)))

        # 候选与“最小权重集合”
        cand_sorted = self._get_candidates(src, dst)
        if not cand_sorted:
            ev = {
                "t": t_arr, "event": "place", "method": "Opt-SEP", "sid": sid,
                "success": 0, "reason": "no_candidates",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "optsep_stage": "none"
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        least_set, others = self._split_least_set(cand_sorted)
        pair = self._build_primary_backup_pair(least_set, others)
        if pair is None:
            ev = {
                "t": t_arr, "event": "place", "method": "Opt-SEP", "sid": sid,
                "success": 0, "reason": "no_disjoint_pair",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "optsep_stage": "fail_LBC_and_OPSI"
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        pSFP, bSFP, stage = pair

        # 资源可行性（DP：主备各预留 bw）
        bundle = [pSFP, bSFP]
        if not self.env.check_paths_feasible(bundle, bw):
            ev = {
                "t": t_arr, "event": "place", "method": "Opt-SEP", "sid": sid,
                "success": 0, "reason": "insufficient_capacity",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 2,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "optsep_stage": stage
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 预留（equal_split: N=2 => bw_each = bw）
        res = self.env.reserve_equal_split(
            sid=sid,
            paths_active=[pSFP],
            path_backup=bSFP,
            bw_each=bw,
            L=L,
            ttl=ttl
        )
        if not res.get("success", 0):
            ev = {
                "t": t_arr, "event": "place", "method": "Opt-SEP", "sid": sid,
                "success": 0, "reason": "reserve_failed",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 2,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "optsep_stage": stage
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        rel_pred = float(self.env.predict_reliability([pSFP], bSFP, L))
        ev = {
            "t": t_arr, "event": "place", "method": "Opt-SEP", "sid": sid,
            "success": 1, "reason": "",
            "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 2,
            "latency_ms": float(res.get("latency_ms", 0.0)),
            "cost_total": float(res.get("cost_total", 0.0)),
            "cost_bw": float(res.get("cost_bw", 0.0)),
            "cost_cpu": float(res.get("cost_cpu", 0.0)),
            "emp_reli_pred": rel_pred,
            "fail_idx": "", "new_active": "",
            "optsep_stage": stage
        }
        self.ev_logger.log(ev); self.agg.ingest(ev)
        self.qm.consume_for_request(used_paths=1, placed=True)
        return ev

    # ---------- failover 统计辅助 ----------
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
        # 回退：逐条判定
        paths = session.get("active_set") or session.get("paths_active") or session.get("paths") or []
        if not hasattr(self.env, "is_path_up"):  # 没有接口
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
        self.agg.reset()
        self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        # 配额（Opt-SEP 为 DP）：N_ref=2
        self.qm.set_epoch_quota(2)

        alive = {}

        for t in range(0, steps):
            # 到达 → 放置
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

            # 仅对“需要切换”的会话尝试 failover
            for s in list(alive.values()):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:
                    continue
                sid_i = int(s.get("sid", -1))

                # 主备同时 down：直接失败并释放
                if down_cnt >= 2:
                    evf = {
                        "t": t, "event": "failover", "method": "Opt-SEP", "sid": sid_i,
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

                # 恰好 1 条在用 down：尝试切到备份
                r = self.env.try_failover(s)
                if r.get("failed", 0) == 0:
                    continue  # 实际无失败（保护）

                evf2 = {
                    "t": t, "event": "failover", "method": "Opt-SEP", "sid": sid_i,
                    "success": 1 if r.get("backup_hit", 0) else 0,
                    "reason": "" if r.get("backup_hit", 0) else "no_usable_backup",
                    "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                    "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": r.get("fail_idx", ""), "new_active": r.get("new_active", ""),
                }
                self.ev_logger.log(evf2); self.agg.ingest(evf2)

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
            print(f"[Opt-SEP EP {ep:03d}] "
                  f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                  f"lat={summ['avg_latency_ms']:.3f}")
        try:
            self.ev_logger.close()
            self.ep_writer.close()
        except Exception:
            pass