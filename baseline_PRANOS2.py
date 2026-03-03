# baseline_PRANOS.py
# -*- coding: utf-8 -*-
"""
PRANOS (Practical Near-Optimal SFC Deployment) —— 轻量工程化基线
- 与现有 env/config 完全兼容
- 采用等分带宽 (N-1 条在用 + 1 条备份) 的容错模型
- 路径选择：候选枚举 + 打分 + 贪心互斥（EDGE/NODE/DZ 不相交）
- 资源预留：调用 env.reserve_equal_split()（与 BEAR 一致）
- 记录：写入 /result 下的 events 与 metrics（字段与 BEAR 保持一致）
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import math
import time

import numpy as np

import csv

# --------- 统一请求字段提取：兼容 dict 与对象(SFCRequest) ---------
def _norm_req(req):
    """
    返回标准化字段字典：sid, src, dst, L, bw, ttl, t_arrive
    支持 dict 或 具备属性的对象（如 SFCRequest）。
    """
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

# --------- 统一请求字段提取：兼容 dict 与对象(SFCRequest) ---------
def _norm_req(req):
    """
    返回标准化字段字典：sid, src, dst, L, bw, ttl, t_arrive
    支持 dict 或 具备属性的对象（如 SFCRequest）。
    """
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

class _SimpleCSVLogger:
    """Very small CSV appender used locally to avoid extra dependencies."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t","algo","event","success","sid","src","dst","bw_req","L","N","num_paths","lat_ms","cost","reason"])
    def log(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row.get("t",""),
                row.get("algo","pranos"),
                row.get("event",""),
                row.get("success",""),
                row.get("sid",""),
                row.get("src",""),
                row.get("dst",""),
                row.get("bw_req",""),
                row.get("L",""),
                row.get("N",""),
                row.get("num_paths",""),
                row.get("lat_ms",""),
                row.get("cost",""),
                row.get("reason",""),
            ])

class _SimpleMetrics:
    """Append per-episode summary to a CSV."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["algo","ep","place_rate","fo_hit","emp_av","avg_cost","avg_lat","attempted","placed","fo_cnt"])
    def add_row(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                row.get("algo","pranos"),
                row.get("ep",""),
                row.get("place_rate",""),
                row.get("fo_hit",""),
                row.get("emp_av",""),
                row.get("avg_cost",""),
                row.get("avg_lat",""),
                row.get("attempted",""),
                row.get("placed",""),
                row.get("fo_cnt",""),
            ])
    def flush(self): 
        # no-op, kept for API parity
        pass


class PRANOSBaseline:
    def __init__(
        self,
        env,
        result_dir: str,
        save_dir: str,
        disjoint_mode: str = "EDGE",
        K_cand_max: int = 16,
        N_fixed: Optional[int] = None,
    ):
        """
        env: 你的 SFCEnv
        out_dir: 输出目录（形如 runs/.../bear-YYYYmmdd-hhMMss）
        disjoint_mode: "EDGE"/"NODE"/"DZ"
        K_cand_max: 候选路径数
        N_fixed: 固定 N（若为 None，取 env.cfg 或 hrl.central.N_min）
        """
        self.env = env
        self.disjoint_mode = disjoint_mode
        self.K_cand_max = int(K_cand_max)
        self.N_fixed = int(N_fixed) if N_fixed is not None else None

        # 输出到 /result 子目录，文件名以 pranos_ 开头
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.ev_path = self.result_dir / "events_pranos.csv"
        self.mt_path = self.result_dir / "metrics_pranos.csv"
        self.ev_logger = _SimpleCSVLogger(self.ev_path)
        self.mt_agg = _SimpleMetrics(self.mt_path)

        # 成本参数（与 env 保持一致）
        self.edge_unit_cost = float(getattr(env, "edge_unit_cost", 1.0))
        self.lambda_backup = float(getattr(env, "lambda_backup", 1.0))
        self.cpu_per_vnf = float(getattr(env, "cpu_per_vnf", 1.0))
        self.cpu_per_backup = float(getattr(env, "cpu_per_backup", 1.0))

        # 中央 N 的默认取值（若未固定）
        if self.N_fixed is None:
            try:
                self.N_fixed = int(getattr(env, "N_min", 3))
            except Exception:
                self.N_fixed = 3

    # --------- 路径评分：越大越好 ----------
    def _score_path(self, path: List[int], bw_req: float, L: int) -> float:
        """
        依据：残余带宽/链路长度/节点CPU裕度/延迟
        """
        G = self.env.G
        # 1) 带宽余量（沿途最小可用带宽 / 需求）
        bw_min = float("inf")
        hop_len = 0.0
        tx_ms = float(getattr(self.env, "tx_delay_per_hop", 0.2))
        prop_km_ms = float(getattr(self.env, "prop_delay_per_km", 0.02))

        # 链路部分
        for u, v in zip(path, path[1:]):
            e = G[u][v]
            cap = float(e.get("bw_capacity", 1.0))
            used_work = float(e.get("bw_used_work", 0.0))
            used_bk = float(e.get("bw_used_bk_resv", 0.0))
            avail = max(0.0, cap - used_work - used_bk)
            bw_min = min(bw_min, avail)
            dist = float(e.get("dist_km", 1.0))
            hop_len += (tx_ms + prop_km_ms * dist)

        # 2) 节点 CPU 裕度（沿途 VNF 部署点的大致裕度，粗略取路径上所有点平均）
        cpu_avgs = []
        for n in path:
            nd = G.nodes[n]
            cap = float(nd.get("cpu_capacity", 0.0))
            used = float(nd.get("cpu_used", 0.0))
            cpu_avgs.append(max(0.0, cap - used))
        cpu_mean = (sum(cpu_avgs) / max(1, len(cpu_avgs))) if cpu_avgs else 0.0

        # 3) 宽度系数/延迟惩罚
        width_ok = (bw_min / max(1e-9, bw_req))
        latency_penalty = 1.0 / (1.0 + hop_len)  # 时延越小越好

        # 4) 归一合成
        #   宽度（0~+∞）→ clip
        w = min(2.0, width_ok)
        #   CPU 裕度规模化一下
        c = math.log1p(cpu_mean)
        #   组合
        score = w * 0.6 + c * 0.3 + latency_penalty * 0.1
        return score

    def _greedy_pick_disjoint(
        self, paths: List[List[int]], scores: List[float], need: int
    ) -> List[List[int]]:
        """
        从候选里按分数高到低，选互不相交的若干条。
        支持 EDGE/NODE/DZ 三类分离（由 env.check_paths_feasible 兜底）
        """
        order = np.argsort(scores)[::-1]
        chosen = []
        for idx in order:
            p = paths[idx]
            # 与已选保持“尽量不相交”，最后仍由 env.check_paths_feasible 严格把关
            conflict = False
            if self.disjoint_mode in ("EDGE", "NODE"):
                for q in chosen:
                    if self.disjoint_mode == "EDGE":
                        # 边相交判定
                        s1 = set(zip(p, p[1:]))
                        s2 = set(zip(q, q[1:]))
                        if s1 & s2:
                            conflict = True
                            break
                    else:
                        # 节点相交判定
                        if set(p[1:-1]) & set(q[1:-1]):
                            conflict = True
                            break
            # DZ 分离/或无法在此处判定，则先不拒；让后续 feasibility 再过滤
            if not conflict:
                chosen.append(p)
                if len(chosen) >= need:
                    break
        return chosen

    # --------- 单请求放置 ----------
    def place_one(self, req) -> Dict[str, Any]:
        """
        输入：SFCRequest
        输出：plan dict（用于事件/指标记录）
        """
        src, dst, L, bw, sid = req.src, req.dst, req.L, req.bw, req.sid

        # 候选路径
        cand = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max)
        if not cand:
            return {"success": 0, "reason": "no_candidate", "sid": sid, "src": src, "dst": dst, "bw": bw, "L": L}

        # 打分
        scores = [self._score_path(p, bw, L) for p in cand]

        # 选择 N 条：前 N-1 在用 + 1条备份
        N = max(3, int(self.N_fixed))  # 至少 3：2条在用 + 1条备份
        chosen = self._greedy_pick_disjoint(cand, scores, need=N)

        if len(chosen) < N:
            return {"success": 0, "reason": "not_enough_disjoint", "sid": sid, "src": src, "dst": dst, "bw": bw, "L": L}

        paths_active, path_backup = chosen[:-1], chosen[-1]

        # 等分带宽 & 资源可行性检查
        bw_each = bw / (N - 1)               # 每条在用分到的带宽
        bw_bk   = bw_each                    # 备份预留的带宽
        ok = self.env.check_paths_feasible(paths_active + [path_backup], bw_each)
        if not ok:
            return {"success": 0, "reason": "insufficient_capacity", "sid": sid, "src": src, "dst": dst, "bw": bw, "L": L}

        # 资源预留（工作带宽 + 备份带宽 + VNF CPU）
        reserve = self.env.reserve_equal_split(
            sid=sid,
            src=src, dst=dst, L=L,
            paths_active=paths_active,
            path_backup=path_backup,
            bw_each=bw_each,
            bw_backup=bw_bk,
            cpu_per_vnf=self.cpu_per_vnf,
            cpu_per_backup=self.cpu_per_backup
        )
        if not reserve.get("success", False):
            return {"success": 0, "reason": "reserve_failed", "sid": sid, "src": src, "dst": dst, "bw": bw, "L": L}

        # 记录成功
        plan = {
            "success": 1,
            "sid": sid, "src": src, "dst": dst, "bw": bw, "L": L,
            "N": N,
            "paths_active": paths_active, "path_backup": path_backup,
            "bw_each": bw_each, "bw_bk": bw_bk,
            "lat_ms": self.env.latency_equal_split(paths_active, L=L),  # 取在用路径的最大时延
            "cost_total": reserve.get("cost_total", 0.0),
            "session": {
                "sid": sid, "src": src, "dst": dst, "bw_req": bw, "L": L,
                "paths_active": paths_active, "path_backup": path_backup
            }
        }
        return plan

    # --------- 单轮 episode ----------
    def run_one_episode(self, ep: int, steps: int) -> Dict[str, float]:
        placed = 0
        attempted = 0
        fo_cnt = 0
        fo_miss = 0
        lat_sum = 0.0
        cost_sum = 0.0

        for t in range(steps):
            req = self.env.maybe_next_request(t)
            # 先释放 TTL 到期
            self.env.release_expired(t)

            if req is None:
                continue

            attempted += 1
            plan = self.place_one(req)
            if plan.get("success", 0) == 1:
                placed += 1
                lat_sum += float(plan.get("lat_ms", 0.0))
                cost_sum += float(plan.get("cost_total", 0.0))
                # 事件日志：place
                self.ev_logger.log({
                    "t": t, "algo": "pranos", "event": "place", "success": 1,
                    "sid": plan["sid"], "src": plan["src"], "dst": plan["dst"],
                    "bw_req": plan["bw"], "L": plan["L"],
                    "N": plan["N"], "num_paths": len(plan["paths_active"])+1,
                    "lat_ms": plan["lat_ms"], "cost": plan["cost_total"], "reason": ""
                })

                # 失效尝试与 failover（若 env 在内部定期注入失效则此处只需尝试）
                sref = plan["session"]
                res = self.env.try_failover(sref)
                if res.get("failed", False):
                    fo_cnt += 1
                    hit = 1 if res.get("backup_hit", False) else 0
                    if hit == 0:
                        fo_miss += 1
                    self.ev_logger.log({
                        "t": t, "algo": "pranos", "event": "failover", "success": hit,
                        "sid": plan["sid"], "src": plan["src"], "dst": plan["dst"],
                        "bw_req": plan["bw"], "L": plan["L"],
                        "N": plan["N"], "num_paths": len(plan["paths_active"])+1,
                        "lat_ms": "", "cost": "", "reason": ("" if hit else "no_usable_backup")
                    })
            else:
                # 放置失败
                self.ev_logger.log({
                    "t": t, "algo": "pranos", "event": "place", "success": 0,
                    "sid": plan.get("sid"), "src": plan.get("src"), "dst": plan.get("dst"),
                    "bw_req": plan.get("bw"), "L": plan.get("L"),
                    "N": "", "num_paths": "", "lat_ms": "", "cost": "",
                    "reason": plan.get("reason", "unknown")
                })

        # 汇总
        avg_lat = (lat_sum / max(1, placed))
        emp_av = (placed - fo_miss) / max(1, placed)  # 经验可用性
        fo_hit = (0.0 if fo_cnt == 0 else (1 - fo_miss/fo_cnt))
        avg_cost = cost_sum / max(1, placed)

        summ = {
            "ep": ep,
            "place_rate": placed / max(1, attempted),
            "fo_hit": fo_hit,
            "emp_av": emp_av,
            "avg_cost": avg_cost,
            "avg_lat": avg_lat,
            "attempted": attempted,
            "placed": placed,
            "fo_cnt": fo_cnt
        }
        self.mt_agg.add_row({"algo":"pranos", **summ})
        print(f"[PRANOS EP {ep:03d}] place={summ['place_rate']:.3f} fo_hit={summ['fo_hit']:.3f} "
              f"emp_av={summ['emp_av']:.3f} cost={summ['avg_cost']:.4f} lat={summ['avg_lat']:.3f}")
        return summ

    # --------- 多轮 ----------
    def run(self, epochs: int, steps: int):
        wall = time.time()
        for ep in range(epochs):
            self.env.reset_episode(seed_offset=ep)
            self.run_one_episode(ep, steps)
        self.mt_agg.flush()
        print(f"[PRANOS] done, wall={time.time()-wall:.2f}s -> {self.res_dir}")