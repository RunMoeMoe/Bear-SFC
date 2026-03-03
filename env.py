from __future__ import annotations
"""
（唯一真源）
•	全局图 G：
•	边属性：bw_cap, dist_km, up∈{0,1}, bw_used（这套方案不再分 work/bk，两者等价）
•	点属性：cpu_cap, up∈{0,1}, cpu_used, dz_id
•	资源 API（原子、对称）
•	can_reserve_path_bw(path, bw) / reserve_path_bw(path, bw) / release_path_bw(path, bw)
•	can_reserve_cpu(node, units) / reserve_cpu(node, units) / release_cpu(node, units)
•	支持批量原子：失败则整批回滚。
•	路径引擎：k_shortest、disjoint_paths(order in {EDGE,NODE,DZ})、rank_paths(...)
•	失效注入：inject_failures(t) 更新 up 标志；is_path_up(path)
•	时延：latency(path) = Σ(α·dist_km + β/跳) + γ·VNF处理
•	成本与可靠性计算
"""
# env.py
# -*- coding: utf-8 -*-
"""
环境真源（唯一状态源）：
- 维护 NetworkX 拓扑图 G：边/点资源与 up/down 状态，带宽与 CPU 使用量
- 候选路径生成：k 最短、按 EDGE/NODE/DZ 过滤不相交
- 资源原子预留/释放：带宽、CPU（失败回滚保证对称性）
- 失效注入与恢复：按配置随机失效并在指定时间后恢复
- 代价/时延/可靠性：与论文一致的等份负载(N-1)+1 设计（每条路径预留 M/(N-1)）
- 请求流生成：泊松到达、持续 TTL、SFC 长度采样、带宽采样
- 评估/训练所需的状态导出钩子：central/edge state
- 适配器 SFCEnv：构造签名与 runner.py 对齐，并转发 bear.py 所需接口
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import random
import numpy as np
import networkx as nx


# ===================== 数据结构 =====================

@dataclass
class SFCRequest:
    sid: int
    t_arrive: int
    src: int
    dst: int
    L: int                 # VNF 数（SFC 长度）
    bw: float              # 业务带宽需求 M
    ttl: int               # 存活时长（步）
    vnf_seq: Optional[List[int]] = None  # 如需具体功能类型，可后续扩展


@dataclass
class ActiveSession:
    """等份负载 N-1 + 1 热备 模式下的会话记账"""
    sid: int
    src: int
    dst: int
    L: int
    bw_each: float                 # 每条路径预留的带宽 M/(N-1)
    N: int                         # 选定路径数
    paths: List[List[int]]         # N 条路径（全部等额预留）
    active_idx: List[int]          # 当前承载的 N-1 条路径索引
    standby_idx: Optional[int]     # 热备路径索引（命中后置空）
    vnf_nodes_per_path: List[List[int]]  # 每条路径上 L 个 VNF 所在节点（由上层挑选后传入）
    t_expire: int                  # 过期时间步


@dataclass
class SessionCPUBook:
    """会话级 CPU 账本，便于精确释放"""
    per_node: Dict[int, float]


# ===================== 工具 =====================

def _product(xs):
    v = 1.0
    for x in xs:
        v *= x
    return v


# ===================== 环境实现 =====================

class Env:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed", 42))
        self._sid = 0

        # ---------- 拓扑 ----------
        self.G = self._build_topology(cfg["topology"], cfg["node"])
        self.disjoint_mode = cfg["topology"].get("disjoint", "EDGE").upper()
        self.k_paths = int(cfg["topology"].get("k_paths", 10))

        # ---------- VNF / 处理时延 / 成本参数 ----------
        sfc_cfg = cfg.get("sfc", {})
        self.cpu_per_vnf = float(sfc_cfg.get("cpu_per_vnf", 1.0))
        self.cpu_per_backup = float(sfc_cfg.get("cpu_per_backup", 0.5))  # 热备降配
        self.vnf_proc_ms = float(sfc_cfg.get("proc_ms", 1.0))

        link = cfg["topology"]["link"]
        self.prop_ms_per_km = float(link.get("prop_delay_per_km", 0.02))
        self.tx_ms_per_hop = float(link.get("tx_delay_per_hop", 0.2))
        self.link_cfg = link

        cost_cfg = cfg.get("cost", {})
        self.cost_lambda_bw = float(cost_cfg.get("lambda_bw", 1.0))
        self.cost_lambda_cpu = float(cost_cfg.get("lambda_cpu", 1.0))

        # ---------- 失效参数 ----------
        self.fail_cfg = cfg.get("failures", {})
        self.node_fail_p = float(self.fail_cfg.get("node_fail_prob", 0.0))
        self.edge_fail_p = float(self.fail_cfg.get("edge_fail_prob", 0.0))
        self.recovery_time = int(self.fail_cfg.get("recovery_time", 200))

        # ---------- 流量 / SFC 生成 ----------
        traffic = cfg.get("traffic", {})
        self.arrival_rate = float(traffic.get("arrival_rate", 0.15))
        self.dur_mean = int(traffic.get("duration_mean", 220))
        self.dur_std = int(traffic.get("duration_std", 60))
        self.bw_mean = float(traffic.get("bw_demand_mean", 0.18))
        self.bw_std = float(traffic.get("bw_demand_std", 0.03))
        self.sfc_len_choices = list(traffic.get("sfc_length_choices", [2, 3, 4, 5]))
        self.sfc_len_probs = list(traffic.get("sfc_length_probs", [0.1, 0.4, 0.3, 0.2]))

        # ---------- 会话与计时 ----------
        self.t = 0
        self.active: Dict[int, ActiveSession] = {}   # sid -> session
        self._cpu_books: Dict[int, Dict[int, float]] = {}
        self._episode_meta: Dict[str, Any] = {}      # 分组统计标签

        self.reset_episode()

    # ----------------- Topology -----------------

    def _build_topology(self, topo: Dict[str, Any], node_cfg: Dict[str, Any]) -> nx.Graph:
        ttype = topo.get("type", "synthetic")
        if ttype != "synthetic":
            raise NotImplementedError("only 'synthetic' is implemented in this env")

        n = int(topo.get("num_nodes", 40))
        m = int(topo.get("num_edges", 120))
        dist_mean = float(topo["link"].get("dist_mean", 50.0))
        dist_std = float(topo["link"].get("dist_std", 10.0))
        bw_cap = float(topo["link"].get("bw_capacity", 5.0))

        G = nx.gnm_random_graph(n, m, seed=self.cfg.get("seed", 42), directed=False)

        # 边属性
        for u, v in G.edges():
            d = max(1.0, self.rng.gauss(dist_mean, dist_std))
            G[u][v]["dist_km"] = d
            G[u][v]["bw_cap"] = bw_cap
            G[u][v]["bw_used"] = 0.0
            G[u][v]["up"] = True
            G[u][v]["down_until"] = -1
            G[u][v]["dz_id"] = 0  # 若后续需要 DZ 区分，可在构建后写入

        # 点属性
        cpu_cap = float(topo.get("cpu_total_per_node", node_cfg.get("cpu_capacity", 16.0)))
        for u in G.nodes():
            G.nodes[u]["cpu_cap"] = cpu_cap
            G.nodes[u]["cpu_used"] = 0.0
            G.nodes[u]["up"] = True
            G.nodes[u]["down_until"] = -1
            G.nodes[u]["dz_id"] = 0

        return G

    # ----------------- Episode Control -----------------

    def reset_episode(self) -> None:
        """清空使用量与 up/down 状态，清空活跃会话，复位时钟"""
        self.t = 0
        for u, v in self.G.edges():
            self.G[u][v]["bw_used"] = 0.0
            self.G[u][v]["up"] = True
            self.G[u][v]["down_until"] = -1
        for u in self.G.nodes():
            self.G.nodes[u]["cpu_used"] = 0.0
            self.G.nodes[u]["up"] = True
            self.G.nodes[u]["down_until"] = -1
        self.active.clear()
        self._cpu_books.clear()

    def sample_scenario_for_episode(self) -> None:
        """按需变更场景元信息（用于分组统计）；此处仅写入元标签"""
        self._episode_meta = {
            "topo_size": f"N{self.G.number_of_nodes()}-E{self.G.number_of_edges()}",
            "sfc_len_group": "mixed",
        }

    def episode_meta(self) -> Dict[str, Any]:
        return dict(self._episode_meta)

    # ----------------- Requests -----------------

    def _new_sid(self) -> int:
        self._sid += 1
        return self._sid

    def maybe_next_request(self, t: int) -> Optional[SFCRequest]:
        """泊松近似：以 arrival_rate 为“每步到达概率”"""
        self.t = t
        if self.rng.random() > self.arrival_rate:
            return None

        src, dst = self._rand_two_nodes()
        L = int(np.random.choice(self.sfc_len_choices, p=self.sfc_len_probs))
        bw = max(0.05, float(self.rng.gauss(self.bw_mean, self.bw_std)))
        ttl = max(50, int(self.rng.gauss(self.dur_mean, self.dur_std)))

        return SFCRequest(
            sid=self._new_sid(),
            t_arrive=t,
            src=src, dst=dst,
            L=L, bw=bw, ttl=ttl, vnf_seq=None
        )

    def _rand_two_nodes(self) -> Tuple[int, int]:
        n = self.G.number_of_nodes()
        a, b = self.rng.randrange(n), self.rng.randrange(n)
        while b == a:
            b = self.rng.randrange(n)
        return a, b

    # ----------------- Failures -----------------

    def inject_failures(self, t: int) -> None:
        """按配置随机失效，并在 down_until 到期后恢复"""
        self.t = t

        # 恢复
        for u, v in self.G.edges():
            if (not self.G[u][v]["up"]) and t >= self.G[u][v]["down_until"]:
                self.G[u][v]["up"] = True
        for u in self.G.nodes():
            if (not self.G.nodes[u]["up"]) and t >= self.G.nodes[u]["down_until"]:
                self.G.nodes[u]["up"] = True

        # 新失效
        for u, v in self.G.edges():
            if self.rng.random() < self.edge_fail_p and self.G[u][v]["up"]:
                self.G[u][v]["up"] = False
                self.G[u][v]["down_until"] = t + self.recovery_time
        for u in self.G.nodes():
            if self.rng.random() < self.node_fail_p and self.G.nodes[u]["up"]:
                self.G.nodes[u]["up"] = False
                self.G.nodes[u]["down_until"] = t + self.recovery_time

    def is_path_up(self, path: List[int]) -> bool:
        """路径上所有节点与边均 up 才算 up"""
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            if (not self.G.nodes[a]["up"]) or (not self.G.nodes[b]["up"]):
                return False
            if not self.G[a][b]["up"]:
                return False
        return True

    # ----------------- Paths -----------------

    def k_shortest(self, src: int, dst: int, k: int) -> List[List[int]]:
        """基于 dist_km 作为权重的 k-短路；过滤不可达与 down 边/点"""
        # 构造“当前 up 的子图”
        H = nx.Graph()
        for u in self.G.nodes():
            if self.G.nodes[u]["up"]:
                H.add_node(u, **self.G.nodes[u])
        for u, v in self.G.edges():
            if self.G[u][v]["up"] and H.has_node(u) and H.has_node(v):
                H.add_edge(u, v, **self.G[u][v])

        if (not H.has_node(src)) or (not H.has_node(dst)):
            return []

        # 若当前 up 子图中无可达路径，直接返回空
        if not nx.has_path(H, src, dst):
            return []

        try:
            gen = nx.shortest_simple_paths(H, src, dst, weight="dist_km")
        except nx.NetworkXNoPath:
            return []

        paths = []
        try:
            for p in gen:
                paths.append(p)
                if len(paths) >= k:
                    break
        except nx.NetworkXNoPath:
            # 生成过程中图已无路径（例如上层在同一步发生了失效切换导致 H 变化）
            pass
        return paths

    def _edges_set(self, p: List[int]) -> set:
        s = set()
        for i in range(len(p)-1):
            u, v = p[i], p[i+1]
            s.add((u, v) if (u, v) in self.G.edges else (v, u))
        return s

    def _dz_set(self, p: List[int]) -> set:
        zs = set()
        for i in range(len(p)-1):
            u, v = p[i], p[i+1]
            for z in (self.G.nodes[u].get("dz_id"), self.G.nodes[v].get("dz_id"), self.G[u][v].get("dz_id", None)):
                if z is not None:
                    zs.add(z)
        return zs

    def filter_disjoint(self, paths: List[List[int]], mode: str) -> List[List[int]]:
        """按 EDGE/NODE/DZ 约束做最大子集选择（贪心）"""
        mode = mode.upper()
        kept: List[List[int]] = []
        for p in paths:
            ok = True
            for q in kept:
                if mode == "EDGE":
                    if len(self._edges_set(p) & self._edges_set(q)) > 0:
                        ok = False; break
                elif mode == "NODE":
                    if len(set(p[1:-1]) & set(q[1:-1])) > 0:
                        ok = False; break
                elif mode == "DZ":
                    if len(self._dz_set(p) & self._dz_set(q)) > 0:
                        ok = False; break
                else:
                    # 无约束
                    pass
            if ok:
                kept.append(p)
        return kept

    def enumerate_candidates(self, src: int, dst: int, disjoint_mode: str, K_max: int) -> List[Dict[str, Any]]:
        """返回候选路径及简单特征：hops, dist, min_residual_bw, availability 以及 ok 标志"""
        disjoint_mode = (disjoint_mode or self.disjoint_mode).upper()
        ks = max(1, int(K_max))
        k_list = self.k_shortest(src, dst, ks * 2)  # 先拉多一些
        if not k_list:
            return []
        disjointed = self.filter_disjoint(k_list, disjoint_mode)
        cand = []
        for p in disjointed[:ks]:
            hops = len(p) - 1
            dist = sum(float(self.G[p[i]][p[i+1]]["dist_km"]) for i in range(hops))
            min_res = min(float(self.G[p[i]][p[i+1]]["bw_cap"]) - float(self.G[p[i]][p[i+1]]["bw_used"]) for i in range(hops)) if hops > 0 else 0.0
            avail = self.path_availability(p)
            ok = self.is_path_up(p)
            feats = np.array([hops, dist, min_res, avail], dtype=np.float32)
            cand.append({"path": p, "feats": feats, "ok": ok})
        return cand

    # ----------------- 资源原子预留 / 释放 -----------------

    def can_reserve_path_bw(self, path: List[int], bw: float) -> bool:
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            cap = float(self.G[u][v]["bw_cap"])
            used = float(self.G[u][v]["bw_used"])
            if used + bw > cap + 1e-9:
                return False
        return True

    def reserve_path_bw(self, path: List[int], bw: float) -> None:
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            self.G[u][v]["bw_used"] = float(self.G[u][v]["bw_used"]) + bw

    def release_path_bw(self, path: List[int], bw: float) -> None:
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            self.G[u][v]["bw_used"] = max(0.0, float(self.G[u][v]["bw_used"]) - bw)

    def can_reserve_cpu(self, node: int, units: float) -> bool:
        cap = float(self.G.nodes[node]["cpu_cap"])
        used = float(self.G.nodes[node]["cpu_used"])
        return (used + units) <= (cap + 1e-9)

    def reserve_cpu(self, node: int, units: float) -> None:
        self.G.nodes[node]["cpu_used"] = float(self.G.nodes[node]["cpu_used"]) + units

    def release_cpu(self, node: int, units: float) -> None:
        self.G.nodes[node]["cpu_used"] = max(0.0, float(self.G.nodes[node]["cpu_used"]) - units)

    def _pick_vnf_nodes_on_path(self, path: List[int], L: int) -> List[int]:
        """在给定路径上选 L 个节点（尽量均匀）；若中间节点不足则退化允许端点"""
        if L <= 0:
            return []
        inner = path[1:-1]
        nodes = inner if len(inner) >= L else path
        idxs = np.linspace(0, len(nodes)-1, num=L, dtype=int)
        chosen = [int(nodes[i]) for i in idxs]
        return chosen

    def reserve_equal_split(self,
                            sid: int,
                            paths_active: List[List[int]],
                            path_backup: Optional[List[int]],
                            bw_each: float,
                            L: int,
                            ttl: int) -> Dict[str, Any]:
        """
        等份带宽原子预留：
        - 路径集合 = 在用 (N-1) + 备用 (1)
        - 每条路径都预留 bw_each
        - 每条路径上放置 L 个 VNF（此处统一用 cpu_per_backup 以体现“热备降配”）
        """
        paths = list(paths_active) + ([path_backup] if path_backup else [])
        if not paths:
            return {"success": 0, "reason": "no_paths"}

        # 1) 可行性复核
        for p in paths:
            if not self.is_path_up(p) or (not self.can_reserve_path_bw(p, bw_each)):
                return {"success": 0, "reason": "not_feasible_now"}

        # 2) 选择每条路径的 VNF 部署节点
        vnf_nodes_per_path: List[List[int]] = []
        for p in paths:
            vnf_nodes_per_path.append(self._pick_vnf_nodes_on_path(p, L))

        # 3) 原子预留：带宽 + CPU（失败则回滚）
        cpu_book: Dict[int, float] = {}
        try:
            # 带宽
            for p in paths:
                self.reserve_path_bw(p, bw_each)
            # CPU（统一用热备降配，保持对称）
            for ns in vnf_nodes_per_path:
                for u in ns:
                    need = self.cpu_per_backup
                    if not self.can_reserve_cpu(u, need):
                        raise RuntimeError("cpu_insufficient")
                    self.reserve_cpu(u, need)
                    cpu_book[u] = cpu_book.get(u, 0.0) + need
        except Exception:
            # 回滚
            for p in paths:
                self.release_path_bw(p, bw_each)
            for u, amt in cpu_book.items():
                self.release_cpu(u, amt)
            return {"success": 0, "reason": "reserve_failed"}

        # 4) 成本与时延（在用 N-1 条的最大时延 + L*proc）
        N = len(paths)
        active_idx = list(range(max(0, N-1)))
        standby_idx = (N-1) if N >= 1 else None
        latencies = [self.latency_ms(p, vnf_count=L) for p in paths[:max(1, N-1)]]
        lat_ms = max(latencies) if latencies else 0.0
        cost = self.cost_sfc(paths, bw_each, vnf_nodes_per_path,
                             cpu_per_vnf=self.cpu_per_backup, cpu_per_vnf_bk=self.cpu_per_backup)

        # 5) 登记会话
        t_exp = self.t + int(ttl)
        self.register_session(
            sid=sid, src=paths[0][0], dst=paths[0][-1], L=L, N=N, paths=paths,
            active_idx=active_idx, standby_idx=standby_idx,
            bw_each=bw_each, vnf_nodes_per_path=vnf_nodes_per_path, t_expire=t_exp
        )
        self._cpu_books[sid] = cpu_book

        return {
            "success": 1,
            "session": self.active[sid].__dict__,
            "latency_ms": float(lat_ms),
            "cost_total": float(cost["total"]),
            "cost_bw": float(cost["bw"]),
            "cost_cpu": float(cost["cpu"]),
        }

    def check_paths_feasible(self, paths: List[List[int]], bw_each: float) -> bool:
        """所有路径均需 up 且每条边留有 bw_each 空间"""
        for p in paths:
            if not self.is_path_up(p):
                return False
            if not self.can_reserve_path_bw(p, bw_each):
                return False
        return True

    # ----------------- 时延 / 成本 / 可靠性 -----------------

    def latency_ms(self, path: List[int], vnf_count: int) -> float:
        """路径传播 + 转发 + VNF 处理开销（VNF 处理按常数 γ 计在每条路径上）"""
        dist = 0.0
        hops = len(path) - 1
        for i in range(hops):
            u, v = path[i], path[i+1]
            dist += float(self.G[u][v]["dist_km"])
        prop = self.prop_ms_per_km * dist
        tx = self.tx_ms_per_hop * hops
        vnf_proc = self.vnf_proc_ms * vnf_count
        return float(prop + tx + vnf_proc)

    def cost_sfc(self,
                 paths: List[List[int]],
                 bw_each: float,
                 vnf_nodes_per_path: List[List[int]],
                 cpu_per_vnf: float,
                 cpu_per_vnf_bk: float) -> Dict[str, float]:
        """
        成本：带宽 + 计算。等份负载方案下，每条路径都预留 bw_each（M/(N-1)）。
        计算成本：每条路径上 L 个 VNF。此处统一视作热备降配（cpu_per_vnf_bk）。
        """
        C_bw = 0.0
        for p in paths:
            hops = len(p) - 1
            C_bw += self.cost_lambda_bw * (bw_each * hops)

        C_cpu = 0.0
        for nodes in vnf_nodes_per_path:
            for _node in nodes:
                C_cpu += self.cost_lambda_cpu * cpu_per_vnf_bk

        return {"bw": float(C_bw), "cpu": float(C_cpu), "total": float(C_bw + C_cpu)}

    def path_availability(self, path: List[int],
                          rho_e: float = 0.01,
                          rho_v: float = 0.005) -> float:
        """
        单条路径的可用性估计（独立假设）；可在 cfg 中扩展不同边/点的失效率字典
        """
        ae = 1.0
        for _ in range(len(path)-1):
            ae *= (1.0 - rho_e)
        av = 1.0
        for _ in path:
            av *= (1.0 - rho_v)
        return ae * av

    def sfc_reliability_dp(self, paths: List[List[int]], need_at_least: int) -> float:
        """
        预期可靠性：至少 need_at_least 条路径可用的概率（泊松二项分布，DP 实现）
        A_j = path_availability(paths[j])
        返回 sum_{k=need_at_least..N} P(K=k)
        """
        N = len(paths)
        if N == 0:
            return 0.0
        A = [self.path_availability(p) for p in paths]
        # DP：dp[k] = 恰好 k 条成功的概率
        dp = [0.0] * (N + 1)
        dp[0] = 1.0
        for a in A:
            nxt = [0.0] * (N + 1)
            for k in range(N + 1):
                if dp[k] == 0.0:
                    continue
                # 失败
                nxt[k] += dp[k] * (1.0 - a)
                # 成功
                if k + 1 <= N:
                    nxt[k + 1] += dp[k] * a
            dp = nxt
        return float(sum(dp[need_at_least:]))

    def predict_reliability(self, paths_active: List[List[int]], path_backup: Optional[List[int]], L: int) -> float:
        """N 条路径至少 N-1 条可用概率（与 L 无关，此处 L 仅作签名占位，便于外层统一调用）"""
        paths = list(paths_active) + ([path_backup] if path_backup else [])
        N = len(paths)
        need = max(1, N-1)
        return self.sfc_reliability_dp(paths, need_at_least=need)

    # ----------------- Failover 与释放 -----------------

    def register_session(self,
                         sid: int,
                         src: int, dst: int,
                         L: int,
                         N: int,
                         paths: List[List[int]],
                         active_idx: List[int],
                         standby_idx: Optional[int],
                         bw_each: float,
                         vnf_nodes_per_path: List[List[int]],
                         t_expire: int) -> None:
        self.active[sid] = ActiveSession(
            sid=sid, src=src, dst=dst, L=L,
            bw_each=bw_each, N=N, paths=paths,
            active_idx=list(active_idx), standby_idx=None if standby_idx is None else int(standby_idx),
            vnf_nodes_per_path=[list(ns) for ns in vnf_nodes_per_path],
            t_expire=int(t_expire)
        )

    def try_failover(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        单故障切换：若任一在用路径down，则用备用顶上；备用若也down则 miss
        session: ActiveSession.__dict__ 视图（bear 外层可直接传入）
        """
        sid = int(session["sid"])
        s = self.active.get(sid)
        if s is None:
            return {"failed": 0}

        # 检查在用路径是否有 down
        failed_idx = None
        for idx in list(s.active_idx):
            p = s.paths[idx]
            if not self.is_path_up(p):
                failed_idx = idx
                break
        if failed_idx is None:
            return {"failed": 0}

        # 备用路径是否可用
        bk_idx = s.standby_idx
        if bk_idx is None:
            return {"failed": 1, "backup_hit": 0}

        if self.is_path_up(s.paths[bk_idx]):
            # 命中：把备用换成在用，备用置空
            s.active_idx.remove(failed_idx)
            s.active_idx.append(bk_idx)
            s.standby_idx = None
            lat = max(self.latency_ms(s.paths[i], s.L) for i in s.active_idx)
            return {"failed": 1, "backup_hit": 1, "latency_ms": float(lat), "fail_idx": int(failed_idx)}
        else:
            return {"failed": 1, "backup_hit": 0}

    def release_expired(self, t: int) -> List[int]:
        """到期释放；返回释放的 sid 列表"""
        self.t = t
        to_del = []
        for sid, s in list(self.active.items()):
            if t >= s.t_expire:
                self.release_session(sid)
                to_del.append(sid)
        return to_del

    def release_session(self, sid: int) -> None:
        """归还该会话占用的带宽与 CPU"""
        s = self.active.pop(int(sid), None)
        if s is None:
            return
        # 释放带宽
        for p in s.paths:
            self.release_path_bw(p, s.bw_each)
        # 释放CPU（按会话账本）
        book = self._cpu_books.pop(int(sid), {})
        for u, amt in book.items():
            self.release_cpu(u, amt)

    # ----------------- 查询接口（bear / runner 需要） -----------------

    def get_session_ref(self, sid: int) -> Optional[Dict[str, Any]]:
        s = self.active.get(int(sid))
        return None if s is None else s.__dict__

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        return [s.__dict__ for s in self.active.values()]

    def bw_utilization(self) -> float:
        bw_used = 0.0; bw_cap = 0.0
        for u, v in self.G.edges():
            bw_used += float(self.G[u][v]["bw_used"])
            bw_cap  += float(self.G[u][v]["bw_cap"])
        return bw_used / max(1e-9, bw_cap)

    def cpu_utilization(self) -> float:
        cpu_used = sum(float(self.G.nodes[u]["cpu_used"]) for u in self.G.nodes())
        cpu_cap  = sum(float(self.G.nodes[u]["cpu_cap"]) for u in self.G.nodes())
        return cpu_used / max(1e-9, cpu_cap)


# ===================== 适配器：SFCEnv =====================

class SFCEnv:
    """
    与 runner.py / bear.py 对齐的适配环境：将扁平化参数组装为 cfg 并委托给 Env。
    同时提供 bear.py 直接使用的转发接口：
      - enumerate_candidates / check_paths_feasible / reserve_equal_split / try_failover
      - release_expired / release_session / predict_reliability / get_session_ref
    """
    def __init__(self,
                 # --- 拓扑 ---
                 num_nodes: int,
                 num_edges: int,
                 disjoint_mode: str,
                 k_paths: int,
                 prop_delay_per_km: float,
                 tx_delay_per_hop: float,
                 vnf_proc_ms: float,
                 link_bw_mean: float,
                 link_bw_std: float,
                 cpu_per_vnf: float,
                 cpu_per_backup: float,
                 cpu_total_per_node: float,
                 dz_fraction: float,
                 # --- 流量 ---
                 arrival_rate: float,
                 sfc_len_choices: List[int],
                 sfc_len_probs: List[float],
                 bw_mean: float,
                 bw_std: float,
                 dur_mean: int,
                 dur_std: int,
                 # --- 失效 ---
                 node_fail_p: float,
                 edge_fail_p: float,
                 repair_mean: int,
                 seed: int,
                 # --- 成本 ---
                 edge_unit_cost: float,
                 lambda_backup: float):
        cfg = {
            "seed": seed,
            "topology": {
                "type": "synthetic",
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "disjoint": disjoint_mode,
                "k_paths": k_paths,
                "link": {
                    "dist_mean": 50.0,
                    "dist_std": 10.0,
                    "bw_capacity": link_bw_mean,   # 简化：用均值作为统一容量；若需更真实可采样
                    "prop_delay_per_km": prop_delay_per_km,
                    "tx_delay_per_hop": tx_delay_per_hop,
                },
                "cpu_total_per_node": cpu_total_per_node,
                "dz_fraction": dz_fraction,      # 当前默认不打 DZ；可在 _build_topology 中扩展
            },
            "node": {
                "cpu_capacity": cpu_total_per_node,
            },
            "failures": {
                "node_fail_prob": node_fail_p,
                "edge_fail_prob": edge_fail_p,
                "recovery_time": repair_mean,
            },
            "traffic": {
                "arrival_rate": arrival_rate,
                "duration_mean": dur_mean,
                "duration_std": dur_std,
                "bw_demand_mean": bw_mean,
                "bw_demand_std": bw_std,
                "sfc_length_choices": sfc_len_choices,
                "sfc_length_probs": sfc_len_probs,
            },
            "sfc": {
                "proc_ms": vnf_proc_ms,
                "cpu_per_vnf": cpu_per_vnf,
                "cpu_per_backup": cpu_per_backup,
            },
            "cost": {
                "lambda_bw": edge_unit_cost,
                "lambda_cpu": 1.0,               # 若需区分，可在配置中单独设置
                "lambda_backup": lambda_backup,  # 预留位置：如需对备用路径单独加权可在 cost_sfc 中使用
            }
        }
        self.e = Env(cfg)

    # ---------- 直接转发（bear / runner 期望的接口） ----------

    def reset_episode(self, seed=None): self.e.reset_episode()
    def episode_meta(self): return self.e.episode_meta()
    def maybe_next_request(self, t: int): return self.e.maybe_next_request(t)
    def inject_failures(self, t: int): return self.e.inject_failures(t)

    def enumerate_candidates(self, src, dst, disjoint_mode, K_max): return self.e.enumerate_candidates(src, dst, disjoint_mode, K_max)
    def check_paths_feasible(self, paths, bw_each): return self.e.check_paths_feasible(paths, bw_each)
    def reserve_equal_split(self, **kw): return self.e.reserve_equal_split(**kw)

    def try_failover(self, session): return self.e.try_failover(session)
    def release_expired(self, t): return self.e.release_expired(t)
    def release_session(self, sid): return self.e.release_session(sid)

    def predict_reliability(self, paths_active, path_backup, L): return self.e.predict_reliability(paths_active, path_backup, L)
    def get_session_ref(self, sid): return self.e.get_session_ref(sid)
    def get_all_sessions(self): return self.e.get_all_sessions()

    def bw_utilization(self): return self.e.bw_utilization()
    def cpu_utilization(self): return self.e.cpu_utilization()