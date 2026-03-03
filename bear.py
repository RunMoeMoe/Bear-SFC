from __future__ import annotations
# -*- coding: utf-8 -*-
"""
BEAR-SFC System with Internal PRANOS-style Preselector
=====================================================
目标：
  - 维持 BEAR 原有接口与指标记录（事件/汇总与 runner 保持一致）；
  - 在 BEAR 内部引入“滑动窗口 LP+Rounding 预筛选器”，为到达请求批量生成
    可行候选组合（N-1 在用 + 1 备用，容量/不相交可行），再由 BEAR 做二次选择；
  - 不需要修改 runner.py：runner 依旧只调用 system.run_one_episode(...)

组件：
  1) BearOrchestrator：单请求编排（放置/故障切换，与 env 对接）
  2) BearSystem      ：训练/评估循环 + 预筛选器接入（窗口缓冲→生成候选池→放置该批）

env 接口（与原版一致，略）
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# 依赖模块（项目内）
from quota import QuotaManager
from algo_central import CentralDQN, DQNConfig
from algo_edge import HeuristicSelector, EdgePPO
from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter, compute_central_reward

# 预筛选器（PRANOS 风格）
from preselector_pranos import PRANOSPreselector


# =========================
# Orchestrator：单请求编排
# =========================

class BearOrchestrator:
    """
    实现“ N 条路径，N-1 在用 + 1 备用”的放置与故障切换逻辑。
    """
    def __init__(self,
                 env,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 32,
                 edge_selector=None):
        self.env = env
        self.disjoint_mode = str(disjoint_mode).upper()   # "EDGE"/"NODE"/"DZ"
        self.K_cand_max = int(K_cand_max)
        # 边策略：启发式或 EdgePPO（统一 act 接口）
        self.edge = edge_selector or HeuristicSelector()
        # 预筛选候选池：由上层注入 {sid: {"request":..., "combos":[...]}}
        self._cand_pool: Dict[int, Dict[str, Any]] = {}

    def set_candidate_pool(self, pool: Dict[int, Dict[str, Any]]) -> None:
        """
        由上层系统注入预筛选结果池。
        """
        if not pool:
            return
        # 合并（同 sid 覆盖）
        for k, v in pool.items():
            self._cand_pool[int(k)] = v

    def _unpack_request(self, request: dict) -> Tuple[int, int, int, int, float, int]:
        """
        兼容两种请求表示：字典或 dataclass(SFCRequest)。
        返回 (sid, src, dst, L, bw, ttl)
        """
        if isinstance(request, dict):
            return (int(request["sid"]), int(request["src"]), int(request["dst"]),
                    int(request["L"]), float(request["bw"]), int(request["ttl"]))
        sid = int(getattr(request, "sid"))
        src = int(getattr(request, "src"))
        dst = int(getattr(request, "dst"))
        L   = int(getattr(request, "L"))
        bw  = float(getattr(request, "bw"))
        ttl = int(getattr(request, "ttl"))
        return sid, src, dst, L, bw, ttl

    # —— 构造边缘状态 —— #
    def _build_edge_state(self, cand_list: List[Dict[str, Any]], N: int) -> Dict[str, Any]:
        """
        cand_list: [{ "path": [...], "feats": np.ndarray(D,), "ok": bool}, ...]
        返回给 Edge 策略的状态（K×D 的特征、mask、目标 N）
        """
        if not cand_list:
            return {"cand_feats": np.zeros((0, 4), dtype=np.float32), "N": N, "mask": None}
        feats = []
        mask = []
        for c in cand_list:
            feats.append(np.asarray(c.get("feats", []), dtype=np.float32))
            mask.append(1.0 if c.get("ok", True) else 0.0)
        feats = np.stack(feats, axis=0)    # (K, D)
        mask = np.asarray(mask, dtype=np.float32)  # (K,)
        return {"cand_feats": feats, "N": N, "mask": mask}

    def select_paths(self,
                     src: int, dst: int,
                     N: int,
                     explore_edge: bool = False) -> Tuple[List[List[int]], Optional[List[int]], Dict[str, Any]]:
        """
        返回：paths_active (N-1)、path_backup (1)、diagnose（诊断信息）
        若候选不足或不相交约束导致不可行，则返回空。
        """
        # 1) 拉取候选
        cand = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max)
        if not cand:
            return [], None, {"reason": "no_candidates"}
        # 2) Edge 选择 N 条
        edge_state = self._build_edge_state(cand, N)
        idx = self.edge.act(edge_state, explore=explore_edge)
        if len(idx) < N:
            return [], None, {"reason": "insufficient_candidates"}

        chosen = [cand[i]["path"] for i in idx[:N]]
        # 3) 切分 在用/备份
        paths_active = chosen[: max(1, N - 1)]
        path_backup = chosen[max(1, N - 1)] if N >= 2 else None

        return paths_active, path_backup, {"chosen_idx": idx, "K": len(cand)}

    def place_one(self,
                  request: Dict[str, Any],
                  N: int,
                  logger: Optional[EventLogger] = None,
                  explore_edge: bool = False) -> Dict[str, Any]:
        """
        对单个请求执行放置：
          - 选择 N 条路径，等份带宽到 N-1 条“在用路径”，第 N 条做“备用路径”
          - 资源预留成功则登记会话并返回事件字典；否则返回失败事件
        """
        sid, src, dst, L, bw, ttl = self._unpack_request(request)
        t_arr = int(request["t_arrive"]) if isinstance(request, dict) else int(getattr(request, "t_arrive"))

        # 1) 选路（优先使用预筛选器给出的可行组合；否则回退到原选择逻辑）
        sid_norm = int(sid)
        use_pool = False
        paths_active: List[List[int]] = []
        path_backup: Optional[List[int]] = None
        diag = {}

        if sid_norm in self._cand_pool:
            combos = self._cand_pool[sid_norm].get("combos", [])
            if combos:
                # 选择策略：最小估计时延（可替换为 RL 价值评分）
                try:
                    best_idx = int(np.argmin([c.get("lat_est", 0.0) for c in combos]))
                except Exception:
                    best_idx = 0
                cand = combos[best_idx]
                paths_active = [list(p) for p in cand.get("active", [])]
                path_backup = list(cand.get("backup")) if cand.get("backup") is not None else None
                bw_each = float(cand.get("bw_each", float(bw) / max(1, N - 1)))
                use_pool = True
                # 用过即删，避免重复
                self._cand_pool.pop(sid_norm, None)

        if not use_pool:
            paths_active, path_backup, diag = self.select_paths(src, dst, N, explore_edge=explore_edge)
            if len(paths_active) < max(1, N - 1) or path_backup is None:
                ev = {
                    "t": t_arr, "event": "place", "method": "bear_sfc", "sid": sid,
                    "success": 0, "reason": diag.get("reason", "select_failed"),
                    "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                    "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                }
                if logger: logger.log(ev)
                return ev
            bw_each = bw / max(1, N - 1)

        # 3) 可行性检查
        if not self.env.check_paths_feasible(paths_active + ([path_backup] if path_backup else []), bw_each):
            ev = {
                "t": t_arr, "event": "place", "method": "bear_sfc", "sid": sid,
                "success": 0, "reason": "insufficient_capacity",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
            }
            if logger: logger.log(ev)
            return ev

        # 4) 资源预留
        place_res = self.env.reserve_equal_split(
            sid=sid, paths_active=paths_active, path_backup=path_backup,
            bw_each=bw_each, L=L, ttl=ttl
        )
        if not place_res.get("success", 0):
            ev = {
                "t": t_arr, "event": "place", "method": "bear_sfc", "sid": sid,
                "success": 0, "reason": "reserve_failed",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
            }
            if logger: logger.log(ev)
            return ev

        # 5) 预期可靠性
        rel_pred = float(self.env.predict_reliability(paths_active, path_backup, L))

        ev = {
            "t": t_arr, "event": "place", "method": "bear_sfc", "sid": sid,
            "success": 1, "reason": "",
            "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(paths_active) + (1 if path_backup else 0),
            "latency_ms": float(place_res.get("latency_ms", 0.0)),
            "cost_total": float(place_res.get("cost_total", 0.0)),
            "cost_bw": float(place_res.get("cost_bw", 0.0)),
            "cost_cpu": float(place_res.get("cost_cpu", 0.0)),
            "emp_reli_pred": rel_pred,
            "fail_idx": "", "new_active": "",
        }
        if logger: logger.log(ev)
        return ev

    def try_failover(self, session: Dict[str, Any], t_now: int,
                     logger: Optional[EventLogger] = None) -> Dict[str, Any]:
        """
        调用 env.try_failover 对单会话执行失败切换；返回并可记录事件
        """
        res = self.env.try_failover(session)
        if res.get("failed", 0) == 0:
            return {"failed": 0}  # 无失败，无需记录

        ev = {
            "t": t_now, "event": "failover", "method": "bear_sfc", "sid": int(session["sid"]),
            "success": 1 if res.get("backup_hit", 0) else 0,
            "reason": "" if res.get("backup_hit", 0) else "no_usable_backup",
            "src": session["src"], "dst": session["dst"], "L": session["L"], "bw": session.get("bw", ""),
            "N": session.get("N", ""), "num_paths": session.get("num_paths", ""),
            "latency_ms": float(res.get("latency_ms", 0.0)),
            "cost_total": "", "cost_bw": "", "cost_cpu": "",
            "emp_reli_pred": "",
            "fail_idx": res.get("fail_idx", ""), "new_active": res.get("new_active", ""),
        }
        if logger: logger.log(ev)
        return ev


# =========================
# BEAR System：训练/评估编排
# =========================

class BearSystem:
    """
    提供统一入口 run(mode, epochs, steps)：
      - mode = "train_central"：冻结边（启发式），训练中央 DQN（决定 N）
      - mode = "train_edge"    ：冻结中央（固定 N=fixed_N），训练边（PPO）
      - mode = "eval"          ：两者固定（从 checkpoint 读），只产出 6 指标与事件日志

    与原版差异：
      - 内部集成 PRANOS 预筛选器：窗口缓冲到达请求 → 生成候选池 → 对该批逐个调用 place_one
    """
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 32,
                 # 中央 DQN
                 central_state_dim: int = 10,
                 N_min: int = 2, N_max: int = 5,
                 central_cfg: Optional[Dict[str, Any]] = None,
                 # 边缘策略（默认启发式；如需 PPO，外部传 EdgePPO）
                 edge_algo=None,
                 # 预筛选器参数
                 window_size: int = 32,
                 top_k: int = 4,
                 ):
        self.env = env
        self.result_dir = Path(result_dir); self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)

        # Orchestrator（默认启发式边；训练边时传 EdgePPO）
        self.edge_algo = edge_algo or HeuristicSelector()
        self.orch = BearOrchestrator(env, disjoint_mode=disjoint_mode,
                                     K_cand_max=K_cand_max, edge_selector=self.edge_algo)

        # Quota（中央的动作 N）
        self.qm = QuotaManager(N_min=N_min, N_max=N_max, smooth_tau=0.9)

        # 中央 DQN
        ccfg = central_cfg or {}
        self.central = CentralDQN(
            DQNConfig(
                in_dim=central_state_dim,
                n_actions=(self.qm.N_max - self.qm.N_min + 1),
                gamma=ccfg.get("gamma", 0.95),
                lr=ccfg.get("lr", 1e-3),
                batch_size=ccfg.get("batch_size", 64),
                buf_size=ccfg.get("buf_size", 50000),
                start_learn_after=ccfg.get("start_learn_after", 1000),
                target_sync=ccfg.get("target_sync", 1000),
                eps_start=ccfg.get("eps_start", 0.2),
                eps_end=ccfg.get("eps_end", 0.02),
                eps_decay_steps=ccfg.get("eps_decay_steps", 50000),
                hidden=tuple(ccfg.get("hidden", (128, 128))),
                device=ccfg.get("device", "cuda" if self._has_cuda() else "cpu"),
            ),
            N_min=self.qm.N_min, N_max=self.qm.N_max
        )

        # 日志器与汇总器（覆盖写）
        for _p in [self.result_dir / "events_bear.csv", self.result_dir / "episode_summary_bear.csv"]:
            try: _p.unlink()
            except FileNotFoundError: pass
        self.ev_logger = EventLogger(self.result_dir / "events_bear.csv")
        self.ep_writer = EpisodeSummaryWriter(self.result_dir / "episode_summary_bear.csv")
        self.agg = SummaryAggregator()

        # 中央奖励权重（可配置）
        self.reward_weights = {
            "w_avail": 1.0,
            "w_hit": 0.6,
            "w_cost": -0.6,
            "w_lat": -0.3,
            "w_place": 0.2,
            "w_rel": 0.2,
            "w_unusd": -0.1,
        }

        # 预筛选器（内部管理，无需 runner 参与）
        self._pre = PRANOSPreselector(env,
                                      disjoint_mode=disjoint_mode,
                                      window_size=int(window_size),
                                      K_cand_max=int(K_cand_max),
                                      top_k=int(top_k))

    # 由预筛选器生成候选池后注入 orchestrator
    def inject_candidate_pool(self, pool: Dict[int, Dict[str, Any]]) -> None:
        if not pool:
            return
        if hasattr(self.orch, "set_candidate_pool"):
            self.orch.set_candidate_pool(pool)

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _count_down_active_paths(self, session: Dict[str, Any]) -> int:
        """
        返回当前会话“在用路径”中宕机的条数。
        优先调用 env 提供的统计接口；若无，则尝试用 env.is_path_up 逐条判断。
        若无法判断，返回 -1（未知）。
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
        paths = (
            session.get("active_set")
            or session.get("paths_active")
            or session.get("paths")
            or []
        )
        if not hasattr(self.env, "is_path_up"):
            return -1
        cnt = 0
        for p in (paths or []):
            try:
                up = bool(self.env.is_path_up(p))
            except Exception:
                continue
            if not up:
                cnt += 1
        return cnt

    def _build_central_state(self) -> np.ndarray:
        # 资源利用度（若无接口则 0）
        bw_util = float(getattr(self.env, "bw_utilization", lambda: 0.0)())
        cpu_util = float(getattr(self.env, "cpu_utilization", lambda: 0.0)())

        last = getattr(self, "_last_summary", None)
        if last is None:
            place_rate = fo_hit = emp_av = rel = cost_n = lat_n = 0.0
        else:
            place_rate = float(last.get("place_rate", 0.0))
            fo_hit = float(last.get("fo_hit_rate", 0.0))
            emp_av = float(last.get("emp_avail", 0.0))
            rel = float(last.get("rel_pred_avg", 0.0))
            cost_n = float(last.get("cost_norm", 0.0))
            lat_n = float(last.get("lat_norm", 0.0))

        Nq = float(self.qm.epoch_quota_N) / float(max(1, self.qm.N_max))

        s = np.array([
            bw_util, cpu_util,
            place_rate, fo_hit, emp_av, rel,
            cost_n, lat_n,
            Nq, 0.0
        ], dtype=np.float32)
        return s

    # —— 一轮（episode）执行 —— #
    def run_one_episode(self,
                        ep_idx: int,
                        steps: int,
                        mode: str = "train_central",
                        fixed_N: Optional[int] = None) -> Dict[str, Any]:
        """
        单集流程：
          - reset env & metrics
          - 决定 N（train_central：central.act；train_edge/eval：fixed_N 必须给）
          - 时间步循环：
              收集到达 -> （若窗口就绪）构建候选池并对该批请求逐个放置
              -> 注入失效 -> failover -> 释放到期 -> 记录
          - 汇总与（可选）训练
        """
        # 1) reset
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset()
        self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        # 2) 决定 N
        explore_central = (mode == "train_central")
        explore_edge = (mode == "train_edge")
        state_C = self._build_central_state()

        if mode == "train_central":
            N = self.central.act(state_C, explore=True)
        else:
            N = int(fixed_N if fixed_N is not None else max(2, self.qm.N_min))
        self.qm.set_epoch_quota(N)

        # 3) 时间步
        alive_sessions: Dict[int, Dict[str, Any]] = {}
        batch_buf: List[Any] = []  # 窗口缓冲

        for t in range(0, steps):
            # 3.1 收集到达请求到窗口
            req = self.env.maybe_next_request(t)
            if req is not None:
                # 记录到达时间（统一）
                if isinstance(req, dict):
                    req.setdefault("t_arrive", t)
                else:
                    try:
                        setattr(req, "t_arrive", t)
                    except Exception:
                        pass
                batch_buf.append(req)
                # 喂给预筛选器
                self._pre.feed(req)

            # 3.2 当窗口就绪：生成候选池，并对该批请求逐个放置
            if self._pre.ready():
                pool = self._pre.run(N=N)
                self.inject_candidate_pool(pool)
                # 逐个调用 orchestrator.place_one；使用候选池则优先用，不可行回退原选路
                for _req in batch_buf:
                    ev = self.orch.place_one(_req, N=N, logger=self.ev_logger, explore_edge=explore_edge)
                    self.agg.ingest(ev)
                    used_paths = int(ev.get("num_paths", 0))
                    self.qm.consume_for_request(used_paths=used_paths, placed=(ev.get("success", 0) == 1))
                    if ev.get("success", 0) == 1:
                        sid = int(ev["sid"])
                        if hasattr(self.env, "get_session_ref"):
                            alive_sessions[sid] = self.env.get_session_ref(sid)
                # 清空该批
                batch_buf = []

            # 3.3 注入失效 & 对所有活跃会话尝试 failover
            self.env.inject_failures(t)
            sess_list = list(alive_sessions.values())
            for s in (sess_list or []):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:
                    continue
                if down_cnt >= 2:
                    sid_i = int(s.get("sid", -1))
                    ev_fail = {
                        "t": t, "event": "failover", "method": "bear_sfc",
                        "sid": sid_i, "success": 0, "reason": "multi_path_down",
                        "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                        "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                        "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                        "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    }
                    if self.ev_logger: self.ev_logger.log(ev_fail)
                    self.agg.ingest(ev_fail)
                    try:
                        if hasattr(self.env, "release_session"):
                            self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)
                    continue
                # 单条在用宕机时尝试 failover
                r = self.orch.try_failover(s, t_now=t, logger=self.ev_logger)
                if isinstance(r, dict):
                    et = str(r.get("event", "")).lower()
                    if et in ("failover", "fo", "fail-over", "switch", "switchover", "switch_over"):
                        if "success" not in r and "hit" in r:
                            r["success"] = 1 if (r.get("hit") in (1, True, "1", "true", "True")) else 0
                        self.agg.ingest(r)
                        if int(r.get("success", 0) or 0) == 0:
                            sid_i = int(s.get("sid", -1))
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

        # 集尾：处理残余窗口
        if batch_buf:
            pool = self._pre.run(N=N)
            self.inject_candidate_pool(pool)
            for _req in batch_buf:
                ev = self.orch.place_one(_req, N=N, logger=self.ev_logger, explore_edge=explore_edge)
                self.agg.ingest(ev)
                used_paths = int(ev.get("num_paths", 0))
                self.qm.consume_for_request(used_paths=used_paths, placed=(ev.get("success", 0) == 1))
                if ev.get("success", 0) == 1:
                    sid = int(ev["sid"])
                    if hasattr(self.env, "get_session_ref"):
                        alive_sessions[sid] = self.env.get_session_ref(sid)
            batch_buf = []

        # 4) 汇总
        summary = self.agg.finalize()
        self._last_summary = summary  # 给下集构造 state_C 时用
        self.ep_writer.write(ep_idx, summary)

        # 5) 训练
        if mode == "train_central":
            rew = compute_central_reward(summary, weights=self.reward_weights,
                                         unused_quota_ratio=self.qm.unused_ratio())
            s_next = self._build_central_state()
            self.central.remember(state_C, action_N=N, reward=rew, next_state=s_next, done=True)
            info = self.central.train_step()
            if (ep_idx + 1) % 10 == 0:
                self.central.save(str(self.save_dir), prefix="central_dqn")
            if info is not None:
                summary.update({f"central_{k}": v for k, v in info.items()})

        elif mode == "train_edge":
            if hasattr(self.edge_algo, "train_step"):
                info = self.edge_algo.train_step()
                if info is not None:
                    summary.update({f"edge_{k}": v for k, v in info.items()})
                if (ep_idx + 1) % 10 and hasattr(self.edge_algo, "save"):
                    self.edge_algo.save(str(self.save_dir), prefix="edge_ppo")

        return summary

    # —— 外部统一入口 —— #
    def run(self,
            mode: str,
            epochs: int,
            steps: int,
            fixed_N: Optional[int] = None) -> None:
        """
        统一入口：训练/评估
        """
        mode = str(mode).lower()
        assert mode in ("train_central", "train_edge", "eval", "alt"), f"bad mode: {mode}"

        for ep in range(epochs):
            if mode == "train_central":
                summ = self.run_one_episode(ep, steps, mode="train_central", fixed_N=fixed_N)
            elif mode == "train_edge":
                summ = self.run_one_episode(ep, steps, mode="train_edge", fixed_N=fixed_N)
            elif mode == "eval":
                summ = self.run_one_episode(ep, steps, mode="eval", fixed_N=fixed_N)
            elif mode == "alt":
                if ep % 2 == 0:
                    summ = self.run_one_episode(ep, steps, mode="train_central", fixed_N=fixed_N)
                else:
                    summ = self.run_one_episode(ep, steps, mode="train_edge", fixed_N=fixed_N)
            # 控制台简报
            msg = (f"[{mode.upper()} EP {ep:03d}] "
                   f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                   f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                   f"lat={summ['avg_latency_ms']:.3f}")
            print(msg)

        # 关闭日志
        self.ev_logger.close()
        self.ep_writer.close()