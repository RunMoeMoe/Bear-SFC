# -*- coding: utf-8 -*-
from __future__ import annotations
"""
baseline_SBD.py
===============
SBD（Selective Backup Deployment）基线实现（工程化版本）

要点：
- 单主路径 + 选择性备份（按关键 VNF 决策是否备份），在工程上通过“备份整条路径”
  的保守实现与 env 接口对齐（reserve_equal_split 需要完整备份路径）；
- 选择性信息（备份 VNF 覆盖、CPU 缩放系数 rho_cpu）完整写入事件日志，便于分析；
- 运行期 failover 与现有口径一致：仅对“需要切换”的会话尝试；>=2 条在用 down 直接失败；
  未命中备份立即释放，避免重复尝试。

与现有工程依赖：
- env: enumerate_candidates / check_paths_feasible / reserve_equal_split / predict_reliability /
       try_failover / inject_failures / release_expired / get_session_ref
- metrics: EventLogger / SummaryAggregator / EpisodeSummaryWriter
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter
from quota import QuotaManager


# ----------------------------
# 工具函数
# ----------------------------
def _paths_disjoint(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode in ("EDGE", "LINK"):
        e1 = {(p1[i], p1[i+1]) for i in range(len(p1)-1)}
        e2 = {(p2[i], p2[i+1]) for i in range(len(p2)-1)}
        return len(e1 & e2) == 0
    s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
    return len(s1 & s2) == 0


def _score_path(cand: Dict[str, Any]) -> float:
    """排序候选路径：优先按估计时延（feats[0]），其次按跳数。"""
    feats = cand.get("feats", [])
    if isinstance(feats, list) and len(feats) >= 1:
        try:
            return float(feats[0])
        except Exception:
            pass
    return float(len(cand.get("path", [])))


def _std_request(req) -> Dict[str, Any]:
    """兼容 dict / 对象请求，标准化字段。"""
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
# SBD 系统
# ----------------------------
class SBDSystem:
    """
    对接 runner 的封装：
      - run_one_episode(ep_idx, steps, mode, fixed_N=None) -> summary
      - run(mode, epochs, steps, fixed_N=None)

    关键参数：
      - disjoint_mode: 备份路径与主路径的不相交模式（EDGE/NODE/DZ）
      - K_cand_max   : 候选路径枚举上限
      - vnf_risk_th  : 判定“关键 VNF”的风险阈值（0~1）；越高越保守
      - rho_cpu      : 对备份 VNF 的 CPU 预留缩放系数（0~1）；日志记录，供将来细化资源模型
    """
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 32,
                 N_min: int = 2, N_max: int = 5,
                 vnf_risk_th: float = 0.3,
                 rho_cpu: float = 0.6):
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
        self.vnf_risk_th = float(vnf_risk_th)
        self.rho_cpu = float(rho_cpu)

        # 配额（与统计对齐）
        self.qm = QuotaManager(N_min=N_min, N_max=N_max, smooth_tau=0.9)
        self._last_summary = None

    # ---------- 候选枚举 ----------
    def _enumerate_sorted_candidates(self, src: int, dst: int) -> List[Dict[str, Any]]:
        cand = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
        good = [c for c in cand if c.get("ok", True)]
        good.sort(key=_score_path)
        return good

    # ---------- 关键 VNF 选择 ----------
    def _select_critical_vnfs(self, path: List[int], L: int) -> List[int]:
        """
        返回需要备份的 VNF 索引列表（0..L-1）。
        这里提供一个工程判据：
          - 若 env 暴露 per-node/per-link 的失效概率或可靠性评分，可在此融合；
          - 简化：按路径长度/中间节点度/候选特征估计风险，超过阈值则标为“关键”。
        当前实现：若 L>=3，备份中间 60% 的 VNF；若 L<3，备份第一个（保守）。
        你可依据 env 的真实风险接口替换该策略。
        """
        if L <= 1:
            return [0]
        if L == 2:
            return [0]  # 仅备份第一个功能
        # 备份中间的 VNF，更易受集中失效影响（示例策略）
        k1 = int(np.floor(0.2 * L))
        k2 = int(np.ceil(0.8 * L))
        idxs = list(range(k1, k2))
        if not idxs:
            idxs = [1] if L >= 2 else [0]
        return sorted(set(idxs))

    # ---------- 放置 ----------
    def _place_request(self, req_raw) -> Dict[str, Any]:
        req = _std_request(req_raw)
        sid, src, dst, L, bw, ttl, t_arr = req["sid"], req["src"], req["dst"], req["L"], req["bw"], req["ttl"], req["t_arrive"]

        cand = self._enumerate_sorted_candidates(src, dst)
        if not cand:
            ev = {
                "t": t_arr, "event": "place", "method": "SBD", "sid": sid,
                "success": 0, "reason": "no_candidates",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "sbd_num_vnf_backup": 0, "sbd_rho_cpu": self.rho_cpu,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 1) 选择主路径（最优）
        main_path = list(cand[0]["path"])

        # 2) 选择性备份 VNF 集合（按策略）
        vnf_backup_idx = self._select_critical_vnfs(main_path, L=L)
        num_vnf_bk = len(vnf_backup_idx)

        # 3) 选择与主路径不相交的备份路径（若找不到，则允许次优：延迟略高但不相交）
        backup_path = None
        for c in cand[1:]:
            p = c.get("path", [])
            if _paths_disjoint(main_path, p, self.disjoint_mode):
                backup_path = list(p)
                break

        # 若没有不相交备份，则可退化为“相交备份”或直接无备份（这里选择：相交备份作为兜底）
        if backup_path is None and len(cand) >= 2:
            backup_path = list(cand[1]["path"])

        # 4) 资源可行性与预留
        #    工程化保守实现：主路径按 bw 全量；备份路径也按 bw 预留（env 以完整路径粒度）
        #    使用 reserve_equal_split 的约定：N=2 => bw_each = bw / (N-1) = bw
        if backup_path is None:
            # 无备份路径：按单路径尝试（仍使用 equal_split，backup=None）
            bundle = [main_path]
            if not self.env.check_paths_feasible(bundle, bw):
                ev = {
                    "t": t_arr, "event": "place", "method": "SBD", "sid": sid,
                    "success": 0, "reason": "insufficient_capacity_main",
                    "src": src, "dst": dst, "L": L, "bw": bw, "N": 1, "num_paths": 1,
                    "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    "sbd_num_vnf_backup": num_vnf_bk, "sbd_rho_cpu": self.rho_cpu,
                }
                self.ev_logger.log(ev); self.agg.ingest(ev)
                self.qm.consume_for_request(used_paths=0, placed=False)
                return ev

            res = self.env.reserve_equal_split(
                sid=sid, paths_active=[main_path], path_backup=None,
                bw_each=bw, L=L, ttl=ttl
            )
            if not res.get("success", 0):
                ev = {
                    "t": t_arr, "event": "place", "method": "SBD", "sid": sid,
                    "success": 0, "reason": "reserve_failed_main",
                    "src": src, "dst": dst, "L": L, "bw": bw, "N": 1, "num_paths": 1,
                    "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    "sbd_num_vnf_backup": num_vnf_bk, "sbd_rho_cpu": self.rho_cpu,
                }
                self.ev_logger.log(ev); self.agg.ingest(ev)
                self.qm.consume_for_request(used_paths=0, placed=False)
                return ev

            rel_pred = float(self.env.predict_reliability([main_path], None, L))
            ev = {
                "t": t_arr, "event": "place", "method": "SBD", "sid": sid,
                "success": 1, "reason": "",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 1, "num_paths": 1,
                "latency_ms": float(res.get("latency_ms", 0.0)),
                "cost_total": float(res.get("cost_total", 0.0)),
                "cost_bw": float(res.get("cost_bw", 0.0)),
                "cost_cpu": float(res.get("cost_cpu", 0.0)),
                "emp_reli_pred": rel_pred,
                "fail_idx": "", "new_active": "",
                "sbd_num_vnf_backup": num_vnf_bk, "sbd_rho_cpu": self.rho_cpu,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=1, placed=True)
            return ev

        # 有备份路径：按 DP 方式进行保守预留，但事件中写入 SBD 的“选择性”元数据
        bundle = [main_path, backup_path]
        if not self.env.check_paths_feasible(bundle, bw):
            ev = {
                "t": t_arr, "event": "place", "method": "SBD", "sid": sid,
                "success": 0, "reason": "insufficient_capacity_main_backup",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 2,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "sbd_num_vnf_backup": num_vnf_bk, "sbd_rho_cpu": self.rho_cpu,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        res = self.env.reserve_equal_split(
            sid=sid, paths_active=[main_path], path_backup=backup_path,
            bw_each=bw, L=L, ttl=ttl
        )
        if not res.get("success", 0):
            ev = {
                "t": t_arr, "event": "place", "method": "SBD", "sid": sid,
                "success": 0, "reason": "reserve_failed_main_backup",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 2,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                "sbd_num_vnf_backup": num_vnf_bk, "sbd_rho_cpu": self.rho_cpu,
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        rel_pred = float(self.env.predict_reliability([main_path], backup_path, L))
        ev = {
            "t": t_arr, "event": "place", "method": "SBD", "sid": sid,
            "success": 1, "reason": "",
            "src": src, "dst": dst, "L": L, "bw": bw, "N": 2, "num_paths": 2,
            "latency_ms": float(res.get("latency_ms", 0.0)),
            "cost_total": float(res.get("cost_total", 0.0)),
            "cost_bw": float(res.get("cost_bw", 0.0)),
            "cost_cpu": float(res.get("cost_cpu", 0.0)),
            "emp_reli_pred": rel_pred,
            "fail_idx": "", "new_active": "",
            # —— SBD 选择性信息（供后续分析/画图）——
            "sbd_num_vnf_backup": num_vnf_bk,  # 备份的 VNF 数量
            "sbd_rho_cpu": self.rho_cpu,       # 备份 CPU 缩放系数（目前仅记录）
        }
        self.ev_logger.log(ev); self.agg.ingest(ev)
        self.qm.consume_for_request(used_paths=1, placed=True)
        return ev

    # ---------- failover ----------
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
                if not self.env.is_path_up(p):
                    cnt += 1
            except Exception:
                continue
        return cnt

    # ---------- 单轮 ----------
    def run_one_episode(self, ep_idx: int, steps: int, mode: str = "eval", fixed_N: Optional[int] = None) -> Dict[str, Any]:
        # reset
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset(); self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        # 名义 N（仅用于配额统计）
        self.qm.set_epoch_quota(2)

        alive_sessions: Dict[int, Dict[str, Any]] = {}

        for t in range(0, steps):
            # 到达
            req = self.env.maybe_next_request(t)
            if req is not None:
                if isinstance(req, dict):
                    req.setdefault("t_arrive", t)
                else:
                    try: setattr(req, "t_arrive", t)
                    except Exception: pass

                ev = self._place_request(req)
                if ev.get("success", 0) == 1 and hasattr(self.env, "get_session_ref"):
                    sid = int(ev["sid"])
                    try:
                        alive_sessions[sid] = self.env.get_session_ref(sid)
                    except Exception:
                        pass

            # 失效注入
            self.env.inject_failures(t)

            # 仅对需要切换的会话尝试 failover
            for s in list(alive_sessions.values()):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:
                    continue
                sid_i = int(s.get("sid", -1))

                if down_cnt >= 2:
                    ev_fail = {
                        "t": t, "event": "failover", "method": "SBD", "sid": sid_i,
                        "success": 0, "reason": "multi_path_down",
                        "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                        "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                        "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                        "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    }
                    self.ev_logger.log(ev_fail); self.agg.ingest(ev_fail)
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)
                    continue

                r = self.env.try_failover(s)
                if r.get("failed", 0) == 0:
                    # 无失败，不产生事件
                    continue

                ev = {
                    "t": t, "event": "failover", "method": "SBD", "sid": sid_i,
                    "success": 1 if r.get("backup_hit", 0) else 0,
                    "reason": "" if r.get("backup_hit", 0) else "no_usable_backup",
                    "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                    "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": r.get("fail_idx", ""), "new_active": r.get("new_active", ""),
                }
                self.ev_logger.log(ev); self.agg.ingest(ev)

                # 未命中即释放
                if int(ev["success"]) == 0:
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)

            # 释放到期
            released = self.env.release_expired(t)
            if released:
                for sid in released:
                    alive_sessions.pop(int(sid), None)

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
            print(f"[SBD EP {ep:03d}] "
                  f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                  f"lat={summ['avg_latency_ms']:.3f}")

        try:
            self.ev_logger.close()
            self.ep_writer.close()
        except Exception:
            pass