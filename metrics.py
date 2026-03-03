from __future__ import annotations
"""
指标与日志）
	•	逐事件日志：place/failover 的 success/0/1、原因、N、num_paths=N、bw_each、latency（见下）。
	•	汇总指标：见 §6。
        •	avg_latency_ms（在用 N−1 条路径的最大 per-path 时延，按会话“最近一次”更新取平均）
        •	place_rate = placed / attempted
        •	fo_hit_rate = fo_hit / fo_cnt
        •	rel_pred_avg（成功放置时的预期可靠性均值）
        •	emp_avail = (placed - fo_miss) / max(1, placed)（按你的定义）
        •	avg_cost_total（成功放置的平均总成本） 
        同时输出一些原始计数，便于排障。
	•	归一化函数、滑动基线、分组统计（按拓扑规模、SFC 长度）。
        •	RunningMinMax 做在线归一化（给中央 DQN 的奖励用）；
        •	compute_central_reward：按权重把“可用性/命中率/成本/时延/未用配额”归一化组装成单标量奖励；
        •	EpisodeSummaryWriter：把每集的汇总指标写入一个 CSV（训练收敛曲线的原始数据，绘图放到 plots.py）。
"""

# metrics.py
# -*- coding: utf-8 -*-
"""
指标与日志工具：
- EventLogger：逐事件 CSV（place/failover/release），成功与失败均记录
- SummaryAggregator：按 episode 汇总六大指标 + 训练奖励归一化入口
- EpisodeSummaryWriter：将每集汇总结果落盘，便于 plots.py 读取画图
"""


import csv
from pathlib import Path
from typing import Dict, Any, List, Optional


# =========================
# 事件级 CSV 记录器
# =========================

class EventLogger:
    """
    逐事件写 CSV，字段统一：见默认 fieldnames
    - 你可以直接传 orchestrator/bear 的事件字典，缺失字段会填空
    """
    DEFAULT_FIELDS = [
        "t", "event", "method", "sid",
        "success", "reason",
        "src", "dst", "L", "bw",
        "N", "num_paths",
        "latency_ms",
        "cost_total", "cost_bw", "cost_cpu",
        "emp_reli_pred",   # 预期可靠性（place 成功时）
        # 失败切换相关
        "fail_idx", "new_active",
        # episode 元标签（可选，由 runner 写入）
        "topo_size", "sfc_len_group",
    ]

    def __init__(self, csv_path: Path, fieldnames: Optional[List[str]] = None) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames or self.DEFAULT_FIELDS

        self._f = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        # 若为空文件，写表头
        if self._f.tell() == 0:
            self._w.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        """写入一条事件；缺字段补空"""
        rec = {k: row.get(k, "") for k in self.fieldnames}
        self._w.writerow(rec)
        # 不在此 flush（让上层控制节奏）；如需安全可取消注释
        # self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


# =========================
# 在线归一化工具
# =========================

class RunningMinMax:
    """简单在线 min-max 归一化器；用于 cost/latency 等非负量的缩放"""
    def __init__(self, init_min: float = 1e9, init_max: float = 0.0) -> None:
        self.min_v = init_min
        self.max_v = init_max

    def update(self, v: float) -> None:
        if v is None:
            return
        if v < self.min_v:
            self.min_v = v
        if v > self.max_v:
            self.max_v = v

    def scale(self, v: float, eps: float = 1e-9) -> float:
        lo, hi = self.min_v, self.max_v
        if hi <= lo + eps:
            return 0.0
        return max(0.0, min(1.0, (v - lo) / (hi - lo)))


# =========================
# Episode 汇总器
# =========================

class SummaryAggregator:
    """
    每集期间收集事件，在 finalize() 时输出六大指标：
      - avg_latency_ms
      - place_rate = placed/attempted
      - fo_hit_rate = fo_hit/fo_cnt
      - rel_pred_avg
      - emp_avail = (placed - fo_miss) / max(1, placed)
      - avg_cost_total
    以及若干原始计数，便于排障。
    同时维护两个 RunningMinMax：cost 与 latency，供训练奖励归一化。
    """
    def __init__(self) -> None:
        self.reset()
        # 在线归一化器（给 DQN 奖励用）
        self.cost_norm = RunningMinMax(init_min=1e9, init_max=0.0)
        self.lat_norm  = RunningMinMax(init_min=1e9, init_max=0.0)

    def reset(self) -> None:
        # place 尝试/成功
        self.attempted = 0
        self.placed = 0
        # failover 统计
        self.fo_cnt = 0
        self.fo_hit = 0
        self.fo_miss = 0
        # 成本/时延/预期可靠性累计
        self.cost_sum = 0.0
        self.lat_sum = 0.0
        self.lat_sid_last: Dict[int, float] = {}  # sid -> last latency
        self.rel_pred_sum = 0.0
        self.rel_pred_cnt = 0
        # 元标签（runner 可在 episode 开始时 set）
        self.meta = {"topo_size": "", "sfc_len_group": ""}

    def set_meta(self, topo_size: str = "", sfc_len_group: str = "") -> None:
        self.meta["topo_size"] = topo_size
        self.meta["sfc_len_group"] = sfc_len_group

    def ingest(self, ev: Dict[str, Any]) -> None:
        """
        摄入一条事件（place/failover/release）
        - place：统计 attempted/placed、成本/时延/预期可靠性
        - failover：统计 fo_cnt/hit/miss，并刷新会话的“最近时延”
        """
        # 事件类型鲁棒解析（兼容 event/type 字段与别名）
        et_raw = ev.get("event", ev.get("type", ""))
        et = str(et_raw).lower()
        is_failover = ("failover" in et) or (et in ("fo", "fo_hit", "fo_miss"))
        # 成功标记鲁棒解析（兼容 success/result/status 且支持命中别名）
        succ_raw = ev.get("success", ev.get("result", ev.get("status", 0)))
        succ_str = str(succ_raw).strip().lower()
        succ = 1 if succ_str in ("1", "true", "hit", "success", "ok", "yes") else 0

        if et == "place":
            self.attempted += 1
            if succ == 1:
                self.placed += 1
                # 成本
                ct = _to_float(ev.get("cost_total", 0.0))
                self.cost_sum += ct
                self.cost_norm.update(ct)
                # 时延：记录“最近一次”的时延，用于后续 failover 替换后刷新；先写入初值
                lat = _to_float(ev.get("latency_ms", 0.0))
                sid = int(ev.get("sid", -1))
                if sid >= 0:
                    self.lat_sid_last[sid] = lat
                # 预期可靠性
                relp = _to_float(ev.get("emp_reli_pred", 0.0))
                if relp > 0:
                    self.rel_pred_sum += relp
                    self.rel_pred_cnt += 1

        elif is_failover:
            self.fo_cnt += 1
            if succ == 1:
                self.fo_hit += 1
                # 切换成功后，可能带来新的时延（若事件中给出）
                sid = int(ev.get("sid", -1))
                lat = ev.get("latency_ms", "")
                if lat != "" and sid >= 0:
                    self.lat_sid_last[sid] = _to_float(lat)
            else:
                self.fo_miss += 1

        # release 只做资源释放确认，不参与指标加总

    def finalize(self) -> Dict[str, Any]:
        """
        计算六大指标并返回字典：
        - avg_latency_ms：对每个会话取“最近一次”的时延，再对会话取平均
        - place_rate, fo_hit_rate, emp_avail, rel_pred_avg, avg_cost_total
        以及一些计数与 meta
        """
        # 平均时延：汇总 lat_sid_last（会话级），若无则 0
        lat_vals = list(self.lat_sid_last.values())
        avg_latency = sum(lat_vals) / max(1, len(lat_vals)) if lat_vals else 0.0
        self.lat_norm.update(avg_latency)

        place_rate = self.placed / max(1, self.attempted)
        fo_hit_rate = self.fo_hit / max(1, self.fo_cnt)
        emp_avail = (self.placed - self.fo_miss) / max(1, self.attempted)
        rel_pred_avg = (self.rel_pred_sum / max(1, self.rel_pred_cnt)) if self.rel_pred_cnt > 0 else 0.0
        avg_cost_total = self.cost_sum / max(1, self.placed)

        return {
            # 六指标
            "avg_latency_ms": round(avg_latency, 6),
            "place_rate": round(place_rate, 6),
            "fo_hit_rate": round(fo_hit_rate, 6),
            "rel_pred_avg": round(rel_pred_avg, 6),
            "emp_avail": round(emp_avail, 6),
            "avg_cost_total": round(avg_cost_total, 6),

            # 原始计数（排障用）
            "attempted": int(self.attempted),
            "placed": int(self.placed),
            "fo_cnt": int(self.fo_cnt),
            "fo_hit": int(self.fo_hit),
            "fo_miss": int(self.fo_miss),

            # 归一化辅助（给 RL 奖励用）
            "cost_norm": self.cost_norm.scale(avg_cost_total),
            "lat_norm": self.lat_norm.scale(avg_latency),

            # 元标签
            "topo_size": self.meta.get("topo_size", ""),
            "sfc_len_group": self.meta.get("sfc_len_group", ""),
        }


# =========================
# 训练奖励（中央 DQN）
# =========================

def compute_central_reward(summary: Dict[str, Any],
                           weights: Dict[str, float],
                           unused_quota_ratio: float = 0.0) -> float:
    """
    把 episode 汇总映射为中央奖励（DQN 用）
    weights 中建议包含：
      - w_avail for emp_avail
      - w_hit   for fo_hit_rate
      - w_cost  for cost_norm（惩罚项，通常 < 0）
      - w_lat   for lat_norm  （惩罚项，通常 < 0）
      - w_place for place_rate
      - w_rel   for rel_pred_avg
      - w_unusd for 未用配额惩罚（负权）
    """
    emp_avail = float(summary.get("emp_avail", 0.0))
    fo_hit    = float(summary.get("fo_hit_rate", 0.0))
    cost_n    = float(summary.get("cost_norm", 0.0))
    lat_n     = float(summary.get("lat_norm", 0.0))
    place_r   = float(summary.get("place_rate", 0.0))
    rel_pred  = float(summary.get("rel_pred_avg", 0.0))

    w_avail = float(weights.get("w_avail", 1.0))
    w_hit   = float(weights.get("w_hit", 0.5))
    w_cost  = float(weights.get("w_cost", -0.5))
    w_lat   = float(weights.get("w_lat", -0.5))
    w_place = float(weights.get("w_place", 0.2))
    w_rel   = float(weights.get("w_rel", 0.2))
    w_unusd = float(weights.get("w_unusd", -0.1))

    reward = (
        w_avail * emp_avail +
        w_hit   * fo_hit +
        w_place * place_r +
        w_rel   * rel_pred +
        w_cost  * cost_n +
        w_lat   * lat_n +
        w_unusd * float(unused_quota_ratio)
    )
    return float(reward)


# =========================
# Episode 汇总 CSV 写入
# =========================

class EpisodeSummaryWriter:
    """
    将每集的汇总结果写入 CSV（收敛曲线数据源）
    """
    DEFAULT_FIELDS = [
        "ep",
        "avg_latency_ms",
        "place_rate",
        "fo_hit_rate",
        "rel_pred_avg",
        "emp_avail",
        "avg_cost_total",
        # 计数
        "attempted", "placed", "fo_cnt", "fo_hit", "fo_miss",
        # 归一化
        "cost_norm", "lat_norm",
        # 元标签
        "topo_size", "sfc_len_group",
        # 也可以在 runner 里扩展写入 N 等控制变量
    ]

    def __init__(self, csv_path: Path, fieldnames: Optional[List[str]] = None) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames or self.DEFAULT_FIELDS

        self._f = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        if self._f.tell() == 0:
            self._w.writeheader()

    def write(self, ep: int, summary: Dict[str, Any]) -> None:
        row = {"ep": int(ep)}
        for k in self.fieldnames:
            if k == "ep":
                continue
            row[k] = summary.get(k, "")
        self._w.writerow(row)
        # 可视需要 flush
        # self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


# =========================
# 小工具
# =========================

def _to_float(x) -> float:
    try:
        if x == "" or x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0