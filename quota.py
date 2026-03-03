from __future__ import annotations
"""
（配额）
	•	此处配额极简：上层只输出 N，不需要段×灾域维度；保留接口便于扩展（例如按段或按 DZ 约束）。
"""
# quota.py
# -*- coding: utf-8 -*-
"""
QuotaManager：中央层“路径数量 N”的配额与利用度跟踪
- 设计目标：为中央 DQN 的奖励提供一个可解释的“未用配额率”惩罚信号
- 使用方式：
    qm = QuotaManager(N_min=2, N_max=5, smooth_tau=0.9)
    qm.set_epoch_quota(N=3)           # 一轮/一周期开始时设置中央动作（决定的 N）
    ...
    qm.consume_for_request(used_paths=N_used)  # 每次 place 后登记“使用了多少条路径”
    ...
    ratio = qm.unused_ratio()         # 计算未用配额率（0~1），用于 compute_central_reward
"""


from dataclasses import dataclass


@dataclass
class _EMA:
    """简单指数滑动平均器：y ← tau*y + (1-tau)*x"""
    tau: float
    value: float = 0.0
    inited: bool = False

    def update(self, x: float) -> None:
        if not self.inited:
            self.value = float(x)
            self.inited = True
        else:
            self.value = float(self.tau * self.value + (1.0 - self.tau) * x)

    def get(self) -> float:
        return float(self.value)


class QuotaManager:
    """
    路径数配额管理：
      - N_min / N_max：合法动作区间
      - epoch_quota_N：本轮中央设置的目标路径数（动作）
      - used_paths_ema：按“每次请求实际选择的路径数”做 EMA，反映边缘/启发式的真实使用度
      - 未用配额率 unused_ratio = max(0, 1 - used_paths_ema/epoch_quota_N)
        （直观解读：中央要 N=4，但边缘平均只用到 3，则存在 25% 未用）
    """
    def __init__(self, N_min: int = 2, N_max: int = 5, smooth_tau: float = 0.9) -> None:
        assert N_min >= 2 and N_max >= N_min
        self.N_min = int(N_min)
        self.N_max = int(N_max)
        self.epoch_quota_N = int(N_min)  # 缺省
        self.used_paths_ema = _EMA(tau=float(smooth_tau))

        # 计数（可选，用于调试/记录）
        self.req_attempted = 0
        self.req_placed = 0

    # —— 生命周期接口 —— #
    def clip_N(self, N: int) -> int:
        """将外部传入的 N 裁剪到合法动作区间"""
        return max(self.N_min, min(self.N_max, int(N)))

    def set_epoch_quota(self, N: int) -> None:
        """在一轮/一周期开始时设置中央动作（决定的路径数 N）"""
        self.epoch_quota_N = self.clip_N(N)

    def consume_for_request(self, used_paths: int, placed: bool = True) -> None:
        """
        在每次 place 决策后调用：
          - used_paths：这次实际使用/预留了多少条路径（通常等于目标 N；若失败或退化则可能小于 N）
          - placed：是否放置成功（仅用于计数）
        """
        self.req_attempted += 1
        if placed:
            self.req_placed += 1

        # 平滑记录“实际使用路径条数”
        up = max(0, int(used_paths))
        self.used_paths_ema.update(up)

    def unused_ratio(self) -> float:
        """未用配额率：1 - (EMA(used_paths) / 设定配额N)"""
        denom = max(1, int(self.epoch_quota_N))
        frac = 1.0 - (self.used_paths_ema.get() / float(denom))
        return float(max(0.0, min(1.0, frac)))

    # —— 可选：统计与重置 —— #
    def stats(self) -> dict:
        return {
            "N_min": self.N_min,
            "N_max": self.N_max,
            "N_quota": self.epoch_quota_N,
            "used_paths_ema": self.used_paths_ema.get(),
            "unused_ratio": self.unused_ratio(),
            "req_attempted": self.req_attempted,
            "req_placed": self.req_placed,
        }

    def reset_episode(self) -> None:
        self.req_attempted = 0
        self.req_placed = 0
        # EMA 不清零，跨集平滑；若你希望每集重置，可在外部手动 new 一个 QuotaManager