from __future__ import annotations
"""
（中央 DQN ⇒ 决策 N）
	•	状态 s_C（周期聚合）：
	•	放置成功率 place_rate、故障重启成功率 fo_hit_rate、归一化成本 cost_norm、归一化时延 lat_norm、经验可用性 emp_avail、资源利用率（CPU/BW）、候选可用路径统计等。
	•	动作 a_C：选择 N\in \{2,\dots,N_{\max}\}。
	•	奖励 r_C（归一化后组合）：
r_C = \lambda_1 \,\widehat{emp\_avail} + \lambda_2\,\widehat{fo\_hit\_rate}
+ \lambda_3\,\widehat{place\_rate}
- \lambda_4\,\widehat{cost}
- \lambda_5\,\widehat{latency}
其中 \widehat{\cdot} 表示 0–1 归一化（以滑动均值或基线常数归一）。
	•	训练/推理分离，周期性更新/下发最新 N。
"""

# algo_central.py
# -*- coding: utf-8 -*-
"""
CentralDQN：中央层 DQN，离散动作空间（动作即路径数 N ∈ [N_min, N_max]）
- act(state, explore)    -> 选择一个 N
- remember/ train_step   -> 经验回放 + MLP DQN 更新
- save / load            -> 模型持久化
说明：
  1) state 的具体构造由 bear_system/bear.py 侧提供（建议包含：资源利用、时延/成本归一化、可用性等）
  2) reward 建议使用 metrics.compute_central_reward(summary, weights, unused_quota_ratio)
  3) N 的合法范围通过 set_action_space(N_min, N_max) 配置
"""


from typing import Tuple, Deque, List, Optional
from dataclasses import dataclass
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ========== MLP 网络 ==========

class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int] = (128, 128)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ========== DQN 主体 ==========

@dataclass
class DQNConfig:
    in_dim: int
    n_actions: int
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buf_size: int = 50_000
    start_learn_after: int = 1_000
    target_sync: int = 1_000
    eps_start: float = 0.2
    eps_end: float = 0.02
    eps_decay_steps: int = 50_000
    hidden: Tuple[int, int] = (128, 128)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CentralDQN:
    """
    离散 DQN，动作索引 a ∈ {0,...,n_actions-1} 映射到 N ∈ [N_min, N_max]
    """
    def __init__(self, cfg: DQNConfig, N_min: int = 2, N_max: int = 5) -> None:
        assert N_max >= N_min >= 2
        self.cfg = cfg
        self.N_min = int(N_min)
        self.N_max = int(N_max)
        self._rebuild_action_map()

        self.device = torch.device(cfg.device)
        self.q = _MLP(cfg.in_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.tgt = _MLP(cfg.in_dim, cfg.n_actions, cfg.hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)

        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=cfg.buf_size)
        self._step = 0
        self._eps = cfg.eps_start

        # 归一化辅助：可外部喂入已归一化 state；如需内部标准化，可在此增加 RunningNorm
        self._last_state_dim = cfg.in_dim

    # —— 动作空间 —— #
    def set_action_space(self, N_min: int, N_max: int) -> None:
        assert N_max >= N_min >= 2
        self.N_min = int(N_min)
        self.N_max = int(N_max)
        self._rebuild_action_map()

    def _rebuild_action_map(self) -> None:
        self._actions: List[int] = list(range(self.N_min, self.N_max + 1))
        self.n_actions = len(self._actions)

    def action_from_index(self, a_idx: int) -> int:
        return int(self._actions[a_idx])

    def index_from_action(self, N: int) -> int:
        N = max(self.N_min, min(self.N_max, int(N)))
        return self._actions.index(N)

    # —— 交互接口 —— #
    @torch.no_grad()
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """
        输入：已归一化/拼装好的状态向量（numpy 1D）
        输出：动作对应的 N（路径数）
        """
        self._step += 1
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        # 维度防御
        if s.shape[1] != self._last_state_dim:
            raise ValueError(f"CentralDQN.act: state dim {s.shape[1]} != configured {self._last_state_dim}")

        # ε-greedy
        if explore and random.random() < self._eps:
            a_idx = random.randrange(self.n_actions)
        else:
            qv = self.q(s)  # (1, n_actions)
            a_idx = int(torch.argmax(qv, dim=1).item())

        return self.action_from_index(a_idx)

    def remember(self, state: np.ndarray, action_N: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """写入经验回放（动作传入“实际的 N”，内部转换为索引）"""
        a_idx = self.index_from_action(action_N)
        self.buf.append((np.array(state, dtype=np.float32),
                         int(a_idx),
                         float(reward),
                         np.array(next_state, dtype=np.float32),
                         bool(done)))

    def train_step(self) -> Optional[dict]:
        """单步训练；当样本不足或未达预热步数时返回 None"""
        if len(self.buf) < self.cfg.start_learn_after:
            self._anneal_epsilon()
            return None

        batch = random.sample(self.buf, k=min(self.cfg.batch_size, len(self.buf)))
        s, a, r, s2, d = zip(*batch)
        s  = torch.as_tensor(np.stack(s),  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,            dtype=torch.long,   device=self.device).view(-1, 1)
        r  = torch.as_tensor(r,            dtype=torch.float32, device=self.device).view(-1, 1)
        s2 = torch.as_tensor(np.stack(s2), dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(d,            dtype=torch.float32, device=self.device).view(-1, 1)

        # Q(s,a)
        q_sa = self.q(s).gather(1, a)  # (B,1)

        with torch.no_grad():
            # Double DQN：a' = argmax_a Q(s',a); y = r + γ Q_tgt(s', a')
            a2 = torch.argmax(self.q(s2), dim=1, keepdim=True)  # (B,1)
            q_tgt_s2_a2 = self.tgt(s2).gather(1, a2)
            y = r + (1.0 - d) * self.cfg.gamma * q_tgt_s2_a2

        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.opt.step()

        # 同步 target
        if self._step % self.cfg.target_sync == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        # 退火 ε
        self._anneal_epsilon()

        return {"loss": float(loss.item()), "eps": float(self._eps)}

    def _anneal_epsilon(self) -> None:
        """线性退火 ε"""
        if self.cfg.eps_decay_steps <= 0:
            return
        frac = min(1.0, self._step / float(self.cfg.eps_decay_steps))
        self._eps = self.cfg.eps_start + (self.cfg.eps_end - self.cfg.eps_start) * frac

    # —— 持久化 —— #
    def save(self, out_dir: str, prefix: str = "central_dqn") -> None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}.pt")
        torch.save({
            "model": self.q.state_dict(),
            "target": self.tgt.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": self.cfg.__dict__,
            "N_min": self.N_min,
            "N_max": self.N_max,
            "step": self._step,
            "eps": self._eps,
        }, path)

    def load(self, out_dir: str, prefix: str = "central_dqn", map_location: Optional[str] = None) -> None:
        path = os.path.join(out_dir, f"{prefix}.pt")
        dev = map_location or self.cfg.device
        ckpt = torch.load(path, map_location=dev)
        self.q.load_state_dict(ckpt["model"])
        self.tgt.load_state_dict(ckpt.get("target", ckpt["model"]))
        self.opt.load_state_dict(ckpt["opt"])
        # 恢复配置（可选，若不一致以当前 cfg 为准）
        self.N_min = int(ckpt.get("N_min", self.N_min))
        self.N_max = int(ckpt.get("N_max", self.N_max))
        self._rebuild_action_map()
        self._step = int(ckpt.get("step", 0))
        self._eps = float(ckpt.get("eps", self.cfg.eps_end))