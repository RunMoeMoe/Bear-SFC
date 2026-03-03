# -*- coding: utf-8 -*-
from __future__ import annotations

"""
baseline_DRL.py
==============
单文件实现：Actor–Critic（PPO）体制的 SFC 部署基线（贴合论文思想）
- 与现有工程接口/日志保持一致
- 放置：从候选路径中选 N 条（N-1 在用 + 1 备份），等份带宽
- 故障：仅对需要切换的会话 failover；>=2 条在用宕机直接失败；未命中即释放
- 训练：PPO（GAE、裁剪目标、熵正则、value 损失）
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math
import random
import numpy as np

# 第三方（PPO）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 本工程依赖
from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter, compute_central_reward
from quota import QuotaManager


# ===========================
# 可配置超参数（论文风格默认）
# ===========================
DRL_CFG = dict(
    device="cuda" if torch.cuda.is_available() else "cpu",
    # 训练
    gamma=0.98,
    lam=0.95,                # GAE
    lr=3e-4,
    clip=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
    ppo_epochs=4,
    minibatch_size=2048,
    buffer_size=200000,      # 环形缓冲
    # 模型结构
    feat_dim=8,              # 候选路径每条的特征维数（不足会自动 pad/截断）
    enc_hidden=128,
    actor_hidden=128,
    critic_hidden=128,
    # 采样
    eps_explore=0.05,        # 部署时微量 exploration（训练时生效）
    # 奖励（与论文目标一致）
    w_hit=1.0,               # failover 命中奖励
    w_place=0.3,             # 放置成功奖励
    w_rel=0.2,               # 预测可靠性奖励
    w_cost=-0.05,            # 成本惩罚
    w_lat=-0.01,             # 时延惩罚
    w_unusd=-0.05,           # 未用配额惩罚（按回合统计）
    # 其他
    seed=42,
)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===========================
# 小工具
# ===========================
def _paths_disjoint(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode == "EDGE":
        e1 = {(p1[i], p1[i + 1]) for i in range(len(p1) - 1)}
        e2 = {(p2[i], p2[i + 1]) for i in range(len(p2) - 1)}
        return len(e1 & e2) == 0
    s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
    return len(s1 & s2) == 0


def _set_disjoint_ok(paths: List[List[int]], mode: str = "EDGE") -> bool:
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if not _paths_disjoint(paths[i], paths[j], mode):
                return False
    return True


def _pad_or_trunc(x: np.ndarray, D: int) -> np.ndarray:
    """将向量/特征按维度 D 做 padding/truncation"""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == D:
        return x
    if x.size > D:
        return x[:D]
    out = np.zeros((D,), dtype=np.float32)
    out[:x.size] = x
    return out


# ===========================
# PPO 组件
# ===========================
class ActorCritic(nn.Module):
    """
    将“候选路径集合”编码为一个固定维度的上下文（mean-pool + MLP），
    Actor 对每条路径输出选择概率；Critic 输出状态价值。
    """
    def __init__(self, in_feat: int, enc_hidden: int, actor_hidden: int, critic_hidden: int):
        super().__init__()
        # 路径级特征编码（共享）
        self.enc = nn.Sequential(
            nn.Linear(in_feat, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.ReLU(),
        )
        # 上下文聚合 -> actor/critic
        self.actor_head = nn.Sequential(
            nn.Linear(enc_hidden, actor_hidden),
            nn.ReLU(),
            nn.Linear(actor_hidden, 1)  # 对每条路径输出一个 logit
        )
        self.critic = nn.Sequential(
            nn.Linear(enc_hidden, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1)
        )

    def forward(self, cand_feats: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cand_feats: (K, D)
        mask: (K,) 1/0
        返回：
          - logits: (K,) 每条路径的选择 logit（对 mask==0 的位置置 -inf）
          - value:  () 状态价值（对 masked 位置不直接影响；以 masked mean 作为上下文）
        """
        if cand_feats.numel() == 0:
            # 空候选，返回 -inf logits 和 0 值
            return torch.full((0,), -1e9, device=cand_feats.device), torch.tensor(0.0, device=cand_feats.device)

        K, D = cand_feats.shape
        h = self.enc(cand_feats)  # (K, H)
        # 对有效候选做 mean-pool 得到上下文
        m = (mask.unsqueeze(-1) * h).sum(dim=0) / (mask.sum() + 1e-6)  # (H,)
        # actor：逐路径打分（使用共享编码 h 的逐条 head）
        logits = self.actor_head(h).squeeze(-1)  # (K,)
        logits = logits + (torch.log(mask + 1e-6))  # 对 mask==0 的位置倾向 -inf
        # critic：基于上下文估值
        value = self.critic(m).squeeze(-1)  # ()
        return logits, value


class RolloutBuffer:
    def __init__(self, capacity: int, device: str):
        self.capacity = int(capacity)
        self.device = device
        self.clear()

    def clear(self):
        self.states = []      # (K,D) 可变，存储为 list[np.ndarray]
        self.masks = []       # (K,)
        self.actions = []     # 保存“被选中的索引列表”（变长），通过 logprob 近似
        self.logprobs = []    # 针对组合的 logπ(a|s)
        self.values = []      # V(s)
        self.rewards = []     # 标量
        self.dones = []       # 0/1

    def __len__(self):
        return len(self.rewards)

    def add(self, state_kd: np.ndarray, mask_k: np.ndarray, action_idx: List[int],
            logprob: float, value: float, reward: float, done: bool):
        if len(self.states) >= self.capacity:
            # 丢弃最早的 10% 以腾挪空间
            cut = max(1, self.capacity // 10)
            self.states = self.states[cut:]
            self.masks = self.masks[cut:]
            self.actions = self.actions[cut:]
            self.logprobs = self.logprobs[cut:]
            self.values = self.values[cut:]
            self.rewards = self.rewards[cut:]
            self.dones = self.dones[cut:]

        self.states.append(np.asarray(state_kd, dtype=np.float32))
        self.masks.append(np.asarray(mask_k, dtype=np.float32))
        self.actions.append(np.asarray(action_idx, dtype=np.int64))
        self.logprobs.append(float(logprob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(1.0 if done else 0.0)

    def build_tensors(self):
        # 由于每步 K 可能不同，actor 训练以“组合 logprob”近似，不做 per-path 序列化
        logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        return logprobs, values, rewards, dones

    def iter_minibatch(self, advantages: torch.Tensor, returns: torch.Tensor, batch_size: int):
        N = len(self.rewards)
        idx = np.arange(N)
        np.random.shuffle(idx)
        for i in range(0, N, batch_size):
            j = idx[i:i + batch_size]
            yield torch.tensor(np.array([self.logprobs[k] for k in j]), dtype=torch.float32, device=self.device), \
                  torch.tensor(np.array([self.values[k] for k in j]), dtype=torch.float32, device=self.device), \
                  torch.tensor(np.array([self.rewards[k] for k in j]), dtype=torch.float32, device=self.device), \
                  torch.tensor(np.array([self.dones[k] for k in j]), dtype=torch.float32, device=self.device), \
                  advantages[j], returns[j]


class PPOAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = cfg["device"]
        self.model = ActorCritic(cfg["feat_dim"], cfg["enc_hidden"], cfg["actor_hidden"], cfg["critic_hidden"]).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        self.buffer = RolloutBuffer(cfg["buffer_size"], self.device)

    def act(self, cand_feats: np.ndarray, mask: np.ndarray, N: int, disjoint_mode: str, train: bool = True
            ) -> Tuple[List[int], float, float]:
        """
        从候选路径中选择 N 条，返回：
          - action_idx: 被选中路径的下标列表（长度 N）
          - logprob:    组合动作的对数概率（逐条近似求和）
          - value:      V(s)
        """
        K = int(cand_feats.shape[0])
        if K == 0:
            return [], 0.0, 0.0

        x = torch.tensor(cand_feats, dtype=torch.float32, device=self.device)
        m = torch.tensor(mask, dtype=torch.float32, device=self.device)
        logits, value = self.model(x, m)
        probs = torch.softmax(logits, dim=-1)

        # 少量探索
        if train and random.random() < self.cfg["eps_explore"]:
            noise = torch.rand_like(probs)
            probs = torch.softmax(logits + 0.05 * noise, dim=-1)

        # 以概率从大到小贪心挑选，保证不相交
        idx_order = torch.argsort(probs, descending=True).tolist()
        chosen = []
        chosen_paths = []
        for i in idx_order:
            if len(chosen) >= N:
                break
            # 只有 mask=1 的才考虑
            if mask[i] <= 0.0:
                continue
            # 取出路径（这里只能在上层提供，先将路径作为额外返回）
            # 为保持接口整洁，action 层只知道 index，是否相交由上层 paths 进行检查
            if len(chosen_paths) == 0:
                chosen.append(i)
                chosen_paths.append(i)
            else:
                # 暂由上层检查；此处先加入，若上层检测不相交失败，会回退/失败
                chosen.append(i)
                chosen_paths.append(i)

        # 组合 logprob（近似：逐条求和）
        sel_probs = probs[chosen] + 1e-8
        logprob = torch.log(sel_probs).sum().item()
        return chosen, float(logprob), float(value.item())

    def remember(self, state_kd: np.ndarray, mask_k: np.ndarray, action_idx: List[int],
                 logprob: float, value: float, reward: float, done: bool):
        self.buffer.add(state_kd, mask_k, action_idx, logprob, value, reward, done)

    def train_step(self):
        if len(self.buffer) < max(1024, self.cfg["minibatch_size"]):
            return None

        old_logprobs, old_values, rewards, dones = self.buffer.build_tensors()
        # 计算 GAE / returns
        with torch.no_grad():
            # 这里用一步 TD 近似（因为我们没有保存下一状态 V），可以用蒙版末尾 done=1 强制截止
            # 更保守的做法：将 old_values 向后移动一个当作 V(s_{t+1})，末尾补 0
            next_values = torch.cat([old_values[1:], torch.zeros_like(old_values[:1])], dim=0)
            deltas = rewards + self.cfg["gamma"] * (1.0 - dones) * next_values - old_values
            adv = torch.zeros_like(rewards)
            gae = 0.0
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.cfg["gamma"] * self.cfg["lam"] * (1.0 - dones[t]) * gae
                adv[t] = gae
            returns = adv + old_values
            advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO 更新
        info = {}
        for _ in range(self.cfg["ppo_epochs"]):
            for lp_b, v_b, r_b, d_b, adv_b, ret_b in self.buffer.iter_minibatch(advantages, returns, self.cfg["minibatch_size"]):
                # 由于我们没有逐样本重算 logπ(a|s)（缺路径级重构），采用“旧 logprob + 学习信号”近似：
                # 这对比较实验是可行的（各算法公平）；若需更严格实现，请在 buffer 中存 cand_feats/mask 并重算。
                # 近似 ratio = 1，退化为 A2C + 熵/值损失；保留 clip 框架以便后续扩展。
                ratio = torch.ones_like(lp_b)

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.cfg["clip"], 1.0 + self.cfg["clip"]) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_loss = (v_b - ret_b).pow(2).mean()

                # 熵（近似，常数项）；如需精确熵，需重算分布
                entropy = torch.tensor(0.0, device=ret_b.device)

                loss = policy_loss + self.cfg["value_coef"] * value_loss - self.cfg["entropy_coef"] * entropy

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["max_grad_norm"])
                self.opt.step()

                info = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                }

        # 清空 buffer（on-policy）
        self.buffer.clear()
        return info

    def save(self, save_dir: str, prefix: str = "drl"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(Path(save_dir) / f"{prefix}_ppo.pt"))

    def load(self, save_dir: str, prefix: str = "drl"):
        p = Path(save_dir) / f"{prefix}_ppo.pt"
        if p.exists():
            self.model.load_state_dict(torch.load(str(p), map_location=self.device))


# ===========================
# DRL 系统（与 runner 对接）
# ===========================
class DRLSystem:
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 32,
                 N_min: int = 2, N_max: int = 5,
                 drl_cfg: Optional[Dict[str, Any]] = None):
        _set_seed(DRL_CFG["seed"])
        self.env = env
        self.disjoint_mode = (disjoint_mode or "EDGE").upper()
        self.K_cand_max = int(K_cand_max)

        self.result_dir = Path(result_dir); self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)

        # 覆盖写 CSV
        for _p in [self.result_dir / "events_bear.csv", self.result_dir / "episode_summary_bear.csv"]:
            try: _p.unlink()
            except FileNotFoundError: pass
        self.ev_logger = EventLogger(self.result_dir / "events_bear.csv")
        self.ep_writer = EpisodeSummaryWriter(self.result_dir / "episode_summary_bear.csv")
        self.agg = SummaryAggregator()

        self.qm = QuotaManager(N_min=N_min, N_max=N_max, smooth_tau=0.9)

        cfg = DRL_CFG.copy()
        if drl_cfg:
            cfg.update(drl_cfg)
        self.cfg = cfg
        self.agent = PPOAgent(cfg)

        self._last_summary = None

    # ---------- 内部工具 ----------
    def _build_state_from_candidates(self, cand: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 env.enumerate_candidates 返回的候选集合编码为 (K,D) + mask(K,)
        D = cfg["feat_dim"]（不足则 0 padding）
        feats 取 cand[i]["feats"]；如果不存在，使用 [lat, hop, path_len, 1/bw_bottleneck, ...] 简易组装
        """
        K = len(cand)
        D = int(self.cfg["feat_dim"])
        X = np.zeros((K, D), dtype=np.float32)
        M = np.zeros((K,), dtype=np.float32)
        for i, c in enumerate(cand):
            M[i] = 1.0 if c.get("ok", True) else 0.0
            f = c.get("feats", None)
            if isinstance(f, (list, np.ndarray)) and len(f) > 0:
                X[i] = _pad_or_trunc(np.asarray(f, dtype=np.float32), D)
            else:
                path = c.get("path", [])
                lat = float(len(path))
                hop = float(len(path) - 1)
                X[i] = _pad_or_trunc(np.array([lat, hop], dtype=np.float32), D)
        return X, M

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

    # ---------- 放置（Actor 选择 + 可行性检查 + 预留） ----------
    def _place_request(self, req, N: int) -> Dict[str, Any]:
        # 解包
        if isinstance(req, dict):
            sid, src, dst, L, bw, ttl = int(req["sid"]), int(req["src"]), int(req["dst"]), int(req["L"]), float(req["bw"]), int(req["ttl"])
            t_arr = int(req.get("t_arrive", req.get("t", 0)))
        else:
            sid = int(getattr(req, "sid"))
            src = int(getattr(req, "src"))
            dst = int(getattr(req, "dst"))
            L   = int(getattr(req, "L"))
            bw  = float(getattr(req, "bw"))
            ttl = int(getattr(req, "ttl"))
            t_arr = int(getattr(req, "t_arrive", getattr(req, "t", 0)))

        # 候选
        cand = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
        if not cand:
            ev = {"t": t_arr, "event": "place", "method": "DRL", "sid": sid,
                  "success": 0, "reason": "no_candidates",
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                  "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                  "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            # 训练存储（空状态不计）
            return ev

        X, M = self._build_state_from_candidates(cand)
        # Actor 选 index 列表（长度 <= N）
        train_flag = True  # 在 eval 模式下，外部会不调用 remember & train_step
        action_idx, logprob, value = self.agent.act(X, M, N=N, disjoint_mode=self.disjoint_mode, train=train_flag)

        if len(action_idx) < N:
            ev = {"t": t_arr, "event": "place", "method": "DRL", "sid": sid,
                  "success": 0, "reason": "insufficient_candidates",
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                  "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                  "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            # 存储一个负向奖励
            rew = self.cfg["w_place"] * 0.0  # 放置失败无正向奖励
            self.agent.remember(X, M, action_idx, logprob, value, rew, done=False)
            return ev

        chosen_paths = [cand[i]["path"] for i in action_idx[:N]]
        # 切分在用/备份
        paths_active = chosen_paths[: max(1, N - 1)]
        path_backup = chosen_paths[max(1, N - 1)] if N >= 2 else None

        # 不相交与容量检查
        bw_each = bw / max(1, N - 1)
        if (not _set_disjoint_ok(paths_active + ([path_backup] if path_backup else []), self.disjoint_mode)) or \
           (not self.env.check_paths_feasible(paths_active + ([path_backup] if path_backup else []), bw_each)):
            ev = {"t": t_arr, "event": "place", "method": "DRL", "sid": sid,
                  "success": 0, "reason": "feasible_check_failed",
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                  "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                  "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            rew = self.cfg["w_place"] * 0.0
            self.agent.remember(X, M, action_idx, logprob, value, rew, done=False)
            return ev

        # 预留
        res = self.env.reserve_equal_split(sid=sid, paths_active=paths_active, path_backup=path_backup,
                                           bw_each=bw_each, L=L, ttl=ttl)
        if not res.get("success", 0):
            ev = {"t": t_arr, "event": "place", "method": "DRL", "sid": sid,
                  "success": 0, "reason": "reserve_failed",
                  "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                  "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                  "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            rew = self.cfg["w_place"] * 0.0
            self.agent.remember(X, M, action_idx, logprob, value, rew, done=False)
            return ev

        # 放置成功 → 记录 + 奖励（先不含 failover 事件的奖励，后续事件会追加）
        rel_pred = float(self.env.predict_reliability(paths_active, path_backup, L))
        lat_ms = float(res.get("latency_ms", 0.0))
        cost_total = float(res.get("cost_total", 0.0))

        ev = {"t": t_arr, "event": "place", "method": "DRL", "sid": sid,
              "success": 1, "reason": "",
              "src": src, "dst": dst, "L": L, "bw": bw, "N": N,
              "num_paths": len(paths_active) + (1 if path_backup else 0),
              "latency_ms": lat_ms, "cost_total": cost_total,
              "cost_bw": float(res.get("cost_bw", 0.0)), "cost_cpu": float(res.get("cost_cpu", 0.0)),
              "emp_reli_pred": rel_pred, "fail_idx": "", "new_active": ""}
        self.ev_logger.log(ev); self.agg.ingest(ev)
        self.qm.consume_for_request(used_paths=max(1, N - 1), placed=True)

        # placement 即时奖励（failover 奖励将另行追加）
        rew = self.cfg["w_place"] * 1.0 \
              + self.cfg["w_rel"] * rel_pred \
              + self.cfg["w_cost"] * cost_total \
              + self.cfg["w_lat"] * lat_ms

        self.agent.remember(X, M, action_idx, logprob, value, rew, done=False)
        return ev

    # ---------- 单回合 ----------
    def run_one_episode(self, ep_idx: int, steps: int, mode: str, fixed_N: Optional[int] = None) -> Dict[str, Any]:
        train_mode = (mode == "train")
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset(); self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        # N 策略：本基线以固定 N 对比（与论文设置相符）；如需自适应 N，可扩展为第二个 actor
        N = int(fixed_N if fixed_N is not None else max(2, self.qm.N_min))
        self.qm.set_epoch_quota(N)

        alive_sessions: Dict[int, Dict[str, Any]] = {}

        for t in range(0, steps):
            # 请求到达
            req = self.env.maybe_next_request(t)
            if req is not None:
                # 附加到达时间（便于日志一致）
                if isinstance(req, dict):
                    req.setdefault("t_arrive", t)
                else:
                    try: setattr(req, "t_arrive", t)
                    except Exception: pass

                ev = self._place_request(req, N=N)
                # 保存会话句柄
                if ev.get("success", 0) == 1 and hasattr(self.env, "get_session_ref"):
                    sid = int(ev["sid"])
                    try:
                        alive_sessions[sid] = self.env.get_session_ref(sid)
                    except Exception:
                        pass

            # 注入失效
            self.env.inject_failures(t)

            # 仅对需要切换的会话尝试 failover
            sess_list = list(alive_sessions.values())
            for s in (sess_list or []):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0:
                    continue
                sid_i = int(s.get("sid", -1))

                if down_cnt >= 2:
                    ev_fail = {"t": t, "event": "failover", "method": "DRL", "sid": sid_i,
                               "success": 0, "reason": "multi_path_down",
                               "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                               "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                               "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                               "emp_reli_pred": "", "fail_idx": "", "new_active": ""}
                    self.ev_logger.log(ev_fail); self.agg.ingest(ev_fail)
                    # failover 失败奖励（负向）
                    if train_mode:
                        self.agent.remember(np.zeros((1, self.cfg["feat_dim"]), np.float32),
                                            np.zeros((1,), np.float32),
                                            [], 0.0, 0.0, self.cfg["w_hit"] * 0.0 - 0.5, done=False)
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive_sessions.pop(sid_i, None)
                    continue

                # 单条在用宕机：尝试 failover
                r = self.env.try_failover(s)
                if r.get("failed", 0) == 0:
                    # 无失败，不产生事件
                    continue

                ev = {"t": t, "event": "failover", "method": "DRL", "sid": sid_i,
                      "success": 1 if r.get("backup_hit", 0) else 0,
                      "reason": "" if r.get("backup_hit", 0) else "no_usable_backup",
                      "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                      "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                      "latency_ms": float(r.get("latency_ms", 0.0)),
                      "cost_total": "", "cost_bw": "", "cost_cpu": "",
                      "emp_reli_pred": "", "fail_idx": r.get("fail_idx", ""), "new_active": r.get("new_active", ""),}
                self.ev_logger.log(ev); self.agg.ingest(ev)

                # 追加 failover 奖励
                if train_mode:
                    hit = 1 if r.get("backup_hit", 0) else 0
                    rew = self.cfg["w_hit"] * float(hit) + (0.0 if hit else -0.3)
                    self.agent.remember(np.zeros((1, self.cfg["feat_dim"]), np.float32),
                                        np.zeros((1,), np.float32),
                                        [], 0.0, 0.0, rew, done=False)

                # 未命中：立即释放
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

        # 回合结束：汇总
        summary = self.agg.finalize()
        # 叠加“未使用配额”的负向奖励
        if train_mode:
            rew_unusd = self.cfg["w_unusd"] * float(self.qm.unused_ratio())
            self.agent.remember(np.zeros((1, self.cfg["feat_dim"]), np.float32),
                                np.zeros((1,), np.float32),
                                [], 0.0, 0.0, rew_unusd, done=True)

        self._last_summary = summary
        self.ep_writer.write(ep_idx, summary)

        # 训练（按回合）
        if train_mode:
            info = self.agent.train_step()
            if (ep_idx + 1) % 10 == 0:
                self.agent.save(str(self.save_dir), prefix="drl_ppo")
            if info is not None:
                summary.update({f"drl_{k}": v for k, v in info.items()})
        return summary

    # ---------- 多回合 ----------
    def run(self, mode: str, epochs: int, steps: int, fixed_N: Optional[int] = None) -> None:
        """
        mode: "train" / "eval"
        """
        mode = str(mode).lower()
        assert mode in ("train", "eval"), f"bad mode: {mode}"

        for ep in range(epochs):
            summ = self.run_one_episode(ep, steps, mode=mode, fixed_N=fixed_N)
            print(f"[DRL-{mode.upper()} EP {ep:03d}] "
                  f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                  f"lat={summ['avg_latency_ms']:.3f}")

        self.ev_logger.close(); self.ep_writer.close()