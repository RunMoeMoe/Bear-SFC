from __future__ import annotations
"""
（边缘 PPO ⇒ 选择具体 N 条路径）
	•	输入状态 s_E：
	•	候选路径集 \mathcal{P}（来自 k_shortest 或 disjoint_paths），每条的特征：跳数、距离、与其他路径的共享边/节点/DZ 关系、当前链路/节点剩余资源、单条路径可用性估计 A_p、单条路径成本等。
	•	动作 a_E：在 \mathcal{P} 中选一个 N 元子集（可用组合策略：先按打分排序后取前 N，或序列化选择逐条加入并约束不相交/不冲突）。
	•	可行性检查：
	•	带宽：对每条被选路径，检查 can_reserve_path_bw(path, M/(N-1))；失败则触发回溯/改选。
	•	CPU：该 SFC 的 L 个 VNF，在被选路径上逐个部署：对每条路径每个 VNF 节点检查 can_reserve_cpu(node, u_cpu)（热备同量或乘 \eta）。
	•	即时奖励（放置时）：
r_E^{place} = \alpha_1\,\widehat{\Delta R} - \alpha_2\,\widehat{cost} - \alpha_3\,\widehat{latency}
其中 \Delta R 可取“与 N-1 条方案相比的可靠性提升”或“与 N 的相邻方案差异”。
	•	即时奖励（故障时）：
r_E^{fail} = \mathbb{1}\{\text{命中}\} - \beta_1\cdot \text{并发失效惩罚} - \beta_2\,\widehat{段成本}
	•	支持 collect_trajectories()（训练）/inference()（评估）。
"""

# algo_edge.py
# -*- coding: utf-8 -*-
"""
Edge 侧路径选择算法：
- HeuristicSelector：启发式 Top-N（无训练，评估/冻结边策略）
- EdgePPO：轻量 PPO，输入 K×D 候选特征，输出 N 个索引（Gumbel-Top-k 近似无放回）
通用接口：
  act(edge_state: dict, explore: bool) -> List[int]
  collect_transition(state, indices, reward, done) -> None
  train_step() -> Optional[dict]
  save(out_dir), load(out_dir)
"""


from typing import List, Dict, Any, Optional, Tuple
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =========================
# 工具：Gumbel-Top-k 采样
# =========================

def gumbel_noise(shape, device):
    u = torch.rand(shape, device=device).clamp_(1e-8, 1.0 - 1e-8)
    return -torch.log(-torch.log(u))

def gumbel_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    logits: (K,)
    返回 top-k 的索引（无放回近似采样）
    """
    g = gumbel_noise(logits.shape, logits.device)
    y = logits + g
    return torch.topk(y, k=min(k, logits.numel()), dim=0).indices


# =========================
# 启发式策略（无训练）
# =========================

class HeuristicSelector:
    """
    对候选做简单线性打分并取 Top-N：
      score = w1 * (-hops) + w2 * (-dist) + w3 * residual_bw + w4 * dz_cross
    你可以在 bear.py 侧构造 cand_feats 的列含义与权重一致：
      e.g. feats = [hops, dist, residual_min, dz_cross_frac, ...]
    """
    def __init__(self, weights: Optional[List[float]] = None):
        # 缺省：越短越好、越低跳数越好、越大剩余带宽越好、跨 DZ 更好
        self.weights = weights  # None 时在 act() 动态匹配前4维

    def act(self, edge_state: Dict[str, Any], explore: bool = False) -> List[int]:
        feats: np.ndarray = edge_state["cand_feats"]  # (K, D)
        N: int = int(edge_state["N"])
        mask: Optional[np.ndarray] = edge_state.get("mask", None)

        K, D = feats.shape
        w = self._get_weights(D)  # (D,)
        scores = (feats @ w.reshape(-1, 1)).reshape(-1)  # (K,)

        if mask is not None:
            scores = np.where(mask > 0.5, scores, -1e9)

        topk = int(min(N, K))
        idx = np.argpartition(-scores, kth=topk - 1)[:topk]
        # 排序
        idx = idx[np.argsort(-scores[idx])]
        return idx.tolist()

    def collect_transition(self, *args, **kwargs):
        pass

    def train_step(self):
        return None

    def save(self, out_dir: str, prefix: str = "edge_heuristic"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}.npz")
        np.savez(path, weights=(self.weights if self.weights is not None else np.array([])))

    def load(self, out_dir: str, prefix: str = "edge_heuristic"):
        path = os.path.join(out_dir, f"{prefix}.npz")
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            w = data.get("weights", np.array([]))
            if w.size > 0:
                self.weights = w

    def _get_weights(self, D: int) -> np.ndarray:
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float32)
            if w.shape[0] != D:
                # 自动扩展/截断
                w2 = np.zeros((D,), dtype=np.float32)
                w2[: min(D, w.shape[0])] = w[: min(D, w.shape[0])]
                return w2
            return w.astype(np.float32)
        # 默认前4维：[-hops, -dist, +residual_bw, +dz_cross]
        w = np.zeros((D,), dtype=np.float32)
        if D >= 1: w[0] = -1.0   # hops
        if D >= 2: w[1] = -1.0   # dist
        if D >= 3: w[2] = +1.0   # residual_min / capacity margin
        if D >= 4: w[3] = +0.5   # dz_cross_frac
        return w


# =========================
# Edge PPO（轻量）
# =========================

class _PolicyNet(nn.Module):
    """逐候选打分的共享 MLP：输入 D -> 输出 1（logit）"""
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (64, 64)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, D) or (K, D)
        orig_shape = x.shape
        if x.dim() == 2:
            y = self.net(x)              # (K, 1)
            return y.squeeze(-1)         # (K,)
        elif x.dim() == 3:
            B, K, D = x.shape
            y = self.net(x.view(B * K, D)).view(B, K, 1)   # (B, K, 1)
            return y.squeeze(-1)         # (B, K)
        else:
            raise ValueError("PolicyNet: bad input shape")


class _ValueNet(nn.Module):
    """对整组候选做一个全局 Value 估计（采用候选均值池化）"""
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (64, 64)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(inplace=True),
            nn.Linear(h1, h2), nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (K, D) 或 (B, K, D)
        先对 K 维做均值池化得到 (D,) 或 (B, D)，再做回归
        """
        if x.dim() == 2:
            feat = x.mean(dim=0, keepdim=True)  # (1, D)
            return self.net(feat).squeeze(-1)   # (1,) -> scalar
        elif x.dim() == 3:
            feat = x.mean(dim=1)                # (B, D)
            y = self.net(feat).squeeze(-1)      # (B,)
            return y
        else:
            raise ValueError("ValueNet: bad input shape")


class EdgePPO:
    """
    轻量版 PPO：
      - act():  logits -> 采样 / Top-N
      - collect_transition(): 保存 (feats, logits, chosen_idx, mask, reward, done)
      - train_step(): PPO 损失（clip policy + value baseline + entropy）
    近似：选择集合的 log-prob = 选中元素在 softmax 上的 log-prob 之和（忽略无放回归一项）
    """
    def __init__(self,
                 feat_dim: int,
                 hidden: Tuple[int, int] = (64, 64),
                 lr_pi: float = 3e-4,
                 lr_v: float = 1e-3,
                 clip_eps: float = 0.2,
                 entropy_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 2.0,
                 device: Optional[str] = None,
                 buffer_size: int = 2048):
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        self.policy = _PolicyNet(feat_dim, hidden).to(self.device)
        self.value  = _ValueNet(feat_dim, hidden).to(self.device)
        self.opt_pi = optim.Adam(self.policy.parameters(), lr=lr_pi)
        self.opt_v  = optim.Adam(self.value.parameters(),  lr=lr_v)

        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.vf_coef = float(vf_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.buffer: List[Dict[str, Any]] = []
        self.buffer_size = int(buffer_size)

        # 记录最近一次 act 的缓存（便于无需重复传 state）
        self._last_state_cache: Optional[Dict[str, Any]] = None

    @torch.no_grad()
    def act(self, edge_state: Dict[str, Any], explore: bool = True) -> List[int]:
        """
        edge_state:
          - cand_feats: np.ndarray, (K, D)
          - N: int
          - mask: 可选，(K,) 0/1
        返回：长度 N 的索引列表
        """
        feats_np: np.ndarray = edge_state["cand_feats"]
        N = int(edge_state["N"])
        mask_np: Optional[np.ndarray] = edge_state.get("mask", None)

        K, D = feats_np.shape
        feats = torch.as_tensor(feats_np, dtype=torch.float32, device=self.device)  # (K, D)
        logits = self.policy(feats)  # (K,)

        if mask_np is not None:
            mask = torch.as_tensor(mask_np, dtype=torch.float32, device=self.device)
            logits = logits + (mask - 1.0) * 1e9  # mask=0 => -1e9，剔除

        if explore:
            idx_t = gumbel_top_k(logits, k=N)  # 采样
        else:
            idx_t = torch.topk(logits, k=min(N, logits.numel())).indices  # 纯利用

        idx = idx_t.detach().cpu().tolist()

        # 缓存，供 collect_transition 使用
        self._last_state_cache = {
            "feats": feats.detach().cpu().numpy(),
            "logits": logits.detach().cpu().numpy(),
            "mask": (mask_np.copy() if mask_np is not None else None),
            "chosen_idx": np.array(idx, dtype=np.int64),
            "N": N,
        }
        return idx

    def collect_transition(self,
                           edge_state: Optional[Dict[str, Any]],
                           chosen_idx: List[int],
                           reward: float,
                           done: bool) -> None:
        """
        存入一条 transition。edge_state 可为 None（默认使用 act() 的缓存）。
        reward：你在 bear/bear_system 里根据 place/failover/成本等组合出来的即时/回合奖励
        done：一个 episode 的结束标记
        """
        if edge_state is None and self._last_state_cache is None:
            return
        if edge_state is not None:
            # 需要重新 forward 一次，得到 logits
            feats_np: np.ndarray = edge_state["cand_feats"]
            mask_np: Optional[np.ndarray] = edge_state.get("mask", None)
            feats = torch.as_tensor(feats_np, dtype=torch.float32, device=self.device)
            logits = self.policy(feats).detach().cpu().numpy()
            cache = {
                "feats": feats_np.copy(),
                "logits": logits.copy(),
                "mask": (mask_np.copy() if mask_np is not None else None),
                "chosen_idx": np.array(chosen_idx, dtype=np.int64),
                "N": int(edge_state["N"]),
            }
        else:
            cache = self._last_state_cache.copy()
            cache["chosen_idx"] = np.array(chosen_idx, dtype=np.int64)

        tr = {
            "feats": cache["feats"],      # (K, D)
            "logits": cache["logits"],    # (K,)
            "mask": cache["mask"],        # (K,) or None
            "chosen_idx": cache["chosen_idx"],  # (N,)
            "reward": float(reward),
            "done": bool(done),
        }
        self.buffer.append(tr)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def train_step(self,
                   ppo_epochs: int = 4,
                   batch_size: int = 64) -> Optional[Dict[str, float]]:
        """
        使用当前 buffer 做一次 PPO 更新（on-policy 近似）。为简化，这里不做 GAE，
        直接用 R - V 作为 advantage。
        """
        if len(self.buffer) == 0:
            return None

        # 打包 batch
        feats = [torch.as_tensor(t["feats"], dtype=torch.float32) for t in self.buffer]  # list(Ki, D)
        logits_old = [torch.as_tensor(t["logits"], dtype=torch.float32) for t in self.buffer]  # list(Ki,)
        masks = [None if t["mask"] is None else torch.as_tensor(t["mask"], dtype=torch.float32) for t in self.buffer]
        chosen = [torch.as_tensor(t["chosen_idx"], dtype=torch.long) for t in self.buffer]
        rewards = torch.as_tensor([t["reward"] for t in self.buffer], dtype=torch.float32)  # (B,)
        dones = torch.as_tensor([t["done"] for t in self.buffer], dtype=torch.float32)

        B = len(self.buffer)
        # 计算旧策略下的 log-prob（作为行为策略）
        with torch.no_grad():
            logps_old = []
            v_old = []
            for i in range(B):
                lo = logits_old[i]                # (Ki,)
                if masks[i] is not None:
                    lo = lo + (masks[i] - 1.0) * 1e9
                logp = (lo - lo.logsumexp(dim=0))  # log-softmax
                ch = chosen[i]                     # (N,)
                logp_sel = logp[ch].sum()          # 选择集合的 log-prob 近似
                logps_old.append(logp_sel)
                # value（对整组候选做均值池化）
                v = self.value(feats[i]).squeeze() # scalar
                v_old.append(v)
            logps_old = torch.stack(logps_old)  # (B,)
            v_old = torch.stack(v_old)          # (B,)

            adv = rewards - v_old
            # 标准化 advantage（有助于稳定）
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ret = rewards  # 无 GAE/多步回报，直接用 R 作为 target

        # 开始多 epoch 训练
        idxs = np.arange(B)
        pi_losses, v_losses, ent_losses = [], [], []
        for _ in range(ppo_epochs):
            np.random.shuffle(idxs)
            for st in range(0, B, batch_size):
                mb = idxs[st: st + batch_size]
                if len(mb) == 0:
                    continue

                # 拼装 mini-batch（不同 Ki，逐条累加 loss）
                pi_loss_mb = 0.0
                ent_mb = 0.0
                v_loss_mb = 0.0

                for i in mb:
                    f = feats[i].to(self.device)      # (Ki, D)
                    lo_old = logits_old[i].to(self.device)  # (Ki,)
                    ch = chosen[i].to(self.device)     # (N,)
                    mask_i = masks[i]
                    # 新策略 logits
                    lo_new = self.policy(f)            # (Ki,)
                    if mask_i is not None:
                        m = mask_i.to(self.device)
                        lo_old = lo_old + (m - 1.0) * 1e9
                        lo_new = lo_new + (m - 1.0) * 1e9

                    logp_old = (lo_old - lo_old.logsumexp(dim=0))[ch].sum()
                    logp_new = (lo_new - lo_new.logsumexp(dim=0))[ch].sum()

                    ratio = torch.exp(logp_new - logp_old)   # (scalar)
                    a = adv[i].to(self.device)
                    # PPO-clip
                    surr1 = ratio * a
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a
                    pi_obj = -torch.min(surr1, surr2)  # 取负做最小化

                    # 熵正则（提高探索）：对整组 logits 做 softmax entropy
                    p = torch.softmax(lo_new, dim=0)
                    ent = -(p * torch.log(p + 1e-8)).sum()

                    # Value 损失
                    v_pred = self.value(f).squeeze()
                    v_tgt = ret[i].to(self.device)
                    v_obj = nn.functional.mse_loss(v_pred, v_tgt)

                    pi_loss_mb += pi_obj
                    ent_mb += ent
                    v_loss_mb += v_obj

                # 归一化
                n = max(1, len(mb))
                pi_loss_mb = pi_loss_mb / n
                ent_mb = ent_mb / n
                v_loss_mb = v_loss_mb / n

                loss = pi_loss_mb + self.vf_coef * v_loss_mb - self.entropy_coef * ent_mb

                self.opt_pi.zero_grad(set_to_none=True)
                self.opt_v.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()),
                                         max_norm=self.max_grad_norm)
                self.opt_pi.step()
                self.opt_v.step()

                pi_losses.append(float(pi_loss_mb.item()))
                v_losses.append(float(v_loss_mb.item()))
                ent_losses.append(float(ent_mb.item()))

        # 清空 buffer（on-policy）
        self.buffer.clear()

        return {
            "pi_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "v_loss": float(np.mean(v_losses)) if v_losses else 0.0,
            "entropy": float(np.mean(ent_losses)) if ent_losses else 0.0,
        }

    def save(self, out_dir: str, prefix: str = "edge_ppo"):
        os.makedirs(out_dir, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }, os.path.join(out_dir, f"{prefix}.pt"))

    def load(self, out_dir: str, prefix: str = "edge_ppo", map_location: Optional[str] = None):
        path = os.path.join(out_dir, f"{prefix}.pt")
        dev = map_location or self.device
        ckpt = torch.load(path, map_location=dev)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])