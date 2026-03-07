# -*- coding: utf-8 -*-
from __future__ import annotations
"""
baseline_BEAR_torch.py
======================
分层 BEAR（PyTorch 版，Central-SAC + Edge-PPO）
- Central（SAC）：学习输出 (N, alpha_logits)，N∈[N_min,N_max]，alpha 为候选偏好权重
- Edge（PPO）：对候选产生评分，与 alpha 结合形成总分，束搜索选 N 条两两不相交路径
- 资源：bw_each = bw/(N-1)，reserve_equal_split(paths_active, path_backup, ...)
- 运行期：仅对“需要切换”的会话尝试；>=2 同时 down -> 失败释放；未命中释放
- 训练：mode='train' 做轻量在线更新；mode='eval' 冻结推理
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import EventLogger, SummaryAggregator, EpisodeSummaryWriter
from quota import QuotaManager


# ----------------------------
# 一些基础工具
# ----------------------------
def _edges_of(path: List[int]) -> List[Tuple[int, int]]:
    return [(path[i], path[i+1]) for i in range(len(path)-1)]

def _disjoint_ok(p1: List[int], p2: List[int], mode: str = "EDGE") -> bool:
    mode = (mode or "EDGE").upper()
    if mode in ("EDGE", "LINK"):
        e1 = set(_edges_of(p1)); e2 = set(_edges_of(p2))
        return len(e1 & e2) == 0
    s1 = set(p1[1:-1]); s2 = set(p2[1:-1])
    return len(s1 & s2) == 0

def _cand_weight(c: Dict[str, Any]) -> float:
    feats = c.get("feats", [])
    if isinstance(feats, list) and len(feats) >= 1:
        try:
            return float(feats[0])
        except Exception:
            pass
    return float(len(c.get("path", [])))

def _std_request(req) -> Dict[str, Any]:
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

def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def _set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ReplayBuffer:
    """
    简单循环 replay buffer，存储 Central SAC 的经验
    每条经验：(obs, N_idx, a_embed, reward)
    """
    def __init__(self, capacity: int = 2000, device=None):
        self.capacity = capacity
        self.device = device
        self.buf_obs = []
        self.buf_N_idx = []
        self.buf_a_embed = []
        self.buf_reward = []
        self._ptr = 0
        self._size = 0

    def push(self, obs: torch.Tensor, N_idx: int,
             a_embed: torch.Tensor, reward: float):
        """存一条经验（detach 后存 CPU tensor 节省显存）"""
        obs_cpu = obs.detach().cpu()
        a_cpu   = a_embed.detach().cpu()

        if self._size < self.capacity:
            self.buf_obs.append(obs_cpu)
            self.buf_N_idx.append(N_idx)
            self.buf_a_embed.append(a_cpu)
            self.buf_reward.append(reward)
            self._size += 1
        else:
            # 循环覆盖
            self.buf_obs[self._ptr]    = obs_cpu
            self.buf_N_idx[self._ptr]  = N_idx
            self.buf_a_embed[self._ptr]= a_cpu
            self.buf_reward[self._ptr] = reward

        self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        """随机采样 batch_size 条，返回 device 上的 tensor"""
        idxs = np.random.randint(0, self._size, size=batch_size)
        obs    = torch.cat([self.buf_obs[i]     for i in idxs], dim=0).to(self.device)
        N_idxs = [self.buf_N_idx[i]              for i in idxs]
        a_emb  = torch.cat([self.buf_a_embed[i] for i in idxs], dim=0).to(self.device)
        rews   = torch.tensor([self.buf_reward[i] for i in idxs],
                               dtype=torch.float32, device=self.device).unsqueeze(1)
        return obs, N_idxs, a_emb, rews

    def __len__(self):
        return self._size

# ----------------------------
# Central SAC：网络定义
# ----------------------------
class CentralActor(nn.Module):
    """
    输入：全局/请求/候选统计特征向量（固定长度）
    输出：
      - logits_N: (n_actions_N,)  对应离散 N 的 logits
      - alpha_logits: (K_max,)    对候选的偏好 logits（对无效候选会 mask）
    """
    def __init__(self, in_dim: int, n_actions_N: int, k_max: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits_N = nn.Linear(hidden, n_actions_N)
        self.alpha_logits = nn.Linear(hidden, k_max)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.logits_N(x), self.alpha_logits(x)


class CentralCritic(nn.Module):
    """
    双 Q 网络，输入为 obs + 动作嵌入（onehot N + 被选候选的聚合统计）
    简化实现：把 (N, alpha_logits masked softmax 后的 top-k summary) 拼到一起
    """
    def __init__(self, in_dim: int, n_actions_N: int, k_max: int, hidden: int = 256):
        super().__init__()
        a_dim = n_actions_N + 2  # N onehot + alpha summary(均值、最大)
        self.q1 = nn.Sequential(
            nn.Linear(in_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs: torch.Tensor, a_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, a_embed], dim=-1)
        return self.q1(x), self.q2(x)


# ----------------------------
# Edge PPO：网络定义（评分器）
# ----------------------------
class EdgePolicy(nn.Module):
    """
    对候选集 (K_max, F) 输出评分向量 (K_max,)
    实际输入为：对每条候选抽取的固定维度特征（长度/估计延迟/链路成本等）
    这里使用轻量 MLP 对 K 条候选逐条打分
    """
    def __init__(self, cand_feat_dim: int, hidden: int = 128):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(cand_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)  # 输出单个得分
        )

    def forward(self, cand_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        cand_feats: (B, K_max, F)
        mask:       (B, K_max)  1=valid, 0=pad
        return:     (B, K_max)  scores（无效位置置极小）
        """
        B, K, Fdim = cand_feats.shape
        x = cand_feats.view(B * K, Fdim)
        s = self.ff(x).view(B, K).squeeze(-1)
        s = s + (mask == 0).float() * (-1e9)
        return s


# ----------------------------
# 束搜索 + 不相交选择
# ----------------------------
def beam_search_disjoint(cands: List[Dict[str, Any]], scores: np.ndarray, N: int,
                         disjoint_mode: str, beam_size: int = 8) -> Optional[List[List[int]]]:
    order = np.argsort(-scores)  # 大到小
    seq = [cands[i] for i in order]
    s_seq = [float(scores[i]) for i in order]
    beams = [([], 0.0)]
    for c, sc in zip(seq, s_seq):
        p = c.get("path", [])
        new_beams = []
        for paths, acc in beams:
            # 不选
            new_beams.append((paths, acc))
            # 选
            ok = True
            for q in paths:
                if not _disjoint_ok(p, q, disjoint_mode):
                    ok = False; break
            if ok:
                new_beams.append((paths + [list(p)], acc + sc))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
        for paths, _ in beams:
            if len(paths) >= N:
                return paths[:N]
    for paths, _ in beams:
        if len(paths) >= N:
            return paths[:N]
    return None


# ----------------------------
# BEAR-TORCH 系统封装
# ----------------------------
class BEARTorchSystem:
    """
    与 runner 对接：
      - run_one_episode(ep_idx, steps, mode, fixed_N=None) -> summary
      - run(mode, epochs, steps, fixed_N=None)

    说明：
    - 在线“轻量版”训练：每次 place/failover 后给 Central/Edge 回传即时奖励并更新
    - 观测设计：用简洁统计量保证与 env 的解耦（无需修改 env）
    """
    def __init__(self,
                 env,
                 result_dir: str,
                 save_dir: str,
                 disjoint_mode: str = "EDGE",
                 K_cand_max: int = 48,
                 N_min: int = 2, N_max: int = 5,
                 k_feat_dim: int = 4,          # 候选特征维度（长度/估计延迟/带宽代价/CPU代价）
                 obs_dim: int = 16,            # 全局/请求统计向量维度（本实现内部构造）
                 beam_size: int = 8,
                 device: str = None):
        _set_seed(42)
        self.env = env
        self.result_dir = Path(result_dir); self.result_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.disjoint_mode = (disjoint_mode or "EDGE").upper()
        self.K_cand_max = int(K_cand_max)
        self.N_min = max(2, int(N_min)); self.N_max = max(self.N_min, int(N_max))
        self.beam_size = int(beam_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 覆盖写 CSV
        for p in [
            self.result_dir / "events_bear.csv",
            self.result_dir / "episode_summary_bear.csv",
            self.result_dir / "update_log_bear.csv",
        ]:
            try: p.unlink()
            except FileNotFoundError: pass
        self.ev_logger = EventLogger(self.result_dir / "events_bear.csv")
        self.ep_writer = EpisodeSummaryWriter(self.result_dir / "episode_summary_bear.csv")
        self.agg = SummaryAggregator()
        # update 日志（step-level）
        import csv
        self.update_log_path = self.result_dir / "update_log_bear.csv"
        self._update_f = open(self.update_log_path, "a", newline="", encoding="utf-8")
        self._update_fields = [
            "t", "sid", "update_type", "reward",
            "critic_loss", "actor_loss", "alpha_loss", "entropy", "td_error",
            "edge_loss",
        ]
        self._update_w = csv.DictWriter(self._update_f, fieldnames=self._update_fields)
        if self._update_f.tell() == 0:
            self._update_w.writeheader()

        # Central SAC
        self.n_actions_N = self.N_max - self.N_min + 1
        self.central_actor = CentralActor(in_dim=obs_dim, n_actions_N=self.n_actions_N, k_max=self.K_cand_max).to(self.device)
        self.central_critic = CentralCritic(in_dim=obs_dim, n_actions_N=self.n_actions_N, k_max=self.K_cand_max).to(self.device)
        self.central_critic_target = CentralCritic(in_dim=obs_dim, n_actions_N=self.n_actions_N, k_max=self.K_cand_max).to(self.device)
        self.central_critic_target.load_state_dict(self.central_critic.state_dict())
        self.opt_actor = torch.optim.Adam(self.central_actor.parameters(), lr=3e-4)
        self.opt_critic = torch.optim.Adam(self.central_critic.parameters(), lr=3e-4)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)  # 温度
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=3e-4)
        # 目标熵（离散 N + α 摘要，近似取 -log(n_actions_N)）
        self.target_entropy = -math.log(self.n_actions_N + 1.0)

        # Edge PPO（评分器）
        self.edge_policy = EdgePolicy(cand_feat_dim=k_feat_dim).to(self.device)
        self.opt_edge = torch.optim.Adam(self.edge_policy.parameters(), lr=3e-4)

        # 轻量经验缓存（只存最近若干步）
        self.buf_central = []  # (obs, a_embed, reward)
        self.buf_edge = []     # (cand_feats, mask, returns)  简化：做监督式回归到“好”的分数

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=5000, device=self.device)
        self.batch_size    = 32    # 每次更新用的 batch 大小
        self.update_every  = 10    # 每积累多少条经验做一次批量更新
        self.warmup_steps  = 50    # buffer 里至少有这么多条才开始更新
        self._step_count   = 0     # 全局步数计数

        # 配额
        self.qm = QuotaManager(N_min=self.N_min, N_max=self.N_max, smooth_tau=0.9)
        self._mode = "eval"
        # 训练/调试统计：记录每步 update 的 loss / entropy / td-error / edge-loss / reward
        from collections import defaultdict
        self.train_stats = defaultdict(list)

    def _log_update(self, row: Dict[str, Any]) -> None:
        try:
            rec = {k: row.get(k, "") for k in self._update_fields}
            self._update_w.writerow(rec)
        except Exception:
            pass

    # ---------- 工程：候选与特征 ----------
    def _enumerate_sorted(self, src: int, dst: int) -> List[Dict[str, Any]]:
        cands = self.env.enumerate_candidates(src, dst, self.disjoint_mode, self.K_cand_max) or []
        good = [c for c in cands if c.get("ok", True) and isinstance(c.get("path", None), list)]
        good.sort(key=_cand_weight)
        return good

    def _build_cand_feats(self, cands: List[Dict[str, Any]]) -> np.ndarray:
        """
        返回 (K_max, F) 候选特征：
          f0 = 归一化 hop 数
          f1 = 归一化 feats[0]（若有）
          f2 = 归一化 cost（若有）
          f3 = 1/（hop+1）
        """
        K = len(cands)
        feats = np.zeros((self.K_cand_max, 4), dtype=np.float32)
        if K == 0:
            return feats
        hops = np.array([len(c["path"]) for c in cands], dtype=np.float32)
        f0 = (hops - hops.min()) / max(1e-6, (hops.max() - hops.min()))
        f0 = np.nan_to_num(f0, nan=0.0)
        # feats[0]
        arr_f1 = []
        for c in cands:
            if "feats" in c and isinstance(c["feats"], list) and len(c["feats"]) > 0:
                try:
                    arr_f1.append(float(c["feats"][0]))
                except Exception:
                    arr_f1.append(float(len(c["path"])))
            else:
                arr_f1.append(float(len(c["path"])))
        arr_f1 = np.array(arr_f1, dtype=np.float32)
        f1 = (arr_f1 - arr_f1.min()) / max(1e-6, (arr_f1.max() - arr_f1.min()))
        f1 = np.nan_to_num(f1, nan=0.0)
        # cost
        arr_cost = []
        for c in cands:
            try: arr_cost.append(float(c.get("cost", len(c["path"]))))
            except Exception: arr_cost.append(float(len(c["path"])))
        arr_cost = np.array(arr_cost, dtype=np.float32)
        f2 = (arr_cost - arr_cost.min()) / max(1e-6, (arr_cost.max() - arr_cost.min()))
        f2 = np.nan_to_num(f2, nan=0.0)
        # inv hop
        f3 = 1.0 / (hops + 1.0)

        feats[:K, 0] = f0; feats[:K, 1] = f1; feats[:K, 2] = f2; feats[:K, 3] = f3
        return feats

    def _build_obs(self, req: Dict[str, Any], cand_feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        构造 Central 的全局/请求统计 obs（长度 obs_dim = 16）：
        [bw, L, K, mean(hop), min(feat0), mean(cost), var(hop), var(cost), ...] 归一化到 [0,1]
        仅作近似统计即可满足学习引导
        """
        bw = req["bw"]; L = req["L"]
        K = int(mask.sum())
        arr = cand_feats[:K, :]
        if K > 0:
            hop_inv = arr[:, 3]
            feat0  = arr[:, 1]
            cost   = arr[:, 2]
            s = np.array([
                bw/100.0, L/10.0, K/self.K_cand_max,
                float(hop_inv.mean()), float(feat0.min()), float(cost.mean()),
                float(hop_inv.var()), float(cost.var()),
                float(feat0.mean()), float(hop_inv.max()), float(cost.max()),
                0.0, 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
        else:
            s = np.zeros((16,), dtype=np.float32)
        return s

    def _a_embed(self, N_idx: int, alpha_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        将离散 N 的 onehot 与 alpha 的摘要拼成动作嵌入：
        [onehot(N), mean(alpha_valid), max(alpha_valid)]
        """
        B = 1
        onehot = torch.zeros((B, self.n_actions_N), device=self.device)
        onehot[0, N_idx] = 1.0
        valid = (mask[0] > 0)
        if valid.any():
            a_valid = alpha_probs[0][valid]
            a_mean = a_valid.mean().unsqueeze(0)
            a_max  = a_valid.max().unsqueeze(0)
        else:
            a_mean = torch.zeros(1, device=self.device)
            a_max  = torch.zeros(1, device=self.device)
        return torch.cat([onehot, a_mean.view(1,1), a_max.view(1,1)], dim=-1)

    # ---------- SAC 推理（一次请求） ----------
    def _central_infer(self, obs_np: np.ndarray, mask_np: np.ndarray) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
          N（数值）、N_idx（索引）、alpha_logits(B, K_max)、alpha_probs(B,K_max)、obs_tensor(B,obs_dim)
        """
        self.central_actor.eval()
        obs = _to_tensor(obs_np.reshape(1, -1), self.device)
        logits_N, alpha_logits = self.central_actor(obs)
        # mask 无效候选
        mask = _to_tensor(mask_np.reshape(1, -1), self.device)
        alpha_logits = alpha_logits + (mask == 0) * (-1e9)

        # 采样 N（离散），这里 eval 也用采样（你可改为 argmax）
        N_dist = torch.distributions.Categorical(logits=logits_N)
        N_idx = int(N_dist.sample().item())
        N_val = self.N_min + N_idx

        # alpha 概率
        alpha_probs = torch.softmax(alpha_logits, dim=-1)
        return N_val, N_idx, alpha_logits, alpha_probs, obs

    # ---------- Edge 推理（评分） ----------
    def _edge_scores(self, cand_feats_np: np.ndarray, mask_np: np.ndarray) -> torch.Tensor:
        self.edge_policy.eval()
        feats = _to_tensor(cand_feats_np.reshape(1, self.K_cand_max, -1), self.device)
        mask = _to_tensor(mask_np.reshape(1, -1), self.device)
        scores = self.edge_policy(feats, mask)  # (1, K_max)
        return scores

    # ---------- 单请求放置 ----------
    def place_request(self, req_raw) -> Dict[str, Any]:
        req = _std_request(req_raw)
        sid, src, dst, L, bw, ttl, t_arr = req["sid"], req["src"], req["dst"], req["L"], req["bw"], req["ttl"], req["t_arrive"]

        cands = self._enumerate_sorted(src, dst)
        K = len(cands)
        if K == 0:
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-TORCH", "sid": sid,
                "success": 0, "reason": "no_candidates",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": 0, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        mask_np = np.zeros((self.K_cand_max,), dtype=np.float32); mask_np[:K] = 1.0
        cand_feats_np = self._build_cand_feats(cands)
        obs_np = self._build_obs(req, cand_feats_np, mask_np)

        # Central 推理 (N, alpha)
        N, N_idx, alpha_logits, alpha_probs, obs_t = self._central_infer(obs_np, mask_np)
        # Edge 评分
        edge_scores = self._edge_scores(cand_feats_np, mask_np)  # (1, K_max)
        # 总分：alpha_probs * edge_scores_softmax
        with torch.no_grad():
            edge_sm = torch.softmax(edge_scores, dim=-1)
            total_scores = (alpha_probs * edge_sm).cpu().numpy()[0][:K]

        # 束搜索挑选 N 条不相交路径
        chosen = beam_search_disjoint(cands, total_scores, N, self.disjoint_mode, beam_size=self.beam_size)
        if chosen is None or len(chosen) < 2:
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-TORCH", "sid": sid,
                "success": 0, "reason": "not_enough_disjoint_paths",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": 0,
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            if self._mode == "train":  # 失败奖励
                self._central_update(
                    obs_t, N_idx, alpha_logits, alpha_probs, mask_np, reward=-1.0,
                    step_t=t_arr, sid=sid, update_type="place_fail_no_disjoint"
                )
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        # 在役/备份划分：以 hop 数排序，N-1 在役，1 备份
        chosen_sorted = sorted(chosen, key=lambda p: len(p))
        active_paths = chosen_sorted[:max(1, N-1)]
        backup_path  = chosen_sorted[max(1, N-1)]

        # bw_each = bw / float(max(1, len(active_paths)))
        bw_each = bw / float(max(1, N - 1))  # 同样除以 N-1
        bundle = list(active_paths) + [backup_path]
        if not self.env.check_paths_feasible(bundle, bw_each):
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-TORCH", "sid": sid,
                "success": 0, "reason": "insufficient_capacity",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            if self._mode == "train":
                self._central_update(
                    obs_t, N_idx, alpha_logits, alpha_probs, mask_np, reward=-0.5,
                    step_t=t_arr, sid=sid, update_type="place_fail_capacity"
                )
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        res = self.env.reserve_equal_split(
            sid=sid,
            paths_active=[list(p) for p in active_paths],
            path_backup=list(backup_path),
            bw_each=bw_each,
            L=L, ttl=ttl
        )
        if not res.get("success", 0):
            ev = {
                "t": t_arr, "event": "place", "method": "BEAR-TORCH", "sid": sid,
                "success": 0, "reason": "reserve_failed",
                "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
                "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                "emp_reli_pred": "", "fail_idx": "", "new_active": "",
            }
            self.ev_logger.log(ev); self.agg.ingest(ev)
            if self._mode == "train":
                self._central_update(
                    obs_t, N_idx, alpha_logits, alpha_probs, mask_np, reward=-0.5,
                    step_t=t_arr, sid=sid, update_type="place_fail_reserve"
                )
            self.qm.consume_for_request(used_paths=0, placed=False)
            return ev

        rel_pred = float(self.env.predict_reliability(active_paths, backup_path, L))
        ev = {
            "t": t_arr, "event": "place", "method": "BEAR-TORCH", "sid": sid,
            "success": 1, "reason": "",
            "src": src, "dst": dst, "L": L, "bw": bw, "N": N, "num_paths": len(bundle),
            "latency_ms": float(res.get("latency_ms", 0.0)),
            "cost_total": float(res.get("cost_total", 0.0)),
            "cost_bw": float(res.get("cost_bw", 0.0)),
            "cost_cpu": float(res.get("cost_cpu", 0.0)),
            "emp_reli_pred": rel_pred,
            "fail_idx": "", "new_active": "",
        }
        self.ev_logger.log(ev); self.agg.ingest(ev)
        self.qm.consume_for_request(used_paths=len(active_paths), placed=True)

        if self._mode == "train":
            # 即时奖励（place 成功 + 成本/延迟/预测可靠性）
            rew = 1.0 + 0.5*rel_pred - 0.01*float(res.get("latency_ms", 0.0)) - 0.005*float(res.get("cost_total", 0.0))
            self._central_update(
                obs_t, N_idx, alpha_logits, alpha_probs, mask_np, reward=rew,
                step_t=t_arr, sid=sid, update_type="place_success"
            )

            # Edge：简单的自监督信号 —— 让评分靠近 total_scores（或命中的候选打更高分）
            self._edge_improve(cand_feats_np, mask_np, chosen, step_t=t_arr, sid=sid, update_type="edge_improve")

        return ev

    # ---------- 训练：Central（SAC 轻量版） ----------
    # def _central_update(self, obs_t: torch.Tensor, N_idx: int,
    #                     alpha_logits: torch.Tensor, alpha_probs: torch.Tensor,
    #                     mask_np: np.ndarray, reward: float,
    #                     step_t: Optional[int] = None, sid: Optional[int] = None,
    #                     update_type: str = "central_update"):
    #     """
    #     单步轻量 SAC 更新：使用 (obs, a_embed, r, obs'≈obs) 的伪目标，主要用于在线自适应
    #     """
    #     self.central_actor.train(); self.central_critic.train()
    #     mask = _to_tensor(mask_np.reshape(1, -1), self.device)

    #     # 动作嵌入 a_embed
    #     a_embed = self._a_embed(N_idx, alpha_probs, mask)

    #     # 目标 Q（无下一状态，近似 r）
    #     r = torch.tensor([[reward]], dtype=torch.float32, device=self.device)

    #     # 更新 Q
    #     q1, q2 = self.central_critic(obs_t, a_embed)
    #     critic_loss = F.mse_loss(q1, r) + F.mse_loss(q2, r)
    #     self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()

    #     # 记录 TD / critic loss 信息
    #     try:
    #         q1v = q1.detach().cpu().numpy()
    #         td_err = float(np.mean(np.abs(q1v - r.detach().cpu().numpy())))
    #     except Exception:
    #         td_err = float('nan')

    #     # 更新 actor（最小化 alpha*entropy - Q）
    #     with torch.no_grad():
    #         # 重新采样动作（但我们简化为用当前动作的近似）
    #         q_min = torch.min(q1, q2)
    #     alpha = self.log_alpha.exp()
    #     # 近似熵项：对 N 分布
    #     logits_N, _ = self.central_actor(obs_t)
    #     pi_N = torch.softmax(logits_N, dim=-1) + 1e-8
    #     ent = -(pi_N * torch.log(pi_N)).sum(dim=-1, keepdim=True)
    #     actor_loss = (alpha * ent - q_min).mean()

    #     self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

    #     # 温度更新
    #     alpha_loss = -(self.log_alpha * (ent.detach() + self.target_entropy)).mean()
    #     self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

    #     # 软更新（这里等价直接拷贝，亦可加 τ）
    #     self.central_critic_target.load_state_dict(self.central_critic.state_dict())
    #     # 记录训练统计
    #     try:
    #         self.train_stats['critic_loss'].append(float(critic_loss.detach().cpu().item()))
    #     except Exception:
    #         pass
    #     try:
    #         self.train_stats['actor_loss'].append(float(actor_loss.detach().cpu().item()))
    #     except Exception:
    #         pass
    #     try:
    #         self.train_stats['alpha_loss'].append(float(alpha_loss.detach().cpu().item()))
    #     except Exception:
    #         pass
    #     try:
    #         self.train_stats['entropy'].append(float(ent.detach().cpu().item()))
    #     except Exception:
    #         pass
    #     try:
    #         self.train_stats['td_error'].append(float(td_err))
    #     except Exception:
    #         pass
    #     # step-level update log
    #     self._log_update({
    #         "t": step_t,
    #         "sid": sid,
    #         "update_type": update_type,
    #         "reward": reward,
    #         "critic_loss": float(critic_loss.detach().cpu().item()),
    #         "actor_loss": float(actor_loss.detach().cpu().item()),
    #         "alpha_loss": float(alpha_loss.detach().cpu().item()),
    #         "entropy": float(ent.detach().cpu().item()),
    #         "td_error": float(td_err),
    #         "edge_loss": "",
    #     })
    def _central_update(self, obs_t: torch.Tensor, N_idx: int,
                        alpha_logits: torch.Tensor, alpha_probs: torch.Tensor,
                        mask_np: np.ndarray, reward: float,
                        step_t=None, sid=None, update_type="central_update"):
        """
        把经验压入 buffer；满足条件时触发一次批量更新
        """
        mask = _to_tensor(mask_np.reshape(1, -1), self.device)
        a_embed = self._a_embed(N_idx, alpha_probs, mask)

        # 存入 buffer
        self.replay_buffer.push(obs_t, N_idx, a_embed, reward)
        self._step_count += 1

        # 条件：buffer 够热身量 且 到了更新周期
        if (len(self.replay_buffer) >= self.warmup_steps and
                self._step_count % self.update_every == 0):
            self._batch_update(step_t=step_t, sid=sid, update_type=update_type)


    def _batch_update(self, step_t=None, sid=None, update_type="batch_update"):
        """
        从 buffer 中采样一个 batch，做一次完整的 SAC 更新
        """
        obs_b, N_idxs, a_emb_b, rew_b = self.replay_buffer.sample(self.batch_size)

        self.central_actor.train()
        self.central_critic.train()

        # ---- Critic 更新 ----
        q1, q2 = self.central_critic(obs_b, a_emb_b)
        critic_loss = F.mse_loss(q1, rew_b) + F.mse_loss(q2, rew_b)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.central_critic.parameters(), 1.0)
        self.opt_critic.step()

        td_err = float(torch.mean(torch.abs(q1.detach() - rew_b)).item())

        # ---- Actor 更新 ----
        with torch.no_grad():
            q_min = torch.min(q1, q2)
        alpha = self.log_alpha.exp()
        logits_N, _ = self.central_actor(obs_b)
        pi_N = torch.softmax(logits_N, dim=-1) + 1e-8
        ent  = -(pi_N * torch.log(pi_N)).sum(dim=-1, keepdim=True)
        actor_loss = (alpha.detach() * ent - q_min).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.central_actor.parameters(), 1.0)
        self.opt_actor.step()

        # ---- 温度更新 ----
        alpha_loss = -(self.log_alpha * (ent.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        alpha_loss.backward()
        self.opt_alpha.step()

        # 软更新 target critic（τ=0.005）
        tau = 0.005
        for p, pt in zip(self.central_critic.parameters(),
                        self.central_critic_target.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

        # 记录统计
        cl = float(critic_loss.detach().item())
        al = float(actor_loss.detach().item())
        aal = float(alpha_loss.detach().item())
        en = float(ent.mean().detach().item())
        for k, v in [('critic_loss', cl), ('actor_loss', al),
                    ('alpha_loss', aal), ('entropy', en), ('td_error', td_err)]:
            self.train_stats[k].append(v)

        self._log_update({
            "t": step_t, "sid": sid, "update_type": update_type, "reward": "",
            "critic_loss": cl, "actor_loss": al, "alpha_loss": aal,
            "entropy": en, "td_error": td_err, "edge_loss": "",
        })

    # ---------- 训练：Edge（PPO 极简替代——监督式评分微调） ----------
    def _edge_improve(self, cand_feats_np: np.ndarray, mask_np: np.ndarray,
                      chosen_paths: List[List[int]],
                      step_t: Optional[int] = None, sid: Optional[int] = None,
                      update_type: str = "edge_update"):
        """
        简化实现：让已选的候选获得更高分（监督式 margin）
        """
        self.edge_policy.train()
        feats = _to_tensor(cand_feats_np.reshape(1, self.K_cand_max, -1), self.device)
        mask = _to_tensor(mask_np.reshape(1, -1), self.device)
        scores = self.edge_policy(feats, mask)  # (1, K_max)

        # 构造目标：已被选中的路径 idx 打 1，其他打 0
        # 需要将 chosen_paths 映射回候选索引；用路径字符串做近似比对
        chosen_set = set()
        for p in chosen_paths:
            chosen_set.add(tuple(p))
        # 假设 feats 前 K 是有效候选，env.enumerate_candidates 内顺序与 cand_feats_np 对齐
        # 因本函数调用处不再有 cands，故此处仅做 margin 推动：置顶 top-k 位置靠近 1
        # 简化：直接推动 scores 的 top-k（=len(chosen_paths)）更大
        K = int(mask_np.sum())
        if K <= 0:
            return
        k_sel = min(K, len(chosen_paths))
        # 排序索引
        s = scores[0, :K]
        topk = torch.topk(s, k_sel, dim=-1).indices
        target = torch.zeros_like(s)
        target[topk] = 1.0
        loss = F.binary_cross_entropy_with_logits(s, target)
        self.opt_edge.zero_grad(); loss.backward(); self.opt_edge.step()
        try:
            self.train_stats['edge_loss'].append(float(loss.detach().cpu().item()))
        except Exception:
            pass
        # step-level update log
        self._log_update({
            "t": step_t,
            "sid": sid,
            "update_type": update_type,
            "reward": "",
            "critic_loss": "",
            "actor_loss": "",
            "alpha_loss": "",
            "entropy": "",
            "td_error": "",
            "edge_loss": float(loss.detach().cpu().item()),
        })

    # ---------- failover 辅助 ----------
    def _count_down_active_paths(self, session: Dict[str, Any]) -> int:  # 计算会话中当前不可用的活动路径数量；返回 -1 表示未知
        if hasattr(self.env, "count_down_active_paths"):  # 如果 env 提供了直接返回计数的接口则优先使用
            try:
                return int(self.env.count_down_active_paths(session))  # 调用该接口并将结果转为 int 返回
            except Exception:
                pass  # 若调用失败则忽略异常，继续下一个回退方案
        if hasattr(self.env, "get_down_active_paths"):  # 如果 env 提供了返回 down 路径列表的接口则作为备用
            try:
                lst = self.env.get_down_active_paths(session)  # 调用接口获取不可用路径列表
                return len(lst) if lst is not None else 0  # 列表非 None 则返回长度，否则认为没有 down 路径返回 0
            except Exception:
                pass  # 若调用失败则忽略异常，继续回退逻辑
        # 回退：若 env 无上述接口则尝试从 session 字段推断活动路径并逐条检测
        paths = session.get("active_set") or session.get("paths_active") or session.get("paths") or []  # 从 session 中尝试提取活动路径字段（多个可能的键）
        if not hasattr(self.env, "is_path_up"):  # 如果 env 无法逐条检查路径状态，则无法判断，返回 -1 表示未知
            return -1
        cnt = 0  # 初始化不可用路径计数器
        for p in (paths or []):  # 遍历所有候选活动路径（容错空列表）
            try:
                if not self.env.is_path_up(p):  # 若该路径当前不可达（is_path_up 返回 False）
                    cnt += 1  # 计数器加一
            except Exception:
                continue  # 若单条路径检查出错则跳过该路径，不影响整体计数
        return cnt  # 返回检测到的不可用活动路径总数
    
    # def _count_down_active_paths(self, session: Dict[str, Any]) -> int:
    #     all_paths = session.get("paths", [])
    #     active_idx = session.get("active_idx", [])
    #     if not all_paths or not active_idx:
    #         return 0
    #     cnt = 0
    #     for idx in active_idx:
    #         if idx < len(all_paths):
    #             try:
    #                 if not self.env.is_path_up(all_paths[idx]):
    #                     cnt += 1
    #             except Exception:
    #                 continue
    #     return cnt


    # ---------- 单轮 ----------
    def run_one_episode(self, ep_idx: int, steps: int, mode: str = "eval", fixed_N: Optional[int] = None) -> Dict[str, Any]:
        self._mode = str(mode or "eval").lower()
        # reset per-episode training stats
        try:
            from collections import defaultdict
            self.train_stats = defaultdict(list)
        except Exception:
            pass
        self.env.reset_episode(seed=None)
        meta = self.env.episode_meta() if hasattr(self.env, "episode_meta") else {"topo_size": "", "sfc_len_group": ""}
        self.agg.reset(); self.agg.set_meta(meta.get("topo_size", ""), meta.get("sfc_len_group", ""))

        # 名义 N（用于配额统计）
        N_ref = max(2, int(round(0.5 * (self.N_min + self.N_max))))
        self.qm.set_epoch_quota(N_ref)

        alive: Dict[int, Dict[str, Any]] = {}

        for t in range(0, steps):
            # 到达
            req = self.env.maybe_next_request(t)
            if req is not None:
                if isinstance(req, dict): req.setdefault("t_arrive", t)
                else:
                    try: setattr(req, "t_arrive", t)
                    except Exception: pass
                evp = self.place_request(req)
                if evp.get("success", 0) == 1 and hasattr(self.env, "get_session_ref"):
                    sid = int(evp["sid"])
                    try: alive[sid] = self.env.get_session_ref(sid)
                    except Exception: pass

            # 注入失效
            self.env.inject_failures(t)

            # 仅对需要切换的会话尝试 failover
            for s in list(alive.values()):
                down_cnt = self._count_down_active_paths(s)
                if down_cnt == 0: continue
                sid_i = int(s.get("sid", -1))

                if down_cnt >= 2:
                    evf = {
                        "t": t, "event": "failover", "method": "BEAR-TORCH", "sid": sid_i,
                        "success": 0, "reason": "multi_path_down",
                        "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                        "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                        "latency_ms": "", "cost_total": "", "cost_bw": "", "cost_cpu": "",
                        "emp_reli_pred": "", "fail_idx": "", "new_active": "",
                    }
                    self.ev_logger.log(evf); self.agg.ingest(evf)
                    # 负向反馈（仅在训练）
                    if self._mode == "train":
                        self._central_update(obs_t=torch.zeros((1,16),device=self.device),
                                             N_idx=0, alpha_logits=torch.zeros((1,self.K_cand_max),device=self.device),
                                             alpha_probs=torch.zeros((1,self.K_cand_max),device=self.device),
                                             mask_np=np.zeros((self.K_cand_max,),dtype=np.float32),
                                             reward=-1.0,
                                             step_t=t, sid=sid_i, update_type="failover_multi_down")
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive.pop(sid_i, None)
                    continue

                r = self.env.try_failover(s)
                if r.get("failed", 0) == 0:
                    continue

                hit = bool(r.get("backup_hit", 0))
                evf2 = {
                    "t": t, "event": "failover", "method": "BEAR-TORCH", "sid": sid_i,
                    "success": 1 if hit else 0,
                    "reason": "" if hit else "no_usable_backup",
                    "src": s.get("src", ""), "dst": s.get("dst", ""), "L": s.get("L", ""),
                    "bw": s.get("bw", ""), "N": s.get("N", ""), "num_paths": s.get("num_paths", ""),
                    "latency_ms": float(r.get("latency_ms", 0.0)),
                    "cost_total": "", "cost_bw": "", "cost_cpu": "",
                    "emp_reli_pred": "", "fail_idx": r.get("fail_idx", ""), "new_active": r.get("new_active", ""),
                }
                self.ev_logger.log(evf2); self.agg.ingest(evf2)

                if not hit:
                    if self._mode == "train":
                        self._central_update(obs_t=torch.zeros((1,16),device=self.device),
                                             N_idx=0, alpha_logits=torch.zeros((1,self.K_cand_max),device=self.device),
                                             alpha_probs=torch.zeros((1,self.K_cand_max),device=self.device),
                                             mask_np=np.zeros((self.K_cand_max,),dtype=np.float32),
                                             reward=-0.5,
                                             step_t=t, sid=sid_i, update_type="failover_miss")
                    try:
                        if hasattr(self.env, "release_session"): self.env.release_session(sid_i)
                    finally:
                        alive.pop(sid_i, None)

            # 释放到期
            released = self.env.release_expired(t)
            if released:
                for sid in released:
                    alive.pop(int(sid), None)

        summary = self.agg.finalize()
        # 把训练统计汇总进 episode summary（均值）
        try:
            import math as _math
            def _mean_or_nan(lst):
                return float(np.mean(lst)) if (lst and len(lst) > 0) else float('nan')

            summary['train_critic_loss_avg'] = _mean_or_nan(self.train_stats.get('critic_loss', []))
            summary['train_actor_loss_avg']  = _mean_or_nan(self.train_stats.get('actor_loss', []))
            summary['train_alpha_loss_avg']  = _mean_or_nan(self.train_stats.get('alpha_loss', []))
            summary['train_entropy_avg']     = _mean_or_nan(self.train_stats.get('entropy', []))
            summary['train_td_error_avg']    = _mean_or_nan(self.train_stats.get('td_error', []))
            summary['train_edge_loss_avg']   = _mean_or_nan(self.train_stats.get('edge_loss', []))
        except Exception:
            pass

        self.ep_writer.write(ep_idx, summary)
        return summary

    # ---------- 多轮 ----------
    def run(self, mode: str, epochs: int, steps: int, fixed_N: Optional[int] = None) -> None:
        mode = str(mode or "eval").lower()
        for ep in range(epochs):
            summ = self.run_one_episode(ep_idx=ep, steps=steps, mode=mode, fixed_N=fixed_N)
            print(f"[BEAR-TORCH EP {ep:03d}] "
                  f"place={summ['place_rate']:.3f} fo_hit={summ['fo_hit_rate']:.3f} "
                  f"emp_av={summ['emp_avail']:.3f} cost={summ['avg_cost_total']:.4f} "
                  f"lat={summ['avg_latency_ms']:.3f}")
        try:
            self.ev_logger.close()
            self.ep_writer.close()
            try:
                self._update_f.flush()
                self._update_f.close()
            except Exception:
                pass
        except Exception:
            pass

    # ---------- 模型保存/加载 ----------
    def save(self, out_dir: str, prefix: str = "bear_torch"):
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        torch.save(self.central_actor.state_dict(), d / f"{prefix}_central_actor.pt")
        torch.save(self.central_critic.state_dict(), d / f"{prefix}_central_critic.pt")
        torch.save(self.edge_policy.state_dict(), d / f"{prefix}_edge_policy.pt")
        torch.save({"log_alpha": self.log_alpha.detach().cpu()}, d / f"{prefix}_alpha.pt")

    def load(self, in_dir: str, prefix: str = "bear_torch"):
        d = Path(in_dir)
        self.central_actor.load_state_dict(torch.load(d / f"{prefix}_central_actor.pt", map_location=self.device))
        self.central_critic.load_state_dict(torch.load(d / f"{prefix}_central_critic.pt", map_location=self.device))
        self.edge_policy.load_state_dict(torch.load(d / f"{prefix}_edge_policy.pt", map_location=self.device))
        st = torch.load(d / f"{prefix}_alpha.pt", map_location="cpu")
        self.log_alpha = torch.tensor(float(st.get("log_alpha", 0.0)), requires_grad=True, device=self.device)