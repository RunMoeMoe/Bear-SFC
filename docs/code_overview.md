# BEAR-SFC 项目代码梳理文档

> 生成时间：2026-03-05
> 代码库路径：`/Users/dianawang/remote-project/`

---

## 1. 整体架构

**项目一句话描述**：BEAR-SFC 是一个用于 SFC（Service Function Chain，服务功能链）备份放置的分层强化学习仿真系统，通过"中央层决策路径数 N + 边缘层选择具体路径"的两层架构，在网络链路/节点失效场景下实现高可靠、低成本的 SFC 部署与故障切换。

### 每个 `.py` 文件职责

| 文件 | 职责 |
|------|------|
| `env.py` | **唯一环境真源**：维护 NetworkX 拓扑图、资源预留/释放 API、失效注入与恢复、SFC 请求生成；包含底层 `Env` 类和面向 `runner/bear` 的适配器 `SFCEnv` |
| `config.yaml` | 全局配置文件，定义拓扑、流量、失效、成本、HRL 超参 |
| `runner.py` | **统一训练/评估入口**：解析配置、构造环境、实例化算法、循环执行 episode，写日志 |
| `algo_central.py` | 中央层 Double-DQN 实现：离散动作（N 的选择）、经验回放、ε-greedy、target 网络同步 |
| `algo_edge.py` | 边缘层算法：`HeuristicSelector`（线性打分无训练）和 `EdgePPO`（轻量 PPO，Gumbel-Top-k 采样） |
| `bear.py` | **BEAR-SFC 完整系统**：DQN（中央）+ PPO/Heuristic（边缘）+ PRANOS 预筛选器，支持 `train_central / train_edge / eval / alt` 四种模式 |
| `baseline_BEAR_torch.py` | **主算法（当前实验目标）**：Central-SAC + Edge-PPO（PyTorch），replay buffer，束搜索不相交路径选择 |
| `baseline_BEAR.py` | 推理版 BEAR 基线：中央层为启发式策略（bw + K 映射 N），边缘层贪心不相交，无训练 |
| `baseline_BEAR_full.py` | 可训练 BEAR 基线：中央层用 Exp3-Bandit 学习 N 和路径权重温度，边缘层束搜索，无 PyTorch 依赖 |
| `baseline_DRL.py` | 纯 PPO 基线：Actor-Critic（mean-pool 编码候选特征），固定 N，对比用 |
| `baseline_PRANOS.py` | PRANOS 基线（完整版）：滑动窗口 + F2-LP 松弛 + 贪心 Rounding，依赖可选 `pulp` 求解器 |
| `baseline_PRANOS0.py` | PRANOS 轻量版（HEU_Cost 风格）：候选打分启发式，N 固定，无 LP 求解 |
| `baseline_PRANOS2.py` | PRANOS 另一变体：同 PRANOS0，结构稍有差异 |
| `baseline_MPD_DCBJOH.py` | MP-DCBJOH 基线：自适应选 h∈[2,k_r] 条路径，推荐 DZ 不相交约束 |
| `baseline_SBD.py` | SBD（选择性备份部署）基线：单主路径 + 按关键 VNF 决定是否备份，保守工程化实现 |
| `baseline_OptSEP.py` | Opt-SEP 基线：LGG 分层图最小权重 SFP + BSI 接口，工程近似为候选集上的最小权重对选择 |
| `preselector_pranos.py` | PRANOS 风格预筛选器：滑动窗口 → LP/贪心生成 top-k 可行候选组合池，供 `bear.py` 消费 |
| `metrics.py` | 指标与日志：`EventLogger`（逐事件 CSV）、`SummaryAggregator`（6 大指标汇总）、`EpisodeSummaryWriter`（收敛曲线 CSV）、`compute_central_reward`（DQN 奖励计算） |
| `quota.py` | 配额管理：跟踪中央决策 N 与边缘实际路径数的差异，输出"未用配额率"惩罚信号 |
| `plot_compare.py` | 多算法结果聚合与绘图：扫描 `result/<ALG>/episode_summary_<steps>.csv`，按步数对比 6 指标 |
| `edit_data.py` | 数据后处理工具：在对比 CSV 中交换列数据（用于实验报告） |

---

## 2. 核心数据流

### 2.1 一个 SFC 请求从到达到放置成功的完整流程

以 `BEARTorchSystem`（`baseline_BEAR_torch.py`）为例：

```
Step t 循环开始
│
├─ [1] 请求到达
│     env.maybe_next_request(t)
│       └─ Bernoulli(arrival_rate) 采样
│       └─ 采样 SFC 长度 L、带宽 bw、TTL ttl、src/dst
│       └─ 返回 SFCRequest(sid, src, dst, L, bw, ttl, t_arrive)
│
├─ [2] 特征构造（place_request）
│     env.enumerate_candidates(src, dst, disjoint_mode, K_cand_max)
│       └─ k_shortest → filter_disjoint → 返回候选列表（含 feats、ok）
│     _build_cand_feats(cands)    → (K_max, 4) 候选特征矩阵
│     _build_obs(req, cand_feats) → (16,) 全局状态向量
│
├─ [3] Central SAC 推理
│     _central_infer(obs_np, mask_np)
│       └─ CentralActor(obs) → logits_N, alpha_logits
│       └─ Categorical(logits_N).sample() → N_idx → N = N_min + N_idx
│       └─ softmax(alpha_logits) → alpha_probs（候选偏好权重）
│
├─ [4] Edge PPO 推理
│     _edge_scores(cand_feats, mask) → EdgePolicy 输出 (1, K_max) 评分
│     total_scores = alpha_probs * softmax(edge_scores)  → (K,)
│
├─ [5] 束搜索选路
│     beam_search_disjoint(cands, total_scores, N, disjoint_mode)
│       └─ 按分数从大到小贪心加入，保持两两不相交（EDGE/NODE/DZ）
│       └─ 返回 N 条路径列表（失败返回 None）
│     按跳数排序：前 N-1 条 → active_paths，第 N 条 → backup_path
│
├─ [6] 可行性检查
│     env.check_paths_feasible(bundle, bw_each)
│       └─ bw_each = bw / (N-1)
│       └─ 检查每条路径的 up 状态与带宽余量
│
├─ [7] 资源原子预留
│     env.reserve_equal_split(sid, paths_active, path_backup, bw_each, L, ttl)
│       └─ 可行性复核 → VNF 节点选择 → 带宽 + CPU 原子预留 → 失败则回滚
│       └─ 登记 ActiveSession → self.active[sid]
│
├─ [8] 指标记录
│     ev_logger.log(ev)     # 写 events_bear.csv
│     agg.ingest(ev)        # 更新 placed/cost/lat/rel 统计
│
└─ [9] 训练更新（mode=train）
      replay_buffer.push(obs_t, N_idx, a_embed, reward)
      _step_count += 1
      if len(buffer) >= warmup_steps and _step_count % update_every == 0:
          _batch_update()   # SAC: Critic + Actor + 温度 + 软更新 target
      _edge_improve(cand_feats, chosen_paths)  # 监督式 BCE margin loss
```

### 2.2 失效发生后的处理流程

**每步都会执行失效注入**：
```
env.inject_failures(t)
  ├─ 恢复：遍历 edges/nodes，若 t >= down_until 则 up=True
  └─ 新失效：Bernoulli(edge_fail_p) / Bernoulli(node_fail_p)
             若 up=True 且采样命中 → up=False, down_until = t + recovery_time
```

**在役路径失效（1 条 down）**：
```
_count_down_active_paths(session) → down_cnt = 1
env.try_failover(session)
  ├─ 遍历 session.active_idx，找到 failed_idx（is_path_up == False）
  ├─ 检查 standby_idx 对应路径是否 up
  ├─ 命中(backup_hit=1)：
  │   active_idx.remove(failed_idx)
  │   active_idx.append(standby_idx)
  │   standby_idx = None          ← 备份路径被消耗，置空
  │   返回 {failed:1, backup_hit:1, latency_ms:...}
  └─ 未命中(backup_hit=0)：
      返回 {failed:1, backup_hit:0}
      → 上层 alive.pop(sid)，调用 env.release_session(sid) 释放资源
```

**多条在役路径同时失效（≥2 条 down）**：
```
_count_down_active_paths(session) → down_cnt >= 2
→ 直接记录 failover 失败事件（reason="multi_path_down"）
→ env.release_session(sid)  释放所有带宽与 CPU
→ alive.pop(sid)
```

**备份路径失效后的处理（设计缺口）**：
```
当 standby_idx 对应路径也失效时：
  → try_failover 返回 {failed:1, backup_hit:0}
  → 上层释放会话，不会尝试重新寻找新备份路径
  → 该场景目前等同于"完全失败"，无重新部署逻辑
```

---

## 3. 关键数据结构

### 3.1 SFCRequest 字段含义

```python
@dataclass
class SFCRequest:
    sid:      int            # 全局唯一会话 ID（自增）
    t_arrive: int            # 到达时间步
    src:      int            # 源节点 ID（NetworkX 节点编号）
    dst:      int            # 目的节点 ID
    L:        int            # SFC 长度（VNF 个数），从 sfc_length_choices 采样
    bw:       float          # 带宽需求 M（单位与 bw_capacity 一致）
    ttl:      int            # 请求存活时长（步数），到期后 release_expired 释放
    vnf_seq:  Optional[List[int]]  # VNF 功能类型序列（当前未使用，留作扩展）
```

### 3.2 ActiveSession 字段含义

```python
@dataclass
class ActiveSession:
    sid:                int              # 会话 ID（与 SFCRequest.sid 一致）
    src:                int              # 源节点
    dst:                int              # 目的节点
    L:                  int              # VNF 数
    bw_each:            float            # 每条路径预留的带宽 = bw/(N-1)
    N:                  int              # 总路径数（在役 N-1 + 热备 1）
    paths:              List[List[int]]  # 所有 N 条路径（nodes 列表）
    active_idx:         List[int]        # 当前在役路径在 paths 中的下标列表（长度 N-1）
    standby_idx:        Optional[int]    # 热备路径在 paths 中的下标（命中后置 None）
    vnf_nodes_per_path: List[List[int]]  # 每条路径上 L 个 VNF 的部署节点列表
    t_expire:           int              # 过期时间步 = t_arrive + ttl
```

### 3.3 active_idx / standby_idx / paths 三者关系与生命周期

```
初始状态（N=3 为例）：
  paths = [P0, P1, P2]    # 3 条不相交路径
  active_idx = [0, 1]     # P0, P1 承载流量（在役，各预留 bw_each 带宽）
  standby_idx = 2          # P2 作热备（已预留带宽，不承载流量）

Failover 发生（P1 失效）：
  paths = [P0, P1, P2]    # paths 不变
  active_idx = [0, 2]     # P1 被移除，P2 升为在役
  standby_idx = None       # 热备已被消耗

会话释放：
  release_session(sid)
    → 遍历 paths（全部 N 条），调用 release_path_bw 归还带宽
    → 按 _cpu_books[sid] 精确归还每个 VNF 节点的 CPU
    → active.pop(sid)

注意：
  - paths 记录全部 N 条路径（包括热备），不会随 failover 缩短
  - active_idx 是动态的，failover 后会更新
  - standby_idx=None 表示热备已被使用，无法再次 failover
  - _count_down_active_paths 检查的是 paths[i] for i in active_idx
```

---

## 4. 所有算法变体对比

| 算法文件 | 系统类名 | 中央层策略 | 边缘层策略 | 在论文中的角色 |
|----------|----------|-----------|-----------|--------------|
| `baseline_BEAR_torch.py` | `BEARTorchSystem` | **Central SAC**（连续动作输出 N + α，离散化）| **Edge PPO**（轻量评分器，束搜索选路）| **主算法**（本文提出的 HRL 方案） |
| `bear.py` | `BearSystem` | **DQN**（离散 N，Double-DQN）| EdgePPO 或 HeuristicSelector（可切换）+ PRANOS 预筛选 | 主算法的完整版（含预筛选器） |
| `baseline_BEAR_full.py` | `BEARFullSystem` | **Exp3-Bandit**（学习 N 的分布）+ WeightedAlphaBandit（学习路径权重温度）| 束搜索 + 自适应回退 N | 消融基线：无 DL 的可训练 BEAR 变体 |
| `baseline_BEAR.py` | `BEARSystem` | **启发式**（bw + 候选数 → 映射 N；softmax 权重 α）| 贪心不相交（按 α 排序）| 消融基线：无训练的推理版 BEAR |
| `baseline_DRL.py` | `DRLSystem` | 无（**固定 N**，由 `fixed_N` 指定）| **PPO**（ActorCritic，mean-pool 编码候选，per-path logit）| 对比基线：纯单层 DRL |
| `baseline_PRANOS.py` | `BaselineSystem` | 无（**固定 N**）| **F2-LP 松弛 + 贪心 Rounding**（可选 pulp 求解器，无则退化为启发式） | 对比基线：PRANOS 论文方法 |
| `baseline_PRANOS0.py` | `PRANOSBaseline` (HEU_Cost) | 无（**固定 N**）| **HEU_Cost 启发式**（lat + hop + cost 线性加权打分）| 对比基线：PRANOS 轻量版 |
| `baseline_PRANOS2.py` | `PRANOSBaseline` | 无（**固定 N**）| **HEU_Cost 启发式**（结构与 PRANOS0 相似，另一版本）| 对比基线：PRANOS 变体 |
| `baseline_MPD_DCBJOH.py` | `MPDCBJOHSystem` | 无（**自适应 h**，按 latency/cost 排名选 h∈[2,k_r]）| 枚举不相交组合，选最优 h 路径 | 对比基线：多路径保护方案 |
| `baseline_SBD.py` | `SBDSystem` | 无（**固定 N=2**，主+备）| **选择性备份**（按关键 VNF 风险打分决定是否部署备份，高风险 VNF 降 CPU 系数）| 对比基线：选择性备份部署 |
| `baseline_OptSEP.py` | `OptSEPSystem` | 无（**固定 N=2**，pSFP+bSFP）| **LBC-SEP → OPSI + BSI**（工程近似：候选集内找最小权重不相交对）| 对比基线：最优 SEP 方案 |

---

## 5. 当前已知的设计缺口

### 5.1 备份路径失效后没有重新部署逻辑

**现象**：当 `standby_idx` 对应的热备路径也失效时，`try_failover` 返回 `{failed:1, backup_hit:0}`，上层直接释放会话。系统不会尝试重新为该会话寻找新的备份路径。

**期望行为**："备份失效 → 释放旧备份带宽 → 重新在网络中寻找一条新备份路径 → 预留资源"。

### 5.2 如需实现"备份失效 → 释放带宽 → 重新找备份 → 预留资源"，需要修改的位置

**改动点一：`env.py`**
- 函数 `try_failover`（第 586 行）：当 `is_path_up(paths[bk_idx]) == False` 时，目前直接返回 `{failed:1, backup_hit:0}`。需在此分支中增加逻辑：释放旧备份路径带宽（调用 `release_path_bw`），返回一个新的信号（如 `backup_expired=True`）告知上层触发重新选路。
- 函数 `register_session` 或新增 `update_session_backup`：支持替换已有会话的备份路径（更新 `paths`、`standby_idx`、`_cpu_books`）。

**改动点二：`baseline_BEAR_torch.py`（主算法）**
- 函数 `run_one_episode` 的 failover 处理段（第 900 行附近）：在 `backup_hit=0` 且为"备份失效"（非多路宕机）的分支中，不直接释放会话，而是调用新的重部署逻辑。
- 可新增 `_replan_backup(session, t)` 方法：调用 `_enumerate_sorted` 重新选路，调用 `env.reserve_equal_split` 或专用接口预留新备份路径。

**改动点三：`baseline_BEAR.py` / `baseline_BEAR_full.py`（若需同步对齐）**
- 这两个基线文件的 `run_one_episode` 中有相同的 failover 处理段（约 400-425 行），需同步修改。

**改动点四：`metrics.py`**
- `SummaryAggregator.ingest`：当前 failover 事件只区分 `backup_hit=1/0`，若新增"备份重新部署"事件类型（如 `event="backup_replan"`），需在此处增加对应计数字段（如 `replan_cnt`, `replan_hit`）。

---

## 6. 配置参数说明

### 6.1 `config.yaml` 每个参数的作用

```yaml
seed: 42               # 全局随机种子（影响拓扑生成、流量采样、故障注入）
sim_time: 2000         # 每个 episode 最大时间步（当前 runner 实际用 CONFIG["steps"] 覆盖）
episode_num: 30        # 训练集数（当前 runner 实际用 CONFIG["epochs"] 覆盖）
logdir: "runs"         # 日志保存路径（当前 runner 使用固定 result/ 目录，此参数未生效）

topology:
  type: "synthetic"        # 拓扑类型（当前仅支持 synthetic 随机图）
  num_nodes: 30            # 节点数量
  num_edges: 80            # 边数量（gnm_random_graph 参数）
  k_paths: 20              # enumerate_candidates 时拉取的候选路径上限
  disjoint: "EDGE"         # 不相交约束：EDGE（边不重叠）/ NODE（中间节点不重叠）/ DZ（灾区不重叠）
  link:
    bw_capacity: 20.0      # 每条边的带宽容量（直接影响 place_rate）
    prop_delay_per_km: 0.02 # 传播延迟系数 (ms/km)
    tx_delay_per_hop: 0.5  # 每跳转发延迟 (ms)
    dist_mean: 50          # 链路平均长度 km（高斯采样）
    dist_std: 10           # 链路长度标准差

node:
  cpu_capacity: 200.0      # 每节点 CPU 总量
  cpu_per_vnf: 1.0         # 部署一个 VNF 占用的 CPU 单位（未实际使用，由 cpu_per_vnf_bk 统一）
  cpu_per_vnf_bk: 1.0      # 热备 VNF 的 CPU 占用（所有路径均按此预留）

traffic:
  arrival_rate: 0.5         # 每步请求到达概率（Bernoulli 近似泊松，0.5 = 每步 50% 概率到达）
  duration_mean: 300        # 请求平均 TTL（步数，高斯采样）
  duration_std: 50          # TTL 标准差
  bw_demand_mean: 1.0       # 请求带宽均值（高斯采样）
  bw_demand_std: 0.2        # 带宽标准差
  sfc_length_choices: [3,4,5,6,7]   # SFC 长度候选集
  sfc_length_probs: [0.2,0.2,0.2,0.2,0.2]  # 各长度采样概率（均匀分布）

failures:
  node_fail_prob: 0.005    # 每步每节点的失效概率（独立 Bernoulli）
  edge_fail_prob: 0.005    # 每步每链路的失效概率
  recovery_time: 200       # 失效后恢复需要的时间步

cost:
  lambda_bw: 1.0           # 带宽成本权重（cost_bw = lambda_bw * bw_each * hops）
  lambda_cpu: 1.0          # CPU 成本权重（cost_cpu = lambda_cpu * cpu_per_vnf * L * N）
  backup_bw_frac: 1.0      # 备份路径带宽比例（兼容字段，当前代码未使用）

hrl:
  central:
    N_min: 3               # 中央层可选的最小路径数
    N_max: 6               # 中央层可选的最大路径数
    gamma: 0.95            # DQN 折扣因子
    lr: 0.001              # DQN 学习率
    eps_start: 1.0         # ε-greedy 初始探索率
    eps_end: 0.1           # ε-greedy 最终探索率
    eps_decay: 0.99        # ε 衰减（当前 runner 未将此值传入 CentralDQN）
    hidden: [64, 64]       # DQN MLP 隐层维度
  edge:
    lr: 0.0003             # Edge PPO 学习率
    gamma: 0.99            # PPO 折扣因子
    clip: 0.2              # PPO clip ε
    epochs: 4              # PPO 内层更新轮数
    batch_size: 64         # PPO 更新 batch 大小
    hidden: [128, 128]     # PPO MLP 隐层维度

runner:
  mode: "train"            # 运行模式（当前由 CONFIG["mode"] 覆盖）
  epochs: 30               # 训练集数（当前由 CONFIG["epochs"] 覆盖）
  steps_per_episode: 2000  # 每集步数（当前由 CONFIG["steps"] 覆盖）
  save_every: 5            # 每隔多少集保存模型（当前由 runner 逐集保存，此字段未用）
  fixed_N: 3               # train_edge 模式下固定的 N（当前由 CONFIG["fixed_N"] 覆盖）
```

### 6.2 `runner.py` 的 `CONFIG` 字典每个字段含义

```python
CONFIG = {
    "algorithm": "BEAR-TORCH",  # 使用的算法，决定实例化哪个 System 类
                                 # 可选：BEAR-TORCH / BEAR-SFC / PRANOS / DRL /
                                 #       MP-DCBJOH / SBD / OPTSEP / BEAR / BEAR-FULL

    "mode": "alt",              # 运行模式：
                                 # "train"         → BEAR-TORCH 中的在线训练（SAC+PPO）
                                 # "train_central"  → BearSystem 中仅训练中央 DQN
                                 # "train_edge"     → BearSystem 中仅训练边缘 PPO
                                 # "alt"            → 交替 train_central / train_edge
                                 # "eval"           → 冻结策略，仅产出指标

    "epochs": 100,              # 训练总集数（每集独立 reset env）

    "steps": 500,               # 每集时间步数（实际在 steps_sweep 中设置为 [100]）

    "fixed_N": None,            # train_edge / eval 模式下固定的路径总数 N
                                 # None 表示使用 N_min

    "out": "runs/default",      # 输出目录（当前实际不生效，使用 result/ 和 save/ 目录）

    "cfg_path": None,           # 外部配置文件路径（.yaml/.json）
                                 # ⚠️ 修复后应设为 str(Path(__file__).parent/"config.yaml")
                                 # None 时所有参数使用 build_env() 中的硬编码默认值

    "seed": 42,                 # 随机种子

    "resume": False,            # 是否从 save_dir 恢复已有模型（True 时加载 checkpoint）
}
```

### 6.3 当 `cfg_path=None` 时各参数的实际生效值

当 `CONFIG["cfg_path"] = None` 时，`load_cfg(None)` 返回 `{}`，所有参数退化为 `build_env()` 中的硬编码默认值，**与 `config.yaml` 的预期值存在偏差**：

| 参数 | `cfg_path=None` 时的实际值 | `config.yaml` 的预期值 | 影响 |
|------|--------------------------|----------------------|------|
| `num_nodes` | 40 | 30 | 拓扑规模偏大 |
| `num_edges` | 120 | 80 | 拓扑边数偏多 |
| `k_paths` | 16 | 20 | 候选路径略少 |
| `link_bw_mean` | **5.0** | **20.0** | ⚠️ 带宽容量相差4倍，place_rate 严重偏低 |
| `cpu_total_per_node` | **100.0** | **200.0** | CPU 容量减半 |
| `arrival_rate` | 0.5 | 0.5 | 巧合相同，无影响 |
| `node_fail_p` | **5e-4 = 0.0005** | **0.005** | ⚠️ 失效率低10倍，fo_cnt≈0 |
| `edge_fail_p` | **5e-4 = 0.0005** | **0.005** | ⚠️ 失效率低10倍，fo_cnt≈0 |
| `repair_mean` | 200 | 200 | 相同，无影响 |
| `dur_mean` | 500 | 300 | TTL 偏长 |
| `N_min`（runner 读 hrl.N_min） | 2 | 3 | N 范围起点偏小 |
| `N_max`（runner 读 hrl.N_max） | 5 | 6 | N 范围终点偏小 |

**修复方法**：在 `runner.py` 的 `CONFIG` 中设置：
```python
"cfg_path": str(Path(__file__).resolve().parent / "config.yaml"),
```

---

## 7. 输出文件说明

| 文件路径 | 内容 |
|----------|------|
| `result/<ALG>/events_bear.csv` | 逐事件流水（place/failover），字段见 `EventLogger.DEFAULT_FIELDS` |
| `result/<ALG>/episode_summary_<steps>.csv` | 每集 6 大汇总指标（place_rate, fo_hit_rate, emp_avail, rel_pred_avg, avg_cost_total, avg_latency_ms）|
| `result/<ALG>/update_log_bear.csv` | 每次 `_batch_update` 的损失与熵（仅 BEAR-TORCH）|
| `save/<ALG>/<timestamp>/train.log` | 控制台日志镜像 |
| `save/<ALG>/<timestamp>/*.pt` | 模型权重（central_actor, central_critic, edge_policy, alpha）|

---

*文档由 Claude Code 自动生成，基于对所有 `.py` 文件的完整阅读，最后更新于 2026-03-05。*
