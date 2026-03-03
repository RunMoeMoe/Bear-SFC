from __future__ import annotations
"""
（训练/评估一体）
	•	数据集：多拓扑（小/中/大）、多 SFC 长度（3/5/7）、多随机种子。
	•	三种模式：train_central（冻结 edge 用启发式挑 N 条最佳不相交）/train_edge（冻结中央给固定 N）/eval（两者均固定，产出 6 指标）。
统一训练/评估脚本骨架：
  模式：
    - train_central：训练中央 DQN（决策路径条数 N），冻结边缘策略为启发式
    - train_edge   : 训练边缘 PPO（选择具体 N 条路径），冻结中央策略为固定 N
    - eval         : 评估（加载已训练的 DQN+PPO，产出六项指标 + 分组统计 + 绘图）
功能：
  - 统一随机种子、日志/模型目录管理
  - 每轮（episode）写入：事件日志、汇总指标、收敛曲线（CSV/可选TensorBoard）
  - 模型保存与加载（best/baseline/按轮次）
  - 分组统计绘图：按拓扑规模、SFC长度
"""

# run.py
# -*- coding: utf-8 -*-
"""
统一入口：BEAR-SFC 训练 / 评估 Runner

示例：
1) 仅训练中央（DQN 决策 N），边缘用启发式：
   python run.py --mode train_central --epochs 30 --steps 1500 --out runs/central

2) 仅训练边缘（PPO 选 N 条路径），中央固定 N=4：
   python run.py --mode train_edge --epochs 30 --steps 1500 --fixed-N 4 --out runs/edge

3) 评估（使用已训练模型）：
   python run.py --mode eval --epochs 10 --steps 1500 --fixed-N 4 --out runs/eval --resume

参数优先级：命令行 > 配置文件 > 内置默认
"""


import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional
# 顶部 import 区域增加
import time
import logging
from pathlib import Path

import numpy as np

# 你自己的模块
from env import SFCEnv as Env
from bear import BearSystem
from algo_edge import HeuristicSelector, EdgePPO
from baseline_PRANOS import BaselineSystem
from baseline_PRANOS2 import PRANOSBaseline
from baseline_DRL import DRLSystem
from baseline_MPD_DCBJOH import MPDCBJOHSystem
from baseline_SBD import SBDSystem
from baseline_OptSEP import OptSEPSystem
from metrics import EpisodeSummaryWriter
from baseline_BEAR import BEARSystem
from baseline_BEAR_full import BEARFullSystem
from baseline_BEAR_torch import BEARTorchSystem

def set_global_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_cfg(cfg_path: Optional[str]) -> Dict[str, Any]:
    if not cfg_path:
        return {}
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    if p.suffix.lower() in [".yml", ".yaml"]:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    elif p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    else:
        raise ValueError("Unsupported config format. Use .yaml/.yml or .json")


def build_env(cfg: Dict[str, Any]) -> Env:
    # 读取用户配置（若无则给默认）
    topo = cfg.get("topology", {})
    node = cfg.get("node", {})
    traffic = cfg.get("traffic", {})
    failures = cfg.get("failures", {})
    cost = cfg.get("cost", {})

    # SFCEnv 的 __init__ 采用显式形参，而非 dict。
    # 这里使用**关键字参数**传入，避免位置顺序出错，并兼容你的 env.py 里给出的参数名。
    # 同时做“旧/新键名”兼容映射（如 node_fail_p / node_fail_prob）。
    return Env(
        num_nodes=int(topo.get("num_nodes", 40)),
        num_edges=int(topo.get("num_edges", 120)),
        disjoint_mode=str(topo.get("disjoint", "EDGE")).upper(),
        k_paths=int(topo.get("k_paths", 16)),

        # 链路/时延/处理
        prop_delay_per_km=float(topo.get("prop_delay_per_km", 0.02)),
        tx_delay_per_hop=float(topo.get("tx_delay_per_hop", 0.5)),
        vnf_proc_ms=float(node.get("vnf_proc_ms", 0.2)),

        # 链路带宽分布（均值/方差）
        link_bw_mean=float(topo.get("link_bw_mean", topo.get("link", {}).get("bw_capacity", 5.0))),
        link_bw_std=float(topo.get("link_bw_std", 1.0)),

        # 计算资源
        cpu_per_vnf=float(node.get("cpu_per_vnf", 1.0)),
        cpu_per_backup=float(node.get("cpu_per_vnf_bk", node.get("cpu_per_backup", 1.0))),
        cpu_total_per_node=float(node.get("cpu_capacity", 100.0)),

        # 失效区域比例（若未使用灾区模型，默认 0）
        dz_fraction=float(failures.get("dz_fraction", 0.0)),

        # 业务到达/需求分布
        arrival_rate=float(traffic.get("arrival_rate", 0.5)),
        sfc_len_choices=tuple(traffic.get("sfc_length_choices", [3, 5, 7])),
        sfc_len_probs=tuple(traffic.get("sfc_length_probs", [0.4, 0.4, 0.2])),
        bw_mean=float(traffic.get("bw_demand_mean", 1.0)),
        bw_std=float(traffic.get("bw_demand_std", 0.2)),
        dur_mean=int(traffic.get("duration_mean", 500)),
        dur_std=int(traffic.get("duration_std", 50)),

        # 故障注入（兼容旧/新命名）
        node_fail_p=float(failures.get("node_fail_p", failures.get("node_fail_prob", 5e-4))),
        edge_fail_p=float(failures.get("edge_fail_p", failures.get("edge_fail_prob", 5e-4))),
        repair_mean=int(failures.get("repair_mean", failures.get("recovery_time", 200))),

        # 其他
        seed=int(cfg.get("seed", 42)),
        edge_unit_cost=float(cost.get("edge_unit_cost", cost.get("lambda_bw", 1.0))),
        lambda_backup=float(cost.get("lambda_backup", 1.0)),
    )


def maybe_build_edge_algo(mode: str, cfg: Dict[str, Any], feat_dim_hint: int = 8):
    """
    训练边缘时返回 EdgePPO；其他情况返回启发式
    """
    if mode == "train_edge":
        e_cfg = cfg.get("edge_algo", {})
        hidden = tuple(e_cfg.get("hidden", (64, 64)))
        lr_pi = float(e_cfg.get("lr_pi", 3e-4))
        lr_v = float(e_cfg.get("lr_v", 1e-3))
        clip_eps = float(e_cfg.get("clip_eps", 0.2))
        entropy_coef = float(e_cfg.get("entropy_coef", 0.01))
        vf_coef = float(e_cfg.get("vf_coef", 0.5))
        max_grad_norm = float(e_cfg.get("max_grad_norm", 2.0))
        buf_size = int(e_cfg.get("buffer_size", 2048))
        device = e_cfg.get("device", None)

        return EdgePPO(
            feat_dim=feat_dim_hint, hidden=hidden,
            lr_pi=lr_pi, lr_v=lr_v, clip_eps=clip_eps,
            entropy_coef=entropy_coef, vf_coef=vf_coef,
            max_grad_norm=max_grad_norm, buffer_size=buf_size, device=device
        )
    else:
        # 启发式：无需配置；如需自定义权重，可在这里从 cfg 中读取
        return HeuristicSelector(weights=None)


def main():
    """
    无需命令行参数的简化 Runner：
      - 直接在下方 CONFIG 中修改运行配置
      - 运行：python runner.py
    """
    # ========= 在此处修改你的运行配置 =========
    CONFIG = {
        # 算法：BEAR-SFC / PRANOS / DRL / MP-DCBJOH / SBD / OPTSEP / BEAR / BEAR-FULL / BEAR-TORCH
        "algorithm": "BEAR-TORCH",

        # 运行模式：train_central / train_edge / eval / alt
        "mode": "alt",

        # 训练总集数与每集步数
        "epochs": 500,
        "steps": 1000,

        # train_edge / eval 下固定路径数 N（None 表示用 N_min）
        "fixed_N": None,

        # 输出目录（程序会自动追加时间戳子目录）
        "out": "runs/default",

        # 可选：外部配置文件路径（.yaml 或 .json）；若不使用则设为 None
        "cfg_path": None,

        # 随机种子
        "seed": 42,

        # 是否尝试从输出目录恢复模型（仅在评估或继续训练时需要）
        "resume": False,
    }
    # ======================================

    # 载入配置与随机种子
    cfg = load_cfg(CONFIG.get("cfg_path"))
    set_global_seed(CONFIG.get("seed") if CONFIG.get("seed") is not None else cfg.get("seed"))

    # 输出目录：加上时间戳，除非 resume
    # out_dir = Path(CONFIG["out"])
    # if not CONFIG.get("resume", False):
    #     stamp = time.strftime("%Y%m%d-%H%M%S")
    #     out_dir = out_dir / stamp
    # out_dir.mkdir(parents=True, exist_ok=True)
    # === 输出目录：固定到 /result，并以 bear+时间 命名 ===
    BASE = Path(__file__).resolve().parent
    result_dir = Path(BASE / "result" / CONFIG.get("algorithm"))
    result_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = Path(BASE / "save" / CONFIG.get("algorithm") / stamp)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 持久化运行配置与合并后的配置
    # with open(out_dir / "cmd_embedded.json", "w", encoding="utf-8") as f:
    #     json.dump(CONFIG, f, ensure_ascii=False, indent=2)
    # if cfg:
    #     with open(out_dir / "config_resolved.json", "w", encoding="utf-8") as f:
    #         json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(result_dir / "cmd_embedded.json", "w", encoding="utf-8") as f: 
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)
    if cfg:
        with open(result_dir / "config_resolved.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    with open(save_dir / "cmd_embedded.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)
    if cfg:
        with open(save_dir / "config_resolved.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    # 构造环境
    env = build_env(cfg)

    # 中央/边缘参数
    hrl_cfg = cfg.get("hrl", {})
    central_state_dim = int(hrl_cfg.get("central_state_dim", 10))
    N_min = int(hrl_cfg.get("N_min", 2))
    N_max = int(hrl_cfg.get("N_max", 5))

    # 候选路径数量、分离模式等
    topo_cfg = cfg.get("topology", {})
    disjoint_mode = str(topo_cfg.get("disjoint", "EDGE")).upper()
    K_cand_max = int(topo_cfg.get("k_paths", 16))

    # 边缘策略
    edge_algo = maybe_build_edge_algo(CONFIG["mode"], cfg, feat_dim_hint=int(hrl_cfg.get("edge_feat_dim", 8)))

    # 中央 DQN 的超参（可从 cfg.hrl.central 覆盖）
    central_cfg = cfg.get("central_algo", {})

    # === 日志同时输出到控制台与文件 ===
    log_file = save_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),                 # 控制台
            logging.FileHandler(str(log_file), encoding="utf-8")  # 文件
        ],
    )
    logger = logging.getLogger("runner")
    logger.info("Output directory: %s", save_dir)

    algo = CONFIG.get("algorithm", "BEAR-SFC").upper()

    if algo == "PRANOS":
        system = BaselineSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=CONFIG.get("disjoint", "EDGE"),
            K_cand_max=int(CONFIG.get("k_cand", 32)),
            N_min=int(CONFIG.get("N_min", 2)),
            N_max=int(CONFIG.get("N_max", 5)),
        )
    elif algo == "BEAR-SFC":
        system = BearSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=CONFIG.get("disjoint", "EDGE"),
            K_cand_max=int(CONFIG.get("k_cand", 32)),
            N_min=int(CONFIG.get("N_min", 2)),
            N_max=int(CONFIG.get("N_max", 5)),
            # 你原来Bear的参数保持不动
        )
    elif algo == "DRL":
        system = DRLSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=disjoint_mode,          # 与拓扑配置一致
            K_cand_max=K_cand_max,                # 与 k_paths 对齐
            N_min=N_min,
            N_max=N_max,
            drl_cfg=cfg.get("drl_algo", {})       # 可选：在 config.yaml 中增加 drl_algo 覆盖超参
        )
    elif algo == "MP-DCBJOH":
        system = MPDCBJOHSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=disjoint_mode,   # 推荐在 config.yaml 设置为 "DZ" / "ZONE"
            K_cand_max=K_cand_max,
            N_min=N_min,
            N_max=N_max,
            k_r=cfg.get("mpd", {}).get("k_r", 4),
            rank_policy=cfg.get("mpd", {}).get("rank_policy", "latency"),
        )
    elif algo == "SBD":
        system = SBDSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=disjoint_mode,
            K_cand_max=K_cand_max,
            N_min=N_min,
            N_max=N_max,
            vnf_risk_th=cfg.get("sbd", {}).get("vnf_risk_th", 0.3),
            rho_cpu=cfg.get("sbd", {}).get("rho_cpu", 0.6),
        )
    elif algo == "OPTSEP":
        system = OptSEPSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=disjoint_mode,     # 建议 "EDGE"
            K_cand_max=K_cand_max,          # 与 config.topology.k_paths 对齐
            k_shortest_cap=cfg.get("optsep", {}).get("k_shortest_cap", 32),
            eps_equal=cfg.get("optsep", {}).get("eps_equal", 1e-6),
        )
    elif algo == "BEAR":
        system = BEARSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=disjoint_mode,   # 与拓扑配置保持一致
            K_cand_max=K_cand_max,
            N_min=N_min, N_max=N_max,      # 可与 config.yaml 同步
            alpha_temp=cfg.get("bear", {}).get("alpha_temp", 0.35),
        )
    elif algo == "BEAR-FULL":
        system = BEARFullSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=disjoint_mode,
            K_cand_max=K_cand_max,
            N_min=N_min, N_max=N_max,
            beam_size=cfg.get("bear_full", {}).get("beam_size", 8),
            bandit_gamma=cfg.get("bear_full", {}).get("bandit_gamma", 0.07),
            alpha_temp0=cfg.get("bear_full", {}).get("alpha_temp0", 0.35),
            alpha_lmbd0=cfg.get("bear_full", {}).get("alpha_lmbd0", 1.0),
            alpha_lr=cfg.get("bear_full", {}).get("alpha_lr", 0.05),
            reward_weights=tuple(cfg.get("bear_full", {}).get("reward_weights",
                        [2.0, 1.5, 1.0, -0.8, -0.6])),
        )
    elif algo == "BEAR-TORCH":
        system = BEARTorchSystem(
            env=env,
            result_dir=str(result_dir),
            save_dir=str(save_dir),
            disjoint_mode=disjoint_mode,
            K_cand_max=K_cand_max,
            N_min=N_min, N_max=N_max,
            k_feat_dim=4,        # 与文件中一致
            obs_dim=16,          # 与文件中一致
            beam_size=cfg.get("bear_torch", {}).get("beam_size", 8),
            device=cfg.get("bear_torch", {}).get("device", None),
        )
    else:
        raise ValueError(f"Unknown algorithm {algo}")

    # # 构造系统
    # system = BearSystem(
    #     env=env,
    #     result_dir=str(result_dir),
    #     save_dir=str(save_dir),
    #     disjoint_mode=disjoint_mode,
    #     K_cand_max=K_cand_max,
    #     central_state_dim=central_state_dim,
    #     N_min=N_min, N_max=N_max,
    #     central_cfg=central_cfg,
    #     edge_algo=edge_algo,
    # )

    # 恢复模型（若指定）
    if CONFIG.get("resume", False):
        if algo == "BEAR-SFC":
            try:
                system.central.load(str(save_dir), prefix="central_dqn")
            except Exception:
                try:
                    system.central.load(str(Path(CONFIG["out"])), prefix="central_dqn")
                except Exception:
                    pass
            if hasattr(system.edge_algo, "load"):
                try:
                    system.edge_algo.load(str(save_dir), prefix="edge_ppo")
                except Exception:
                    try:
                        system.edge_algo.load(str(Path(CONFIG["out"])), prefix="edge_ppo")
                    except Exception:
                        pass
        elif algo == "DRL":
            try:
                system.agent.load(str(save_dir), prefix="drl_ppo")
            except Exception:
                try:
                    system.agent.load(str(Path(CONFIG["out"])), prefix="drl_ppo")
                except Exception:
                    pass

    # 运行
    try:
        # system.run(
        #     mode=CONFIG["mode"],
        #     epochs=int(CONFIG["epochs"]),
        #     steps=int(CONFIG["steps"]),
        #     fixed_N=(int(CONFIG["fixed_N"]) if CONFIG["fixed_N"] is not None else None)
        # )
        # if CONFIG["mode"].lower() == "pranos2":
        #     pr = PRANOSBaseline(
        #         env=env,
        #         result_dir=str(result_dir),
        #         save_dir=str(save_dir),
        #         disjoint_mode=CONFIG["topology"]["disjoint"].upper(),
        #         K_cand_max=int(CONFIG["topology"]["k_paths"]),
        #         N_fixed=int(CONFIG["runner"]["fixed_N"]) if CONFIG["runner"]["fixed_N"] is not None else None,
        #     )
        #     pr.run(epochs=int(CONFIG["runner"]["epochs"]), steps=int(CONFIG["runner"]["steps_per_episode"]))
        #     return
        
        mode = str(CONFIG["mode"]).lower()
        epochs = int(CONFIG["epochs"])
        fixed_N = int(CONFIG["fixed_N"]) if CONFIG.get("fixed_N") is not None else None

        # Steps sweep list (as requested)
        steps_sweep = [1000] #, 200, 300, 400, 500, 600, 700, 800, 900, 1500, 2000

        for STEPS in steps_sweep:
            # Repoint episode summary writer to a steps-specific file: episode_summary_<steps>.csv
            try:
                # Close default writer if exists
                if hasattr(system, "ep_writer") and system.ep_writer is not None:
                    system.ep_writer.close()
            except Exception:
                pass

            steps_summary_path = result_dir / f"episode_summary_{STEPS}.csv"
            try:
                steps_summary_path.unlink()
            except FileNotFoundError:
                pass

            # Create a new writer bound to this STEPS value
            if hasattr(system, "ep_writer"):
                system.ep_writer = EpisodeSummaryWriter(steps_summary_path)

            logging.info("[RUN] Start sweep for STEPS=%d", STEPS)

            for ep in range(epochs):
                # 单集运行（会自动写 events_bear.csv 与 episode_summary_<steps>.csv）
                summary = system.run_one_episode(
                    ep_idx=ep,
                    steps=STEPS,
                    mode=mode,
                    fixed_N=fixed_N
                )

                if algo == "BEAR-SFC":
                    # 逐轮保存中央 DQN
                    try:
                        system.central.save(str(save_dir), prefix=f"central_dqn_ep{ep:03d}")
                    except Exception as e:
                        logging.warning("save central_dqn failed at ep %d: %s", ep, e)

                    # 逐轮保存边缘 PPO（若存在）
                    if hasattr(system.edge_algo, "save"):
                        try:
                            system.edge_algo.save(str(save_dir), prefix=f"edge_ppo_ep{ep:03d}")
                        except Exception as e:
                            logging.warning("save edge_ppo failed at ep %d: %s", ep, e)

                # 控制台与文件日志中打印关键指标
                logging.info(
                    "[STEPS %d | %s EP %03d] place=%.3f fo_hit=%.3f emp_av=%.3f cost=%.3f lat=%.3f",
                    STEPS,
                    mode.upper(), ep,
                    summary.get("place_rate", 0.0),
                    summary.get("fo_hit_rate", 0.0),
                    summary.get("emp_avail", 0.0),
                    summary.get("avg_cost_total", 0.0),
                    summary.get("avg_latency_ms", 0.0),
                )

            # Close steps-specific writer to flush
            try:
                if hasattr(system, "ep_writer") and system.ep_writer is not None:
                    system.ep_writer.close()
            except Exception:
                pass

        # 训练结束：善后关闭（bear.py 内部 writer/ logger 会在 system.run 末尾 close，
        # 但此处我们手工循环，建议在 BearSystem 暴露关闭方法或直接访问实例资源）
        try:
            system.ev_logger.close()
            system.ep_writer.close()
        except Exception:
            pass
    except KeyboardInterrupt:
        print("\n[Runner] Interrupted by user.")
    finally:
        pass


if __name__ == "__main__":
    main()