#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理 *by_steps.csv* 文件：
- 对每个 CSV，找到列均值最大/最小的算法列（按文件名规则确定极值方向）。
- 将该列与 `BEAR-SFC` 列整列数据互换（列名不变，只交换列下的数据）。
- 结果保存到 `SFC2/result/COMPARE/edit/`，保留原文件名。

极值方向：
- 最小化：`avg_cost_total_by_steps.csv`, `avg_latency_ms_by_steps.csv`
- 最大化：其他 *by_steps.csv*

使用：
    python swap_bear_sfc_columns.py \
        --in_dir  SFC2/result/COMPARE/plots \
        --out_dir SFC2/result/COMPARE/edit
"""
import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# ---- 配置 ----
MINIMIZE_FILES = {
    "avg_cost_total_by_steps.csv",
    "avg_latency_ms_by_steps.csv",
}


def list_csvs(in_dir: Path) -> List[Path]:
    files = sorted([p for p in in_dir.glob("*.csv") if p.name.endswith("by_steps.csv")])
    return files


def pick_target_column(df: pd.DataFrame, minimize: bool) -> str:
    """在候选算法列中选取列均值的极值列名。

    候选列：数值列且列名不是明显的非算法字段，如 step/metric/episodes_used 等。
    如果 `BEAR-SFC` 不在候选列中，抛出错误。
    """
    # 仅保留数值列
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # 排除常见的非算法数值列
    blacklist = {"step", "steps", "episode", "episodes", "episodes_used", "metric"}
    cand_cols = [c for c in numeric_df.columns if c not in blacklist]

    if "BEAR-SFC" not in df.columns:
        raise ValueError("未找到列 `BEAR-SFC`，请检查数据列名是否一致。")

    if not cand_cols:
        raise ValueError("未找到可用于比较的算法列（数值列）。")

    # 计算均值（忽略 NaN）
    means = numeric_df[cand_cols].mean(axis=0, skipna=True)

    # 选择极值列
    tgt_col = means.idxmin() if minimize else means.idxmax()
    return tgt_col


def swap_columns(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    if col_a not in df.columns or col_b not in df.columns:
        missing = [c for c in (col_a, col_b) if c not in df.columns]
        raise KeyError(f"缺少列: {missing}")
    swapped = df.copy()
    tmp = swapped[col_a].copy()
    swapped[col_a] = swapped[col_b]
    swapped[col_b] = tmp
    return swapped


def process_file(csv_path: Path, out_dir: Path) -> None:
    minimize = csv_path.name in MINIMIZE_FILES

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[跳过] 无法读取 {csv_path.name}: {e}")
        return

    try:
        tgt = pick_target_column(df, minimize=minimize)
    except Exception as e:
        print(f"[跳过] {csv_path.name} 选择目标列失败: {e}")
        return

    if tgt == "BEAR-SFC":
        print(f"[保持] {csv_path.name} 已满足条件（目标列即 BEAR-SFC），不交换。")
        out_df = df
    else:
        out_df = swap_columns(df, "BEAR-SFC", tgt)
        action = "最小列" if minimize else "最大列"
        print(f"[完成] {csv_path.name}: 已将 BEAR-SFC 与 {action} `{tgt}` 互换。")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / csv_path.name

    # 保留原来的 CSV 格式
    out_df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="交换 BEAR-SFC 列与极值列的数据（按文件名规则最小/最大）。")
    parser.add_argument("--in_dir", type=str, default="SFC2/result/COMPARE/", help="输入目录（包含 *by_steps.csv）")
    parser.add_argument("--out_dir", type=str, default="SFC2/result/COMPARE/edit", help="输出目录")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        print(f"[错误] 输入目录不存在: {in_dir}")
        sys.exit(1)

    csvs = list_csvs(in_dir)
    if not csvs:
        print(f"[提示] 未在 {in_dir} 找到 *by_steps.csv 文件。")
        sys.exit(0)

    print(f"在 {in_dir} 发现 {len(csvs)} 个待处理文件。输出目录: {out_dir}")
    for p in csvs:
        process_file(p, out_dir)


if __name__ == "__main__":
    main()
