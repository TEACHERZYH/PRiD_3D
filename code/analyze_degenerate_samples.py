"""近空/退化样本诊断：连通性指标对近空集失真的量化。

背景：多 seed 实验中发现"每 seed 常有 1 个退化样本"（no_bridge 结果近空，甚至 2 体素），
而连通性（最大连通分量占材料比）在这种近空集上会"平凡地"等于 1.0，从而虚高均值。

本脚本量化：
  1. 各方法的占用体素分布（分位数 + 近空样本计数）
  2. 近空样本的连通性是否确实虚高
  3. 剔除近空样本后，各方法的连通性均值如何变化（敏感性）

用法：
    python code/analyze_degenerate_samples.py --csv data/multiseed15_detail.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


METHODS = ["unrefined", "one_shot", "no_bridge", "prid"]
THRESHOLDS = [0, 10, 50, 100]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/multiseed15_detail.csv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 82)
    p("近空/退化样本诊断（连通性指标对近空集失真）")
    p("=" * 82)
    p(f"数据：{args.csv}  行数={len(df)}  种子数={df['seed'].nunique()}"
      f"  每 seed 样本数={len(df) // df['seed'].nunique()}")

    # 1. 占用体素分布
    p("\n" + "-" * 82)
    p("1. 占用体素分布（>0.5 的体素数；设计域 16^3=4096）")
    p("-" * 82)
    p(f"{'方法':<12}{'均值':>9}{'P5':>8}{'P25':>8}{'中位':>8}{'P75':>9}{'最小':>8}")
    for m in METHODS:
        occ = df[f"{m}_occ"].to_numpy()
        p(f"{m:<12}{occ.mean():>9.1f}{np.percentile(occ,5):>8.0f}"
          f"{np.percentile(occ,25):>8.0f}{np.median(occ):>8.0f}"
          f"{np.percentile(occ,75):>9.0f}{occ.min():>8.0f}")

    # 2. 近空样本计数 + 其连通性
    p("\n" + "-" * 82)
    p("2. 近空样本（occ ≤ 阈值）：计数 与 其连通性均值")
    p("-" * 82)
    p(f"{'方法':<12}" + "".join(f"{'≤' + str(t):>16}" for t in THRESHOLDS))
    for m in METHODS:
        occ = df[f"{m}_occ"].to_numpy()
        conn = df[f"{m}_conn"].to_numpy()
        row = f"{m:<12}"
        for t in THRESHOLDS:
            mask = occ <= t
            n = int(mask.sum())
            c = conn[mask].mean() if n else float("nan")
            row += f"{f'{n}({c:.3f})' if n else '0':>16}"
        p(row)
    p("（格式：样本数(这些样本的连通性均值)；阈值 0 = 完全空结构，其连通性定义为 0）")

    # 3. 剔除近空后的敏感性
    p("\n" + "-" * 82)
    p("3. 剔除近空样本后连通性均值的变化（敏感性分析）")
    p("-" * 82)
    p(f"{'方法':<12}{'全部':>10}{'occ>0':>10}{'occ>10':>10}{'occ>50':>10}{'occ>100':>11}")
    for m in METHODS:
        occ = df[f"{m}_occ"].to_numpy()
        conn = df[f"{m}_conn"].to_numpy()
        row = f"{m:<12}{conn.mean():>10.4f}"
        for t in THRESHOLDS[1:]:
            mask = occ > t
            row += f"{conn[mask].mean():>10.4f}" if mask.any() else f"{'—':>10}"
        p(row)

    # 4. 每 seed 的近空样本数（验证"每 seed 1 个"之说）
    p("\n" + "-" * 82)
    p("4. 每 seed 的近空样本数（occ ≤ 50）")
    p("-" * 82)
    p(f"{'seed':>8}" + "".join(f"{m:>12}" for m in METHODS))
    for s, g in df.groupby("seed"):
        p(f"{s:>8}" + "".join(
            f"{int((g[f'{m}_occ'] <= 50).sum()):>12}" for m in METHODS))

    # 5. worst-case 口径
    p("\n" + "-" * 82)
    p("5. worst-case 口径：每 seed 取该 seed 内最差的样本连通性，再跨 seed 平均")
    p("-" * 82)
    p(f"{'方法':<12}{'均值':>10}{'sd':>9}{'最差 seed':>12}")
    for m in METHODS:
        per_seed_min = df.groupby("seed")[f"{m}_conn"].min()
        p(f"{m:<12}{per_seed_min.mean():>10.4f}{per_seed_min.std(ddof=1):>9.4f}"
          f"{per_seed_min.min():>12.4f}")

    # 6. 塌缩/近空事件的配对一致性检验（McNemar）
    p("\n" + "-" * 82)
    p("6. 近空事件（occ ≤ 阈值）的配对 McNemar 检验（同一批样本、同一噪声配对）")
    p("-" * 82)
    try:
        from scipy.stats import binomtest
        has_sm = True
    except Exception:
        has_sm = False
    for t in [0, 50]:
        p(f"  阈值 occ ≤ {t}")
        for a, b in [("no_bridge", "prid"), ("one_shot", "prid"), ("one_shot", "no_bridge")]:
            ea = df[f"{a}_occ"].to_numpy() <= t
            eb = df[f"{b}_occ"].to_numpy() <= t
            n10 = int((ea & ~eb).sum())   # a 坏、b 好
            n01 = int((~ea & eb).sum())   # a 好、b 坏
            note = ""
            if has_sm and (n01 + n10) > 0:
                # 精确 McNemar：不一致对中 b 更好的比例是否偏离 0.5
                pv = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
                note = f"  p={pv:.4f}"
            p(f"    {a} vs {b}: {a} 近空 {int(ea.sum())} / {b} 近空 {int(eb.sum())}"
              f"  不一致对 {a}坏={n10} vs {b}坏={n01}{note}")
    if not has_sm:
        p("    (scipy 不可用，仅报计数)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[写入] {args.out}")


if __name__ == "__main__":
    main()
