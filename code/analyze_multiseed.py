"""多 seed 稳定性结果统计：读 multiseed_stability.csv 做方差齐性与配对检验。

用法: python code/analyze_multiseed.py --csv data/multiseed_stability.csv
"""
import argparse
import csv

import numpy as np
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/multiseed_stability.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    g = lambda k: np.array([float(r[k]) for r in rows])

    nb = g("no_bridge_conn")
    pr = g("prid_conn")
    os_ = g("one_shot_conn")
    un = g("unrefined_conn")
    d = pr - nb
    n = len(rows)

    print(f"=== 跨 {n} 个种子（每个 30 样本，严格噪声配对）===")
    for name, v in [("unrefined", un), ("one_shot", os_), ("no_bridge", nb), ("PRiD", pr)]:
        print(f"  {name:10s} 连通性 {v.mean():.4f} ± {v.std(ddof=1):.4f}   min {v.min():.4f}  max {v.max():.4f}")

    print(f"\n=== PRiD vs no_bridge ===")
    t, p = stats.ttest_rel(pr, nb)
    print(f"  逐样本配对 t 检验（合并 {n*len(pr)//n} 样本级）：t={t:.3f}, p={p:.4f}")
    t1, p1 = stats.ttest_1samp(d, 0.0)
    print(f"  种子级 单样本 t（Δ vs 0）：t={t1:.3f}, p={p1:.4f}, Cohen's d={d.mean()/d.std(ddof=1):.2f}")
    print(f"  Δ 均值 {d.mean():+.4f}  跨种子 sd {d.std(ddof=1):.4f}  (|Δ| < sd → 均值优势不稳健)")
    print(f"  种子级胜负：PRiD 更优 {int((d>0).sum())}/{n}，更差 {int((d<0).sum())}/{n}")

    print(f"\n=== 方差齐性（核心：投影是否降低不确定性）===")
    F = nb.var(ddof=1) / pr.var(ddof=1)
    pf = 2 * min(stats.f.cdf(F, n - 1, n - 1), 1 - stats.f.cdf(F, n - 1, n - 1))
    print(f"  no_bridge sd {nb.std(ddof=1):.4f}  vs  PRiD sd {pr.std(ddof=1):.4f}  "
          f"(sd 比 {nb.std(ddof=1)/pr.std(ddof=1):.2f}x)")
    print(f"  F({n-1},{n-1})={F:.2f}, p={pf:.4f}  → {'方差差异显著' if pf < 0.05 else '方差差异不显著'}")

    print(f"\n=== 柔度（PRiD - no_bridge，正=PRiD 更差）===")
    dc = g("prid_minus_nobridge_comp")
    print(f"  各种子 Δ: {np.round(dc, 4).tolist()}")
    print(f"  Δ 均值 {dc.mean():+.4f}  同号种子 {int((dc>0).sum())}/{n}")

    print(f"\n=== 材料量（占用体素）===")
    print(f"  unrefined {g('unrefined_occ').mean():.1f}  no_bridge {g('no_bridge_occ').mean():.1f}  "
          f"PRiD {g('prid_occ').mean():.1f}")
    print(f"  退化样本(no_bridge occ<=10) 各种子: {g('n_degenerate').astype(int).tolist()}")


if __name__ == "__main__":
    main()
