"""Z1 训练-输出可行性差距配对检验 + Z2 floating material ratio 重算。

Z1（P1 的统计检验）：同一设计条件下，模型生成输出的可行性是否系统性高于
    训练标签（SIMP 解）？按样本配对（先对每个样本跨 seed 求均值，再与标签配对，
    保证样本独立性）。
Z2（文献可比口径）：把连通性结论换算为 floating material ratio（两种口径），
    并统计退化（近空）样本，评估指标失真的影响范围。

用法:
    python code/analyze_feasibility_gap.py \
        --data data/train_16.npz --detail data/multiseed15_detail.csv
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy.ndimage import label as cc, generate_binary_structure
from scipy.stats import ttest_rel, wilcoxon

METHODS = ["unrefined", "one_shot", "no_bridge", "prid"]


def conn_stats(binary):
    total = int(binary.sum())
    if total == 0:
        return 0.0, 1.0, 0
    lab, n = cc(binary, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max())
    return largest / total, 1.0 - largest / total, int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--detail", default="data/multiseed15_detail.csv")
    ap.add_argument("--degen_occ", type=int, default=10)
    ap.add_argument("--out_summary", default="data/feasibility_gap_summary.csv")
    args = ap.parse_args()

    # ---------- 训练标签 ----------
    d = np.load(args.data)
    labels = d["labels"]
    nvox = int(np.prod(labels.shape[2:]))
    lab_conn, lab_occ = [], []
    for i in range(len(labels)):
        b = labels[i, 0] > 0.5
        c, f, n = conn_stats(b)
        lab_conn.append(c)
        lab_occ.append(int(b.sum()))
    lab_conn = np.array(lab_conn)
    lab_occ = np.array(lab_occ)

    print("=" * 74)
    print(f"训练标签（SIMP 解，{len(labels)} 样本，设计域 {nvox} 体素）")
    print("=" * 74)
    print(f"  连通性          均值 {lab_conn.mean():.4f} ± {lab_conn.std(ddof=1):.4f}  "
          f"min {lab_conn.min():.4f}")
    print(f"  完全连通样本    {int((lab_conn > 0.999).sum())}/{len(labels)}")
    print(f"  有缺陷样本      {int((lab_conn < 0.999).sum())}/{len(labels)} "
          f"（其中 <0.95 有 {int((lab_conn < 0.95).sum())} 个）")
    print(f"  退化样本(occ≤{args.degen_occ})  {int((lab_occ <= args.degen_occ).sum())} 个"
          f"（occ=0 有 {int((lab_occ == 0).sum())} 个）")
    print(f"  浮材率(占材料)  均值 {(1-lab_conn).mean():.4f}  中位 {np.median(1-lab_conn):.4f}")
    print(f"  浮材率(占设计域) 均值 {((1-lab_conn)*lab_occ/nvox).mean():.4f}")

    # ---------- 模型输出 ----------
    rows = list(csv.DictReader(open(args.detail, encoding="utf-8")))
    seeds = sorted({int(r["seed"]) for r in rows})
    samples = sorted({int(r["sample"]) for r in rows})
    print(f"\n模型输出：{len(rows)} 条（{len(seeds)} seed × {len(samples)} 样本）")

    # 每个样本跨 seed 平均
    out = {}
    for m in METHODS:
        out[m] = {}
        for s in samples:
            sub = [r for r in rows if int(r["sample"]) == s]
            out[m][s] = {
                "conn": np.mean([float(r[f"{m}_conn"]) for r in sub]),
                "occ": np.mean([float(r[f"{m}_occ"]) for r in sub]),
            }

    print("\n" + "=" * 74)
    print("Z2：连通性与 floating material ratio（按样本跨 seed 平均）")
    print("=" * 74)
    print(f"{'方法':<12}{'连通性':>10}{'FM(占材料)':>12}{'FM(占设计域)':>13}{'占用体素':>10}")
    for m in METHODS:
        c = np.mean([out[m][s]["conn"] for s in samples])
        o = np.mean([out[m][s]["occ"] for s in samples])
        fm_mat = 1 - c
        fm_dom = np.mean([(1 - out[m][s]["conn"]) * out[m][s]["occ"] / nvox for s in samples])
        print(f"{m:<12}{c:>10.4f}{fm_mat:>12.4f}{fm_dom:>13.4f}{o:>10.1f}")
    print(f"{'训练标签':<12}{lab_conn.mean():>10.4f}"
          f"{(1-lab_conn).mean():>12.4f}"
          f"{((1-lab_conn)*lab_occ/nvox).mean():>13.4f}{lab_occ.mean():>10.1f}")

    # ---------- Z1 配对检验 ----------
    print("\n" + "=" * 74)
    print("Z1：训练标签 vs 模型输出（同条件配对，n=样本数）")
    print("=" * 74)
    summary = []
    lab_by_sample = {s: lab_conn[s] for s in samples}
    lab_occ_by_sample = {s: lab_occ[s] for s in samples}

    groups = {
        "全部样本": list(samples),
        "排除标签退化": [s for s in samples if lab_occ_by_sample[s] > args.degen_occ],
    }
    for gname, gs in groups.items():
        if len(gs) < 3:
            continue
        print(f"\n  [{gname}] n={len(gs)}")
        for m in METHODS:
            o = np.array([out[m][s]["conn"] for s in gs])
            l = np.array([lab_by_sample[s] for s in gs])
            diff = o - l
            t, p = ttest_rel(o, l)
            try:
                w, pw = wilcoxon(o, l)
            except ValueError:
                pw = float("nan")
            print(f"    {m:<11} 输出 {o.mean():.4f} vs 标签 {l.mean():.4f}  "
                  f"Δ={diff.mean():+.4f}  t={t:.2f} p={p:.4g}  "
                  f"输出更高 {int((diff>1e-9).sum())}/{len(gs)}  wilcoxon p={pw:.4g}")
            summary.append({"group": gname, "n": len(gs), "method": m,
                            "output_conn": o.mean(), "label_conn": l.mean(),
                            "delta": diff.mean(), "t": t, "p": p,
                            "n_higher": int((diff > 1e-9).sum()), "p_wilcoxon": pw})

    # ---------- 退化样本影响 ----------
    print("\n" + "=" * 74)
    print(f"退化（近空）样本影响：no_bridge 输出 occ ≤ {args.degen_occ}")
    print("=" * 74)
    for m in METHODS:
        cnt = sum(1 for s in samples if out[m][s]["occ"] <= args.degen_occ)
        print(f"  {m:<12} {cnt}/{len(samples)} 个样本近空"
              f"（其中 conn=1.0 的 {sum(1 for s in samples if out[m][s]['occ'] <= args.degen_occ and out[m][s]['conn'] > 0.999)} 个为平凡连通）")

    with open(args.out_summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\n汇总 CSV: {args.out_summary}")


if __name__ == "__main__":
    main()
