"""改进方案探索结果分析：变体排名 + Pareto 前沿 + 配对检验。

用法：
    python code/analyze_variants.py
输出：
    paper/v2_rewrite_20260915/output/改进方案对比.txt
"""
from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "paper/v2_rewrite_20260915/output/改进方案对比.txt"
LINES: list[str] = []

BASE = "recon30"          # 论文当前主力配方，作为改进的基准
CAND = "volcons"          # 候选改进
SEEDS15 = [12000, 22000, 32000, 42000, 52000, 60000, 61000, 62000, 63000,
           64000, 65000, 66000, 67000, 68000, 69000]


def p(s: str = "") -> None:
    print(s)
    LINES.append(s)


def load():
    rows = []
    for f in sorted(glob.glob(str(DATA / "explore_variants_*.csv"))):
        if "_smoke" in f:
            continue
        with open(f, encoding="utf-8-sig") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="", help="覆盖默认 data/ 目录（用于 OOD 重跑）")
    ap.add_argument("--seeds", default="",
                    help="逗号分隔，只统计这些 seed（留空=各变体各自可用的全部）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    global DATA
    if args.data_dir:
        DATA = Path(args.data_dir)
    seed_filter = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    rows = load()
    if not rows:
        print("未找到 data/explore_variants_*.csv")
        return 1
    if seed_filter:
        rows = [r for r in rows if int(r["seed"]) in set(seed_filter)]
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    # 不同文件覆盖的变体不同（15 seed 确认只跑 4 个变体）→ 每个变体用各自可用的配对
    vkeys: dict[str, list] = {}
    for v in variants:
        vkeys[v] = sorted({(int(r["seed"]), int(r["sample"])) for r in rows
                           if r["variant"] == v})
    idx = {(int(r["seed"]), int(r["sample"]), r["variant"]): r for r in rows}
    seeds = sorted({k[0] for k in vkeys[variants[0]]})
    n_unit = len(vkeys[variants[0]])

    p("=" * 100)
    p("改进方案探索：精炼变体对比")
    p("=" * 100)
    p(f"样本：每个变体用其自身覆盖的配对单元（最大 {n_unit} 个，{len(seeds)} seed）；"
      f"变体 {len(variants)} 个")
    p("")
    p(f"{'变体':<15}{'C':>9}{'sd样本':>9}{'sd跨seed':>10}{'worst C':>9}{'F':>8}{'占用':>8}"
      f"{'IoU':>7}{'柔度':>8}{'降幅':>8}{'近空':>6}{'全空':>6}{'n':>6}")
    p("-" * 110)

    stat = {}
    for v in variants:
        ks = vkeys[v]
        c = np.array([float(idx[k + (v,)]["conn"]) for k in ks])
        comp = np.array([float(idx[k + (v,)]["comp"]) for k in ks])
        occ = np.array([float(idx[k + (v,)]["occ"]) for k in ks])
        iou = np.array([float(idx[k + (v,)]["iou_vs_unref"]) for k in ks])
        fl = np.array([float(idx[k + (v,)]["floating"]) for k in ks])
        ok = np.isfinite(comp) & (comp > 0)
        vseeds = sorted({k[0] for k in ks})
        per_seed_min = [c[[j for j, k in enumerate(ks) if k[0] == s]].min()
                        for s in vseeds]
        # 与论文表 4 一致的口径：跨 seed 的**seed 均值**标准差
        per_seed_mean = [c[[j for j, k in enumerate(ks) if k[0] == s]].mean()
                         for s in vseeds]
        stat[v] = dict(c=c, comp=comp, occ=occ, iou=iou, fl=fl, ok=ok,
                       worst=float(np.mean(per_seed_min)),
                       sd_seed=float(np.std(per_seed_mean, ddof=1))
                       if len(per_seed_mean) > 1 else float("nan"),
                       n_empty=int((occ <= 50).sum()),
                       n_zero=int((occ == 0).sum()), n=len(ks),
                       n_seed=len(vseeds), raw_mean=None)  # 稍后填
    raw_mean = stat["unref"]["comp"][stat["unref"]["ok"]].mean()
    for v in variants:
        stat[v]["raw_mean"] = raw_mean

    for v in variants:
        s = stat[v]
        cm = s["comp"][s["ok"]].mean() if s["ok"].any() else float("nan")
        drop = 1 - cm / raw_mean
        p(f"{v:<15}{s['c'].mean():>9.4f}{s['c'].std(ddof=1):>8.4f}"
          f"{s['sd_seed']:>9.4f}{s['worst']:>9.4f}"
          f"{s['fl'].mean():>8.4f}{s['occ'].mean():>8.1f}{s['iou'].mean():>7.3f}"
          f"{cm:>8.2f}{100*drop:>7.1f}%{s['n_empty']:>6}{s['n_zero']:>6}"
          f"{s['n']:>6}{s['n_seed']:>6}")
    p("-" * 110)
    p(f"柔度降幅相对**原始采样**（{raw_mean:.2f}，n_valid={int(stat['unref']['ok'].sum())}）")
    p("")

    # ---- 配对检验：候选 vs 基准 ----
    p("=" * 100)
    p(f"配对检验（基准 = {BASE}）")
    p("=" * 100)
    for v in variants:
        if v in (BASE, "unref"):
            continue
        common = sorted(set(vkeys[v]) & set(vkeys[BASE]))
        if len(common) < 5:
            p(f"  {v:<15} （与基准的共同配对仅 {len(common)} 个，跳过）")
            continue
        diffs = {}
        for metric, key in (("连通性", "conn"), ("柔度", "comp"), ("IoU", "iou_vs_unref"),
                            ("占用", "occ")):
            a = np.array([float(idx[k + (v,)][key]) for k in common])
            b = np.array([float(idx[k + (BASE,)][key]) for k in common])
            if metric == "柔度":
                m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            else:
                m = np.ones(len(a), dtype=bool)
            if m.sum() < 5:
                continue
            t, pv = ttest_rel(a[m], b[m])
            diffs[metric] = (np.mean(a[m] - b[m]), pv, int(m.sum()))
        parts = "  ".join(f"{k}: Δ={d:+.4f} p={pv:.2g}(n={n})"
                          for k, (d, pv, n) in diffs.items())
        p(f"  {v:<15} {parts}")

    # ---- Pareto 前沿：零塌缩前提下 (柔度↓, C↑, IoU↑) ----
    p("")
    p("=" * 100)
    p("Pareto 前沿（约束：近空样本数 ≤ 1，即基本不塌缩）")
    p("=" * 100)
    ok_v = [v for v in variants if stat[v]["n_empty"] <= 1]
    front = []
    for v in ok_v:
        s = stat[v]
        cm = s["comp"][s["ok"]].mean()
        dominated = False
        for u in ok_v:
            if u == v:
                continue
            t = stat[u]
            cu = t["comp"][t["ok"]].mean()
            if (cu <= cm + 1e-9 and t["c"].mean() >= s["c"].mean() - 1e-9
                    and t["iou"].mean() >= s["iou"].mean() - 1e-9
                    and (cu < cm - 1e-6 or t["c"].mean() > s["c"].mean() + 1e-6
                         or t["iou"].mean() > s["iou"].mean() + 1e-6)):
                dominated = True
                break
        if not dominated:
            front.append(v)
    p("  前沿变体：" + ", ".join(front))
    if CAND in front:
        p(f"  ⇒ 候选改进「{CAND}」位于前沿；相对 {BASE}："
          f"柔度 {stat[BASE]['comp'][stat[BASE]['ok']].mean():.2f} → "
          f"{stat[CAND]['comp'][stat[CAND]['ok']].mean():.2f}，"
          f"C {stat[BASE]['c'].mean():.4f} → {stat[CAND]['c'].mean():.4f}，"
          f"IoU {stat[BASE]['iou'].mean():.3f} → {stat[CAND]['iou'].mean():.3f}")
    elif CAND not in ok_v:
        p(f"  ⚠️ 候选「{CAND}」近空样本数 {stat[CAND]['n_empty']} 超限")
    p("")
    p("判读建议：改进成立需同时满足 ①近空样本数不增加 ②柔度显著更低 ③C 不降低 ④IoU 可接受")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
