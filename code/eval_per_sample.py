"""逐样本统计表：对每个验证样本输出三种方法（无条件/采样后投影/投影精炼）的
连通性、漂浮体比例、连通分量数，配对设计（固定 seed，共享初始噪声）。

输出：CSV 文件（data/per_sample.csv）+ 控制台 markdown 表格。

用法:
    python code/eval_per_sample.py --n_samples 30
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from scipy.ndimage import label as cc

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection


def metrics(rho: np.ndarray):
    """返回 (连通性, 漂浮体比例, 连通分量数)。"""
    bn = rho > 0.5
    if not bn.any():
        return 0.0, 1.0, 0
    ls, nc = cc(bn)
    sizes = np.bincount(ls.ravel())
    sizes[0] = 0
    mc = int(np.argmax(sizes))
    conn = (ls == mc).sum() / bn.sum()
    return conn, 1.0 - conn, nc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--out", default="data/per_sample.csv")
    args = ap.parse_args()

    data = np.load(args.data)
    conds = torch.from_numpy(data["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]

    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    N = min(args.n_samples, conds.shape[0])
    rows = []
    header = [
        "sample", "volfrac",
        "uncond_conn", "uncond_float", "uncond_ncomp",
        "posthoc_conn", "posthoc_float", "posthoc_ncomp",
        "refine_conn", "refine_float", "refine_ncomp",
        "gain_conn_refine_vs_posthoc",
    ]

    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        vf = float(conds[i, 0, 0, 0, 0])
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            ph = manifold_projection(s, source_mask=src, support_mask=sup)
            rf = diff.refine(s, cond, src, sup, K=5, t_refine=30)
        uc, uf, un = metrics(s[0, 0].numpy())
        pc, pf, pn = metrics(ph[0, 0].numpy())
        rc, rfl, rn = metrics(rf[0, 0].numpy())
        rows.append([i, vf, uc, uf, un, pc, pf, pn, rc, rfl, rn, rc - pc])

    # 写 CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}")

    # 控制台 markdown 表格
    print("\n=== 逐样本统计表 ===")
    print("| idx | vf | 无条件conn | 采样后conn | 精炼conn | 精炼-采样后 |")
    print("|-----|-----|-----------|-----------|---------|------------|")
    for r in rows:
        print(f"| {r[0]:3d} | {r[1]:.2f} | {r[2]:.4f} | {r[5]:.4f} | {r[8]:.4f} | {r[11]:+.4f} |")

    # 均值
    uc_m = np.mean([r[2] for r in rows])
    pc_m = np.mean([r[5] for r in rows])
    rc_m = np.mean([r[8] for r in rows])
    d = np.array([r[11] for r in rows])
    t = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
    print(f"\n=== 均值汇总 ===")
    print(f"无条件: {uc_m:.4f} | 采样后: {pc_m:.4f} | 精炼: {rc_m:.4f}")
    print(f"精炼-采样后 差异: {d.mean():+.4f}, t={t:.3f}, 正样本占比={(d>0).mean():.3f}")


if __name__ == "__main__":
    main()
