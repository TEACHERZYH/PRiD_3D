"""t_r 敏感性曲线：低噪重构的"保真–修复"权衡。

目的：把"低噪重构"与 SDEdit 的 t0 权衡定量对接。
    SDEdit：t0 小 → 忠实输入（保结构）；t0 大 → 逼真但偏离输入。
    本实验：t_r 小 → 结构变化小但修复能力受限；t_r 大 → 修复强但结构被重新生成。

配置：固定 unrefined 场（同 seed），只变重构噪声水平 t_r ∈ {10,20,30,50,80,120,160,199}，
    K=5 轮、bridge=False（纯重构）。度量连通性、柔度、占用体素、浮材率，
    以及与 unrefined 的结构保真度 IoU。

用法:
    python code/eval_tr_sensitivity.py --n_samples 30 --out data/tr_sensitivity.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from diffusion.diffusion import UNet3D, Diffusion
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index
from eval_bridge_recon_factorial import build_ls, compliance, connectivity_6


def iou(a, b, thresh=0.5):
    A, B = a > thresh, b > thresh
    u = int(np.logical_or(A, B).sum())
    return float(np.logical_and(A, B).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--refine_seed", type=int, default=77000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--t_r_list", default="10,20,30,50,80,120,160,199")
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--out", default="data/tr_sensitivity.csv")
    args = ap.parse_args()

    data = np.load(args.data)
    conds = torch.from_numpy(data["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]

    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    ke = hex8_stiffness(1.0, 0.3)
    ke_flat = ke.ravel()
    edof = build_edof(D, H, W)
    iK, jK = build_k_index(edof)
    ndof = 3 * (D + 1) * (H + 1) * (W + 1)

    N = min(args.n_samples, conds.shape[0])
    shape = (1, 1, D, H, W)
    tr_list = [int(x) for x in args.t_r_list.split(",") if x.strip()]

    rows = []
    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, shape, ddim_steps=args.ddim_steps)
        s_np = s[0, 0].numpy()
        c0, f0, _ = connectivity_6(s_np)
        comp0 = compliance(s_np, loads, supports, ke_flat, iK, jK, ndof)
        occ0 = int((s_np > 0.5).sum())

        for tr in tr_list:
            torch.manual_seed(args.refine_seed + i)
            with torch.no_grad():
                xr = diff.refine(s, cond, src, sup, K=args.K,
                                 t_refine=tr, bridge=False, reconstruct=True)
            arr = xr[0, 0].numpy()
            c, f, nc = connectivity_6(arr)
            rows.append([
                i, tr, c, f, nc,
                compliance(arr, loads, supports, ke_flat, iK, jK, ndof),
                int((arr > 0.5).sum()),
                iou(s_np, arr),
                c0, comp0, occ0,
            ])

        print(f"[sample {i}] " + " ".join(
            f"tr{r[1]}:{r[2]:.3f}/{r[7]:.2f}" for r in rows if r[0] == i))

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "t_r", "conn", "floating", "ncomp", "comp",
                    "occ", "iou_vs_unrefined", "unrefined_conn",
                    "unrefined_comp", "unrefined_occ"])
        w.writerows(rows)

    print("\n" + "=" * 78)
    print("t_r 敏感性：保真–修复权衡（no-bridge 纯重构，K=5，30 样本）")
    print("=" * 78)
    print(f"{'t_r':>5}{'连通性':>10}{'浮材率':>10}{'IoU(保真)':>12}{'占用体素':>10}{'柔度':>10}")
    base_conn = np.mean([r[8] for r in rows if r[1] == tr_list[0]])
    base_occ = np.mean([r[10] for r in rows if r[1] == tr_list[0]])
    print(f"{'unref':>5}{base_conn:>10.4f}{1-base_conn:>10.4f}{1.0:>12.4f}{base_occ:>10.1f}"
          f"{np.mean([r[9] for r in rows if r[1]==tr_list[0]]):>10.2f}")
    for tr in tr_list:
        sub = [r for r in rows if r[1] == tr]
        valid = [r for r in sub if 0 < r[5] < 1e6]
        comp_m = np.mean([r[5] for r in valid]) if valid else float("nan")
        print(f"{tr:>5}{np.mean([r[2] for r in sub]):>10.4f}"
              f"{np.mean([r[3] for r in sub]):>10.4f}"
              f"{np.mean([r[7] for r in sub]):>12.4f}"
              f"{np.mean([r[6] for r in sub]):>10.1f}"
              f"{comp_m:>10.2f}")

    print("\n注：IoU = 与 unrefined 阈值化结构的交并比（1.0 = 完全保留）。")
    print("    低 t_r 应保真但修复不足；高 t_r 修复强但结构被重新生成。")
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
