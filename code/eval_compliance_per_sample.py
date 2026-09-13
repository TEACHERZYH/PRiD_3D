"""柔度逐样本统计：对每个验证样本输出无条件/采样后投影/投影精炼的柔度，
配对设计（固定 seed）。柔度用稀疏 FEM 重算。

用法:
    python code/eval_compliance_per_sample.py --n_samples 20
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index


def build_ls(cond_np, D, H, W, ndof):
    """从条件通道构建荷载向量和支座自由度掩码。

    坐标约定（与 synthesize_parallel.py 一致）：数组维度 (x, y, z)，
    节点编号 = x + (nx+1)*y + (nx+1)*(ny+1)*z。
    支座：x=0 面节点全固定；荷载：顶面 z=nz 节点，-z 集中力。
    """
    loads = np.zeros(ndof)
    supports = np.zeros(ndof, dtype=bool)
    # 支座（通道 3）：x=0 面 → 节点 x=0（复刻数据生成循环）
    if cond_np[3].any():
        for z in range(W + 1):
            for y in range(H + 1):
                nid = (D + 1) * (H + 1) * z + (D + 1) * y  # 节点 (x=0, y, z)
                supports[3 * nid:3 * nid + 3] = True
    # 荷载（通道 2）：顶面单元 → 顶面节点 (x, y, W)，-z 方向
    lp = np.argwhere(cond_np[2] > 0.5)
    if len(lp) > 0:
        x, y, z = lp[0]  # 单元坐标 (x, y, z)
        nid = x + (D + 1) * y + (D + 1) * (H + 1) * W  # 节点 (x, y, W)
        loads[3 * nid + 2] = -1.0
    return loads, supports


def compliance(rho, loads, supports, ke_flat, iK, jK, ndof):
    """稀疏 FEM 重算柔度。"""
    E = 1e-9 + (1 - 1e-9) * rho.ravel() ** 3
    vals = (ke_flat[None, :] * E[:, None]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = ~supports
    Uf = spsolve(K[free][:, free], loads[free])
    U = np.zeros(ndof)
    U[free] = Uf
    return float(loads @ U)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--out", default="data/compliance_per_sample.csv")
    args = ap.parse_args()

    data = np.load(args.data)
    conds = torch.from_numpy(data["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]

    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    # 预计算 FEM 结构
    ke = hex8_stiffness(1.0, 0.3)
    ke_flat = ke.ravel()
    edof = build_edof(D, H, W)
    iK, jK = build_k_index(edof)
    ndof = 3 * (D + 1) * (H + 1) * (W + 1)

    N = min(args.n_samples, conds.shape[0])
    rows = []
    header = ["sample", "volfrac", "uncond_comp", "posthoc_comp", "refine_comp",
              "refine_vs_uncond_pct", "refine_vs_posthoc_pct"]

    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        vf = float(conds[i, 0, 0, 0, 0])
        loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            ph = manifold_projection(s, source_mask=src, support_mask=sup)
            rf = diff.refine(s, cond, src, sup, K=5, t_refine=30)
        uc = compliance(s[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        pc = compliance(ph[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        rc = compliance(rf[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        rows.append([i, vf, uc, pc, rc,
                     100 * (rc - uc) / (uc + 1e-12), 100 * (rc - pc) / (pc + 1e-12)])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}")

    # 过滤奇异值（柔度 >1e6 视为 FEM 奇异）
    valid = [r for r in rows if all(0 < r[j] < 1e6 for j in [2, 3, 4])]
    print(f"\n=== 柔度逐样本表（{len(valid)}/{len(rows)} 个有效样本） ===")
    print("| idx | vf | 无条件 | 采样后 | 精炼 | 精炼vs无条件% |")
    print("|-----|-----|--------|--------|------|-------------|")
    for r in valid:
        print(f"| {r[0]:3d} | {r[1]:.2f} | {r[2]:8.1f} | {r[3]:8.1f} | {r[4]:8.1f} | {r[5]:+8.1f}% |")

    if valid:
        uc_m = np.mean([r[2] for r in valid])
        pc_m = np.mean([r[3] for r in valid])
        rc_m = np.mean([r[4] for r in valid])
        print(f"\n=== 均值（过滤奇异后） ===")
        print(f"无条件: {uc_m:.1f} | 采样后: {pc_m:.1f} | 精炼: {rc_m:.1f} | 降幅: {(rc_m-uc_m)/uc_m*100:.1f}%")


if __name__ == "__main__":
    main()
