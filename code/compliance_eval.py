"""柔度评估（P2 最优性）：对比流形保持采样 vs SIMP 真值的柔度。

命题 P2：合法性与最优性在合法流形上不冲突——流形保持投影在满足
连通性约束的同时，柔度不显著劣于无约束采样。

方法：从条件通道提取每个样本的实际工况（荷载+支座），用 SIMP 求解器
对采样结果重算柔度，对比真值柔度。

用法:
    python code/compliance_eval.py --data data/train_16.npz --ckpt results/checkpoint_16.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index


def build_loads_supports_from_cond(cond, nx: int, ny: int, nz: int):
    """从条件的工况掩码通道构建 loads 和 supports 数组。

    坐标约定（与 synthesize_parallel.py 一致）：数组维度 (x, y, z)，
    节点编号 = x + (nx+1)*y + (nx+1)*(ny+1)*z。
    cond[2] = 荷载掩码，cond[3] = 支座掩码（x=0 面）。
    """
    ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    loads = np.zeros(ndof)
    supports = np.zeros(ndof, dtype=bool)

    # 支座：x=0 面节点全固定（复刻数据生成循环）
    if cond[3].any():
        for z in range(nz + 1):
            for y in range(ny + 1):
                nid = (nx + 1) * (ny + 1) * z + (nx + 1) * y  # 节点 (x=0, y, z)
                supports[3 * nid:3 * nid + 3] = True

    # 荷载：顶面单元 → 顶面节点 (x, y, nz)，-z 方向集中力
    load_positions = np.argwhere(cond[2] > 0.5)
    if len(load_positions) > 0:
        x, y, z = load_positions[0]  # 单元坐标 (x, y, z)
        nid = x + (nx + 1) * y + (nx + 1) * (ny + 1) * nz  # 节点 (x, y, nz)
        loads[3 * nid + 2] = -1.0

    return loads, supports


def compute_compliance(rho, nx, ny, nz, loads, supports):
    """用稀疏 FEM 重算密度场 rho 的柔度。"""
    n_elem = nx * ny * nz
    ndof = loads.shape[0]
    ke = hex8_stiffness(1.0, 0.3)
    ke_flat = ke.ravel()
    edof = build_edof(nx, ny, nz)
    iK, jK = build_k_index(edof)
    E = 1e-9 + (1 - 1e-9) * rho.ravel() ** 3
    vals = (ke_flat[None, :] * E[:, None]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = ~supports
    Uf = spsolve(K[free][:, free], loads[free])
    U = np.zeros(ndof)
    U[free] = Uf
    return float(loads @ U)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_16.npz")
    parser.add_argument("--ckpt", type=str, default="results/checkpoint_16.pt")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=200)
    args = parser.parse_args()

    data = np.load(args.data)
    conds, labels_all = data["conds"], data["labels"]
    n_val = max(1, conds.shape[0] // 10)
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]
    nx = ny = nz = D

    model = UNet3D(in_channels=5, base=args.base, depth=args.depth)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=args.timesteps)

    gt_c, uncond_c, manifold_c = [], [], []

    for i in range(min(n_val, args.n_samples)):
        cond = torch.from_numpy(conds[i:i + 1])
        gt = labels_all[i, 0]
        src = conds[i, 2]
        sup = conds[i, 3]

        # 从条件提取真实工况
        loads, supports = build_loads_supports_from_cond(conds[i], nx, ny, nz)

        with torch.no_grad():
            sample = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20)
            sample_np = sample[0, 0].numpy()
            proj = manifold_projection(
                sample,
                source_mask=torch.from_numpy(src[None, None]),
                support_mask=torch.from_numpy(sup[None, None]),
            )
            proj_np = proj[0, 0].numpy()

        try:
            c_gt = compute_compliance(gt, nx, ny, nz, loads, supports)
            c_uncond = compute_compliance(sample_np, nx, ny, nz, loads, supports)
            c_proj = compute_compliance(proj_np, nx, ny, nz, loads, supports)
            gt_c.append(c_gt)
            uncond_c.append(c_uncond)
            manifold_c.append(c_proj)
            print(f"样本 {i+1}: 真值={c_gt:.2f}, 无条件={c_uncond:.2f}, 流形保持={c_proj:.2f}")
        except Exception as e:
            print(f"样本 {i+1} 求解失败: {e}")

    print("\n=== 柔度对比（P2 最优性） ===")
    print(f"SIMP 真值:    均值 {np.mean(gt_c):.2f}")
    print(f"无条件采样:    均值 {np.mean(uncond_c):.2f}")
    print(f"流形保持采样:  均值 {np.mean(manifold_c):.2f}")

    err_uncond = np.mean([abs(u - g) / g for u, g in zip(uncond_c, gt_c) if g > 0])
    err_manifold = np.mean([abs(m - g) / g for m, g in zip(manifold_c, gt_c) if g > 0])
    print(f"\n相对柔度误差: 无条件 {err_uncond:.4f}, 流形保持 {err_manifold:.4f}")


if __name__ == "__main__":
    main()