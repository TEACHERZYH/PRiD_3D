"""验证柔度感知精炼：每轮去噪后比较柔度，保留更低者，保证柔度单调不增。

对比原精炼（diff.refine）与柔度感知精炼在样本 2、17 及整体上的柔度。

用法:
    python code/verify_fix.py --n_samples 20
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import label as cc

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
    if cond_np[3].any():
        for z in range(W + 1):
            for y in range(H + 1):
                nid = (D + 1) * (H + 1) * z + (D + 1) * y
                supports[3 * nid:3 * nid + 3] = True
    lp = np.argwhere(cond_np[2] > 0.5)
    if len(lp) > 0:
        x, y, z = lp[0]
        nid = x + (D + 1) * y + (D + 1) * (H + 1) * W
        loads[3 * nid + 2] = -1.0
    return loads, supports


def compliance(rho, loads, supports, ke_flat, iK, jK, ndof):
    E = 1e-9 + (1 - 1e-9) * rho.ravel() ** 3
    vals = (ke_flat[None, :] * E[:, None]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = ~supports
    Uf = spsolve(K[free][:, free], loads[free])
    U = np.zeros(ndof)
    U[free] = Uf
    return float(loads @ U)


def connectivity(rho):
    bn = rho > 0.5
    if not bn.any():
        return 0.0
    ls, nc = cc(bn)
    sizes = np.bincount(ls.ravel())
    sizes[0] = 0
    mc = int(np.argmax(sizes))
    return (ls == mc).sum() / bn.sum()


def refine_compliance_aware(diff, x0, cond, src, sup, K, t_ref, loads, supports,
                            ke_flat, iK, jK, ndof):
    """柔度感知精炼：每轮选择「投影后」与「去噪后」中柔度更低者。"""
    for _ in range(K):
        x_proj = manifold_projection(x0, source_mask=src, support_mask=sup)
        t = torch.tensor([t_ref], device=x0.device)
        x_t, _ = diff.q_sample(x_proj, t)
        x_in = torch.cat([x_t, cond], dim=1)
        eps = diff.model(x_in, t)
        a = diff.sqrt_alphas_cumprod[t_ref]
        b = diff.sqrt_one_minus[t_ref]
        x_den = torch.clamp((x_t - b * eps) / a, 0.0, 1.0)

        cp = compliance(x_proj[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        cd = compliance(x_den[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        x0 = x_den if cd <= cp else x_proj
    return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12000)
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
    print(f"\n{'idx':>4} {'vf':>5} {'无条件':>10} {'原精炼':>10} {'感知精炼':>10} {'原-无条件%':>12} {'感知-无条件%':>12}")
    rows = []
    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        vf = float(conds[i, 0, 0, 0, 0])
        loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            r_orig = diff.refine(s, cond, src, sup, K=5, t_refine=30)
            r_aware = refine_compliance_aware(diff, s, cond, src, sup, 5, 30, loads,
                                              supports, ke_flat, iK, jK, ndof)
        cu = compliance(s[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        co = compliance(r_orig[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        ca = compliance(r_aware[0, 0].numpy(), loads, supports, ke_flat, iK, jK, ndof)
        conn_u = connectivity(s[0, 0].numpy())
        conn_o = connectivity(r_orig[0, 0].numpy())
        conn_a = connectivity(r_aware[0, 0].numpy())
        valid = all(0 < c < 1e6 for c in [cu, co, ca])
        tag = "" if valid else " (奇异)"
        denom = cu if cu > 1e-9 else 1.0
        print(f"{i:4d} {vf:5.2f} {cu:10.1f} {co:10.1f} {ca:10.1f} "
              f"{(co-cu)/denom*100:12.1f} {(ca-cu)/denom*100:12.1f}{tag}")
        rows.append((i, vf, cu, co, ca, conn_u, conn_o, conn_a, valid))

    # 统计有效样本
    valid_rows = [r for r in rows if r[8]]
    print(f"\n=== 有效样本统计（{len(valid_rows)}/{len(rows)}） ===")
    if valid_rows:
        cu_m = np.median([r[2] for r in valid_rows])
        co_m = np.median([r[3] for r in valid_rows])
        ca_m = np.median([r[4] for r in valid_rows])
        conn_o_m = np.mean([r[6] for r in valid_rows])
        conn_a_m = np.mean([r[7] for r in valid_rows])
        # 恶化样本数
        worse_orig = sum(1 for r in valid_rows if r[3] > r[2])
        worse_aware = sum(1 for r in valid_rows if r[4] > r[2])
        print(f"柔度中位数: 无条件={cu_m:.1f} | 原精炼={co_m:.1f} | 感知精炼={ca_m:.1f}")
        print(f"连通性均值: 原精炼={conn_o_m:.4f} | 感知精炼={conn_a_m:.4f}")
        print(f"恶化样本数: 原精炼={worse_orig} | 感知精炼={worse_aware}")


if __name__ == "__main__":
    main()
