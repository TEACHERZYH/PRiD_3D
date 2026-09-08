"""诊断：逐步跟踪样本 2、17 的精炼过程，定位柔度恶化的环节。

对每个目标样本，逐轮记录（投影前/投影后/去噪后）的柔度、连通性、材料量，
判断柔度恶化是由「桥接」还是「去噪」导致。

用法:
    python code/diagnose_degradation.py --idx 2 17
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


def material(rho):
    return int((rho > 0.5).sum())


def denoise_step(diff, x0, cond, t_ref=30):
    t = torch.tensor([t_ref], device=x0.device)
    x_t, _ = diff.q_sample(x0, t)
    x_in = torch.cat([x_t, cond], dim=1)
    eps = diff.model(x_in, t)
    a = diff.sqrt_alphas_cumprod[t_ref]
    b = diff.sqrt_one_minus[t_ref]
    x0 = (x_t - b * eps) / a
    return torch.clamp(x0, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--idx", type=int, nargs="+", default=[2, 17])
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--K", type=int, default=5)
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

    for idx in args.idx:
        cond = conds[idx:idx + 1]
        src = conds[idx, 2:3]
        sup = conds[idx, 3:4]
        vf = float(conds[idx, 0, 0, 0, 0])
        loads, supports = build_ls(conds[idx].numpy(), D, H, W, ndof)

        torch.manual_seed(args.seed + idx)
        with torch.no_grad():
            x0 = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)

        print(f"\n========== 样本 idx={idx} (vf={vf:.3f}) ==========")
        rho = x0[0, 0].numpy()
        print(f"[K=0 无条件] 柔度={compliance(rho, loads, supports, ke_flat, iK, jK, ndof):.1f} "
              f"连通性={connectivity(rho):.4f} 材料={material(rho)}")

        for k in range(args.K):
            with torch.no_grad():
                # 投影步
                x0 = manifold_projection(x0, source_mask=src, support_mask=sup)
                rho_p = x0[0, 0].numpy()
                # 去噪步
                x0 = denoise_step(diff, x0, cond)
                rho_d = x0[0, 0].numpy()

            cp = compliance(rho_p, loads, supports, ke_flat, iK, jK, ndof)
            cd = compliance(rho_d, loads, supports, ke_flat, iK, jK, ndof)
            print(f"[K={k+1} 投影后] 柔度={cp:.1f} 连通性={connectivity(rho_p):.4f} 材料={material(rho_p)}")
            print(f"[K={k+1} 去噪后] 柔度={cd:.1f} 连通性={connectivity(rho_d):.4f} 材料={material(rho_d)}")


if __name__ == "__main__":
    main()
