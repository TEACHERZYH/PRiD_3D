"""消融实验：桥接 vs 删除 + 投影精炼收敛性分析。

1. 桥接 vs 删除：投影算子把漂浮体"桥接回主体"还是"删除"？
   —— 支撑论文 C1 贡献（桥接保留材料与承载潜力，优于删除）。
2. 收敛性：连通性随精炼轮数 K 的收敛曲线（K=0~5）。
   —— 印证"投影梯度下降"的理论定位。

用法（远程 CPU）:
    python code/ablation_analysis.py --n_samples 30
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.ndimage import label as cc

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection, _label_components


def delete_floating(x: torch.Tensor) -> torch.Tensor:
    """删除漂浮体（对照）：保留最大连通分量，其余材料置 0。x: (D,H,W)。"""
    binary = x > 0.5
    labels = _label_components(binary)
    flat = labels.ravel()
    nz = flat[flat > 0]
    if nz.numel() == 0:
        return x.clone()
    sizes = torch.bincount(nz)
    main_comp = int(torch.argmax(sizes[1:]) + 1)
    main = labels == main_comp
    out = x.clone()
    out[~main] = 0.0
    return out


def metrics(rho: np.ndarray):
    """连通性（最大分量占比）、漂浮体比例、连通分量数。"""
    bn = rho > 0.5
    if not bn.any():
        return 0.0, 1.0, 0
    ls, nc = cc(bn)
    sizes = np.bincount(ls.ravel())
    sizes[0] = 0
    mc = int(np.argmax(sizes))
    conn = (ls == mc).sum() / bn.sum()
    return conn, 1.0 - conn, nc


def refine_loop(diff, x0, cond, src, sup, K, t_refine, mode):
    """投影 + 精炼循环。mode='bridge' 桥接，'delete' 删除。"""
    for _ in range(K):
        if mode == "bridge":
            x0 = manifold_projection(x0, source_mask=src, support_mask=sup)
        else:
            outs = [delete_floating(x0[b, 0]).unsqueeze(0).unsqueeze(0) for b in range(x0.shape[0])]
            x0 = torch.cat(outs, dim=0)
        t = torch.tensor([t_refine], device=x0.device)
        x_t, _ = diff.q_sample(x0, t)
        x_in = torch.cat([x_t, cond], dim=1)
        eps = diff.model(x_in, t)
        a = diff.sqrt_alphas_cumprod[t_refine]
        b = diff.sqrt_one_minus[t_refine]
        x0 = (x_t - b * eps) / a
        x0 = torch.clamp(x0, 0.0, 1.0)
    return x0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13000)
    args = ap.parse_args()

    data = np.load(args.data)
    conds = torch.from_numpy(data["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]

    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    N = min(args.n_samples, conds.shape[0])
    uncond_conn, bridge_conn, delete_conn = [], [], []
    bridge_float, delete_float = [], []
    bridge_nc, delete_nc = [], []
    k_conn = {k: [] for k in [0, 1, 2, 3, 5]}

    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            sn = s[0, 0].numpy()

            c0, _, _ = metrics(sn)
            uncond_conn.append(c0)
            k_conn[0].append(c0)

            # 桥接精炼（K=5）
            br = refine_loop(diff, s, cond, src, sup, K=5, t_refine=30, mode="bridge")
            c, f, nc = metrics(br[0, 0].numpy())
            bridge_conn.append(c); bridge_float.append(f); bridge_nc.append(nc)
            k_conn[5].append(c)

            # 删除精炼（K=5）
            de = refine_loop(diff, s, cond, src, sup, K=5, t_refine=30, mode="delete")
            c, f, nc = metrics(de[0, 0].numpy())
            delete_conn.append(c); delete_float.append(f); delete_nc.append(nc)

            # K 收敛曲线（桥接，K=1,2,3）
            for k in [1, 2, 3]:
                brk = refine_loop(diff, s, cond, src, sup, K=k, t_refine=30, mode="bridge")
                c, _, _ = metrics(brk[0, 0].numpy())
                k_conn[k].append(c)

    def mean(xs):
        return float(np.mean(xs))

    print("\n=== 桥接 vs 删除（K=5，配对） ===")
    print(f"{'方法':<12} {'连通性':<10} {'漂浮体':<10} {'分量数':<10}")
    print(f"{'无条件':<12} {mean(uncond_conn):<10.4f} {1-mean(uncond_conn):<10.4f} {'-':<10}")
    print(f"{'桥接(本文)':<12} {mean(bridge_conn):<10.4f} {mean(bridge_float):<10.4f} {mean(bridge_nc):<10.2f}")
    print(f"{'删除(对照)':<12} {mean(delete_conn):<10.4f} {mean(delete_float):<10.4f} {mean(delete_nc):<10.2f}")

    d = np.array(bridge_conn) - np.array(delete_conn)
    t = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)
    print(f"\n桥接 vs 删除 连通性差异: 均值={d.mean():+.4f}, 正占比={(d>0).mean():.3f}, t={t:.3f}")

    print("\n=== 收敛性（桥接精炼，K=0~5） ===")
    for k in [0, 1, 2, 3, 5]:
        print(f"K={k}: 连通性={mean(k_conn[k]):.4f}")


if __name__ == "__main__":
    main()
