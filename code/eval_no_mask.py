"""工况编码消融评估：对比 4 通道（含荷载/支座掩码）与 2 通道（无掩码）模型。

验证显式工况编码（论文 C3 贡献）对连通性的必要性：
若去掉掩码通道后无条件采样连通性显著下降，则说明工况编码是关键设计。

用法:
    python code/eval_no_mask.py --n_samples 30
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.ndimage import label as cc

from diffusion.diffusion import UNet3D, Diffusion


def connectivity(rho: np.ndarray) -> float:
    bn = rho > 0.5
    if not bn.any():
        return 0.0
    ls, nc = cc(bn)
    sizes = np.bincount(ls.ravel())
    sizes[0] = 0
    mc = int(np.argmax(sizes))
    return (ls == mc).sum() / bn.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt_full", default="results/checkpoint_16.pt")
    ap.add_argument("--ckpt_nomask", default="results/checkpoint_16_nomask.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=14000)
    args = ap.parse_args()

    data = np.load(args.data)
    conds = data["conds"]
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]
    conds_full = torch.from_numpy(conds).float()
    conds_nomask = torch.from_numpy(conds[:, :2]).float()

    model_full = UNet3D(in_channels=5, base=32, depth=2)
    model_full.load_state_dict(torch.load(args.ckpt_full, map_location="cpu"))
    model_full.eval()
    diff_full = Diffusion(model_full, timesteps=200)

    model_nomask = UNet3D(in_channels=3, base=32, depth=2)
    model_nomask.load_state_dict(torch.load(args.ckpt_nomask, map_location="cpu"))
    model_nomask.eval()
    diff_nomask = Diffusion(model_nomask, timesteps=200)

    N = min(args.n_samples, conds.shape[0])
    full_conn, nomask_conn = [], []
    for i in range(N):
        cf = conds_full[i:i + 1]
        cn = conds_nomask[i:i + 1]
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            sf = diff_full.sample(cf, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            torch.manual_seed(args.seed + i)
            sn = diff_nomask.sample(cn, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
        full_conn.append(connectivity(sf[0, 0].numpy()))
        nomask_conn.append(connectivity(sn[0, 0].numpy()))

    full_m = float(np.mean(full_conn))
    nomask_m = float(np.mean(nomask_conn))
    d = np.array(full_conn) - np.array(nomask_conn)
    t = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-12)

    print("\n=== 工况编码消融（无条件采样连通性） ===")
    print(f"含掩码(4通道, 本文): 连通性={full_m:.4f}")
    print(f"无掩码(2通道, 对照): 连通性={nomask_m:.4f}")
    print(f"差异(含掩码-无掩码): {d.mean():+.4f}, t={t:.3f}, 正占比={(d>0).mean():.3f}")


if __name__ == "__main__":
    main()
