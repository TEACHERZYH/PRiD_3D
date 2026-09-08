"""评估脚本：加载 checkpoint，采样生成 3D 拓扑，验证流形保持效果。

验证命题 P1（合法性）的核心指标：
1. 连通性：从工况掩码（荷载+支座）seed 的主连通分量覆盖的材料占比；
2. 漂浮体比例：孤立于主体之外的材料占比（越低越好）；
3. 体积分数。

对比三组：SIMP 真值 / 无条件采样（无投影）/ 流形保持采样（有投影）。

工况掩码从条件通道读取：通道2=荷载掩码，通道3=支座掩码。

用法:
    python code/evaluate.py --data data/train_16.npz --ckpt results/checkpoint_16.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.ndimage import label as connected_components

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection


def compute_metrics(rho: np.ndarray, src: np.ndarray, sup: np.ndarray):
    """计算连通性、漂浮体比例、体积分数。"""
    binary = rho > 0.5
    total_material = binary.sum()
    volfrac = total_material / binary.size

    if total_material == 0:
        return {"volfrac": 0.0, "connectivity": 0.0, "floating_ratio": 1.0, "n_components": 0}

    labels, n_comp = connected_components(binary)
    seed = (src > 0.5) | (sup > 0.5)
    main_comps = set(labels[seed & binary].tolist())
    if not main_comps:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        main_comps = {int(np.argmax(sizes))}
    main_material = sum((labels == c).sum() for c in main_comps if c > 0)

    connectivity = main_material / total_material
    floating_ratio = 1.0 - connectivity
    return {
        "volfrac": float(volfrac),
        "connectivity": float(connectivity),
        "floating_ratio": float(floating_ratio),
        "n_components": int(n_comp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_16.npz")
    parser.add_argument("--ckpt", type=str, default="results/checkpoint_16.pt")
    parser.add_argument("--out", type=str, default="results/eval")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    data = np.load(args.data)
    conds = data["conds"]  # (N,4,D,H,W)
    labels = data["labels"]
    n = conds.shape[0]
    n_val = max(1, n // 10)
    val_conds = conds[:n_val]
    val_labels = labels[:n_val]
    D, H, W = val_conds.shape[2], val_conds.shape[3], val_conds.shape[4]
    print(f"[eval] val 样本 {n_val}, 网格 {D}x{H}x{W}")

    model = UNet3D(in_channels=5, base=args.base, depth=args.depth)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=args.timesteps)

    gt_metrics, uncond_metrics, posthoc_metrics, manifold_metrics = [], [], [], []

    for i in range(min(n_val, args.n_samples)):
        cond = torch.from_numpy(val_conds[i:i + 1])
        gt = val_labels[i, 0]
        src = val_conds[i, 2]
        sup = val_conds[i, 3]

        with torch.no_grad():
            # 无条件采样（全程无投影）
            uncond = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            uncond_np = uncond[0, 0].numpy()
            # 采样后投影（事后修正，post-hoc）
            posthoc = manifold_projection(
                uncond,
                source_mask=torch.from_numpy(src[None, None]),
                support_mask=torch.from_numpy(sup[None, None]),
            )
            posthoc_np = posthoc[0, 0].numpy()
            # 采样中投影（流形保持，本文方法）
            manifold = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=True)
            manifold_np = manifold[0, 0].numpy()

        gt_metrics.append(compute_metrics(gt, src, sup))
        uncond_metrics.append(compute_metrics(uncond_np, src, sup))
        posthoc_metrics.append(compute_metrics(posthoc_np, src, sup))
        manifold_metrics.append(compute_metrics(manifold_np, src, sup))

        if i == 0:
            np.savez_compressed(
                os.path.join(args.out, "sample_viz.npz"),
                gt=gt, uncond=uncond_np, posthoc=posthoc_np, manifold=manifold_np, cond=val_conds[i],
            )

    def mean(ms):
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    print("\n=== 评估结果（均值） ===")
    print(f"{'指标':<14} {'SIMP真值':<10} {'无条件':<10} {'采样后投影':<10} {'流形保持(本文)':<12}")
    for k in ["volfrac", "connectivity", "floating_ratio", "n_components"]:
        g = mean(gt_metrics)[k]
        u = mean(uncond_metrics)[k]
        ph = mean(posthoc_metrics)[k]
        m = mean(manifold_metrics)[k]
        print(f"{k:<14} {g:<10.4f} {u:<10.4f} {ph:<10.4f} {m:<12.4f}")

    print("\n关键结论（核心主张验证）：")
    print(f"  漂浮体比例: 无条件 {mean(uncond_metrics)['floating_ratio']:.4f} -> 采样后投影 {mean(posthoc_metrics)['floating_ratio']:.4f} -> 流形保持 {mean(manifold_metrics)['floating_ratio']:.4f}")
    print(f"  连通性:     无条件 {mean(uncond_metrics)['connectivity']:.4f} -> 采样后投影 {mean(posthoc_metrics)['connectivity']:.4f} -> 流形保持 {mean(manifold_metrics)['connectivity']:.4f}")


if __name__ == "__main__":
    main()
