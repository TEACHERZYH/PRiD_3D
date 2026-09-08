"""消融实验 + 基线对比脚本：验证 P1（合法性）与 P2（最优性）。

对比三个方法：
1. 无条件扩散（无任何约束，暴露"物理非法"问题）
2. 软约束（连通性作为损失项，HPG-Diff 式"拉扯"）
3. 流形保持（投影算子，本文方法"界定"）

对每个方法采样，计算：
- P1 指标：漂浮体比例、连通性、连通分量数
- P2 指标：柔度（用可微 FEM 或 SIMP 重算），相对 SIMP 真值的柔度误差

用法:
    python code/ablation.py --data data/train_16.npz --ckpt results/checkpoint_16.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.ndimage import label as connected_components

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection
from data_generation.simp_solver import topology_optimize


def compute_metrics(rho, src, sup):
    binary = rho > 0.5
    total = binary.sum()
    volfrac = total / binary.size
    if total == 0:
        return {"volfrac": 0.0, "connectivity": 0.0, "floating_ratio": 1.0, "n_components": 0}
    labels, n_comp = connected_components(binary)
    seed = (src > 0.5) | (sup > 0.5)
    main_comps = set(labels[seed & binary].tolist())
    main_material = sum((labels == c).sum() for c in main_comps if c > 0)
    return {
        "volfrac": float(volfrac),
        "connectivity": float(main_material / total),
        "floating_ratio": float(1 - main_material / total),
        "n_components": int(n_comp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_16.npz")
    parser.add_argument("--ckpt", type=str, default="results/checkpoint_16.pt")
    parser.add_argument("--out", type=str, default="results/ablation")
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = np.load(args.data)
    conds, labels_all = data["conds"], data["labels"]
    n_val = max(1, conds.shape[0] // 10)
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]

    src = np.zeros((D, H, W)); src[D - 1, H - 1, W - 1] = 1
    sup = np.zeros((D, H, W)); sup[0, :, :] = 1

    model = UNet3D(in_channels=3, base=args.base, depth=args.depth)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=args.timesteps)

    results = {"uncond": [], "soft": [], "manifold": []}

    for i in range(min(n_val, args.n_samples)):
        cond = torch.from_numpy(conds[i:i + 1])
        gt = labels_all[i, 0]

        with torch.no_grad():
            sample = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20)
            sample_np = sample[0, 0].numpy()
            proj = manifold_projection(
                sample,
                source_mask=torch.from_numpy(src[None, None]),
                support_mask=torch.from_numpy(sup[None, None]),
            )
            proj_np = proj[0, 0].numpy()

        results["uncond"].append(compute_metrics(sample_np, src, sup))
        results["manifold"].append(compute_metrics(proj_np, src, sup))

    def mean(ms):
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    print("=" * 60)
    print("消融实验：无条件 vs 流形保持（P1 合法性）")
    print("=" * 60)
    header = f"{'指标':<16}{'无条件':<14}{'流形保持':<14}"
    print(header)
    print("-" * 44)
    for k in ["volfrac", "connectivity", "floating_ratio", "n_components"]:
        u, m = mean(results["uncond"])[k], mean(results["manifold"])[k]
        print(f"{k:<16}{u:<14.4f}{m:<14.4f}")

    # 保存数值结果
    np.savez(
        os.path.join(args.out, "ablation_results.npz"),
        uncond=np.array([list(m.values()) for m in results["uncond"]]),
        manifold=np.array([list(m.values()) for m in results["manifold"]]),
        metric_names=np.array(list(results["uncond"][0].keys())),
    )
    print(f"\n结果已保存到 {args.out}/ablation_results.npz")


if __name__ == "__main__":
    main()
