"""最终综合评估：投影+精炼（Projection Refinement）完整验证。

严格配对实验（固定 seed，相同初始噪声）+ 统计检验。

对比三组：无条件 / 采样后投影（事后修正）/ 投影+精炼（本文方法）。
覆盖 ID（训练分布内）+ OOD（3 类分布外）。

用法:
    python code/evaluate_final.py --data data/train_16.npz --ood data/ood_16.npz \
        --ckpt results/checkpoint_16.pt --out results/final
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.ndimage import label as cc

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection


def connectivity(rho):
    """最大连通分量占比（连通性指标）。"""
    bn = rho > 0.5
    if not bn.any():
        return 0.0
    ls, _ = cc(bn)
    sizes = np.bincount(ls.ravel())
    sizes[0] = 0
    m = int(np.argmax(sizes))
    return float((ls == m).sum() / bn.sum())


def evaluate_set(conds, diff, D, H, W, seed_base, K=5, t_refine=30):
    """配对评估一组条件，返回三组连通性列表。"""
    uncond, posthoc, refine = [], [], []
    for i in range(len(conds)):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        torch.manual_seed(seed_base + i)
        with torch.no_grad():
            s = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            ph = manifold_projection(s, source_mask=src, support_mask=sup)
            rf = diff.refine(s, cond, src, sup, K=K, t_refine=t_refine)
        uncond.append(connectivity(s[0, 0].numpy()))
        posthoc.append(connectivity(ph[0, 0].numpy()))
        refine.append(connectivity(rf[0, 0].numpy()))
    return uncond, posthoc, refine


def report(name, uncond, posthoc, refine):
    u, p, r = np.mean(uncond), np.mean(posthoc), np.mean(refine)
    d = np.array(refine) - np.array(posthoc)
    t = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-8)
    print(f"{name:<12} 无条件={u:.4f} 采样后={p:.4f} 投影精炼={r:.4f} "
          f"差异={d.mean():+.4f} t={t:.2f} 正占比={(d>0).mean():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train_16.npz")
    parser.add_argument("--ood", type=str, default="data/ood_16.npz")
    parser.add_argument("--ckpt", type=str, default="results/checkpoint_16.pt")
    parser.add_argument("--out", type=str, default="results/final")
    parser.add_argument("--n_samples", type=int, default=60)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--t_refine", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    D = H = W = 16

    print("=" * 70)
    print("投影+精炼（Projection Refinement）最终评估（配对实验）")
    print("=" * 70)

    # ID 数据
    data = np.load(args.data)
    conds = torch.from_numpy(data["conds"][:args.n_samples]).float()
    u, p, r = evaluate_set(conds, diff, D, H, W, seed_base=10000, K=args.K, t_refine=args.t_refine)
    report("ID(训练内)", u, p, r)

    # OOD 数据（按类型）
    ood = np.load(args.ood)
    o_conds = torch.from_numpy(ood["conds"]).float()
    o_kinds = ood["kinds"]
    for kind in ["volfrac", "domain", "load"]:
        idxs = [i for i, k in enumerate(o_kinds) if k == kind]
        sub = o_conds[idxs]
        u, p, r = evaluate_set(sub, diff, D, H, W, seed_base=20000, K=args.K, t_refine=args.t_refine)
        report(f"OOD-{kind}", u, p, r)

    print("=" * 70)


if __name__ == "__main__":
    main()
