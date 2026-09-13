"""多 seed 重训诊断：小数据集扩散模型训练的"采样质量方差"。

背景：同一配置、同一数据（clean 标签）、120 epochs，新训模型 val_loss 0.0099
    与原模型 0.0094 接近，但采样连通性仅 0.7533（原模型 0.9511）。
本脚本用多个 seed 重训 clean 模型，每个训练后立即做采样诊断，判断：
    (a) 只是特定 seed 运气差 → 换 seed 即可；
    (b) 训练本身高方差 → 这是一个需要报告的稳定性问题（也解释单次结果不可复现）。

用法:
    python code/train_multiseed_clean.py --seeds 1,2,3,4,5 --epochs 120 --out_dir results/ms_clean/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from diffusion.diffusion import UNet3D, Diffusion
from eval_bridge_recon_factorial import connectivity_6
from train_clean_vs_dirty import train_model


def diagnose(state, conds, shape, n=10, seed=12000, device="cpu"):
    """用内存中的 state_dict 做采样诊断（不读磁盘，避免文件系统延迟问题）。"""
    model = UNet3D(in_channels=5, base=32, depth=2).to(device)
    model.load_state_dict(state)
    model.eval()
    diff = Diffusion(model, timesteps=200).to(device)
    cs, occs = [], []
    for i in range(n):
        torch.manual_seed(seed + i)
        with torch.no_grad():
            s = diff.sample(conds[i:i + 1].to(device), shape, ddim_steps=20)
        arr = s[0, 0].cpu().numpy()
        cs.append(connectivity_6(arr)[0])
        occs.append(int((arr > 0.5).sum()))
    return float(np.mean(cs)), float(np.std(cs)), float(np.mean(occs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="results/ms_clean")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.data)
    conds_np, labels = d["conds"], d["labels"]
    conds_t = torch.from_numpy(conds_np).float()
    D, H, W = conds_np.shape[2], conds_np.shape[3], conds_np.shape[4]
    shape = (1, 1, D, H, W)
    os.makedirs(args.out_dir, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    results = []
    for seed in seeds:
        path = os.path.join(args.out_dir, f"clean_seed{seed}.pt")
        best, state = train_model(conds_np, labels, seed, args.epochs, args.batch,
                                  args.lr, 200, path, f"seed{seed}")
        mc, sc, occ = diagnose(state, conds_t, shape, device=device)
        results.append((seed, best, mc, sc, occ))
        print(f"[SUMMARY] seed={seed} val={best:.4f} 采样连通性={mc:.4f}±{sc:.4f} 占用={occ:.1f}",
              flush=True)

    print("\n" + "=" * 70)
    print("多 seed clean 训练的采样质量分布")
    print("=" * 70)
    print(f"{'seed':>6}{'val_loss':>12}{'采样连通性':>14}{'±sd':>10}{'占用体素':>10}")
    vals = np.array([r[2] for r in results])
    for s, b, mc, sc, occ in results:
        print(f"{s:>6}{b:>12.4f}{mc:>14.4f}{sc:>10.4f}{occ:>10.1f}")
    print(f"\n采样连通性：均值 {vals.mean():.4f}  范围 [{vals.min():.4f}, {vals.max():.4f}]  "
          f"极差 {vals.max()-vals.min():.4f}")
    print("参考：原 checkpoint_16.pt 采样连通性 0.9511")
    print(f"\n结论判据：若极差 > 0.10 → 训练高方差（需报告）；否则属正常随机性。")


if __name__ == "__main__":
    main()
