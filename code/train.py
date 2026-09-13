"""主训练脚本：流形保持扩散（Manifold-Preserving Diffusion）。

核心：在 DDPM 去噪链中插入 manifold_projection，实现采样空间约束。

数据来自预合成的 .npz 文件（由 synthesize_parallel.py 生成）。

用法（远程 GPU）:
    python code/train.py --data data/train_16.npz --epochs 50 --batch 8 --device cuda
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection


def load_data(path: str, split_ratio: float = 0.9):
    """加载 .npz 数据，返回 (train_conds, train_labels, val_conds, val_labels)。"""
    data = np.load(path)
    conds = data["conds"]  # (N, 2, D, H, W)
    labels = data["labels"]  # (N, 1, D, H, W)
    n = conds.shape[0]
    n_train = int(n * split_ratio)
    idx = np.random.permutation(n)
    tr, va = idx[:n_train], idx[n_train:]
    return (
        torch.from_numpy(conds[tr]),
        torch.from_numpy(labels[tr]),
        torch.from_numpy(conds[va]),
        torch.from_numpy(labels[va]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--out", type=str, default="results/checkpoint.pt")
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # 加载数据
    tr_c, tr_l, va_c, va_l = load_data(args.data)
    n_train = tr_c.shape[0]
    nx, ny, nz = tr_c.shape[2], tr_c.shape[3], tr_c.shape[4]
    print(f"[train] data: train {n_train}, val {va_c.shape[0]}, grid {nx}x{ny}x{nz}")

    tr_c, tr_l = tr_c.to(device), tr_l.to(device)
    va_c, va_l = va_c.to(device), va_l.to(device)

    # 模型（输入通道 = 1 噪声 + 4 条件 = 5）
    model = UNet3D(in_channels=5, base=args.base, depth=args.depth).to(device)
    diff = Diffusion(model, timesteps=args.timesteps)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_batches = max(1, n_train // args.batch)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch:(b + 1) * args.batch]
            x0 = tr_l[idx]
            cond = tr_c[idx]
            loss = diff.train_loss(x0, cond)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg = total_loss / n_batches

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            # 验证集损失
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for b in range(max(1, va_c.shape[0] // args.batch)):
                    idx = slice(b * args.batch, (b + 1) * args.batch)
                    val_loss += diff.train_loss(va_l[idx], va_c[idx]).item()
                val_avg = val_loss / max(1, va_c.shape[0] // args.batch)

                # 采样 + 投影，验证流形保持（seed 从条件第2/3通道提取）
                cond0 = va_c[:1]
                src = cond0[:, 2:3]   # 荷载掩码通道
                sup = cond0[:, 3:4]   # 支座掩码通道
                sample = diff.sample(cond0, (1, 1, nx, ny, nz), ddim_steps=20)
                proj = manifold_projection(sample, source_mask=src, support_mask=sup)
                print(
                    f"[train] epoch {epoch+1}/{args.epochs} "
                    f"train_loss={avg:.4f} val_loss={val_avg:.4f} "
                    f"sample_mean={sample.mean().item():.3f}->proj={proj.mean().item():.3f}"
                )

            if val_avg < best_val:
                best_val = val_avg
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                torch.save(model.state_dict(), args.out)

    print(f"[train] done. best val_loss = {best_val:.4f}, checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
