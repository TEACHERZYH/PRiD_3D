"""工况编码消融：训练无荷载/支座掩码的 2 通道条件模型。

对比 4 通道（体积分数 + von Mises + 荷载掩码 + 支座掩码）与
2 通道（仅体积分数 + von Mises，无显式工况位置）模型的连通性，
验证显式工况编码（论文 C3 贡献）的必要性。

用法:
    python code/train_ablation.py --n_cond 2 --epochs 100 --out results/checkpoint_16_nomask.pt
    python code/train_ablation.py --n_cond 4 --epochs 100 --out results/checkpoint_16_full.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from diffusion.diffusion import UNet3D, Diffusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--n_cond", type=int, default=2, help="条件通道数（2=无掩码，4=含掩码）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out", type=str, default="results/checkpoint_ablation.pt")
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    data = np.load(args.data)
    conds = data["conds"][:, :args.n_cond]  # 只取前 n_cond 通道
    labels = data["labels"]
    n = conds.shape[0]
    n_train = int(n * 0.9)
    idx = np.random.permutation(n)
    tr, va = idx[:n_train], idx[n_train:]

    tr_c = torch.from_numpy(conds[tr]).float()
    tr_l = torch.from_numpy(labels[tr]).float()
    va_c = torch.from_numpy(conds[va]).float()
    va_l = torch.from_numpy(labels[va]).float()

    print(f"[ablation] n_cond={args.n_cond}, train={n_train}, val={n-n_train}")

    model = UNet3D(in_channels=1 + args.n_cond, base=args.base, depth=args.depth)
    diff = Diffusion(model, timesteps=args.timesteps)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_batches = max(1, n_train // args.batch)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for b in range(n_batches):
            ii = perm[b * args.batch:(b + 1) * args.batch]
            loss = diff.train_loss(tr_l[ii], tr_c[ii])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / n_batches

        if (epoch + 1) % args.log_every == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    diff.train_loss(va_l[b * args.batch:(b + 1) * args.batch],
                                    va_c[b * args.batch:(b + 1) * args.batch]).item()
                    for b in range(max(1, va_c.shape[0] // args.batch))
                ) / max(1, va_c.shape[0] // args.batch)
            print(f"[ablation] epoch {epoch+1}/{args.epochs} train={avg:.4f} val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
                torch.save(model.state_dict(), args.out)

    print(f"[ablation] done. best val={best_val:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
