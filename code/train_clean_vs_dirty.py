"""对照训练：训练数据连通性是否为"可行性恢复上限"的因果决定因素。

设计（唯一变量 = 训练数据的连通性）：
  * 模型-C（clean）：原始 SIMP 标签训练
  * 模型-D（dirty）：在标签中注入孤立材料块（浮材），使训练分布本身不连通
  两者使用 **相同的数据划分、相同的 seed、相同的超参**，差异只在标签。

预期：若"恢复能力来自训练分布"，则模型-D 的低噪重构应显著弱于模型-C。

用法（远程 GPU）:
    python code/train_clean_vs_dirty.py --data data/train_16.npz \
        --epochs 50 --batch 8 --float_ratio 0.15 --out_dir results/
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.ndimage import label as cc, generate_binary_structure, distance_transform_edt

from diffusion.diffusion import UNet3D, Diffusion


def conn_stats(binary):
    total = int(binary.sum())
    if total == 0:
        return 0.0, 0
    lab, n = cc(binary, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    return int(sizes.max()) / total, n


def inject_floaters(labels, ratio=0.15, block=2, seed=0):
    """在标签主体之外注入孤立材料块，注入量为自身材料量的 ratio。"""
    rng = np.random.default_rng(seed)
    out = labels.copy()
    N, _, D, H, W = labels.shape
    added = []
    for i in range(N):
        m = labels[i, 0] > 0.5
        occ = int(m.sum())
        if occ == 0:
            added.append(0)
            continue
        lab, n = cc(m, structure=generate_binary_structure(3, 1))
        sizes = np.bincount(lab.ravel())[1:]
        main = lab == (int(np.argmax(sizes)) + 1)
        dt = distance_transform_edt(~main)
        cand = np.argwhere(dt > block)
        if len(cand) == 0:
            added.append(0)
            continue
        n_target = int(round(ratio * occ))
        n_blocks = max(1, n_target // (block ** 3))
        idx = rng.choice(len(cand), size=min(n_blocks, len(cand)), replace=False)
        for j in idx:
            cx, cy, cz = cand[j]
            x0 = int(np.clip(cx - block // 2, 0, D - block))
            y0 = int(np.clip(cy - block // 2, 0, H - block))
            z0 = int(np.clip(cz - block // 2, 0, W - block))
            out[i, 0, x0:x0 + block, y0:y0 + block, z0:z0 + block] = 1.0
        added.append(int(out[i, 0].sum() - occ))
    return out, added


def dataset_conn(labels):
    c = [conn_stats(labels[i, 0] > 0.5)[0] for i in range(len(labels))]
    c = np.array(c)
    return c.mean(), int((c > 0.999).sum()), len(c)


def train_model(conds, labels, seed, epochs, batch, lr, timesteps, out_path, tag,
                device=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[{tag}] device={device}", flush=True)
    n = len(conds)
    n_tr = int(n * 0.9)
    perm = np.random.permutation(n)
    tr, va = perm[:n_tr], perm[n_tr:]

    tr_c = torch.from_numpy(conds[tr]).float().to(device)
    tr_l = torch.from_numpy(labels[tr]).float().to(device)
    va_c = torch.from_numpy(conds[va]).float().to(device)
    va_l = torch.from_numpy(labels[va]).float().to(device)

    model = UNet3D(in_channels=5, base=32, depth=2).to(device)
    diff = Diffusion(model, timesteps=timesteps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = max(1, n_tr // batch)
    best = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        p = torch.randperm(n_tr, device=device)
        tot = 0.0
        for b in range(n_batches):
            idx = p[b * batch:(b + 1) * batch]
            loss = diff.train_loss(tr_l[idx], tr_c[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        avg = tot / n_batches
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = 0.0
                nb = max(1, va_c.shape[0] // batch)
                for b in range(nb):
                    s = slice(b * batch, (b + 1) * batch)
                    vl += diff.train_loss(va_l[s], va_c[s]).item()
                vl /= nb
                smp = diff.sample(va_c[:1], (1, 1) + tuple(conds.shape[2:]), ddim_steps=20)
            print(f"[{tag}] epoch {ep+1}/{epochs} train={avg:.4f} val={vl:.4f} "
                  f"sample_mean={smp.mean().item():.3f}", flush=True)
            if vl < best:
                best = vl
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                tmp = out_path + ".tmp"
                with open(tmp, "wb") as fh:
                    torch.save(best_state, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, out_path)
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    print(f"[{tag}] done, best val={best:.4f} -> {out_path}", flush=True)
    return best, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--timesteps", type=int, default=200)
    ap.add_argument("--float_ratio", type=float, default=0.15,
                    help="注入浮材量占自身材料量的比例")
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--device", default="auto", help="auto / cpu / cuda")
    args = ap.parse_args()
    dev = None if args.device == "auto" else args.device

    print(f"[setup] device={'cuda' if torch.cuda.is_available() else 'cpu'}"
          f" cuda_avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[setup] gpu = {torch.cuda.get_device_name(0)}")

    d = np.load(args.data)
    conds, labels = d["conds"], d["labels"]

    mc, n1, N = dataset_conn(labels)
    print(f"\n[data] clean 标签：连通性 {mc:.4f}，完全连通 {n1}/{N}")

    dirty, added = inject_floaters(labels, ratio=args.float_ratio,
                                   block=args.block, seed=args.seed)
    md, n2, _ = dataset_conn(dirty)
    print(f"[data] dirty 标签：连通性 {md:.4f}，完全连通 {n2}/{N}"
          f"（每样本注入 {np.mean(added):.0f} 体素，占材料 {100*args.float_ratio:.0f}%）")
    np.savez_compressed(os.path.join(args.out_dir, "train_16_dirty.npz"),
                        conds=conds, labels=dirty)
    print(f"[data] dirty 数据集已保存：{args.out_dir}/train_16_dirty.npz")

    print("\n" + "=" * 60)
    print("训练模型-C（clean 标签）")
    print("=" * 60)
    train_model(conds, labels, args.seed, args.epochs, args.batch, args.lr,
                args.timesteps,
                os.path.join(args.out_dir, "checkpoint_16_clean_cpu.pt"), "clean_cpu",
                device=dev)

    print("\n" + "=" * 60)
    print("训练模型-D（dirty 标签，含浮材）")
    print("=" * 60)
    train_model(conds, dirty, args.seed, args.epochs, args.batch, args.lr,
                args.timesteps,
                os.path.join(args.out_dir, "checkpoint_16_dirty_cpu.pt"), "dirty_cpu",
                device=dev)

    print("\n[all done] 两个 checkpoint 已就绪，可运行 eval_clean_vs_dirty.py 对照评估")


if __name__ == "__main__":
    main()
