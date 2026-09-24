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


def remove_equal_mass(labels, conds, n_remove, thresh=0.5, stress_channel=1):
    """从主体中删除与注入量相同的材料，使材料总量保持不变。

    为什么必须做这一步（2026-09-17 修正）：原 dirty 条件在标签里注入占材料 15% 的
    孤立块，**同时把训练集材料量抬高了 15%**。而"材料多 → 生成得多 → 更易连通"
    与"训练分布改变了可行先验"是两条完全不同的因果路径，原设计无法区分
    （实测 CPU/GPU 两条件下 dirty 模型的采样连通性都反而更高，正是该混淆的表现）。

    删除策略：在**主体**内按归一化 von Mises 应力**升序**删除（先删最不承力的材料，
    即柔度最小化自己也会先删的那部分），每删一个即检查主体是否仍连通，
    不连通则回退该体素。这样保证：材料总量相等、且主体始终连通。

    返回 (labels_out, removed_counts)。
    """
    rng_guard = np.random.default_rng(0)  # 仅用于打散同应力值的候选顺序
    out = labels.copy()
    removed = []
    for i in range(len(labels)):
        b = labels[i, 0] > thresh
        occ = int(b.sum())
        if occ == 0:
            removed.append(0)
            continue
        lab, n = cc(b, structure=generate_binary_structure(3, 1))
        sizes = np.bincount(lab.ravel())[1:]
        main_id = int(np.argmax(sizes)) + 1
        main = lab == main_id
        stress = conds[i, stress_channel]
        cand = np.argwhere(main)
        if len(cand) == 0:
            removed.append(0)
            continue
        order = np.argsort(stress[tuple(cand.T)], kind="stable")
        # 同应力值内部随机化，避免系统性地在某一侧挖空
        key = stress[tuple(cand.T)][order]
        jitter = rng_guard.random(len(order)) * 1e-6
        order = order[np.argsort(key + jitter, kind="stable")]
        cand = cand[order]

        n_done = 0
        for (cx, cy, cz) in cand[: n_remove * 3]:
            if n_done >= n_remove:
                break
            if not out[i, 0, cx, cy, cz] > thresh:
                continue
            out[i, 0, cx, cy, cz] = 0.0
            nb = out[i, 0] > thresh
            lab2, _ = cc(nb, structure=generate_binary_structure(3, 1))
            # 判据：主体剩余体素必须仍属**同一个**连通分量
            # （不能要求"整个场只有一个分量"—— dirty 标签本身就带注入的孤立块）
            rest = main.copy()
            rest[cx, cy, cz] = False
            vals = lab2[rest]
            if vals.size == 0 or len(np.unique(vals)) != 1:
                out[i, 0, cx, cy, cz] = labels[i, 0, cx, cy, cz]   # 回退，保连通
            else:
                main = rest
                n_done += 1
        removed.append(n_done)
    return out, removed


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
    # 修复（2026-09-17）：原代码用未播种的 np.random.permutation 生成划分，
    # 导致同一 seed 下 clean 与 dirty 的**数据划分并不相同**（注释所称"相同划分"不成立）。
    # 现改为由 seed 决定的确定性划分 + 确定性初始化，使两条件只差标签。
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    n = len(conds)
    n_tr = int(n * 0.9)
    perm = np.random.default_rng(seed).permutation(n)
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
    ap.add_argument("--match_volume", action="store_true",
                    help="等材料量注入：注入浮材后从主体按最低 von Mises 应力删除等量材料，"
                         "消除「材料量 +15%」混淆（2026-09-17 修正）")
    ap.add_argument("--tag", default="",
                    help="输出文件名标签，如 seed1  → checkpoint_16_seed1_clean.pt")
    ap.add_argument("--labels_only", action="store_true",
                    help="只构造并保存标签数据集，不训练（用于本机快速验证）")
    args = ap.parse_args()
    dev = None if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)

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

    mass_clean = float((labels > 0.5).sum())
    mass_dirty = float((dirty > 0.5).sum())
    print(f"[data] 材料量（体素总数）：clean={mass_clean:.0f}  dirty={mass_dirty:.0f}"
          f"（{100*(mass_dirty/mass_clean - 1):+.1f}%）")

    if args.match_volume:
        n_remove = int(round(np.mean(added)))
        print(f"\n[data] 等材料量注入：从主体按最低 von Mises 应力删除 {n_remove} 体素/样本")
        dirty, removed = remove_equal_mass(dirty, conds, n_remove)
        md2, n2b, _ = dataset_conn(dirty)
        mass_dirty2 = float((dirty > 0.5).sum())
        print(f"[data] 等量后 dirty 标签：连通性 {md2:.4f}，完全连通 {n2b}/{N}"
              f"（实际删除 {np.mean(removed):.0f} 体素/样本）")
        print(f"[data] 等量后材料量：dirty={mass_dirty2:.0f}"
              f"（相对 clean {100*(mass_dirty2/mass_clean - 1):+.1f}%）")

    sfx = f"_{args.tag}" if args.tag else ""
    np.savez_compressed(os.path.join(args.out_dir, f"train_16_dirty{sfx}.npz"),
                        conds=conds, labels=dirty)
    print(f"[data] dirty 数据集已保存：{args.out_dir}/train_16_dirty{sfx}.npz")

    if args.labels_only:
        print("[labels_only] 已跳过训练")
        return

    print("\n" + "=" * 60)
    print("训练模型-C（clean 标签）")
    print("=" * 60)
    ck_c = os.path.join(args.out_dir, f"checkpoint_16{sfx}_clean.pt")
    train_model(conds, labels, args.seed, args.epochs, args.batch, args.lr,
                args.timesteps, ck_c, f"clean_cpu{sfx}", device=dev)

    print("\n" + "=" * 60)
    print("训练模型-D（dirty 标签，含浮材）")
    print("=" * 60)
    ck_d = os.path.join(args.out_dir, f"checkpoint_16{sfx}_dirty.pt")
    train_model(conds, dirty, args.seed, args.epochs, args.batch, args.lr,
                args.timesteps, ck_d, f"dirty_cpu{sfx}", device=dev)

    print("\n[all done] 两个 checkpoint 已就绪，可运行 eval_clean_vs_dirty.py 对照评估")


if __name__ == "__main__":
    main()
