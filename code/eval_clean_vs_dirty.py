"""对照评估：clean vs dirty 训练数据的"可行性恢复能力"。

比较两个模型（同配置训练，仅训练标签的连通性不同）在相同 30 个条件下的：
    unrefined 连通性 / 低噪重构（no-bridge）连通性 / 恢复增益 / 柔度 / 材料量

核心观测量：**恢复增益 = conn(重构) − conn(采样)**
    若"恢复能力来自训练分布"，则 dirty 模型（训练分布本身含浮材）的恢复增益
    应显著低于 clean 模型。

用法:
    python code/eval_clean_vs_dirty.py --n_samples 30 --out data/clean_vs_dirty.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

from diffusion.diffusion import UNet3D, Diffusion
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index
from eval_bridge_recon_factorial import build_ls, compliance, connectivity_6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--clean", default="results/checkpoint_16_clean.pt")
    ap.add_argument("--dirty", default="results/checkpoint_16_dirty.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--refine_seed", type=int, default=77000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--t_refine", type=int, default=30)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--out", default="data/clean_vs_dirty.csv")
    args = ap.parse_args()

    d = np.load(args.data)
    conds = torch.from_numpy(d["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]
    shape = (1, 1, D, H, W)

    ke = hex8_stiffness(1.0, 0.3)
    ke_flat = ke.ravel()
    edof = build_edof(D, H, W)
    iK, jK = build_k_index(edof)
    ndof = 3 * (D + 1) * (H + 1) * (W + 1)
    N = min(args.n_samples, conds.shape[0])

    rows = []
    for tag, path in [("clean", args.clean), ("dirty", args.dirty)]:
        if not os.path.isfile(path):
            print(f"[skip] {tag}: {path} 不存在")
            continue
        model = UNet3D(in_channels=5, base=32, depth=2)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        diff = Diffusion(model, timesteps=200)

        for i in range(N):
            cond = conds[i:i + 1]
            src = conds[i, 2:3]
            sup = conds[i, 3:4]
            loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)

            torch.manual_seed(args.seed + i)
            with torch.no_grad():
                s = diff.sample(cond, shape, ddim_steps=args.ddim_steps)
                torch.manual_seed(args.refine_seed + i)
                nb = diff.refine(s, cond, src, sup, K=args.K,
                                 t_refine=args.t_refine,
                                 bridge=False, reconstruct=True)

            s_np, nb_np = s[0, 0].numpy(), nb[0, 0].numpy()
            c_s, f_s, _ = connectivity_6(s_np)
            c_n, f_n, _ = connectivity_6(nb_np)
            rows.append([
                tag, i, c_s, c_n, c_n - c_s, f_s, f_n,
                compliance(s_np, loads, supports, ke_flat, iK, jK, ndof),
                compliance(nb_np, loads, supports, ke_flat, iK, jK, ndof),
                int((s_np > 0.5).sum()), int((nb_np > 0.5).sum()),
            ])
        print(f"[{tag}] 完成 {N} 样本")

    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "sample", "unrefined_conn", "nobridge_conn",
                    "recovery_gain", "unrefined_float", "nobridge_float",
                    "unrefined_comp", "nobridge_comp",
                    "unrefined_occ", "nobridge_occ"])
        w.writerows(rows)

    print("\n" + "=" * 92)
    print("clean vs dirty 训练数据：可行性恢复能力对照")
    print("=" * 92)
    print(f"{'模型':<10}{'采样conn':>10}{'重构conn':>10}{'采样浮材率':>12}{'重构浮材率':>12}"
          f"{'恢复增益':>10}{'柔度(重构)':>11}{'材料(重构)':>11}")
    seen = []
    for r in rows:
        if r[0] not in seen:
            seen.append(r[0])
    for tag in seen:
        sub = [r for r in rows if r[0] == tag]
        cs = np.mean([r[2] for r in sub])
        cn = np.mean([r[3] for r in sub])
        gain = np.mean([r[4] for r in sub])
        fs = np.mean([r[5] for r in sub])
        fn = np.mean([r[6] for r in sub])
        comps = [r[8] for r in sub if 0 < r[8] < 1e6]
        print(f"{tag:<10}{cs:>10.4f}{cn:>10.4f}{fs:>12.4f}{fn:>12.4f}"
              f"{gain:>+10.4f}{np.mean(comps):>11.2f}"
              f"{np.mean([r[10] for r in sub]):>11.1f}")

    # 配对比较（两模型用相同条件与采样种子）
    cl = {r[1]: r for r in rows if r[0] == "clean"}
    dt = {r[1]: r for r in rows if r[0] == "dirty"}
    common = sorted(set(cl) & set(dt))
    if common:
        from scipy.stats import ttest_rel
        g_cl = np.array([cl[i][4] for i in common])
        g_dt = np.array([dt[i][4] for i in common])
        t, p = ttest_rel(g_cl, g_dt)
        print(f"\n恢复增益 配对比较（n={len(common)}）：clean {g_cl.mean():+.4f} "
              f"vs dirty {g_dt.mean():+.4f}  Δ={g_cl.mean()-g_dt.mean():+.4f}  "
              f"t={t:.2f}  p={p:.4g}")
        f_cl = np.array([cl[i][5] for i in common])
        f_dt = np.array([dt[i][5] for i in common])
        t2, p2 = ttest_rel(f_cl, f_dt)
        print(f"采样浮材率 配对比较：clean {f_cl.mean():.4f} vs dirty {f_dt.mean():.4f}  "
              f"Δ={f_cl.mean()-f_dt.mean():+.4f}  t={t2:.2f}  p={p2:.4g}")
        print("→ 浮材率不受材料量混淆影响，是『训练先验是否被学到』的直接指标")
        print("→ 若 clean 的恢复增益显著更高，则『恢复能力来自训练分布』成立")

    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
