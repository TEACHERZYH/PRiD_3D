"""训练分布对照的**体积归一化**评估（消除"生成占用差异"这一残余混淆）。

背景（2026-09-17 增补实验发现）：
  等材料量注入（`--match_volume`）已把**训练集**材料量对齐到 +0.1%，
  但 dirty 模型**生成**的占用仍比 clean 低 46%（639 vs 1190）。
  而连通性天然依赖占用（结构越稀疏越易断开），两组占用分布几乎不重叠
  （匹配占用后仅剩 15 对，n 不足），因此原口径下的连通性差无法归因。

修正办法：**按工况给定的目标体积分数二值化**（取密度最高的 k = V0·N 个体素）。
  同一设计工况下 V0 相同 ⇒ 两条件的占用**按构造完全相等**，
  连通性差异不再能由"生成占用"解释。

用法:
    python code/eval_dist_volnorm.py --n_samples 30 \
        --clean results/checkpoint_16_eq1_clean.pt \
        --dirty results/checkpoint_16_eq1_dirty.pt \
        --out data/dist_volnorm_eq1.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label as cc

from diffusion.diffusion import UNet3D, Diffusion
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index
from eval_bridge_recon_factorial import build_ls, compliance, connectivity_6


def conn_topk(field, k):
    """取密度最高的 k 个体素作为占用（k 由工况目标体积分数决定），再算 6 邻域连通性。"""
    flat = field.ravel()
    n = flat.size
    k = int(np.clip(k, 1, n))
    thr = np.partition(flat, n - k)[n - k]
    binary = flat >= thr
    if binary.sum() > k:                      # 并列值可能多取，做确定性裁剪
        idx = np.argsort(-flat, kind="stable")
        sel = np.zeros(n, dtype=bool)
        sel[idx[:k]] = True
        binary = sel
    total = int(binary.sum())
    lab, _ = cc(binary.reshape(field.shape),
                structure=generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    return (largest / total if total else 0.0,
            1.0 - (largest / total if total else 1.0), total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--dirty", required=True)
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--refine_seed", type=int, default=77000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--t_refine", type=int, default=30)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.data)
    conds = torch.from_numpy(d["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]
    N_vox = D * H * W
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
            print(f"[skip] {tag}: {path} 不存在", flush=True)
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
            v0 = float(conds[i, 0, 0, 0, 0])
            k = int(round(v0 * N_vox))

            torch.manual_seed(args.seed + i)
            with torch.no_grad():
                s = diff.sample(cond, shape, ddim_steps=args.ddim_steps)
                torch.manual_seed(args.refine_seed + i)
                nb = diff.refine(s, cond, src, sup, K=args.K,
                                 t_refine=args.t_refine,
                                 bridge=False, reconstruct=True)

            s_np, nb_np = s[0, 0].numpy(), nb[0, 0].numpy()
            c_s, f_s, o_s = conn_topk(s_np, k)
            c_n, f_n, o_n = conn_topk(nb_np, k)
            rows.append([
                tag, i, v0, k,
                c_s, c_n, c_n - c_s, f_s, f_n, o_s, o_n,
                compliance(s_np, loads, supports, ke_flat, iK, jK, ndof),
                compliance(nb_np, loads, supports, ke_flat, iK, jK, ndof),
                connectivity_6(nb_np)[0],                    # 原 0.5 阈值，留作对照
            ])
        print(f"[{tag}] 完成 {N} 样本", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "sample", "volfrac", "k_target",
                    "unref_conn_vn", "nobr_conn_vn", "gain_vn",
                    "unref_float_vn", "nobr_float_vn",
                    "unref_occ_vn", "nobr_occ_vn",
                    "unref_comp_vn", "nobr_comp_vn", "nobr_conn_half"])
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}")

    for tag in ("clean", "dirty"):
        sub = [r for r in rows if r[0] == tag]
        if not sub:
            continue
        print(f"[{tag}] 体积归一化：采样conn={np.mean([r[4] for r in sub]):.4f}  "
              f"重构conn={np.mean([r[5] for r in sub]):.4f}  "
              f"占用={np.mean([r[10] for r in sub]):.1f}  "
              f"（原口径重构conn={np.mean([r[13] for r in sub]):.4f}）")


if __name__ == "__main__":
    main()
