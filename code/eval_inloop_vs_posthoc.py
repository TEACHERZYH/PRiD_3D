"""采样中投影（in-loop）vs 采样后投影（post-hoc）的配对对照。

修复的缺陷：论文 §5.5 的"in-loop 仅 13.3% 配对占优"与结论的"87% 配对更差"
来自旧 P0 实验的**单 seed 记录**（experiments/实验记录（补做P0）.md），本机不可重算。
本脚本用当前模型、严格配对噪声重做该对照，产出可归档的逐样本 CSV。

设计：
  * 配对噪声：同一 (seed, sample) 下三个条件的初始噪声完全一致
    （torch.manual_seed(seed + i) 后立即采样）；
  * post-hoc : 普通 DDIM 采样 → 采样结束后施加一次 manifold_projection（+ 相同平滑）；
  * in-loop  : 采样过程中在 project_from_frac 之后逐步投影（anneal + smooth），
               这是文献主流的"推理期干预"做法；
  * 另设 project_from_frac=0.5 的 in-loop 变体作为敏感性检查（干预更早、更强）；
  * 指标：6 邻域连通性 C、浮材率 F、占用体素、柔度 c；
  * 统计：配对差值、in-loop 占优的配对比例、配对 t 检验。

用法:
    python code/eval_inloop_vs_posthoc.py --n_samples 30 --seeds 12000,22000,32000
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import generate_binary_structure, label as cc
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import ttest_rel

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index

CONFIGS = ["posthoc", "inloop90", "inloop50"]


def conn_stats(binary):
    total = int(binary.sum())
    if total == 0:
        return 0.0, 1.0, 0
    lab, _ = cc(binary, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
    return largest / total, 1.0 - largest / total, int(largest > 0)


def build_ls(cond_np, D, H, W, ndof):
    """从条件通道重建荷载/支座向量。通道 2=荷载掩码，3=支座掩码。"""
    load_mask = cond_np[2] > 0.5
    sup_mask = cond_np[3] > 0.5
    loads = np.zeros(ndof)
    supports = np.zeros(ndof, dtype=bool)
    for ix in range(D):
        for iy in range(H):
            for iz in range(W):
                if load_mask[ix, iy, iz]:
                    nid = ix + (D + 1) * iy + (D + 1) * (H + 1) * (iz + 1)
                    if 3 * nid + 2 < ndof:
                        loads[3 * nid + 2] = -1.0
                if sup_mask[ix, iy, iz]:
                    for dx in (0, 1):
                        for dy in (0, 1):
                            for dz in (0, 1):
                                nid = (ix + dx) + (D + 1) * (iy + dy) \
                                      + (D + 1) * (H + 1) * (iz + dz)
                                if 3 * nid + 2 < ndof:
                                    supports[3 * nid:3 * nid + 3] = True
    return loads, supports


def compliance(rho, loads, supports, ke_flat, iK, jK, ndof, penal=3.0, emin=1e-3):
    rho = np.clip(rho.ravel(), 0.0, 1.0)
    E = emin + (1.0 - emin) * rho ** penal
    vals = (E[:, None] * ke_flat[None, :]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsr()
    free = ~supports
    if free.sum() == 0:
        return float("nan")
    Kff = K[free][:, free].tocsc()
    u = np.zeros(ndof)
    try:
        u[free] = spsolve(Kff, loads[free])
    except Exception:  # noqa: BLE001
        return float("nan")
    c = float(loads @ u)
    if not np.isfinite(c) or c <= 0:
        return float("nan")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seeds", default="12000,22000,32000")
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--out", default="data/inloop_vs_posthoc.csv")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    data = np.load(args.data)
    conds_np = data["conds"]
    D, H, W = conds_np.shape[2], conds_np.shape[3], conds_np.shape[4]

    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    ke = hex8_stiffness(1.0, 0.3)
    ke_flat = ke.ravel()
    edof = build_edof(D, H, W)
    iK, jK = build_k_index(edof)
    ndof = 3 * (D + 1) * (H + 1) * (W + 1)

    N = min(args.n_samples, conds_np.shape[0])
    header = ["seed", "sample", "config", "conn", "floating", "occ", "comp"]
    rows = []

    print(f"=== in-loop vs post-hoc 配对对照：{len(seeds)} seed × {N} 样本 × {len(CONFIGS)} 条件 ===")
    for seed in seeds:
        for i in range(N):
            cond = torch.from_numpy(conds_np[i:i + 1]).float()
            src = cond[:, 2:3]
            sup = cond[:, 3:4]
            loads, supports = build_ls(conds_np[i], D, H, W, ndof)

            outs = {}
            # post-hoc：普通采样 + 事后一次投影（与 in-loop 用同样的投影与平滑）
            torch.manual_seed(seed + i)
            with torch.no_grad():
                x = diff.sample(cond, (1, 1, D, H, W), ddim_steps=args.ddim_steps,
                                use_projection=False)
                x = manifold_projection(x, source_mask=src, support_mask=sup)
                x = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
            outs["posthoc"] = x

            # in-loop：采样过程中干预（两种起始强度）
            for name, frac in (("inloop90", 0.9), ("inloop50", 0.5)):
                torch.manual_seed(seed + i)
                with torch.no_grad():
                    outs[name] = diff.sample(
                        cond, (1, 1, D, H, W), ddim_steps=args.ddim_steps,
                        use_projection=True, project_from_frac=frac,
                        projection_anneal=True, projection_smooth=True)

            for name in CONFIGS:
                rho = outs[name][0, 0].numpy()
                b = rho > 0.5
                c, f, _ = conn_stats(b)
                rows.append([seed, i, name, c, f, int(b.sum()),
                             compliance(rho, loads, supports, ke_flat, iK, jK, ndof)])
            if (i + 1) % 5 == 0:
                print(f"  seed={seed} 完成 {i+1}/{N}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}（{len(rows)} 行）")

    # ---- 统计 ----
    def col(config, key):
        return np.array([r[3 + key] for r in rows if r[2] == config])

    idx = {"conn": 0, "floating": 1, "occ": 2, "comp": 3}
    print("\n=== 均值（%d 样本/条件）===" % N if len(seeds) == 1
          else f"\n=== 均值（{len(seeds)*N} 样本/条件）===")
    for name in CONFIGS:
        v = col(name, idx["conn"])
        print(f"  {name:<9} conn={v.mean():.4f}  "
              f"floating={col(name, idx['floating']).mean():.4f}  "
              f"occ={col(name, idx['occ']).mean():.1f}")

    print("\n=== 配对比较（post-hoc 为基准）===")
    base = col("posthoc", idx["conn"])
    for name in ("inloop90", "inloop50"):
        v = col(name, idx["conn"])
        d = v - base
        n_inloop_better = int((d > 0).sum())
        t, p = ttest_rel(v, base)
        print(f"  {name}: Δconn={d.mean():+.4f}  配对 t={t:+.3f} p={p:.4g}  "
              f"in-loop 占优 {n_inloop_better}/{len(d)} = "
              f"{100.0*n_inloop_better/len(d):.1f}%")
        cb = col("posthoc", idx["comp"])
        cv = col(name, idx["comp"])
        m = np.isfinite(cb) & np.isfinite(cv)
        if m.sum() > 3:
            t2, p2 = ttest_rel(cv[m], cb[m])
            print(f"            柔度 Δ={np.mean(cv[m]-cb[m]):+.3f}  t={t2:+.3f} p={p2:.4g}")


if __name__ == "__main__":
    main()
