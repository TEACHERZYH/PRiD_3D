"""no-bridge + reconstruction 对照：验证桥接（连通性投影）的必要性。

问题②的关键实验：PRiD 的柔度/连通性优势，到底来自「桥接加材料」还是「重构本身」？
对每个样本（配对、固定 seed）比较三种后处理：
1. unrefined：条件 DDIM 采样（无后处理）
2. no-bridge：只做 K 轮低噪重构，跳过桥接投影（refine(bridge=False)）
3. PRiD：桥接 + 重构（refine(bridge=True)）

若 no-bridge 的连通性显著低于 PRiD，说明桥接是连通性提升的必要环节；
若 no-bridge 的柔度也大幅下降，说明柔度优势主要来自重构而非桥接。

用法:
    python code/eval_no_bridge_control.py --n_samples 30 --seed 12000 --out data/no_bridge_control.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from scipy.ndimage import label as connected_components, generate_binary_structure
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import ttest_rel

from diffusion.diffusion import UNet3D, Diffusion
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index


# --------------------------------------------------------------------------- #
# 工况与柔度（与 eval_compliance_per_sample.py 一致，坐标约定对齐数据生成）
# --------------------------------------------------------------------------- #
def build_ls(cond_np, D, H, W, ndof):
    """从条件通道构建荷载向量和支座自由度掩码。

    坐标约定（与 synthesize_parallel.py 一致）：数组维度 (x, y, z)，
    节点编号 = x + (nx+1)*y + (nx+1)*(ny+1)*z。
    支座：x=0 面节点全固定；荷载：顶面 z=nz 节点，-z 集中力。
    """
    loads = np.zeros(ndof)
    supports = np.zeros(ndof, dtype=bool)
    if cond_np[3].any():
        for z in range(W + 1):
            for y in range(H + 1):
                nid = (D + 1) * (H + 1) * z + (D + 1) * y  # 节点 (x=0, y, z)
                supports[3 * nid:3 * nid + 3] = True
    lp = np.argwhere(cond_np[2] > 0.5)
    if len(lp) > 0:
        x, y, z = lp[0]  # 单元坐标 (x, y, z)
        nid = x + (D + 1) * y + (D + 1) * (H + 1) * W  # 节点 (x, y, W)
        loads[3 * nid + 2] = -1.0
    return loads, supports


def compliance(rho, loads, supports, ke_flat, iK, jK, ndof):
    """稀疏 FEM 重算柔度。"""
    E = 1e-9 + (1 - 1e-9) * rho.ravel() ** 3
    vals = (ke_flat[None, :] * E[:, None]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = ~supports
    Uf = spsolve(K[free][:, free], loads[free])
    U = np.zeros(ndof)
    U[free] = Uf
    return float(loads @ U)


def connectivity_6(rho, threshold=0.5):
    """6 邻域最大连通分量材料占比（论文 eq:connectivity）。

    返回 (connectivity, floating_fraction, n_components)。
    """
    binary = rho > threshold
    total = int(binary.sum())
    if total == 0:
        return 0.0, 1.0, 0
    s = generate_binary_structure(3, 1)  # 6 邻域
    labels, n_comp = connected_components(binary, structure=s)
    sizes = np.bincount(labels.ravel())
    if len(sizes) <= 1:
        return 1.0, 0.0, 1
    largest = int(sizes[1:].max())
    return largest / total, 1.0 - largest / total, int(n_comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--t_refine", type=int, default=30)
    ap.add_argument("--out", default="data/no_bridge_control.csv")
    args = ap.parse_args()

    data = np.load(args.data)
    conds = torch.from_numpy(data["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]

    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    ke = hex8_stiffness(1.0, 0.3)
    ke_flat = ke.ravel()
    edof = build_edof(D, H, W)
    iK, jK = build_k_index(edof)
    ndof = 3 * (D + 1) * (H + 1) * (W + 1)

    N = min(args.n_samples, conds.shape[0])
    rows = []
    header = ["sample", "vf",
              "unrefined_conn", "no_bridge_conn", "prid_conn",
              "unrefined_comp", "no_bridge_comp", "prid_comp",
              "unrefined_occ", "no_bridge_occ", "prid_occ"]

    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        vf = float(conds[i, 0, 0, 0, 0])
        loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
            nb = diff.refine(s, cond, src, sup, K=args.K, t_refine=args.t_refine, bridge=False)
            rf = diff.refine(s, cond, src, sup, K=args.K, t_refine=args.t_refine, bridge=True)

        s_np = s[0, 0].numpy()
        nb_np = nb[0, 0].numpy()
        rf_np = rf[0, 0].numpy()

        uc, _, _ = connectivity_6(s_np)
        nbc, _, _ = connectivity_6(nb_np)
        rc, _, _ = connectivity_6(rf_np)

        u_occ = int((s_np > 0.5).sum())
        nb_occ = int((nb_np > 0.5).sum())
        rf_occ = int((rf_np > 0.5).sum())

        u_comp = compliance(s_np, loads, supports, ke_flat, iK, jK, ndof)
        nb_comp = compliance(nb_np, loads, supports, ke_flat, iK, jK, ndof)
        rf_comp = compliance(rf_np, loads, supports, ke_flat, iK, jK, ndof)

        rows.append([i, vf, uc, nbc, rc, u_comp, nb_comp, rf_comp,
                     u_occ, nb_occ, rf_occ])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}")

    # ---- 连通性（全部样本，配对 t 检验）----
    u_conn = np.array([r[2] for r in rows])
    nb_conn = np.array([r[3] for r in rows])
    rf_conn = np.array([r[4] for r in rows])
    t_nb_rf, p_nb_rf = ttest_rel(rf_conn, nb_conn)
    t_nb_u, p_nb_u = ttest_rel(nb_conn, u_conn)

    print(f"\n=== 连通性（{len(rows)} 样本，配对） ===")
    print(f"unrefined  均值 {u_conn.mean():.4f}")
    print(f"no-bridge  均值 {nb_conn.mean():.4f}  (vs unrefined: t={t_nb_u:.2f}, p={p_nb_u:.4g})")
    print(f"PRiD       均值 {rf_conn.mean():.4f}  (vs no-bridge: t={t_nb_rf:.2f}, p={p_nb_rf:.4g})")
    print(f"PRiD 胜过 no-bridge 的样本: {int((rf_conn > nb_conn).sum())}/{len(rows)}")

    # ---- 占用体素（全部样本）----
    u_occ = np.array([r[8] for r in rows], dtype=float)
    nb_occ = np.array([r[9] for r in rows], dtype=float)
    rf_occ = np.array([r[10] for r in rows], dtype=float)
    print(f"\n=== 占用体素（{len(rows)} 样本） ===")
    print(f"unrefined  均值 {u_occ.mean():.1f}")
    print(f"no-bridge  均值 {nb_occ.mean():.1f}  ({100*(nb_occ.mean()/u_occ.mean()-1):+.1f}%)")
    print(f"PRiD       均值 {rf_occ.mean():.1f}  ({100*(rf_occ.mean()/u_occ.mean()-1):+.1f}%)")

    # ---- 柔度（仅有效样本：0 < comp < 1e6）----
    valid = [r for r in rows if all(0 < r[j] < 1e6 for j in [5, 6, 7])]
    if valid:
        u_comp = np.array([r[5] for r in valid])
        nb_comp = np.array([r[6] for r in valid])
        rf_comp = np.array([r[7] for r in valid])
        t_nb_rf_c, p_nb_rf_c = ttest_rel(rf_comp, nb_comp)
        print(f"\n=== 柔度（{len(valid)}/{len(rows)} 有效样本，配对） ===")
        print(f"unrefined  均值 {u_comp.mean():.1f}")
        print(f"no-bridge  均值 {nb_comp.mean():.1f}  ({(nb_comp.mean()/u_comp.mean()-1)*100:+.1f}%)")
        print(f"PRiD       均值 {rf_comp.mean():.1f}  ({(rf_comp.mean()/u_comp.mean()-1)*100:+.1f}%)")
        print(f"PRiD vs no-bridge 柔度: t={t_nb_rf_c:.2f}, p={p_nb_rf_c:.4g}, "
              f"PRiD 更低 {int((rf_comp < nb_comp).sum())}/{len(valid)}")
    else:
        print("\n（无有效柔度样本）")


if __name__ == "__main__":
    main()
