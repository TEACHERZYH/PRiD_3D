"""2x2 因子对照：分离「桥接投影」与「低噪重构」对连通性/柔度的贡献。

背景：no_bridge_control 的初版对照只有三组（unrefined / no-bridge / PRiD），
缺少 one-shot（只桥接）一格，无法完成变量分离；且两次 refine 依次消耗不同
噪声序列，配对不严格。本脚本修复这两点：

1. 完整 2x2 因子设计（每格都用同一初始场，配对）：
   |              | 不重构            | 重构(K 轮)        |
   |--------------|-------------------|-------------------|
   | 不桥接       | unrefined         | no-bridge         |
   | 桥接(K 轮)   | one-shot (K=1)    | PRiD              |

2. 严格噪声配对：同一 unrefined 场上做后处理，且每次 refine 前重置到同一
   随机种子（seed+offset），使各方法消耗完全相同的噪声序列。

3. 可选采样强度扫描（--ddim_scan）：unrefined 在 ddim_steps in {20,50,100}
   下的连通性，用于确认「unrefined 是否最优采样」这一疑问。

主效应（配对）：
    桥接主效应  = mean[(PRiD + one-shot) - (no-bridge + unrefined)] / 2
    重构主效应  = mean[(PRiD + no-bridge) - (one-shot + unrefined)] / 2
    交互效应    = mean[(PRiD - one-shot) - (no-bridge - unrefined)]

用法:
    python code/eval_bridge_recon_factorial.py \
        --n_samples 30 --seed 12000 --out data/bridge_recon_factorial.csv
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


METHODS = ["unrefined", "one_shot", "no_bridge", "prid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--refine_seed", type=int, default=77000,
                    help="refine 各方法共享的噪声种子（严格配对）")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--t_refine", type=int, default=30)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--ddim_scan", default="20,50,100",
                    help="unrefined 采样强度扫描，逗号分隔；空串关闭")
    ap.add_argument("--out", default="data/bridge_recon_factorial.csv")
    ap.add_argument("--scan_out", default="data/ddim_strength_scan.csv")
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
    header = ["sample", "vf"]
    for m in METHODS:
        header += [f"{m}_conn", f"{m}_comp", f"{m}_occ"]

    scan_rows = []
    scan_steps = [int(s) for s in args.ddim_scan.split(",") if s.strip()]

    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        vf = float(conds[i, 0, 0, 0, 0])
        loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)

        # ---- 共享的 unrefined 初始场（所有后处理方法由此出发）----
        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, (1, 1, D, H, W),
                            ddim_steps=args.ddim_steps, use_projection=False)

        out = {}
        with torch.no_grad():
            # unrefined: 无后处理
            out["unrefined"] = s
            # one-shot: 只桥接一次（bridge=True, reconstruct=False, K=1）
            torch.manual_seed(args.refine_seed + i)
            out["one_shot"] = diff.refine(s, cond, src, sup, K=1,
                                          t_refine=args.t_refine,
                                          bridge=True, reconstruct=False)
            # no-bridge: 只重构 K 轮
            torch.manual_seed(args.refine_seed + i)
            out["no_bridge"] = diff.refine(s, cond, src, sup, K=args.K,
                                           t_refine=args.t_refine,
                                           bridge=False, reconstruct=True)
            # PRiD: 桥接 + 重构交替 K 轮
            torch.manual_seed(args.refine_seed + i)
            out["prid"] = diff.refine(s, cond, src, sup, K=args.K,
                                      t_refine=args.t_refine,
                                      bridge=True, reconstruct=True)

        row = [i, vf]
        for m in METHODS:
            arr = out[m][0, 0].numpy()
            c, _, _ = connectivity_6(arr)
            comp = compliance(arr, loads, supports, ke_flat, iK, jK, ndof)
            occ = int((arr > 0.5).sum())
            row += [c, comp, occ]
        rows.append(row)

        # ---- 采样强度扫描（仅 unrefined）----
        for ds in scan_steps:
            torch.manual_seed(args.seed + i)
            with torch.no_grad():
                sd = diff.sample(cond, (1, 1, D, H, W),
                                 ddim_steps=ds, use_projection=False)
            arr = sd[0, 0].numpy()
            c, _, _ = connectivity_6(arr)
            scan_rows.append([i, ds, c, int((arr > 0.5).sum())])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}")

    if scan_rows:
        with open(args.scan_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample", "ddim_steps", "conn", "occ"])
            w.writerows(scan_rows)
        print(f"扫描 CSV 已保存: {args.scan_out}")

    # ------------------------------------------------------------------ #
    # 2x2 因子分析
    # ------------------------------------------------------------------ #
    def col(m, k):
        idx = 2 + METHODS.index(m) * 3 + k  # k: 0=conn, 1=comp, 2=occ
        return np.array([r[idx] for r in rows], dtype=float)

    print(f"\n=== 连通性（{len(rows)} 样本，配对；同一初始场 + 相同 refine 噪声）===")
    conn = {m: col(m, 0) for m in METHODS}
    for m in METHODS:
        print(f"  {m:10s} 均值 {conn[m].mean():.4f}   =1.0 样本 {int((conn[m] > 0.999).sum())}/{len(rows)}")

    eff_bridge_conn = ((conn["prid"] + conn["one_shot"])
                       - (conn["no_bridge"] + conn["unrefined"])) / 2.0
    eff_recon_conn = ((conn["prid"] + conn["no_bridge"])
                      - (conn["one_shot"] + conn["unrefined"])) / 2.0
    inter_conn = (conn["prid"] - conn["one_shot"]) - (conn["no_bridge"] - conn["unrefined"])
    print(f"\n  桥接主效应 {eff_bridge_conn.mean():+.4f}  (t={ttest_rel(conn['prid'] + conn['one_shot'], conn['no_bridge'] + conn['unrefined'])[0]:.2f}, p={ttest_rel(conn['prid'] + conn['one_shot'], conn['no_bridge'] + conn['unrefined'])[1]:.4g})")
    print(f"  重构主效应 {eff_recon_conn.mean():+.4f}  (t={ttest_rel(conn['prid'] + conn['no_bridge'], conn['one_shot'] + conn['unrefined'])[0]:.2f}, p={ttest_rel(conn['prid'] + conn['no_bridge'], conn['one_shot'] + conn['unrefined'])[1]:.4g})")
    print(f"  交互效应   {inter_conn.mean():+.4f}")

    print("\n  关键配对对比：")
    for a, b in [("one_shot", "unrefined"), ("no_bridge", "unrefined"),
                 ("prid", "unrefined"), ("prid", "one_shot"), ("prid", "no_bridge")]:
        t, p = ttest_rel(conn[a], conn[b])
        print(f"    {a:10s} vs {b:10s}: Δ={conn[a].mean()-conn[b].mean():+.4f}  t={t:.2f}  p={p:.4g}  "
              f"胜 {int((conn[a] > conn[b]).sum())}/{len(rows)}")

    print(f"\n=== 占用体素（{len(rows)} 样本）===")
    occ = {m: col(m, 2) for m in METHODS}
    base = occ["unrefined"].mean()
    for m in METHODS:
        print(f"  {m:10s} 均值 {occ[m].mean():.1f}  ({100*(occ[m].mean()/base-1):+.1f}%)")

    print(f"\n=== 柔度（仅有效样本：0 < comp < 1e6，四组都有效）===")
    comp = {m: col(m, 1) for m in METHODS}
    valid_mask = np.array([all(0 < r[2 + METHODS.index(m) * 3 + 1] < 1e6
                              for m in METHODS) for r in rows])
    n_valid = int(valid_mask.sum())
    if n_valid:
        cbase = comp["unrefined"][valid_mask].mean()
        for m in METHODS:
            v = comp[m][valid_mask]
            print(f"  {m:10s} 均值 {v.mean():.1f}  ({(v.mean()/cbase-1)*100:+.1f}%)")
        eff_bridge_comp = ((comp["prid"] + comp["one_shot"])
                           - (comp["no_bridge"] + comp["unrefined"]))[valid_mask] / 2.0
        eff_recon_comp = ((comp["prid"] + comp["no_bridge"])
                          - (comp["one_shot"] + comp["unrefined"]))[valid_mask] / 2.0
        print(f"\n  桥接主效应 {eff_bridge_comp.mean():+.3f}")
        print(f"  重构主效应 {eff_recon_comp.mean():+.3f}")
        for a, b in [("prid", "no_bridge"), ("prid", "one_shot")]:
            t, p = ttest_rel(comp[a][valid_mask], comp[b][valid_mask])
            print(f"    {a:10s} vs {b:10s}: t={t:.2f}, p={p:.4g}")
    else:
        print("  （无有效柔度样本）")

    # ------------------------------------------------------------------ #
    # 采样强度扫描
    # ------------------------------------------------------------------ #
    if scan_rows:
        print(f"\n=== 采样强度扫描（unrefined，{N} 样本）===")
        for ds in scan_steps:
            vals = [r[2] for r in scan_rows if r[1] == ds]
            occs = [r[3] for r in scan_rows if r[1] == ds]
            v = np.array(vals)
            print(f"  ddim_steps={ds:3d}: 连通性均值 {v.mean():.4f}  =1.0 样本 {int((v > 0.999).sum())}/{len(v)}  "
                  f"占用体素 {np.mean(occs):.1f}")


if __name__ == "__main__":
    main()
