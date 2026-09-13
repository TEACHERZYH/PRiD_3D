"""P0 实验：界定（投影精炼）vs 拉扯（软约束引导）配对对照 —— 修正版。

修正内容（2026-09-08 审查）：
  1. 热扩散数值稳定性：显式格式 kappa=1.0 六邻域更新中心系数 = -5（负权重振荡，
     被 clamp 掩盖）。改用半隐式格式（对角隐式，系数全非负，无条件稳定）。
  2. 软约束防删材取巧：原目标 mean(density*(1-T)) 无体积约束，全零密度即零损失。
     增加体积保持惩罚 vol_weight*(mean(x)-vol0)^2，迫使软约束在等体积下
     「重新分配」材料（桥接）而非「删除」材料。
  3. 统计口径：配对 t 检验由 np.std(ddof=0) 改为 scipy.stats.ttest_rel（ddof=1）。
  4. 材料用量解释：报告材料量变化百分比与比柔度（柔度/材料量），柔度优势按实际
     材料变化解释，不包装成等材料效率优势。

三组对照（配对 seed，相同初始噪声）：
  unconditional   无条件采样
  soft_guide      软约束引导（拉扯，等体积）
  refine          投影精炼（界定，桥接）

用法:
    python code/soft_constraint_baseline.py --n_samples 30
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label as cc
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import ttest_rel

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index


# --------------------------------------------------------------------------- #
# 软约束（拉扯）实现 —— 修正版
# --------------------------------------------------------------------------- #
NEIGHBOR_KERNEL = torch.tensor(
    [[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
      [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]],
    dtype=torch.float32,
).unsqueeze(0)  # 六邻域求和核（中心 0，邻居 1）


def heat_diffusion(density, src, sup, kappa=1.0, iters=100):
    """半隐式锚点热扩散（邻居 conductivity 加权，无条件稳定）。

    温度 T 从荷载+支座锚点（Dirichlet 边界，T=1）沿材料扩散。
    关键修正：邻居贡献按邻居密度加权（sum_j c[j]*T[j]），使热只在材料内
    传播、不向空隙泄漏；半隐式格式系数全非负，任意 kappa 稳定
    （修复显式格式负权重振荡 + 热向空隙泄漏两个问题）。
    漂浮体与锚点不连通 → 温度低；主体 → 温度高。
    """
    conductivity = density.clamp(0.0, 1.0)
    heat = (src + sup).clamp(0.0, 1.0)  # 锚点（Dirichlet）
    T = heat.clone()
    kernel = NEIGHBOR_KERNEL.to(density.device)
    # 邻居密度和（不随 T 变化，预计算）
    c_pad = F.pad(conductivity, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)
    sum_c = F.conv3d(c_pad, kernel, padding=0)
    for _ in range(iters):
        cT = conductivity * T
        cT_pad = F.pad(cT, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)
        sum_cT = F.conv3d(cT_pad, kernel, padding=0)
        T_update = (T + kappa * sum_cT) / (1.0 + kappa * sum_c)
        T = T_update * (1.0 - heat) + heat  # 锚点保持 1
    return T


def soft_guide(x0, src, sup, steps=100, lr=0.5, kappa=1.0, iters=100, vol_weight=1.0):
    """软约束引导（拉扯）：等体积下最小化漂浮体可微代理。

    漂浮体 = 密度高 + 温度低。损失 = mean(density*(1-T)) + vol_weight*(mean-mean0)^2。
    体积惩罚迫使软约束「重新分配」材料（桥接）而非「删除」（全零密度取巧）。
    """
    x = x0.clone().requires_grad_(True)
    vol0 = x0.mean().detach()
    for _ in range(steps):
        T = heat_diffusion(x, src, sup, kappa=kappa, iters=iters)
        floating = (x * (1.0 - T)).mean()
        vol_penalty = (x.mean() - vol0) ** 2
        loss = floating + vol_weight * vol_penalty
        g = torch.autograd.grad(loss, x)[0]
        x = (x - lr * g).clamp(0.0, 1.0)
        x = x.detach().requires_grad_(True)
    return x.detach()


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def connectivity(rho):
    """连通性 = 最大连通分量材料占比。"""
    bn = rho > 0.5
    if not bn.any():
        return 0.0, 1.0, 0
    labeled, n = cc(bn)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    conn = float((labeled == largest).sum() / bn.sum())
    return conn, 1.0 - conn, n


def build_ls(cond_np, D, H, W, ndof):
    """修复后的工况映射：支座 x=0 面，荷载顶面 z=W 节点。"""
    loads = np.zeros(ndof)
    supports = np.zeros(ndof, dtype=bool)
    if cond_np[3].any():
        for z in range(W + 1):
            for y in range(H + 1):
                nid = (D + 1) * (H + 1) * z + (D + 1) * y
                supports[3 * nid:3 * nid + 3] = True
    lp = np.argwhere(cond_np[2] > 0.5)
    if len(lp) > 0:
        x, y, z = lp[0]
        nid = x + (D + 1) * y + (D + 1) * (H + 1) * W
        loads[3 * nid + 2] = -1.0
    return loads, supports


def compliance(rho, loads, supports, ke_flat, iK, jK, ndof):
    E = 1e-9 + (1 - 1e-9) * rho.ravel() ** 3
    vals = (ke_flat[None, :] * E[:, None]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = ~supports
    Uf = spsolve(K[free][:, free], loads[free])
    U = np.zeros(ndof)
    U[free] = Uf
    return float(loads @ U)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--guide_steps", type=int, default=100)
    ap.add_argument("--guide_lr", type=float, default=0.5)
    ap.add_argument("--guide_iters", type=int, default=100)
    ap.add_argument("--vol_weight", type=float, default=1.0)
    ap.add_argument("--out", default="data/soft_vs_proj_v2.csv")
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
    header = ["sample", "volfrac",
              "uncond_conn", "soft_conn", "refine_conn",
              "uncond_float", "soft_float", "refine_float",
              "uncond_mass", "soft_mass", "refine_mass",
              "uncond_comp", "soft_comp", "refine_comp"]
    rows = []

    print(f"=== P0 界定 vs 拉扯（修正版，配对 seed={args.seed}，{N} 样本） ===")
    print(f"软约束: steps={args.guide_steps}, lr={args.guide_lr}, "
          f"iters={args.guide_iters}, vol_weight={args.vol_weight}")
    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]
        vf = float(conds[i, 0, 0, 0, 0])
        loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            uncond = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s_for_soft = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
        soft = soft_guide(s_for_soft, src, sup, steps=args.guide_steps, lr=args.guide_lr,
                          iters=args.guide_iters, vol_weight=args.vol_weight)

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s_for_ref = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
        refine = diff.refine(s_for_ref, cond, src, sup, K=5, t_refine=30)

        un = uncond[0, 0].numpy()
        sn = soft[0, 0].numpy()
        rn = refine[0, 0].numpy()

        uc, uf, _ = connectivity(un)
        sc, sf, _ = connectivity(sn)
        rc, rf, _ = connectivity(rn)

        um = float((un > 0.5).sum())
        sm = float((sn > 0.5).sum())
        rm = float((rn > 0.5).sum())

        u_comp = compliance(un, loads, supports, ke_flat, iK, jK, ndof)
        s_comp = compliance(sn, loads, supports, ke_flat, iK, jK, ndof)
        r_comp = compliance(rn, loads, supports, ke_flat, iK, jK, ndof)

        rows.append([i, vf, uc, sc, rc, uf, sf, rf, um, sm, rm, u_comp, s_comp, r_comp])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}")

    uc = np.array([r[2] for r in rows])
    sc = np.array([r[3] for r in rows])
    rc = np.array([r[4] for r in rows])
    uf = np.array([r[5] for r in rows])
    sf = np.array([r[6] for r in rows])
    rf = np.array([r[7] for r in rows])
    um = np.array([r[8] for r in rows])
    sm = np.array([r[9] for r in rows])
    rm = np.array([r[10] for r in rows])

    print("\n=== 连通性（均值） ===")
    print(f"无条件:      {uc.mean():.4f}")
    print(f"软约束引导:  {sc.mean():.4f}")
    print(f"投影精炼:    {rc.mean():.4f}")

    print("\n=== 漂浮体比例（均值） ===")
    print(f"无条件:      {uf.mean():.4f}")
    print(f"软约束引导:  {sf.mean():.4f}")
    print(f"投影精炼:    {rf.mean():.4f}")

    print("\n=== 材料量（均值，二值化体素数） ===")
    print(f"无条件:      {um.mean():.1f}")
    print(f"软约束引导:  {sm.mean():.1f}  (相对无条件 {100*(sm-um).mean()/um.mean():+.1f}%)")
    print(f"投影精炼:    {rm.mean():.1f}  (相对无条件 {100*(rm-um).mean()/um.mean():+.1f}%)")

    # 配对 t 检验（scipy ttest_rel，ddof=1 正确口径）
    print("\n=== 配对 t 检验（连通性，scipy ttest_rel） ===")
    for name, a, b in [("软约束引导 vs 无条件", sc, uc),
                       ("投影精炼 vs 无条件", rc, uc),
                       ("投影精炼 vs 软约束引导", rc, sc)]:
        t, p = ttest_rel(a, b)
        wins = (a > b).sum()
        ties = (a == b).sum()
        print(f"{name}: diff={a.mean()-b.mean():+.4f}, t({len(a)-1})={t:.3f}, "
              f"p={p:.6f}, 胜={wins} 平={ties} 负={len(a)-wins-ties}")

    # 柔度（过滤奇异）+ 材料归一化
    u_comp = np.array([r[11] for r in rows])
    s_comp = np.array([r[12] for r in rows])
    r_comp = np.array([r[13] for r in rows])
    mask = (u_comp > 0) & (u_comp < 1e6) & (s_comp > 0) & (s_comp < 1e6) & (r_comp > 0) & (r_comp < 1e6)
    uf_c, sf_c, rf_c = u_comp[mask], s_comp[mask], r_comp[mask]
    um_m, sm_m, rm_m = um[mask], sm[mask], rm[mask]
    print(f"\n=== 柔度（{len(uf_c)} 个有效样本，过滤奇异） ===")
    print(f"无条件:      {uf_c.mean():.2f}")
    print(f"软约束引导:  {sf_c.mean():.2f}")
    print(f"投影精炼:    {rf_c.mean():.2f}")

    # 比柔度 = 柔度 / 材料量（单位材料的柔度，越小越"省材料"）
    print("\n=== 比柔度（柔度/材料量，材料归一化） ===")
    print(f"无条件:      {uf_c.mean()/um_m.mean():.4f}")
    print(f"软约束引导:  {sf_c.mean()/sm_m.mean():.4f}")
    print(f"投影精炼:    {rf_c.mean()/rm_m.mean():.4f}")


if __name__ == "__main__":
    main()
