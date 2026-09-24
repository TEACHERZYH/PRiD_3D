"""体积重标定精炼的细粒度扫描：把「一行后处理」逼近 SIMP 参考解。

背景：在 OOD 留出集上，体积重标定（K=5 轮、t_r=30、每轮把密度质量重标定到 V0·N）
给出柔度 2.99，而 SIMP 参考解（同工况标签）柔度 2.63 —— 仍有 ~13% 空间。
本脚本扫描三个旋钮，找最低柔度的配置：
  * 重标定系数 f：目标质量 = f·V0·N（f=0.8..1.2，找最优材料预算）
  * 循环次数 K（5 / 10）
  * 重构噪声水平 t_r（30 / 50 / 80）

变体：volcons_f{0.8,0.9,1.0,1.1,1.2} / volcons_K10 / volcons_tr{50,80}，
加 unref、recon30 作基线。评价：C / F / 占用 / IoU / 柔度，并与 SIMP 参考柔度对比。

用法:
    python code/explore_budget_refinement.py --data data/ood_16.npz \
        --n_samples 30 --seeds 12000,22000,32000,42000,52000 --out data/ood_budget_sweep.csv
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
from eval_bridge_recon_factorial import build_ls
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


def compliance_cg(rho, loads, supports, ke_flat, iK, jK, ndof, maxiter=800):
    """CG 求解柔度。32³ 下 spsolve(LU) 因 3D 带宽过大而爆；
    CG 虽难收敛到 rtol=1e-6（病态），但柔度(loads@U)早已稳定到 ~2e-5。
    16³ 与 spsolve 对拍相对差 2.5e-5。"""
    E = 1e-9 + (1 - 1e-9) * rho.ravel() ** 3
    vals = (ke_flat[None, :] * E[:, None]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
    free = ~supports
    Uf, _ = cg(K[free][:, free], loads[free], rtol=1e-6, maxiter=maxiter)
    U = np.zeros(ndof)
    U[free] = Uf
    return float(loads @ U)

VARIANTS = ["unref", "recon30",
            "volcons_f0.8", "volcons_f0.9", "volcons_f1.0",
            "volcons_f1.1", "volcons_f1.2",
            "volcons_K10", "volcons_tr50", "volcons_tr80"]


def conn_stats(binary):
    total = int(binary.sum())
    if total == 0:
        return 0.0, 1.0
    lab, _ = cc(binary, structure=generate_binary_structure(3, 1))
    sz = np.bincount(lab.ravel())[1:]
    largest = int(sz.max()) if sz.size else 0
    return largest / total, 1.0 - largest / total


@torch.no_grad()
def denoise_cycle(diff, x, cond, t_refine):
    t = torch.tensor([t_refine], device=x.device)
    x_t, _ = diff.q_sample(x, t)
    eps = diff.model(torch.cat([x_t, cond], dim=1), t)
    a = diff.sqrt_alphas_cumprod[t_refine]
    b = diff.sqrt_one_minus[t_refine]
    return torch.clamp((x_t - b * eps) / a, 0.0, 1.0)


@torch.no_grad()
def run_variant(diff, x0, cond, name, budget):
    if name == "unref":
        return x0
    if name == "recon30":
        x = x0
        for _ in range(5):
            x = denoise_cycle(diff, x, cond, 30)
        return x
    # 体积重标定系列
    K = 5
    t_r = 30
    f = 1.0
    if name.startswith("volcons_f"):
        f = float(name.split("f")[1])
    elif name == "volcons_K10":
        K = 10
    elif name == "volcons_tr50":
        t_r = 50
    elif name == "volcons_tr80":
        t_r = 80
    x = x0
    for _ in range(K):
        x = denoise_cycle(diff, x, cond, t_r)
        m = float(x.sum())
        if m > 0:
            x = torch.clamp(x * (f * budget / m), 0.0, 1.0)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ood_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seeds", default="12000,22000,32000,42000,52000")
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--device", default="cpu", help="采样/去噪设备（32³ 建议 cuda）")
    ap.add_argument("--base", type=int, default=32, help="模型 base 通道（须与 checkpoint 一致）")
    ap.add_argument("--depth", type=int, default=2, help="模型 depth（须与 checkpoint 一致）")
    ap.add_argument("--out", default="data/ood_budget_sweep.csv")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[budget] device = {device}", flush=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    d = np.load(args.data)
    conds_np = d["conds"]
    labels = d["labels"]
    D, H, W = conds_np.shape[2], conds_np.shape[3], conds_np.shape[4]
    Nvox = D * H * W
    shape = (1, 1, D, H, W)

    model = UNet3D(in_channels=5, base=args.base, depth=args.depth)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()
    diff = Diffusion(model, timesteps=200)
    diff.to(device)

    ke = hex8_stiffness(1.0, 0.3)
    ke_flat = ke.ravel()
    edof = build_edof(D, H, W)
    iK, jK = build_k_index(edof)
    ndof = 3 * (D + 1) * (H + 1) * (W + 1)
    N = min(args.n_samples, conds_np.shape[0])

    rows = []
    print(f"=== 体积重标定扫描：{len(seeds)} seed × {N} 样本 × {len(VARIANTS)} 变体 ===",
          flush=True)
    for seed in seeds:
        for i in range(N):
            cond = torch.from_numpy(conds_np[i:i + 1]).float().to(device)
            v0 = float(conds_np[i, 0, 0, 0, 0])
            budget = v0 * Nvox
            loads, supports = build_ls(conds_np[i], D, H, W, ndof)
            simp_ref = compliance_cg(labels[i, 0], loads, supports,
                                  ke_flat, iK, jK, ndof)

            torch.manual_seed(seed + i)
            with torch.no_grad():
                x0 = diff.sample(cond, shape, ddim_steps=args.ddim_steps)
            base = x0[0, 0].cpu().numpy()
            base_b = base > 0.5

            for name in VARIANTS:
                torch.manual_seed(77000 + i)   # 每变体前重置，保证噪声一致
                xv = run_variant(diff, x0, cond, name, budget)
                f = xv[0, 0].cpu().numpy()
                b = f > 0.5
                c, fl = conn_stats(b)
                inter = int((b & base_b).sum())
                union = int((b | base_b).sum())
                iou = inter / union if union else 0.0
                comp = compliance_cg(f, loads, supports, ke_flat, iK, jK, ndof)
                rows.append([seed, i, name, v0, c, fl, int(b.sum()), iou, comp,
                             simp_ref])
            if (i + 1) % 10 == 0:
                print(f"  seed={seed} {i+1}/{N}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "sample", "variant", "volfrac", "conn", "floating",
                    "occ", "iou_vs_unref", "comp", "simp_ref_comp"])
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}（{len(rows)} 行）", flush=True)

    print(f"\n{'变体':<14}{'C':>9}{'F':>8}{'占用':>8}{'IoU':>7}{'柔度':>8}"
          f"{'vsSIMP':>8}{'近空':>6}")
    print("-" * 70)
    for name in VARIANTS:
        r = [x for x in rows if x[2] == name]
        c = np.array([x[4] for x in r])
        fl = np.array([x[5] for x in r])
        occ = np.array([x[6] for x in r])
        iou = np.array([x[7] for x in r])
        comp = np.array([x[8] for x in r])
        ref = np.array([x[9] for x in r])
        ok = (comp > 0) & (comp < 1e6)
        ratio = np.mean(comp[ok] / ref[ok])
        print(f"{name:<14}{c.mean():>9.4f}{fl.mean():>8.4f}{occ.mean():>8.1f}"
              f"{iou.mean():>7.3f}{comp[ok].mean():>8.3f}{ratio:>7.2f}x"
              f"{int((occ <= 50).sum()):>6}")
    print("-" * 70)
    print("vsSIMP = 柔度 / SIMP参考柔度（越小越好；1.0 = 追平参考）")


if __name__ == "__main__":
    main()
