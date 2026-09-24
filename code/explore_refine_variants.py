"""改进方案探索：在同一个采样输出上对比一组精炼变体，找更优的可行性/性能折中。

动机（现状的弱点）：
  * 只重构（no-bridge）柔度最好（−56%）但**会塌缩**（450 样本里 13 个近空、7 个全空，
    worst-case C 0.4681）；
  * 桥接+重构（PRiD）永不塌缩，但对柔度有**稳定小幅损害**（+0.0152，15/15 seed 同号）；
  * 于是"零塌缩"和"柔度最优"目前不可兼得 —— 这正是可改进的地方。

探索的变体（全部作用在**同一个采样输出**上，严格配对）：
  unref        原始采样（基准）
  recon30      只重构 K=5, t_r=30（论文主力配方）
  prid         桥接+重构 K=5, t_r=30（论文完整循环）
  sel_occ      **选择性修复**：仅当占用 < 0.5·V0·N 时才桥接，然后重构 K=5
  sel_conn     **选择性修复**：仅当连通性 < 0.8 时才桥接，然后重构 K=5
  anneal       退火重构：t_r 从 120 线性降到 30（保真-性能折中的改进尝试）
  volcons      体积约束重构：每轮把密度质量重标定到目标 V0·N（抑制材料漂移）
  recon120     只重构 t_r=120（性能上界参照）

评价：C / F / 占用 / IoU（相对原始采样）/ 柔度 / 塌缩标记。
目标：在不增加塌缩的前提下降低柔度；或在同柔度下提高 C 与 IoU。

用法（远程 CPU）:
    python code/explore_refine_variants.py --n_samples 30 --seeds 12000,22000,32000 \
        --out data/explore_variants.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label as cc

from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index
from eval_bridge_recon_factorial import build_ls, compliance


def conn_stats(binary):
    total = int(binary.sum())
    if total == 0:
        return 0.0, 1.0
    lab, _ = cc(binary, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max()) if sizes.size else 0
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
def run_variant(diff, x0, cond, src, sup, name, budget):
    """在给定采样输出 x0 上执行一个精炼变体，返回处理后的密度场。"""
    x = x0
    if name == "unref":
        return x

    if name == "recon30":
        for _ in range(5):
            x = denoise_cycle(diff, x, cond, 30)
        return x

    if name == "prid":
        for _ in range(5):
            x = manifold_projection(x, source_mask=src, support_mask=sup)
            x = denoise_cycle(diff, x, cond, 30)
        return x

    if name in ("sel_occ", "sel_conn"):
        xn = x[0, 0].numpy()
        occ = int((xn > 0.5).sum())
        c, _ = conn_stats(xn > 0.5)
        need = (occ < 0.5 * budget) if name == "sel_occ" else (c < 0.8)
        if need:                      # 只在"确实坏"的样本上付出桥接代价
            x = manifold_projection(x, source_mask=src, support_mask=sup)
        for _ in range(5):
            x = denoise_cycle(diff, x, cond, 30)
        return x

    if name == "anneal":
        ts = np.linspace(120, 30, 5).round().astype(int)
        for t_k in ts:
            x = denoise_cycle(diff, x, cond, int(t_k))
        return x

    if name == "volcons":
        for _ in range(5):
            x = denoise_cycle(diff, x, cond, 30)
            m = float(x.sum())
            if m > 0:
                x = torch.clamp(x * (budget / m), 0.0, 1.0)
        return x

    if name in ("volcons60", "volcons120"):
        t_r = 60 if name == "volcons60" else 120
        for _ in range(5):
            x = denoise_cycle(diff, x, cond, t_r)
            m = float(x.sum())
            if m > 0:
                x = torch.clamp(x * (budget / m), 0.0, 1.0)
        return x

    if name == "volcons_anneal":
        ts = np.linspace(120, 30, 5).round().astype(int)
        for t_k in ts:
            x = denoise_cycle(diff, x, cond, int(t_k))
            m = float(x.sum())
            if m > 0:
                x = torch.clamp(x * (budget / m), 0.0, 1.0)
        return x

    if name == "volcons_prid":
        for _ in range(5):
            x = manifold_projection(x, source_mask=src, support_mask=sup)
            x = denoise_cycle(diff, x, cond, 30)
            m = float(x.sum())
            if m > 0:
                x = torch.clamp(x * (budget / m), 0.0, 1.0)
        return x

    if name == "volcons_sel":
        xn = x[0, 0].numpy()
        if conn_stats(xn > 0.5)[0] < 0.8:
            x = manifold_projection(x, source_mask=src, support_mask=sup)
        for _ in range(5):
            x = denoise_cycle(diff, x, cond, 30)
            m = float(x.sum())
            if m > 0:
                x = torch.clamp(x * (budget / m), 0.0, 1.0)
        return x

    if name == "recon120":
        for _ in range(5):
            x = denoise_cycle(diff, x, cond, 120)
        return x

    raise ValueError(name)


VARIANTS = ["unref", "recon30", "prid", "sel_occ", "sel_conn", "anneal",
            "volcons", "volcons60", "volcons120", "volcons_anneal",
            "volcons_prid", "volcons_sel", "recon120"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seeds", default="12000,22000,32000")
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--variants", default="",
                    help="逗号分隔，只跑指定变体（留空=全部）。用于大规模 seed 确认")
    ap.add_argument("--refine_seed", type=int, default=77000,
                    help="精炼阶段噪声种子基数：**每个变体前都重置**，"
                         "使各变体经历完全相同的精炼噪声（否则结果会依赖变体的执行顺序）")
    ap.add_argument("--out", default="data/explore_variants.csv")
    args = ap.parse_args()

    global VARIANTS
    if args.variants:
        want = [v.strip() for v in args.variants.split(",") if v.strip()]
        bad = [v for v in want if v not in VARIANTS]
        if bad:
            raise SystemExit(f"未知变体: {bad}；可选 {VARIANTS}")
        VARIANTS = want

    seeds = [int(s) for s in args.seeds.split(",")]
    d = np.load(args.data)
    conds_np = d["conds"]
    D, H, W = conds_np.shape[2], conds_np.shape[3], conds_np.shape[4]
    Nvox = D * H * W
    shape = (1, 1, D, H, W)

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

    rows = []
    print(f"=== 改进方案探索：{len(seeds)} seed × {N} 样本 × {len(VARIANTS)} 变体 ===",
          flush=True)
    for seed in seeds:
        for i in range(N):
            cond = torch.from_numpy(conds_np[i:i + 1]).float()
            src = cond[:, 2:3]
            sup = cond[:, 3:4]
            v0 = float(conds_np[i, 0, 0, 0, 0])
            budget = v0 * Nvox
            loads, supports = build_ls(conds_np[i], D, H, W, ndof)

            torch.manual_seed(seed + i)
            with torch.no_grad():
                x0 = diff.sample(cond, shape, ddim_steps=args.ddim_steps)
            base = x0[0, 0].numpy()
            base_b = base > 0.5

            for name in VARIANTS:
                # 关键：每个变体前把 RNG 重置到同一状态，使精炼阶段的噪声完全相同。
                # 否则 denoise_cycle 里的 q_sample 会消耗全局 RNG，变体结果将依赖
                # 它在列表中的位置（2026-09-17 由校验脚本发现该缺陷）。
                torch.manual_seed(args.refine_seed + i)
                xv = run_variant(diff, x0, cond, src, sup, name, budget)
                f = xv[0, 0].numpy()
                b = f > 0.5
                c, fl = conn_stats(b)
                inter = int((b & base_b).sum())
                union = int((b | base_b).sum())
                iou = inter / union if union else 0.0
                rows.append([
                    seed, i, name, v0, c, fl, int(b.sum()), iou,
                    compliance(f, loads, supports, ke_flat, iK, jK, ndof),
                ])
            if (i + 1) % 5 == 0:
                print(f"  seed={seed} {i+1}/{N}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "sample", "variant", "volfrac", "conn", "floating",
                    "occ", "iou_vs_unref", "comp"])
        w.writerows(rows)
    print(f"CSV 已保存: {args.out}（{len(rows)} 行）", flush=True)

    # ---- 汇总 ----
    def col(name, idx):
        return np.array([r[idx] for r in rows if r[2] == name])

    print("\n" + "=" * 96)
    print(f"{'变体':<10}{'C':>9}{'sd(C)':>8}{'worst C':>9}{'F':>8}{'占用':>8}"
          f"{'IoU':>8}{'柔度':>9}{'近空':>6}{'全空':>6}")
    print("-" * 96)
    for name in VARIANTS:
        c = col(name, 4)
        comp = col(name, 8)
        ok = np.isfinite(comp) & (comp > 0)
        occ = col(name, 6)
        per_seed_min = [min(col(name, 4)[k::N]) for k in range(len(seeds))]
        print(f"{name:<10}{c.mean():>9.4f}{c.std(ddof=1):>8.4f}"
              f"{np.mean(per_seed_min):>9.4f}{col(name,5).mean():>8.4f}"
              f"{occ.mean():>8.1f}{col(name,7).mean():>8.3f}"
              f"{comp[ok].mean() if ok.any() else float('nan'):>9.2f}"
              f"{int((occ <= 50).sum()):>6}{int((occ == 0).sum()):>6}")
    print("=" * 96)
    print("近空 = occ ≤ 50；全空 = occ = 0；worst C = 「每 seed 内最小 C」再跨 seed 平均")


if __name__ == "__main__":
    main()
