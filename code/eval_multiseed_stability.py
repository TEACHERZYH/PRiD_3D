"""多 seed 稳定性验证：确认「桥接 vs 重构」的贡献结论是否稳健。

动机：单次 30 样本的 2x2 因子对照（严格噪声配对）显示
    no_bridge conn=1.0000 (30/30)  >  PRiD conn=0.9927 (28/30)
而此前非严格配对的初版显示相反（PRiD 1.0 > no_bridge 0.9667）。
结论随噪声配对翻转 => 必须做多 seed 重复，区分「真实效应」与「噪声方差」。

本脚本对 S 个独立采样种子各跑一遍四格对照，报告：
  - 每个种子的四方法均值与 PRiD-vs-no_bridge 配对差
  - 跨种子汇总（均值 ± std，PRiD 胜/负/平 的种子数）
  - 近空样本统计（occ <= --degenerate_occ 视为退化/平凡连通）

用法:
    python code/eval_multiseed_stability.py \
        --seeds 12000,22000,32000,42000,52000 --n_samples 30 \
        --out data/multiseed_stability.csv --detail_out data/multiseed_detail.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from scipy.stats import ttest_rel

from diffusion.diffusion import UNet3D, Diffusion
from data_generation.simp_solver import hex8_stiffness, build_edof, build_k_index
from eval_bridge_recon_factorial import (
    build_ls, compliance, connectivity_6, METHODS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--seeds", default="12000,22000,32000,42000,52000")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--refine_offset", type=int, default=65000,
                    help="refine 噪声种子 = sample seed + offset")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--t_refine", type=int, default=30)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--degenerate_occ", type=int, default=10,
                    help="occ <= 该值视为退化/近空样本（平凡连通）")
    ap.add_argument("--out", default="data/multiseed_stability.csv")
    ap.add_argument("--detail_out", default="data/multiseed_detail.csv")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

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
    detail_rows = []
    summary_rows = []

    for seed in seeds:
        # 缓存每个样本的 unrefined 场，供四方法共享
        per = {m: {"conn": [], "comp": [], "occ": []} for m in METHODS}
        n_deg = 0

        for i in range(N):
            cond = conds[i:i + 1]
            src = conds[i, 2:3]
            sup = conds[i, 3:4]
            loads, supports = build_ls(conds[i].numpy(), D, H, W, ndof)

            torch.manual_seed(seed + i)
            with torch.no_grad():
                s = diff.sample(cond, (1, 1, D, H, W),
                                ddim_steps=args.ddim_steps, use_projection=False)

            outs = {"unrefined": s}
            with torch.no_grad():
                torch.manual_seed(seed + args.refine_offset + i)
                outs["one_shot"] = diff.refine(s, cond, src, sup, K=1,
                                               t_refine=args.t_refine,
                                               bridge=True, reconstruct=False)
                torch.manual_seed(seed + args.refine_offset + i)
                outs["no_bridge"] = diff.refine(s, cond, src, sup, K=args.K,
                                                t_refine=args.t_refine,
                                                bridge=False, reconstruct=True)
                torch.manual_seed(seed + args.refine_offset + i)
                outs["prid"] = diff.refine(s, cond, src, sup, K=args.K,
                                           t_refine=args.t_refine,
                                           bridge=True, reconstruct=True)

            row = {"seed": seed, "sample": i}
            for m in METHODS:
                arr = outs[m][0, 0].numpy()
                c, _, _ = connectivity_6(arr)
                comp = compliance(arr, loads, supports, ke_flat, iK, jK, ndof)
                occ = int((arr > 0.5).sum())
                per[m]["conn"].append(c)
                per[m]["comp"].append(comp)
                per[m]["occ"].append(occ)
                row[f"{m}_conn"] = c
                row[f"{m}_comp"] = comp
                row[f"{m}_occ"] = occ
            if row["no_bridge_occ"] <= args.degenerate_occ:
                n_deg += 1
            detail_rows.append(row)

        # ---- 单 seed 汇总 ----
        def mean(m, k):
            return float(np.mean(per[m][k]))

        conn_nb = np.array(per["no_bridge"]["conn"])
        conn_pr = np.array(per["prid"]["conn"])
        conn_os = np.array(per["one_shot"]["conn"])
        conn_un = np.array(per["unrefined"]["conn"])
        t_pn, p_pn = ttest_rel(conn_pr, conn_nb)
        t_on, p_on = ttest_rel(conn_os, conn_un)

        valid = np.array([all(0 < per[m]["comp"][j] < 1e6 for m in METHODS)
                          for j in range(N)])
        comp_nb = np.array(per["no_bridge"]["comp"])
        comp_pr = np.array(per["prid"]["comp"])
        if valid.sum() > 1:
            tc, pc = ttest_rel(comp_pr[valid], comp_nb[valid])
        else:
            tc, pc = float("nan"), float("nan")

        summary_rows.append({
            "seed": seed,
            "unrefined_conn": mean("unrefined", "conn"),
            "one_shot_conn": mean("one_shot", "conn"),
            "no_bridge_conn": mean("no_bridge", "conn"),
            "prid_conn": mean("prid", "conn"),
            "n_conn1_nobridge": int((conn_nb > 0.999).sum()),
            "n_conn1_prid": int((conn_pr > 0.999).sum()),
            "prid_minus_nobridge_conn": float((conn_pr - conn_nb).mean()),
            "t_prid_vs_nobridge": t_pn,
            "p_prid_vs_nobridge": p_pn,
            "n_prid_win": int((conn_pr > conn_nb).sum()),
            "n_prid_loss": int((conn_pr < conn_nb).sum()),
            "one_shot_minus_unrefined_conn": float((conn_os - conn_un).mean()),
            "p_oneshot_vs_unrefined": p_on,
            "unrefined_occ": mean("unrefined", "occ"),
            "no_bridge_occ": mean("no_bridge", "occ"),
            "prid_occ": mean("prid", "occ"),
            "n_degenerate": n_deg,
            "prid_minus_nobridge_comp": float((comp_pr[valid] - comp_nb[valid]).mean()) if valid.sum() else float("nan"),
            "p_comp_prid_vs_nobridge": pc,
        })
        print(f"[seed {seed}] unrefined {conn_un.mean():.4f} | one_shot {conn_os.mean():.4f} | "
              f"no_bridge {conn_nb.mean():.4f} ({int((conn_nb>0.999).sum())}/{N}) | "
              f"prid {conn_pr.mean():.4f} ({int((conn_pr>0.999).sum())}/{N}) | "
              f"PRiD-NB Δ{(conn_pr-conn_nb).mean():+.4f} p={p_pn:.3g} "
              f"[胜{int((conn_pr>conn_nb).sum())} 负{int((conn_pr<conn_nb).sum())}] | "
              f"退化样本 {n_deg}")

    # ---- 跨 seed 汇总 ----
    print(f"\n=== 跨 {len(seeds)} 个种子汇总 ===")
    for key in ["unrefined_conn", "one_shot_conn", "no_bridge_conn", "prid_conn",
                "prid_minus_nobridge_conn"]:
        v = np.array([r[key] for r in summary_rows], dtype=float)
        print(f"  {key:28s} {v.mean():.4f} ± {v.std(ddof=1):.4f}  (min {v.min():.4f}, max {v.max():.4f})")
    n_win = sum(r["n_prid_win"] for r in summary_rows)
    n_loss = sum(r["n_prid_loss"] for r in summary_rows)
    total = n_win + n_loss
    print(f"\n  PRiD vs no_bridge 逐样本胜负：PRiD 胜 {n_win} / 负 {n_loss}（共 {total} 个非平局）")
    n_seed_win = sum(1 for r in summary_rows if r["prid_minus_nobridge_conn"] > 0)
    n_seed_loss = sum(1 for r in summary_rows if r["prid_minus_nobridge_conn"] < 0)
    print(f"  种子级：PRiD 更优 {n_seed_win}/{len(seeds)} 个种子，更差 {n_seed_loss}/{len(seeds)}")
    print(f"  退化样本(no_bridge occ<={args.degenerate_occ})总数：{sum(r['n_degenerate'] for r in summary_rows)}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(args.detail_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)
    print(f"\n汇总 CSV: {args.out}\n明细 CSV: {args.detail_out}")


if __name__ == "__main__":
    main()
