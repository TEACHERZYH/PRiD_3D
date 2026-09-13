"""可行性吸引子（feasibility attractor）与去噪轨迹可行性演化实验。

支撑两条命题：
  P2（吸引子）：从一个被破坏的可行结构出发，加噪到 t0 再去噪，动力学能否
      把不可行态拉回可行域？扫描 t0 得到"保真-恢复"权衡曲线。
  P1（内生性，直接证据）：自然采样轨迹上，x0 估计的可行性如何随去噪推进演化。

设计要点：
  * 可行起点 x_feas = 模型自身输出的重构结果（bridge=False, K=5），保证属于模型分布；
  * 破坏方式 = 沿三轴搜索"使 6 邻域连通性最低的单层切片"并删除，得到确定的断裂；
  * 恢复 = q_sample 加噪到 t0 → DDIM 从 t0 去噪回 0；
  * 严格可复现：每个样本固定 seed，不同 t0 从同一噪声序列起点出发。

用法:
  python code/eval_feasibility_projection.py --mode attractor  --n_samples 30
  python code/eval_feasibility_projection.py --mode trajectory --n_samples 10
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from scipy.ndimage import label as connected_components, generate_binary_structure

from diffusion.diffusion import UNet3D, Diffusion


def conn_stats(binary):
    """6 邻域最大连通分量占比 (connectivity, floating_fraction, n_components)。"""
    total = int(binary.sum())
    if total == 0:
        return 0.0, 1.0, 0
    lab, n = connected_components(binary, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max())
    return largest / total, 1.0 - largest / total, int(n)


def find_worst_break(rho, thresh=0.5, margin=3):
    """沿三个轴搜索使连通性最低的单层切片并删除。

    返回 (axis, layer, rho_broken, conn_broken, occ_broken)。
    """
    best = None
    for axis in range(3):
        n = rho.shape[axis]
        for L in range(margin, n - margin):
            x = rho.copy()
            if axis == 0:
                x[L, :, :] = 0.0
            elif axis == 1:
                x[:, L, :] = 0.0
            else:
                x[:, :, L] = 0.0
            b = x > thresh
            if b.sum() == 0:
                continue
            c, _, _ = conn_stats(b)
            if best is None or c < best[3]:
                best = (axis, L, x, c, int(b.sum()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["attractor", "trajectory"], default="attractor")
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--ckpt", default="results/checkpoint_16.pt")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12000)
    ap.add_argument("--refine_seed", type=int, default=77000)
    ap.add_argument("--rec_seed", type=int, default=31000, help="加噪/去噪的配对种子")
    ap.add_argument("--t0_list", default="20,40,60,90,120,160")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--t_refine", type=int, default=30)
    ap.add_argument("--ddim_steps", type=int, default=20)
    ap.add_argument("--min_feas_conn", type=float, default=0.99,
                    help="可行起点要求的最低连通性（否则跳过该样本）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    data = np.load(args.data)
    conds = torch.from_numpy(data["conds"]).float()
    D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]

    model = UNet3D(in_channels=5, base=32, depth=2)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    diff = Diffusion(model, timesteps=200)

    N = min(args.n_samples, conds.shape[0])
    shape = (1, 1, D, H, W)

    if args.mode == "trajectory":
        rows = []
        for i in range(N):
            cond = conds[i:i + 1]
            trace = []

            def fn(step, x0_pred, t, _tr=trace):
                arr = x0_pred[0, 0].detach().cpu().numpy()
                c, f, n = conn_stats(arr > 0.5)
                _tr.append((step, t, c, f, int((arr > 0.5).sum())))

            torch.manual_seed(args.seed + i)
            with torch.no_grad():
                diff.sample(cond, shape, ddim_steps=args.ddim_steps, trace_fn=fn)
            for step, t, c, f, occ in trace:
                rows.append([i, step, t, c, f, occ])
            print(f"[sample {i}] 起点 conn={rows[-len(trace)][3]:.4f} -> 终点 conn={trace[-1][2]:.4f}")

        out = args.out or "data/trajectory_feasibility.csv"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sample", "step", "t", "conn", "floating", "occ"])
            w.writerows(rows)

        arr = np.array([[r[3] for r in rows if r[0] == i] for i in range(N)], dtype=float)
        print(f"\n=== 轨迹可行性演化（{N} 样本，{arr.shape[1]} 步）===")
        print(f"{'step':>5} {'t':>5} {'conn均值':>10} {'conn中位':>10}")
        for s in range(arr.shape[1]):
            t_val = [r[2] for r in rows if r[0] == 0][s]
            print(f"{s:>5} {t_val:>5} {arr[:, s].mean():>10.4f} {np.median(arr[:, s]):>10.4f}")
        print(f"\nCSV: {out}")
        return

    # ---------------- attractor ----------------
    t0_list = [int(x) for x in args.t0_list.split(",") if x.strip()]
    rows = []
    skipped = 0
    for i in range(N):
        cond = conds[i:i + 1]
        src = conds[i, 2:3]
        sup = conds[i, 3:4]

        torch.manual_seed(args.seed + i)
        with torch.no_grad():
            s = diff.sample(cond, shape, ddim_steps=args.ddim_steps)
            torch.manual_seed(args.refine_seed + i)
            x_feas = diff.refine(s, cond, src, sup, K=args.K,
                                 t_refine=args.t_refine,
                                 bridge=False, reconstruct=True)
        rho_feas = x_feas[0, 0].numpy()
        c_feas, f_feas, _ = conn_stats(rho_feas > 0.5)
        if c_feas < args.min_feas_conn:
            skipped += 1
            print(f"[sample {i}] 可行起点连通性 {c_feas:.4f} < {args.min_feas_conn}，跳过")
            continue

        axis, L, rho_broken, c_broken, occ_broken = find_worst_break(rho_feas)
        x_broken = torch.from_numpy(rho_broken[None, None]).float()

        for t0 in t0_list:
            torch.manual_seed(args.rec_seed + i)
            with torch.no_grad():
                x_noisy, _ = diff.q_sample(x_broken, torch.tensor([t0]))
                torch.manual_seed(args.rec_seed + i)
                x_rec = diff.sample(cond, shape, ddim_steps=args.ddim_steps,
                                    x_init=x_noisy, t_start=t0)
            rho_rec = x_rec[0, 0].numpy()
            c_rec, f_rec, n_rec = conn_stats(rho_rec > 0.5)
            occ_rec = int((rho_rec > 0.5).sum())
            denom = c_feas - c_broken
            gain = (c_rec - c_broken) / denom if abs(denom) > 1e-6 else float("nan")
            rows.append([i, t0, axis, L, c_feas, c_broken, c_rec,
                         f_feas, f_rec, n_rec, occ_broken, occ_rec, gain])
        print(f"[sample {i}] feas {c_feas:.4f} | broken {c_broken:.4f} (axis{axis} L{L}) | "
              f"rec " + " ".join(f"t0={r[1]}:{r[6]:.3f}" for r in rows if r[0] == i))

    out = args.out or "data/feasibility_attractor.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "t0", "break_axis", "break_layer", "conn_feas", "conn_broken",
                    "conn_rec", "floating_feas", "floating_rec", "ncomp_rec",
                    "occ_broken", "occ_rec", "recovery_gain"])
        w.writerows(rows)

    print(f"\n=== 可行性吸引子（{len(rows)} 条记录，跳过 {skipped} 个样本）===")
    print(f"{'t0':>5} {'conn_broken':>12} {'conn_rec':>10} {'恢复率':>8} {'恢复度':>8} {'浮材率':>8}")
    for t0 in t0_list:
        sub = [r for r in rows if r[1] == t0]
        if not sub:
            continue
        cb = np.mean([r[5] for r in sub])
        cr = np.mean([r[6] for r in sub])
        rate = np.mean([1.0 if r[6] > r[5] + 1e-9 else 0.0 for r in sub])
        gain = np.nanmean([r[12] for r in sub])
        fl = np.mean([r[8] for r in sub])
        print(f"{t0:>5} {cb:>12.4f} {cr:>10.4f} {rate:>8.2f} {gain:>8.3f} {fl:>8.4f}")

    # 与"自然采样输出"对照（不需要加噪的基线）
    base = [r for r in rows if r[1] == t0_list[0]]
    print(f"\n参考：可行起点连通性均值 {np.mean([r[4] for r in base]):.4f}")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
