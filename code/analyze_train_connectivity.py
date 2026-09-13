"""统计训练集（SIMP 标签）本身的连通性分布。

关键问题：no_bridge（纯重构）能达到 30/30 全连通，是因为
  (a) 训练数据本身就是连通的（去噪器只是学到了数据分布的属性），还是
  (b) 去噪器修复了训练数据里存在的缺陷？
本脚本用与评估完全相同的口径（6 邻域、阈值 0.5）统计 train 集的
连通性 / 浮材率 / 占用体素，以界定实验结论的适用范围。

用法: python code/analyze_train_connectivity.py --data data/train_16.npz
"""
import argparse

import numpy as np
from scipy.ndimage import label, generate_binary_structure


def conn_stats(binary):
    total = int(binary.sum())
    if total == 0:
        return 0.0, 1.0, 0
    lab, n = label(binary, structure=generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max())
    return largest / total, 1.0 - largest / total, int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_16.npz")
    ap.add_argument("--n", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    d = np.load(args.data)
    labels = d["labels"]
    conds = d.get("conds")
    N = len(labels) if args.n <= 0 else min(args.n, len(labels))
    print(f"样本数 {N}（总 {len(labels)}）| labels shape {labels.shape}")

    conns, floats, occs, ncomp = [], [], [], []
    for i in range(N):
        rho = labels[i, 0]
        binary = rho > 0.5
        c, f, nc = conn_stats(binary)
        conns.append(c)
        floats.append(f)
        occs.append(int(binary.sum()))
        ncomp.append(nc)

    conns = np.array(conns)
    floats = np.array(floats)
    occs = np.array(occs)
    ncomp = np.array(ncomp)

    print("\n=== 训练集（SIMP 标签）连通性（6 邻域，阈值 0.5）===")
    print(f"  连通性  均值 {conns.mean():.4f} ± {conns.std(ddof=1):.4f}  "
          f"min {conns.min():.4f}  max {conns.max():.4f}")
    print(f"  =1.0 的样本      {int((conns > 0.999).sum())}/{N}")
    print(f"  <0.99 的样本     {int((conns < 0.99).sum())}/{N}")
    print(f"  <0.95 的样本     {int((conns < 0.95).sum())}/{N}")
    print(f"  浮材率(体素口径) 均值 {floats.mean():.4f}  中位 {np.median(floats):.4f}  max {floats.max():.4f}")
    print(f"  连通分量数       均值 {ncomp.mean():.2f}  max {ncomp.max()}")
    print(f"  占用体素         均值 {occs.mean():.1f}  min {occs.min()}  max {occs.max()}")

    if conds is not None:
        vf = conds[:, 0, 0, 0, 0]
        print(f"\n  体积分数（条件通道0）均值 {vf[:N].mean():.4f} 范围 [{vf[:N].min():.3f}, {vf[:N].max():.3f}]")

    print("\n=== 与评估结果对比 ===")
    print(f"  训练标签连通性        {conns.mean():.4f}")
    print(f"  unrefined（采样）     0.9348")
    print(f"  no_bridge（只重构）   0.9798")
    print(f"  PRiD（桥接+重构）     0.9967")

    # 小分量规模分布（判断浮材是"真孤岛"还是"1-2 体素噪声"）
    small = 0
    for i in range(N):
        rho = labels[i, 0] > 0.5
        if rho.sum() == 0:
            continue
        lab, n = label(rho, structure=generate_binary_structure(3, 1))
        sizes = np.bincount(lab.ravel())[1:]
        if len(sizes) > 1:
            small += int((np.sort(sizes)[:-1] <= 4).sum())
    print(f"\n  训练标签中 ≤4 体素的小分量总数：{small}")


if __name__ == "__main__":
    main()
