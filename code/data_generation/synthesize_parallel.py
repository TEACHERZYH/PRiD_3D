"""并行数据合成脚本：用多进程批量生成 (条件, 标签) 训练样本。

数据合成是 CPU 密集 + 天然并行的（每个样本独立），用 multiprocessing
在远程多个 CPU 核上并行，大幅缩短合成时间。

用法:
    python code/data_generation/synthesize_parallel.py \
        --nx 16 --ny 16 --nz 16 --n_samples 300 --n_proc 8 \
        --out data/train_16.npz
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from data_generation.simp_solver import topology_optimize


def _worker(args):
    """单个样本的合成（子进程执行）。

    随机工况：荷载位置随机（顶面），支座固定 x=0 面。
    条件通道：0=体积分数标量场，1=归一化应力场，2=荷载掩码，3=支座掩码。
    """
    idx, nx, ny, nz, volfrac_range, max_iter = args
    rng = np.random.default_rng(idx)
    vf = volfrac_range[0] + (volfrac_range[1] - volfrac_range[0]) * rng.random()

    n_elem = nx * ny * nz
    nn = (nx + 1) * (ny + 1) * (nz + 1)
    ndof = 3 * nn

    # 随机荷载：顶面（z=nz）随机节点，-z 集中力
    lx = rng.integers(0, nx + 1)
    ly = rng.integers(0, ny + 1)
    load_node = lx + (nx + 1) * ly + (nx + 1) * (ny + 1) * nz
    loads = np.zeros(ndof)
    loads[3 * load_node + 2] = -1.0

    # 支座：固定 x=0 左端面
    supports = np.zeros(ndof, dtype=bool)
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            nid = (nx + 1) * (ny + 1) * iz + (nx + 1) * iy
            supports[3 * nid:3 * nid + 3] = True

    rho, _, von_mises = topology_optimize(
        nx, ny, nz, volfrac=vf, max_iter=max_iter,
        loads=loads, supports=supports, return_stress=True,
    )
    rho = rho.reshape(nx, ny, nz).astype(np.float32)
    vm = von_mises.reshape(nx, ny, nz).astype(np.float32)
    vm_norm = vm / (vm.max() + 1e-8)

    # 工况掩码通道
    load_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    # 荷载节点坐标映射到单元（节点比单元多一个，越界时钳制到边界单元）
    load_mask[min(lx, nx - 1), min(ly, ny - 1), nz - 1] = 1.0
    sup_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    sup_mask[0, :, :] = 1.0  # 支座面

    cond = np.stack(
        [
            np.full((nx, ny, nz), vf, dtype=np.float32),
            vm_norm,
            load_mask,
            sup_mask,
        ],
        axis=0,
    )
    label = rho[None, ...]
    return idx, cond, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=300)
    parser.add_argument("--n_proc", type=int, default=8)
    parser.add_argument("--volfrac_min", type=float, default=0.2)
    parser.add_argument("--volfrac_max", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--out", type=str, default="data/train.npz")
    args = parser.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    volfrac_range = (args.volfrac_min, args.volfrac_max)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"[synthesize] 并行合成 {args.n_samples} 样本, {args.n_proc} 进程, 网格 {nx}x{ny}x{nz}")
    t0 = time.time()

    conds = np.zeros((args.n_samples, 4, nx, ny, nz), dtype=np.float32)
    labels = np.zeros((args.n_samples, 1, nx, ny, nz), dtype=np.float32)

    tasks = [(i, nx, ny, nz, volfrac_range, args.max_iter) for i in range(args.n_samples)]

    if args.n_proc > 1:
        from multiprocessing import Pool
        with Pool(args.n_proc) as pool:
            done = 0
            for idx, cond, label in pool.imap_unordered(_worker, tasks, chunksize=4):
                conds[idx] = cond
                labels[idx] = label
                done += 1
                if done % 50 == 0:
                    print(f"  ... {done}/{args.n_samples} ({time.time()-t0:.0f}s)")
    else:
        for task in tasks:
            idx, cond, label = _worker(task)
            conds[idx] = cond
            labels[idx] = label

    np.savez_compressed(args.out, conds=conds, labels=labels)
    print(f"[synthesize] 完成: {args.n_samples} 样本, 耗时 {time.time()-t0:.1f}s, 保存到 {args.out}")


if __name__ == "__main__":
    main()
