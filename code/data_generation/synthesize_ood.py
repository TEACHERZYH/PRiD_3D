"""OOD 测试数据合成：生成分布外的工况样本（用于命题 P3 泛化）。

OOD 维度：
1. 跨体积分数：训练用 vf∈[0.2,0.5]，OOD 用 vf=0.15 和 0.6；
2. 跨设计域：训练用立方体，OOD 用 L 形域（掩码掉一角）；
3. 跨荷载：训练用右下角集中力，OOD 用多点荷载。

用法:
    python code/data_generation/synthesize_ood.py --nx 16 --n_samples 50 --out data/ood_16.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from data_generation.simp_solver import topology_optimize


def synthesize_one(idx, nx, ny, nz, kind):
    """合成一个 OOD 样本。kind: 'volfrac' / 'domain' / 'load'。

    条件通道：0=体积分数, 1=应力, 2=荷载掩码, 3=支座掩码。
    """
    rng = np.random.default_rng(idx)
    n_elem = nx * ny * nz
    nn = (nx + 1) * (ny + 1) * (nz + 1)
    ndof = 3 * nn

    # 默认支座：x=0 左端面
    supports = np.zeros(ndof, dtype=bool)
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            nid = (nx + 1) * (ny + 1) * iz + (nx + 1) * iy
            supports[3 * nid:3 * nid + 3] = True

    loads = np.zeros(ndof)
    domain_mask = None
    load_nodes = []  # 记录荷载节点位置，用于生成 load_mask

    if kind == "volfrac":
        vf = 0.15 if idx % 2 == 0 else 0.6
        node_id = nx + (nx + 1) * ny + (nx + 1) * (ny + 1) * nz
        loads[3 * node_id + 2] = -1.0
        load_nodes.append((nx, ny))
    elif kind == "domain":
        vf = 0.35
        domain_mask = np.ones((nx, ny, nz), dtype=bool)
        domain_mask[nx // 2:, ny // 2:, :] = False  # 去掉右下角
        # 荷载放在 L 形域的合法位置（右上角，y 在上方未被掩码）
        lx, ly = nx - 1, ny // 4
        node_id = lx + (nx + 1) * ly + (nx + 1) * (ny + 1) * nz
        loads[3 * node_id + 2] = -1.0
        load_nodes.append((lx, ly))
    else:  # load
        vf = 0.35
        for i in range(3):
            lx = nx * (i + 1) // 4
            ly = ny * (i + 1) // 4
            nid = lx + (nx + 1) * ly + (nx + 1) * (ny + 1) * nz
            loads[3 * nid + 2] = -1.0
            load_nodes.append((lx, ly))

    rho, _, vm = topology_optimize(
        nx, ny, nz, volfrac=vf, max_iter=40, loads=loads, supports=supports, return_stress=True
    )
    rho = rho.reshape(nx, ny, nz).astype(np.float32)
    vm_norm = vm.reshape(nx, ny, nz).astype(np.float64)
    vm_norm = vm_norm / (vm_norm.max() + 1e-8)

    if domain_mask is not None:
        rho = rho * domain_mask
        vm_norm = vm_norm * domain_mask

    # 4通道条件：体积分数 + 应力 + 荷载掩码 + 支座掩码
    load_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    for (lx, ly) in load_nodes:
        load_mask[min(lx, nx - 1), min(ly, ny - 1), nz - 1] = 1.0
    sup_mask = np.zeros((nx, ny, nz), dtype=np.float32)
    sup_mask[0, :, :] = 1.0

    cond = np.stack(
        [
            np.full((nx, ny, nz), vf, dtype=np.float32),
            vm_norm,
            load_mask,
            sup_mask,
        ],
        axis=0,
    )
    return cond, rho[None, ...], kind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--out", type=str, default="data/ood_16.npz")
    args = parser.parse_args()
    nx = args.nx
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    kinds = ["volfrac", "domain", "load"]
    per_kind = args.n_samples // len(kinds)
    conds, labels, kind_labels = [], [], []
    print(f"[OOD] 合成 {per_kind}x3 = {per_kind*3} 样本")

    for i in range(per_kind):
        for k in kinds:
            cond, label, kind = synthesize_one(i, nx, nx, nx, k)
            conds.append(cond)
            labels.append(label)
            kind_labels.append(k)

    conds = np.stack(conds)
    labels = np.stack(labels)
    np.savez_compressed(args.out, conds=conds, labels=labels, kinds=np.array(kind_labels))
    print(f"[OOD] 完成，保存到 {args.out}")


if __name__ == "__main__":
    main()
