"""数据合成管线：批量生成 (条件, 密度场标签) 训练样本。

零数据采集的核心：随机采样工况 → SIMP 拓扑优化 → 配对成样本。

样本结构：
    condition: [密度场初始化/工况编码, 应力场] 等通道
    label: 优化后的密度场 (1, D, H, W)
"""

from __future__ import annotations

import torch

from data_generation.simp_solver import topology_optimize


def synthesize_batch(
    n_samples: int,
    nx: int,
    ny: int,
    nz: int,
    volfrac_range=(0.2, 0.5),
    device: str = "cpu",
    verbose: bool = False,
):
    """生成一批 (condition, label) 样本。

    返回:
        conds: (n_samples, 2, nx, ny, nz) —— 通道0: 体积分数标量场；通道1: 应力场
        labels: (n_samples, 1, nx, ny, nz)
    """
    conds = torch.zeros(n_samples, 2, nx, ny, nz, dtype=torch.float32)
    labels = torch.zeros(n_samples, 1, nx, ny, nz, dtype=torch.float32)

    for i in range(n_samples):
        vf = volfrac_range[0] + (volfrac_range[1] - volfrac_range[0]) * torch.rand(1).item()
        rho, _, von_mises = topology_optimize(
            nx, ny, nz, volfrac=vf, max_iter=80, device=device,
            verbose=False, return_stress=True,
        )
        rho = rho.reshape(nx, ny, nz)
        # 归一化 von Mises 应力到 [0,1]（作为物理特征通道）
        vm = von_mises.reshape(nx, ny, nz)
        vm_norm = vm / (vm.max() + 1e-8)

        # 通道0：体积分数标量场（工况条件编码）
        conds[i, 0] = torch.full((nx, ny, nz), vf)
        # 通道1：归一化 von Mises 应力场（真实物理特征）
        conds[i, 1] = vm_norm.float()
        labels[i, 0] = rho.float()

        if verbose and (i + 1) % 10 == 0:
            print(f"generated {i+1}/{n_samples}")

    return conds, labels


if __name__ == "__main__":
    conds, labels = synthesize_batch(2, nx=8, ny=8, nz=8, verbose=True)
    print("conds shape:", conds.shape, "labels shape:", labels.shape)
    print("label value range:", labels.min().item(), labels.max().item())
