"""可微 FEM 柔度计算（PyTorch autograd 包 SIMP）。

首版采用"PyTorch autograd 包 SIMP 解析灵敏度"的方式实现可微柔度：
- 复用 data_generation/simp_solver.py 的单元刚度与组装逻辑；
- 把柔度 c = U^T K U 写成 x（密度场）的可微函数；
- 训练时作为合法流形上的最优性度量（命题 P2 的核心）。

后续 M2 里程碑可评估切换 JAX-SSO 做真正的可微有限元。
"""

from __future__ import annotations

import torch

from data_generation.simp_solver import hex8_stiffness


def assemble_edof(nx: int, ny: int, nz: int, device: str = "cpu") -> torch.Tensor:
    """返回 (nx*ny*nz, 24) 的单元自由度映射。"""
    edof_mat = torch.zeros(nx * ny * nz, 24, dtype=torch.long, device=device)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                el = iz * (nx * ny) + iy * nx + ix
                n1 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix
                n2 = n1 + 1
                n3 = n1 + (nx + 1) + 1
                n4 = n1 + (nx + 1)
                n5 = n1 + (nx + 1) * (ny + 1)
                n6 = n5 + 1
                n7 = n5 + (nx + 1) + 1
                n8 = n5 + (nx + 1)
                nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
                for j in range(8):
                    edof_mat[el, 3 * j : 3 * j + 3] = torch.tensor(
                        [3 * nodes[j], 3 * nodes[j] + 1, 3 * nodes[j] + 2],
                        dtype=torch.long,
                        device=device,
                    )
    return edof_mat


def compliance(
    x: torch.Tensor,
    edof: torch.Tensor,
    loads: torch.Tensor,
    free: torch.Tensor,
    E0: float = 1.0,
    Emin: float = 1e-9,
    nu: float = 0.3,
    penal: float = 3.0,
) -> torch.Tensor:
    """可微柔度。

    参数:
        x: 密度场 (n_elems,)，可微。
        edof: (n_elems, 24) 自由度映射。
        loads: 荷载向量 (ndof,)。
        free: 自由自由度掩码 (ndof,)。
        E0, Emin, nu, penal: SIMP 参数。

    返回:
        柔度标量（可微，x 的梯度即灵敏度）。
    """
    ndof = loads.shape[0]
    ke = hex8_stiffness(E0, nu).to(x.device)

    n_elems = x.shape[0]
    x_penal = x ** penal
    E = Emin + (E0 - Emin) * x_penal  # (n_elems,)

    # 组装全局刚度（用 dense 简化，最小测试规模足够）
    K = torch.zeros(ndof, ndof, dtype=x.dtype, device=x.device)
    for el in range(n_elems):
        ed = edof[el]
        K[ed.unsqueeze(1), ed.unsqueeze(0)] += E[el] * ke

    # 施加支座（free 掩码）
    Kff = K[free][:, free]
    Uf = torch.linalg.solve(Kff, loads[free])

    U = torch.zeros(ndof, dtype=x.dtype, device=x.device)
    U[free] = Uf

    c = loads @ U
    return c


if __name__ == "__main__":
    # 最小自测：可微性验证
    nx, ny, nz = 4, 4, 4
    ndof = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    edof = assemble_edof(nx, ny, nz)

    loads = torch.zeros(ndof, dtype=torch.float64)
    node_id = nx + (nx + 1) * ny + (nx + 1) * (ny + 1) * nz
    loads[3 * node_id + 2] = -1.0

    free = torch.ones(ndof, dtype=torch.bool)
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            nid = (nx + 1) * (ny + 1) * iz + (nx + 1) * iy
            for d in range(3):
                free[3 * nid + d] = False

    x = torch.full((nx * ny * nz,), 0.5, dtype=torch.float64, requires_grad=True)
    c = compliance(x, edof, loads, free)
    print("compliance:", c.item())
    c.backward()
    print("grad shape:", x.grad.shape, "grad sample:", x.grad[:3])
    print("differentiable FEM OK")
