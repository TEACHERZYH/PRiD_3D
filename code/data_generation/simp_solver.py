"""3D SIMP 拓扑优化求解器（numpy + scipy.sparse 稀疏求解版）。

本模块是零数据采集方案的地基：
1. 作为数据合成管线，生成 (条件, 密度场标签) 训练样本；
2. 同时输出 von Mises 应力场作为物理特征通道。

性能关键：使用 scipy.sparse 稀疏刚度矩阵 + 直接稀疏求解器（spsolve），
避免 dense 求解在 3D 网格上 O(n^3) 爆炸。

设计要点：
- 纯 numpy + scipy 实现（数据合成不需要 autograd，追求速度）；
- 8 节点六面体单元（trilinear），SIMP 幂律插值 + 密度滤波 + OC 更新；
- 密度场 x ∈ [0,1]^{nx*ny*nz}。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, cg
from scipy.ndimage import convolve


# --------------------------------------------------------------------------- #
# 单元刚度矩阵（8 节点六面体，trilinear，各向同性材料）
# --------------------------------------------------------------------------- #
def hex8_stiffness(E: float, nu: float) -> np.ndarray:
    """返回 24x24 的 8 节点六面体单元刚度矩阵（单位边长 [0,1]^3）。

    物理单元为 [0,1]^3，自然坐标 [-1,1]^3，雅可比映射因子 2（每方向），
    8 点高斯积分，积分权重 1/8（|J|）。
    """
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ]
    ) * c

    gp = 1.0 / np.sqrt(3.0)
    gauss = [-gp, gp]

    ke = np.zeros((24, 24))
    for xi in gauss:
        for eta in gauss:
            for zeta in gauss:
                dN = hex8_shape_derivs(xi, eta, zeta)  # (8,3) 自然坐标导数
                B = np.zeros((6, 24))
                for i in range(8):
                    dNdx, dNdy, dNdz = 2.0 * dN[i]  # 自然 -> 物理坐标乘 2
                    B[0, 3 * i] = dNdx
                    B[1, 3 * i + 1] = dNdy
                    B[2, 3 * i + 2] = dNdz
                    B[3, 3 * i] = dNdy
                    B[3, 3 * i + 1] = dNdx
                    B[4, 3 * i + 1] = dNdz
                    B[4, 3 * i + 2] = dNdy
                    B[5, 3 * i] = dNdz
                    B[5, 3 * i + 2] = dNdx
                ke += (1.0 / 8.0) * (B.T @ D @ B)
    return ke


def hex8_shape_derivs(xi: float, eta: float, zeta: float) -> np.ndarray:
    """8 节点六面体形函数对 (xi,eta,zeta) 的导数，形状 (8,3)。"""
    coords = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ]
    )
    dN = np.zeros((8, 3))
    for i in range(8):
        dN[i, 0] = 0.125 * coords[i, 0] * (1 + coords[i, 1] * eta) * (1 + coords[i, 2] * zeta)
        dN[i, 1] = 0.125 * coords[i, 1] * (1 + coords[i, 0] * xi) * (1 + coords[i, 2] * zeta)
        dN[i, 2] = 0.125 * coords[i, 2] * (1 + coords[i, 0] * xi) * (1 + coords[i, 1] * eta)
    return dN


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def build_edof(nx: int, ny: int, nz: int) -> np.ndarray:
    """返回 (n_elem, 24) 的单元自由度映射。"""
    n_elem = nx * ny * nz
    edof = np.zeros((n_elem, 24), dtype=np.int64)
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
                    edof[el, 3 * j:3 * j + 3] = [3 * nodes[j], 3 * nodes[j] + 1, 3 * nodes[j] + 2]
    return edof


def build_k_index(edof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """预计算稀疏刚度矩阵的 COO 索引 (iK, jK)，长度 n_elem*576。"""
    iK = np.repeat(edof, 24, axis=1).ravel()      # 行：每个 dof 重复 24 次
    jK = np.tile(edof, (1, 24)).ravel()           # 列：整个 dof 向量重复 24 次
    return iK, jK


# --------------------------------------------------------------------------- #
# SIMP 拓扑优化
# --------------------------------------------------------------------------- #
def topology_optimize(
    nx: int,
    ny: int,
    nz: int,
    volfrac: float,
    penal: float = 3.0,
    rmin: float = 1.5,
    max_iter: int = 60,
    E0: float = 1.0,
    Emin: float = 1e-9,
    nu: float = 0.3,
    loads: np.ndarray | None = None,
    supports: np.ndarray | None = None,
    return_stress: bool = False,
    verbose: bool = False,
):
    """3D SIMP 拓扑优化。

    返回:
        x: 优化后的密度场 (nx*ny*nz,) float64。
        c_history: 柔度历史 (ndarray)。
        （可选）von_mises: von Mises 应力场 (nx*ny*nz,)。
    """
    n_elem = nx * ny * nz
    nn = (nx + 1) * (ny + 1) * (nz + 1)
    ndof = 3 * nn

    # 荷载（默认右下角顶部节点 -z 集中力）
    if loads is None:
        loads = np.zeros(ndof)
        node_id = nx + (nx + 1) * ny + (nx + 1) * (ny + 1) * nz
        loads[3 * node_id + 2] = -1.0

    # 支座（默认 x=0 左端面全固定）
    if supports is None:
        supports = np.zeros(ndof, dtype=bool)
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                node_id = (nx + 1) * (ny + 1) * iz + (nx + 1) * iy
                supports[3 * node_id:3 * node_id + 3] = True
    free = ~supports

    # 单元刚度 + 组装索引
    ke = hex8_stiffness(E0, nu)
    ke_flat = ke.ravel()
    edof = build_edof(nx, ny, nz)
    iK, jK = build_k_index(edof)

    # 密度滤波核（3x3x3 均值）
    filt = np.ones((3, 3, 3)) / 27.0

    def density_filter(x):
        return convolve(x.reshape(nx, ny, nz), filt, mode="constant", cval=0.0).ravel()

    x = np.full(n_elem, volfrac)
    c_history = []
    Uf_prev = None  # warm start：前一次位移解作为 CG 初值

    for it in range(max_iter):
        xf = density_filter(x)
        xp = xf ** penal
        E = Emin + (E0 - Emin) * xp

        # 组装稀疏刚度矩阵
        vals = (ke_flat[None, :] * E[:, None]).ravel()
        K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()

        # 求解位移：CG 迭代 + warm start（前次解作初值）
        Kff = K[free][:, free]
        Uf, info = cg(Kff, loads[free], x0=Uf_prev, rtol=1e-6, maxiter=500)
        Uf_prev = Uf
        U = np.zeros(ndof)
        U[free] = Uf

        # 柔度
        c_val = float(loads @ U)

        # 灵敏度（向量化）
        Ue = U[edof]  # (n_elem, 24)
        ce = np.einsum("ei,ij,ej->e", Ue, ke, Ue)
        dc = -penal * xp ** ((penal - 1) / penal) * ce
        dc = density_filter(dc)

        # OC 更新
        l1, l2 = 0.0, 1e9
        move = 0.2
        eta = 0.5
        while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-6:
            lmid = 0.5 * (l1 + l2)
            xnew = x * (-dc / (lmid + 1e-12)) ** eta
            xnew = np.clip(np.clip(xnew, x - move, x + move), 0.001, 1.0)
            if xnew.sum() - volfrac * n_elem > 0:
                l1 = lmid
            else:
                l2 = lmid
        x = xnew
        c_history.append(c_val)
        if verbose and (it + 1) % 20 == 0:
            print(f"iter {it+1:3d}: compliance = {c_val:.4e}, volfrac = {x.mean():.3f}")

    if return_stress:
        von_mises = compute_von_mises(
            x, edof, ke, ke_flat, iK, jK, loads, free, E0, Emin, nu, penal, ndof
        )
        return x, np.array(c_history), von_mises

    return x, np.array(c_history)


def compute_von_mises(
    x, edof, ke, ke_flat, iK, jK, loads, free, E0, Emin, nu, penal, ndof
):
    """计算每个单元的 von Mises 应力（用单元应变能密度近似）。"""
    n_elem = x.shape[0]
    E = Emin + (E0 - Emin) * (x ** penal)
    vals = (ke_flat[None, :] * E[:, None]).ravel()
    K = coo_matrix((vals, (iK, jK)), shape=(ndof, ndof)).tocsc()
    Uf = spsolve(K[free][:, free], loads[free])
    U = np.zeros(ndof)
    U[free] = Uf

    Ue = U[edof]  # (n_elem, 24)
    energy = 0.5 * np.einsum("ei,ij,ej->e", Ue, ke, Ue)
    von_mises = np.sqrt(2.0 * E * np.maximum(energy, 0.0))
    return von_mises


if __name__ == "__main__":
    import time
    t0 = time.time()
    x, c = topology_optimize(16, 16, 16, volfrac=0.3, max_iter=40, verbose=False)
    print(f"N=16 耗时 {time.time()-t0:.2f}s, 柔度 {c[-1]:.2f}, volfrac {x.mean():.3f}")
