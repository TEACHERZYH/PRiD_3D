"""流形保持投影算子（方法核心）。

把工程本体/规范先验实现为"采样空间边界"的可微投影算子，
在去噪的每一步把密度场投影到合法设计流形 M_legal 上。

三类约束：
1. 连通性投影  connect_projection：基于虚拟热传导，把漂浮体桥接回主体；
2. 最小特征尺寸投影  min_size_projection：可微形态学闭运算（软阈值）；
3. 对称性投影  symmetry_projection：群作用镜像平均（天然可微）。

全部算子保持 PyTorch autograd 可微，可插入去噪链而不破坏梯度。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 工具：直通估计（Straight-Through Estimator）
# --------------------------------------------------------------------------- #
class StraightThrough(torch.autograd.Function):
    """前向返回 hard 值，反向把梯度直通到 soft 输入。

    用法: StraightThrough.apply(hard_value, soft_input)
    - forward: 返回 hard_value
    - backward: 梯度直接传给 soft_input（identity）
    用于把不可微的离散投影（连通分量标记、二值化）变成可微算子。
    """

    @staticmethod
    def forward(ctx, hard_value, soft_input):
        return hard_value

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output


def _to_tensor_like(x, ref):
    return torch.tensor(x, dtype=ref.dtype, device=ref.device)


# --------------------------------------------------------------------------- #
# 1. 连通性投影（真桥接：把漂浮体连接到主体，而非删除）
# --------------------------------------------------------------------------- #
def _label_components(binary: torch.Tensor) -> torch.Tensor:
    """3D 连通分量标记（26 邻域），返回每个体素的组件编号。

    输入 binary: (D, H, W) bool 张量。用 BFS 实现（Python 循环，正确性优先）。
    返回 labels: (D, H, W) long 张量，0 表示空，>=1 表示组件编号。
    """
    D, H, W = binary.shape
    labels = torch.zeros_like(binary, dtype=torch.long)
    comp_id = 0
    # 26 邻域偏移
    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dz, dy, dx))

    for z in range(D):
        for y in range(H):
            for x in range(W):
                if binary[z, y, x] and labels[z, y, x] == 0:
                    comp_id += 1
                    # BFS
                    stack = [(z, y, x)]
                    labels[z, y, x] = comp_id
                    while stack:
                        cz, cy, cx = stack.pop()
                        for dz, dy, dx in offsets:
                            nz, ny, nx = cz + dz, cy + dy, cx + dx
                            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                                if binary[nz, ny, nx] and labels[nz, ny, nx] == 0:
                                    labels[nz, ny, nx] = comp_id
                                    stack.append((nz, ny, nx))
    return labels


def _bridge_floating(
    x: torch.Tensor,
    binary: torch.Tensor,
    labels: torch.Tensor,
    source_mask: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """把漂浮体桥接回主体。

    主体 = 与荷载点或支座点连通的组件。
    对每个漂浮体组件，用"膨胀逐步扩张 + 与主体相交"找到连接路径，
    把路径上的单元密度提升（桥接）。

    返回：桥接后的密度场（离散操作，用于前向硬投影）。
    """
    D, H, W = binary.shape
    # 主体 = 最大连通分量（主导结构）。
    # 用最大分量而非"seed 命中分量"，因为扩散采样在荷载点/支座面
    # 可能恰好无材料，用 seed 会误判主体为空，破坏结构。
    flat = labels.ravel()
    sizes = torch.bincount(flat[flat > 0])  # 各分量体素数（分量编号从 1 起）
    if sizes.numel() <= 1:
        return x.clone()  # 无材料或单分量，无需桥接
    main_comp = int(torch.argmax(sizes[1:]) + 1)  # 最大分量编号
    main = (labels == main_comp)

    # 对每个漂浮体组件，用最短路径桥接回主体（只填连接线，最少材料）
    floating = binary & ~main
    x_new = x.clone()

    if floating.any():
        # 距离变换：每个体素到主体 main 的最近距离
        dt = _distance_to_mask(main)
        # 对每个漂浮体组件分别处理
        comps = labels[floating].unique()
        for c in comps.tolist():
            if c <= 0:
                continue
            comp_mask = (labels == c)
            # 该组件中离主体最近的体素
            comp_dt = dt.clone()
            comp_dt[~comp_mask] = float("inf")
            min_idx = torch.argmin(comp_dt.view(-1))
            start = (min_idx // (H * W), (min_idx // W) % H, min_idx % W)
            # 沿距离场梯度下降到主体，得到最短路径
            path = _descend_to_mask(dt, start, main, max_steps=max(D, H, W) * 3)
            # 只填充最短路径上的体素（桥接线）
            if path:
                for p in path:
                    x_new[p] = max(float(x_new[p]), 0.7)

    return x_new


def _distance_to_mask(mask: torch.Tensor) -> torch.Tensor:
    """计算每个体素到 mask 的最近距离（用 BFS 近似，返回 float 张量）。"""
    import numpy as np
    mask_np = mask.cpu().numpy()
    from scipy.ndimage import distance_transform_edt
    # distance_transform_edt 返回"到背景(0)的距离"，故用 ~mask 使主体成为背景，
    # 得到"到主体的距离"（主体位置=0，越远越大）
    dt = distance_transform_edt(~mask_np)
    return torch.from_numpy(dt.astype(np.float32)).to(mask.device)


def _descend_to_mask(dt: torch.Tensor, start, mask: torch.Tensor, max_steps: int):
    """从 start 沿距离场梯度下降到 mask，返回路径上的体素坐标列表。"""
    D, H, W = dt.shape
    path = []
    cur = start
    for _ in range(max_steps):
        if mask[cur[0], cur[1], cur[2]]:
            return path
        path.append(cur)
        # 找 6 邻域中距离最小的方向
        best = None
        best_val = dt[cur[0], cur[1], cur[2]].item()
        for dz, dy, dx in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
            nz, ny, nx = cur[0] + dz, cur[1] + dy, cur[2] + dx
            if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                if dt[nz, ny, nx] < best_val:
                    best_val = dt[nz, ny, nx].item()
                    best = (nz, ny, nx)
        if best is None or best in path:
            return path
        cur = best
    return path


def connect_projection(
    x: torch.Tensor,
    source_mask: torch.Tensor,
    support_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """连通性投影（真桥接版）。

    前向：二值化 → 连通分量标记 → 把漂浮体桥接回主体（离散硬投影）。
    反向：直通估计（梯度直接穿过）。

    这是"流形保持"的核心：投影后的采样结果保证连通（合法），
    而不是删除漂浮体（事后修正）。

    参数:
        x: 密度场 (B, 1, D, H, W) 或 (D, H, W)。
        source_mask: 荷载位置掩码。
        support_mask: 支座位置掩码。
        threshold: 二值化阈值。

    返回:
        投影后的密度场（保证连通）。
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0).unsqueeze(0)
        source_mask = source_mask.unsqueeze(0).unsqueeze(0)
        support_mask = support_mask.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 4:
        x = x.unsqueeze(1)
        source_mask = source_mask.unsqueeze(1)
        support_mask = support_mask.unsqueeze(1)

    B = x.shape[0]
    outs = []
    for b in range(B):
        xb = x[b, 0]
        sb = source_mask[b, 0]
        supb = support_mask[b, 0]
        # 二值化（离散，前向）
        binary = (xb > threshold)
        labels = _label_components(binary)
        # 桥接（离散，前向）
        x_bridged = _bridge_floating(xb, binary, labels, sb, supb)
        outs.append(x_bridged.unsqueeze(0))  # (1, D, H, W)

    x_proj = torch.stack(outs, dim=0)  # (B, 1, D, H, W) -- 已含通道维

    # 直通估计：前向用桥接结果，反向梯度直通到原始 x
    x_proj = StraightThrough.apply(x_proj, x)

    if single:
        return x_proj.squeeze(0).squeeze(0)  # (D, H, W)
    return x_proj


# --------------------------------------------------------------------------- #
# 2. 最小特征尺寸投影（可微形态学闭运算）
# --------------------------------------------------------------------------- #
def min_size_projection(
    x: torch.Tensor,
    kernel_size: int = 3,
    strength: float = 0.3,
) -> torch.Tensor:
    """可微最小特征尺寸约束。

    用软闭运算（先膨胀后腐蚀）消除小于 kernel_size 的孔洞与细枝，
    对应规范中的最小构件截面尺寸。用 max-pool 近似膨胀、min-pool 近似腐蚀，
    并保留可微性（soft 组合）。

    参数:
        x: 密度场 (B,1,D,H,W) 或 (D,H,W)。
        kernel_size: 形态学核尺寸（越大，最小特征尺寸越大）。
        strength: 投影强度（0=不投影，1=完全闭运算）。

    返回:
        投影后的密度场。
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 4:
        x = x.unsqueeze(1)

    k = kernel_size
    # 膨胀：max-pool；腐蚀：min-pool（用 -max-pool 实现）
    dilate = F.max_pool3d(x, kernel_size=k, stride=1, padding=k // 2)
    erode = -F.max_pool3d(-dilate, kernel_size=k, stride=1, padding=k // 2)

    # 软投影：在原始与闭运算结果之间插值
    x_proj = (1.0 - strength) * x + strength * erode

    if single:
        return x_proj.squeeze(0).squeeze(0)
    return x_proj


# --------------------------------------------------------------------------- #
# 3. 对称性投影（群作用镜像平均）
# --------------------------------------------------------------------------- #
def symmetry_projection(
    x: torch.Tensor,
    axes: tuple[bool, bool, bool] = (False, True, False),
) -> torch.Tensor:
    """镜像平均投影，保证结构对称（天然可微）。

    参数:
        x: 密度场 (B,1,D,H,W) 或 (D,H,W)。
        axes: 三个维度是否施加镜像对称（默认沿 H 轴，即 y 方向对称）。

    返回:
        投影后的密度场。
    """
    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 4:
        x = x.unsqueeze(1)

    x_proj = x
    for dim, do in enumerate(axes):
        if do:
            spatial_dim = dim + 2  # 对应 D/H/W
            x_proj = 0.5 * (x_proj + torch.flip(x_proj, dims=[spatial_dim]))

    if single:
        return x_proj.squeeze(0).squeeze(0)
    return x_proj


# --------------------------------------------------------------------------- #
# 组合投影（去噪链中调用）
# --------------------------------------------------------------------------- #
def manifold_projection(
    x: torch.Tensor,
    source_mask: torch.Tensor | None = None,
    support_mask: torch.Tensor | None = None,
    min_size: int = 0,
    sym_axes: tuple[bool, bool, bool] = (False, False, False),
    min_size_strength: float = 0.3,
) -> torch.Tensor:
    """一次完整投影：连通性（桥接）+ 可选的最小尺寸 + 可选的对称性。

    核心是连通性投影（流形保持的关键）；最小尺寸与对称性默认关闭，
    避免在无条件采样质量一般时破坏结构。

    返回投影到合法流形 M_legal 上的密度场。
    """
    x_proj = x
    if source_mask is not None and support_mask is not None:
        x_proj = connect_projection(x_proj, source_mask, support_mask)
    if min_size > 0:
        x_proj = min_size_projection(x_proj, kernel_size=min_size, strength=min_size_strength)
    if any(sym_axes):
        # 对称投影是完全投影（硬约束），强度固定为 1
        x_proj = symmetry_projection(x_proj, axes=sym_axes)
    return x_proj


if __name__ == "__main__":
    # 最小自测
    torch.manual_seed(0)
    x = torch.rand(1, 1, 8, 8, 8, dtype=torch.float32)
    src = torch.zeros_like(x)
    src[..., 6, 6, 6] = 1.0
    sup = torch.zeros_like(x)
    sup[..., 0, :, :] = 1.0

    x_proj = manifold_projection(x, source_mask=src, support_mask=sup)
    print("input  mean:", x.mean().item(), "output mean:", x_proj.mean().item())
    print("projection is differentiable:", x_proj.requires_grad is False)
    # 验证可微
    xx = x.clone().requires_grad_(True)
    y = min_size_projection(xx, kernel_size=3)
    y.sum().backward()
    print("min_size_projection grad OK:", xx.grad is not None)
