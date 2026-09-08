"""绘制方法框架图：投影精炼扩散（Projection-Refined Diffusion）的流程示意图。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 52)
ax.axis("off")

# 颜色
c_cond = "#2f6fba"      # 条件（蓝）
c_samp = "#6b7280"      # 采样（灰）
c_proj = "#c0392b"      # 投影（红，核心）
c_den  = "#1e8e5a"      # 去噪（绿）
c_out  = "#1f2937"      # 输出（深灰）

def box(x, y, w, h, fc, ec, lw=1.6, r=1.6, ls="-"):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2)
    ax.add_patch(b)

def text(x, y, s, size=9.5, weight="normal", color="black", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, style=style, zorder=3)

def arrow(x1, y1, x2, y2, color="black", lw=1.8, ls="-", style="-|>", ms=14):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms,
                        color=color, linewidth=lw, linestyle=ls, zorder=1)
    ax.add_patch(a)

# ---------------- 阶段 1：条件输入 ----------------
box(2, 22, 24, 24, "#eef4fb", c_cond)
text(14, 42.5, "Conditioning (4 channels)", size=10, weight="bold", color=c_cond)
text(14, 37.5, "volume fraction $v_f$", size=9)
text(14, 33, "von Mises stress $\\sigma$", size=9)
text(14, 28.5, "load mask $\\mathbf{s}$", size=9, color=c_proj, weight="bold")
text(14, 24, "support mask $\\mathbf{u}$", size=9, color=c_den, weight="bold")

# ---------------- 阶段 2：无条件采样 ----------------
box(32, 22, 26, 24, "#f3f4f6", c_samp)
text(45, 42.5, "Unconditional DDIM", size=10, weight="bold", color=c_samp)
text(45, 37.5, "$\\mathbf{x}_T\\to\\cdots\\to\\mathbf{x}_0$", size=10)
text(45, 32, "sample $\\mathbf{x}_0$", size=9)
text(45, 27, "floating material,", size=9, color=c_proj)
text(45, 23.5, "disconnected", size=9, color=c_proj)

# ---------------- 阶段 3：投影精炼循环 ----------------
box(64, 16, 34, 34, "#fdf0ef", c_proj)
text(81, 46.5, "Projection-Refinement", size=10, weight="bold", color=c_proj)
text(81, 43.0, "($K$ rounds)", size=9, color=c_proj)

# 循环内部两个节点
box(68, 22, 12, 12, "#fdecea", c_proj)
text(74, 28, "$\\mathcal{P}$", size=12, weight="bold", color=c_proj)
text(74, 24.6, "bridge floating", size=8, color=c_proj)

box(86, 22, 10, 12, "#eaf6ef", c_den)
text(91, 28, "Denoise", size=9.5, weight="bold", color=c_den)
text(91, 24.6, "naturalize", size=8, color=c_den)

# 投影 → 去噪
arrow(80, 28, 86, 28, color="black", lw=1.8)
# 去噪 → 投影（循环返回）
arrow(91, 22, 91, 18.5, color="black", lw=1.8)
arrow(91, 18.5, 74, 18.5, color="black", lw=1.8)
arrow(74, 18.5, 74, 22, color="black", lw=1.8)
text(81, 17.2, "repeat $K$ times", size=8, color="black", style="italic")

# ---------------- 阶段 4：输出 ----------------
box(64, 4, 34, 8, "#eef4fb", c_out)
text(81, 8, "feasible & natural topology", size=10, weight="bold", color=c_out)

# ---------------- 主流程箭头 ----------------
arrow(26, 34, 32, 34, color="black", lw=2.2)
arrow(58, 34, 64, 34, color="black", lw=2.2)
arrow(81, 16, 81, 12, color="black", lw=2.2)

# ---------------- 顶部标题说明 ----------------
text(50, 50.5, "Projection-Refined Diffusion for Physically-Constrained 3D Topology Optimization",
     size=12.5, weight="bold", color="#111827")

plt.tight_layout(pad=0.4)
plt.savefig("paper/figures/framework.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved framework.png")
