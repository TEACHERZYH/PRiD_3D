"""将所有论文图重新渲染到 300dpi（EAAI 投稿要求彩色图 >= 300dpi）。"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 1. compliance.png（柔度柱状图） ----------
methods = ["Unconditional", "Post-hoc Proj.", "Proj.-Refined"]
compliance = [22252, 22241, 10851]
colors = ["#8c8c8c", "#5b9bd5", "#c0392b"]
fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor="white")
bars = ax.bar(methods, compliance, color=colors, width=0.55)
for b, v in zip(bars, compliance):
    ax.text(b.get_x() + b.get_width() / 2, v + 500, f"{v:,}", ha="center",
            va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Mean compliance (N\u00b7mm)")
ax.set_title("Compliance comparison")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("paper/figures/compliance.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# ---------- 2. convergence.png（收敛曲线） ----------
K = [0, 1, 2, 3, 5]
conn = [0.952, 0.987, 0.998, 1.000, 0.9995]
fig, ax = plt.subplots(figsize=(4.6, 3.2), facecolor="white")
ax.plot(K, conn, marker="o", color="#c0392b", linewidth=2, markersize=7, zorder=3)
for x, y in zip(K, conn):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Refinement rounds K")
ax.set_ylabel("Connectivity")
ax.set_ylim(0.93, 1.02)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Convergence of projection-refined diffusion", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("paper/figures/convergence.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# ---------- 3. final_comparison.png（四组对比切片） ----------
d = np.load("results/final_viz.npz")
gt, uncond, posthoc, refine = d["gt"], d["uncond"], d["posthoc"], d["refine"]
D, H, W = gt.shape
fig, axes = plt.subplots(3, 4, figsize=(12, 9), facecolor="white")
for r in range(3):
    sl = [lambda a: a[D // 2], lambda a: a[:, H // 2, :], lambda a: a[:, :, W // 2]][r]
    for c, arr in enumerate([gt, uncond, posthoc, refine]):
        axes[r, c].imshow(sl(arr), cmap="gray_r", vmin=0, vmax=1)
        if r == 0:
            axes[r, c].set_title(["SIMP GT", "Unconditional", "Post-hoc Proj.",
                                  "Projection-Refined"][c], fontweight="bold", fontsize=11)
    axes[r, 0].set_ylabel(["z-mid", "y-mid", "x-mid"][r])
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("Projection-Refined Diffusion (black = material, volfrac=0.39)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("paper/figures/final_comparison.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()

print("done: compliance.png, convergence.png, final_comparison.png @ 300dpi")
