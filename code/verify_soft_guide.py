"""验证修正后的软约束：半隐式热扩散判别 + 体积约束防删材。"""
import numpy as np
import torch

from soft_constraint_baseline import heat_diffusion, soft_guide, connectivity

D, H, W = 16, 16, 16
x = torch.zeros(1, 1, D, H, W)
x[..., 0, :, :] = 1.0
x[..., :, 4, 4] = 1.0
x[..., 15, 4, 4] = 1.0
x[..., 12:15, 10:13, 0:3] = 1.0  # 漂浮体
src = torch.zeros(1, 1, D, H, W); src[..., 15, 4, 4] = 1
sup = torch.zeros(1, 1, D, H, W); sup[..., 0, :, :] = 1

# 1. 热扩散判别
T = heat_diffusion(x, src, sup, kappa=1.0, iters=100)
T_np = T[0, 0].numpy()
body_T = float(T_np[8, 4, 4])
float_T = float(T_np[13, 11, 1])
print("=== 热扩散判别（半隐式 kappa=1.0 iters=100）===")
print(f"主体温度={body_T:.3f} (应接近1), 漂浮体温度={float_T:.3f} (应接近0)")
print(f"判别有效: {body_T > 0.5 > float_T}")

# 2. 体积约束防删材
print("\n=== 体积约束防删材 ===")
mass0 = int((x > 0.5).sum().item())
for vol_weight in [0.0, 1.0, 10.0]:
    sg = soft_guide(x, src, sup, steps=100, lr=0.5, iters=100, vol_weight=vol_weight)
    mass1 = int((sg > 0.5).sum().item())
    conn, _, _ = connectivity(sg[0, 0].numpy())
    tag = "删材!" if mass1 < mass0 * 0.9 else "保持"
    print(f"vol_weight={vol_weight:4.1f}: 材料 {mass0} -> {mass1} ({tag}) 连通性={conn:.3f}")
