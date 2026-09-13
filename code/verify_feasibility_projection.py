"""模块级验证：sample 新增参数（x_init/t_start/trace_fn）+ 破坏算子。

用随机权重 UNet，只验证参数行为与几何算子，不涉及训练结果。
"""
import sys

sys.path.insert(0, "code")

import numpy as np
import torch

from diffusion.diffusion import UNet3D, Diffusion
from eval_feasibility_projection import conn_stats, find_worst_break

ok = True


def check(name, cond, extra=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        ok = False


# ---- 1. conn_stats 在已知结构上正确 ----
solid = np.ones((8, 8, 8), dtype=bool)
check("全材料 -> conn=1.0", abs(conn_stats(solid)[0] - 1.0) < 1e-9)

two = np.zeros((8, 8, 8), dtype=bool)
two[0:2, 0:2, 0:2] = True
two[5:7, 5:7, 5:7] = True
c2, f2, n2 = conn_stats(two)
check("两个等大分离块 -> conn=0.5, ncomp=2",
      abs(c2 - 0.5) < 1e-9 and n2 == 2, f"(got conn={c2:.3f}, n={n2})")

# ---- 2. find_worst_break 确实制造断裂 ----
rho = np.zeros((16, 16, 16), dtype=np.float32)
rho[6:10, :, :] = 1.0
rho[6:10, :, 6:10] = 1.0
axis, L, rho_b, c_b, occ_b = find_worst_break(rho)
c0, _, _ = conn_stats(rho > 0.5)
check("破坏算子降低连通性", c_b < c0 - 1e-6, f"(before {c0:.3f} -> after {c_b:.3f}, axis{axis} L{L})")

# ---- 3. sample 参数行为 ----
torch.manual_seed(0)
model = UNet3D(in_channels=5, base=8, depth=1)
model.eval()
diff = Diffusion(model, timesteps=200)
cond = torch.rand(1, 4, 8, 8, 8)
shape = (1, 1, 8, 8, 8)

with torch.no_grad():
    torch.manual_seed(123)
    a = diff.sample(cond, shape, ddim_steps=8)
    torch.manual_seed(123)
    b = diff.sample(cond, shape, ddim_steps=8, x_init=None, t_start=None, trace_fn=None)
    check("默认调用 == 显式 None（向后兼容）", torch.allclose(a, b))

    x0 = torch.rand(1, 1, 8, 8, 8)
    torch.manual_seed(123)
    c = diff.sample(cond, shape, ddim_steps=8, x_init=x0, t_start=None)
    check("给定 x_init 后输出不同（参数生效）", not torch.allclose(a, c))

    calls = []
    torch.manual_seed(123)
    _ = diff.sample(cond, shape, ddim_steps=8,
                    trace_fn=lambda step, x0p, t: calls.append((step, t, tuple(x0p.shape))))
    check("trace_fn 调用次数 == ddim_steps", len(calls) == 8, f"(got {len(calls)})")
    check("trace_fn 收到正确的 x0_pred 形状",
          calls[0][2] == (1, 1, 8, 8, 8), f"(got {calls[0][2]})")
    ts = [c[1] for c in calls]
    check("轨迹时间步单调下降", all(ts[i] > ts[i + 1] for i in range(len(ts) - 1)), f"(t={ts})")

    # t_start 生效：从 t0 出发的轨迹长度不变，但起点时间不同
    calls2 = []
    torch.manual_seed(123)
    _ = diff.sample(cond, shape, ddim_steps=8, x_init=x0, t_start=100,
                    trace_fn=lambda step, x0p, t: calls2.append(t))
    check("t_start=100 时轨迹起点时间为 100", calls2[0] == 100, f"(got {calls2[0]})")

print("\nRESULT:", "ALL PASS" if ok else "HAS FAILURE")
