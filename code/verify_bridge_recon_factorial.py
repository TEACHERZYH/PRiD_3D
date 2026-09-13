"""模块级最小验证：refine 的 bridge/reconstruct 开关 + 严格噪声配对。

不加载 checkpoint，用随机权重 UNet 只验证分支逻辑与随机数消耗行为。
"""
import sys
sys.path.insert(0, "code")

import torch
from diffusion.diffusion import UNet3D, Diffusion
from projection.manifold_projection import manifold_projection

ok = True


def check(name, cond):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        ok = False


torch.manual_seed(0)
model = UNet3D(in_channels=5, base=8, depth=1)
model.eval()
diff = Diffusion(model, timesteps=200)

x = torch.rand(1, 1, 8, 8, 8)
cond = torch.rand(1, 4, 8, 8, 8)  # 与 conds 一致：4 通道，cat(x_t) 后为 5 通道输入
src = cond[:, 2:3]
sup = cond[:, 3:4]

with torch.no_grad():
    # 1. bridge=False, reconstruct=False => 恒等映射
    out = diff.refine(x, cond, src, sup, K=3, bridge=False, reconstruct=False)
    check("bridge=F,recon=F 为恒等映射", torch.allclose(out, x))

    # 2. one-shot：K=1, bridge=T, reconstruct=F == 直接调用一次 manifold_projection
    out1 = diff.refine(x, cond, src, sup, K=1, bridge=True, reconstruct=False)
    direct = manifold_projection(x, source_mask=src, support_mask=sup)
    check("one-shot == 一次 manifold_projection", torch.allclose(out1, direct))

    # 3. K=1 与 K=3 只桥接结果应相同（桥接幂等/收敛到同一修复场）
    out3 = diff.refine(x, cond, src, sup, K=3, bridge=True, reconstruct=False)
    print(f"     (只桥接 K=1 vs K=3 最大差 {float((out1-out3).abs().max()):.4e})")

    # 4. 严格配对：相同 seed 两次 refine(bridge=True,recon=True) 输出一致
    torch.manual_seed(123)
    a = diff.refine(x, cond, src, sup, K=2, bridge=True, reconstruct=True)
    torch.manual_seed(123)
    b = diff.refine(x, cond, src, sup, K=2, bridge=True, reconstruct=True)
    check("同 seed 两次 PRiD refine 输出一致（噪声可复现）", torch.allclose(a, b))

    # 5. 同 seed 下 no_bridge 与 prid 的噪声消耗起点相同（都应消耗 K 组 randn_like）
    torch.manual_seed(7)
    nb = diff.refine(x, cond, src, sup, K=2, bridge=False, reconstruct=True)
    st_after_nb = torch.rand(1).item()
    torch.manual_seed(7)
    rf = diff.refine(x, cond, src, sup, K=2, bridge=True, reconstruct=True)
    st_after_rf = torch.rand(1).item()
    check("no_bridge 与 PRiD 消耗相同数量随机数（配对成立）",
          abs(st_after_nb - st_after_rf) < 1e-9)

    # 6. 回归：新签名默认值与旧行为一致（bridge=True, reconstruct=True）
    torch.manual_seed(7)
    c = diff.refine(x, cond, src, sup, K=2)
    torch.manual_seed(7)
    d = diff.refine(x, cond, src, sup, K=2, bridge=True, reconstruct=True)
    check("默认调用 == 显式 bridge=T,recon=T（向后兼容）", torch.allclose(c, d))

print("\nRESULT:", "ALL PASS" if ok else "HAS FAILURE")
