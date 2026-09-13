"""扫描修正后软约束引导的参数（真实数据 10 样本，精简组合）。"""
import numpy as np
import torch

from diffusion.diffusion import UNet3D, Diffusion
from soft_constraint_baseline import soft_guide, connectivity

data = np.load("data/train_16.npz")
conds = torch.from_numpy(data["conds"][:10]).float()
D, H, W = conds.shape[2], conds.shape[3], conds.shape[4]
model = UNet3D(in_channels=5, base=32, depth=2)
model.load_state_dict(torch.load("results/checkpoint_16.pt", map_location="cpu"))
model.eval()
diff = Diffusion(model, timesteps=200)

samples = []
for i in range(10):
    cond = conds[i:i + 1]
    src = conds[i, 2:3]
    sup = conds[i, 3:4]
    torch.manual_seed(12000 + i)
    with torch.no_grad():
        s = diff.sample(cond, (1, 1, D, H, W), ddim_steps=20, use_projection=False)
    samples.append((s, src, sup))

base_conn = np.mean([connectivity(s[0, 0].numpy())[0] for s, _, _ in samples])
print(f"无条件基线: 连通性={base_conn:.4f}", flush=True)

print("=== 软约束引导参数扫描（vol_weight x lr，steps=120，iters=60） ===", flush=True)
for vol_weight in [0.5, 2.0]:
    for lr in [0.3, 0.7]:
        conns = []
        mass_ratio = []
        for s, src, sup in samples:
            sg = soft_guide(s, src, sup, steps=120, lr=lr, iters=60, vol_weight=vol_weight)
            sn = sg[0, 0].numpy()
            conns.append(connectivity(sn)[0])
            m0 = (s[0, 0].numpy() > 0.5).sum()
            m1 = (sn > 0.5).sum()
            mass_ratio.append(m1 / max(m0, 1))
        print(f"vol_weight={vol_weight:4.1f}, lr={lr:.1f}: "
              f"连通性={np.mean(conns):.4f}, 材料比={np.mean(mass_ratio):.3f}", flush=True)
