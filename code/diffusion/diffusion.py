"""3D 条件扩散模型（DDPM + 3D U-Net 骨干，含时间步嵌入）。

骨干是"工具"而非"贡献"——本文的贡献在于流形保持投影算子
（见 projection/manifold_projection.py），扩散骨干保持标准实现。

本模块提供：
1. 3D U-Net（含时间步 FiLM 调制）；
2. DDPM 训练与 DDIM 采样。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 时间步编码
# --------------------------------------------------------------------------- #
def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t, self.mlp[0].in_features))


# --------------------------------------------------------------------------- #
# 3D U-Net（含时间步 FiLM 调制）
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.act2 = nn.SiLU()
        # FiLM：时间步嵌入 → scale + shift
        self.emb_proj = nn.Linear(emb_dim, out_c * 2)

    def forward(self, x, emb):
        h = self.conv1(x)
        h = self.norm1(h)
        scale, shift = self.emb_proj(emb).chunk(2, dim=1)
        h = h * (1 + scale.view(-1, scale.shape[1], 1, 1, 1)) + shift.view(-1, shift.shape[1], 1, 1, 1)
        h = self.act1(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        return h


class Down(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init__()
        self.block = ConvBlock(in_c, out_c, emb_dim)
        self.pool = nn.Conv3d(out_c, out_c, 2, stride=2)

    def forward(self, x, emb):
        h = self.block(x, emb)
        return h, self.pool(h)


class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c, emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_c, out_c, 2, stride=2)
        self.block = ConvBlock(out_c + skip_c, out_c, emb_dim)

    def forward(self, x, skip, emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, emb)


class UNet3D(nn.Module):
    """3D U-Net，含时间步 FiLM 调制。

    输入：x 为 (B, in_channels, D, H, W)（噪声 + 条件通道拼接），t 为 (B,) 时间步。
    """

    def __init__(self, in_channels=2, base=32, depth=2, emb_dim=128):
        super().__init__()
        self.time_embed = TimeEmbed(emb_dim)
        self.inc = ConvBlock(in_channels, base, emb_dim)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        skip_channels = []
        c = base
        for _ in range(depth):
            skip_channels.append(c * 2)
            self.downs.append(Down(c, c * 2, emb_dim))
            c *= 2
        self.mid = ConvBlock(c, c, emb_dim)
        for sc in reversed(skip_channels):
            self.ups.append(Up(c, sc, c // 2, emb_dim))
            c //= 2
        self.outc = nn.Conv3d(c, 1, 1)

    def forward(self, x, t):
        emb = self.time_embed(t)
        h = self.inc(x, emb)
        skips = []
        for down in self.downs:
            h = down.block(h, emb)
            skips.append(h)
            h = down.pool(h)
        h = self.mid(h, emb)
        for up, skip in zip(self.ups, reversed(skips)):
            h = up(h, skip, emb)
        return self.outc(h)


# --------------------------------------------------------------------------- #
# DDPM
# --------------------------------------------------------------------------- #
class Diffusion:
    """DDPM 训练与 DDIM 采样封装。"""

    def __init__(self, model: nn.Module, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus[t].view(-1, 1, 1, 1, 1)
        return a * x0 + b * noise, noise

    def train_loss(self, x0, cond, t=None):
        b = x0.shape[0]
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        x_in = torch.cat([x_t, cond], dim=1)
        pred = self.model(x_in, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def refine(self, x0, cond, src, sup, K=3, t_refine=30, compliance_fn=None):
        """投影 + 精炼循环：交替投影（修正连通性）与低噪声去噪（修正自然性）。

        思路：投影把 x0 桥接到合法流形，但桥接结构是离散的、不自然的；
        随后用模型在低噪声水平去噪，把 x0 拉回自然密度分布。交替进行，
        使最终解同时满足"合法"与"自然"。

        参数:
            x0: (B,1,D,H,W) 采样得到的密度场。
            cond: 条件。src/sup: 工况掩码（通道2/3）。
            K: 精炼迭代次数。
            t_refine: 精炼去噪所用的噪声水平（越小越轻微）。
            compliance_fn: 可选，callable(rho_np)->float，返回该密度场的柔度。
                若提供，则每轮去噪后比较「投影后」与「去噪后」的柔度，仅当
                去噪降低柔度时才采用去噪结果，否则回退到投影结果。这是
                "柔度感知精炼"，保证柔度在精炼过程中单调不增（消除去噪
                偶尔破坏脆弱传力路径导致的柔度恶化）。
        """
        from projection.manifold_projection import manifold_projection
        for _ in range(K):
            x_proj = manifold_projection(x0, source_mask=src, support_mask=sup)
            t = torch.tensor([t_refine], device=x0.device)
            x_t, _ = self.q_sample(x_proj, t)
            x_in = torch.cat([x_t, cond], dim=1)
            eps = self.model(x_in, t)
            a = self.sqrt_alphas_cumprod[t_refine]
            b = self.sqrt_one_minus[t_refine]
            x_den = torch.clamp((x_t - b * eps) / a, 0.0, 1.0)

            if compliance_fn is None:
                x0 = x_den
            else:
                c_proj = float(compliance_fn(x_proj[0, 0].cpu().numpy()))
                c_den = float(compliance_fn(x_den[0, 0].cpu().numpy()))
                x0 = x_den if c_den <= c_proj else x_proj
        return x0

    @torch.no_grad()
    def sample(self, cond, shape, ddim_steps=50, use_projection=False,
               project_from_frac=0.9, projection_anneal=True, projection_smooth=True):
        """DDIM 确定性采样，可选"流形保持"投影。

        核心主张的体现：use_projection=True 时，在去噪过程中逐步把 x0_pred
        投影到合法流形上（采样中投影），而非采样完成后一次性投影（事后修正）。

        渐进强度（projection_anneal=True）：投影强度 λ 从 project_from_frac 起
        由 0 线性增至 1，早期弱投影避免破坏采样轨迹，后期强投影保证合法性。

        参数:
            cond: (B, C_cond, D, H, W)，通道 2=荷载掩码，通道 3=支座掩码。
            use_projection: 是否在采样中逐步投影（流形保持）。
            project_from_frac: 从采样进度多少比例开始投影。
            projection_anneal: 是否渐进增强投影强度（True）还是立即硬投影（False）。
        """
        device = cond.device
        b = shape[0]
        x_t = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, ddim_steps).long().to(device)

        # 从条件提取工况掩码（通道 2=荷载，通道 3=支座）
        has_proj = use_projection and cond.shape[1] >= 4
        if has_proj:
            from projection.manifold_projection import manifold_projection
            src = cond[:, 2:3]
            sup = cond[:, 3:4]
            project_start = int(ddim_steps * project_from_frac)
            n_project_steps = max(ddim_steps - project_start, 1)

        for i in range(ddim_steps):
            t = times[i]
            t_next = times[i + 1] if i + 1 < ddim_steps else -1
            x_in = torch.cat([x_t, cond], dim=1)
            t_batch = torch.full((b,), t, device=device)
            eps = self.model(x_in, t_batch)
            a_t = self.alphas_cumprod[t]
            a_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0)
            x0_pred = (x_t - torch.sqrt(1 - a_t) * eps) / torch.sqrt(a_t)

            # 流形保持：逐步投影 x0_pred 到合法流形
            if has_proj and i >= project_start:
                proj = manifold_projection(x0_pred, source_mask=src, support_mask=sup)
                if projection_smooth:
                    # 投影后平滑：消除桥接路径的离散跳变，让输出接近自然密度场
                    proj = F.avg_pool3d(proj, kernel_size=3, stride=1, padding=1)
                if projection_anneal:
                    # 渐进强度：从 0 线性增至 1
                    lam = (i - project_start + 1) / n_project_steps
                    x0_pred = (1.0 - lam) * x0_pred + lam * proj
                else:
                    x0_pred = proj
                # 关键：投影后反推 eps，保持 x_t = sqrt(a_t)*x0 + sqrt(1-a_t)*eps 分解一致
                eps = (x_t - torch.sqrt(a_t) * x0_pred) / torch.sqrt(1 - a_t).clamp(min=1e-6)

            x_t = torch.sqrt(a_next) * x0_pred + torch.sqrt(1 - a_next) * eps

        return torch.clamp(x_t, 0.0, 1.0)


if __name__ == "__main__":
    model = UNet3D(in_channels=3, base=16, depth=2)
    diff = Diffusion(model, timesteps=100)
    x0 = torch.rand(2, 1, 16, 16, 16)
    cond = torch.rand(2, 2, 16, 16, 16)
    loss = diff.train_loss(x0, cond)
    print("train loss:", loss.item())
    loss.backward()
    print("backward OK")
    with torch.no_grad():
        s = diff.sample(cond[:1], (1, 1, 16, 16, 16), ddim_steps=10)
    print("sample shape:", s.shape, "range:", s.min().item(), s.max().item())
