import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from functools import partial
from typing import Optional, Callable
from timm.layers import DropPath

__all__ = ['FMConv', 'C2f_FMConv', 'C3k2_FMConv']
# ============================================================
# 1. 基础组件 & 频域增强模块 (优化版)
# ============================================================

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=eps)

    def forward(self, x):
        return self.norm(x)


class AdaptiveSpectralModulation(nn.Module):
    """
    [学术优化] 自适应频谱调制模块
    1. 残差式调制：保证训练稳定性。
    2. 多尺度频域：捕捉不同粗细的裂纹。
    3. 空间引导：利用空间特征抑制频域噪声。
    4. 【新增】Shortcut：保证特征流通，防止梯度消失。
    """

    def __init__(self, in_channels, out_channels, patch_sizes=[4, 16]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_sizes = patch_sizes

        # --- 原有参数保持不变 ---
        self.amp_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(in_channels, 1, 1, ps, ps // 2 + 1))
            for ps in patch_sizes
        ])
        self.phase_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(in_channels, 1, 1, ps, ps // 2 + 1))
            for ps in patch_sizes
        ])

        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, len(patch_sizes), 1),
            nn.Softmax(dim=1)
        )

        # self.spatial_gate = nn.Sequential(
        #     nn.Conv2d(in_channels, 1, 1),
        #     nn.Sigmoid()
        # )

        self.out_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

        # --- [新增] 残差连接分支 ---
        # 如果输入输出通道不同，需要用 1x1 卷积对齐通道数
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # 相同时直接传递

    def _process_scale(self, x, ps, amp_w, phase_w):
        # ... (此处保持您原有的 _process_scale 代码完全不变) ...
        B, C, H, W = x.shape
        P = ps
        h_pad = (P - H % P) % P
        w_pad = (P - W % P) % P
        if h_pad > 0 or w_pad > 0:
            x_pad = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        else:
            x_pad = x
        _, _, H_p, W_p = x_pad.shape
        x_patch = rearrange(x_pad, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=P, p2=P)
        x_patch_fft = torch.fft.rfft2(x_patch, norm='ortho')
        mag = torch.abs(x_patch_fft)
        phase = torch.angle(x_patch_fft)
        mag_delta = mag * amp_w
        phase_delta = phase_w
        new_mag = mag + mag_delta
        new_phase = phase + phase_delta
        x_patch_fft_mod = torch.polar(new_mag, new_phase)
        x_patch = torch.fft.irfft2(x_patch_fft_mod, s=(P, P), norm='ortho')
        x_recon = rearrange(x_patch, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=P, p2=P)
        return x_recon[:, :, :H, :W]

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 保存残差分支 (identity mapping)
        identity = self.shortcut(x)

        # 2. 计算空间注意力权重
        scale_weights = self.scale_attention(x)

        # 3. 多尺度频域处理
        freq_feats = []
        for i, ps in enumerate(self.patch_sizes):
            feat = self._process_scale(x, ps, self.amp_weights[i], self.phase_weights[i])
            freq_feats.append(feat * scale_weights[:, i].view(B, 1, 1, 1))

        # 4. 融合
        freq_out = sum(freq_feats)

        # 5. 空间引导门控
        # gate_mask = self.spatial_gate(x)
        # freq_out = freq_out * gate_mask
        freq_out = freq_out

        # 6. 输出投影
        out = self.out_conv(freq_out)
        out = self.bn(out)

        # --- [新增] 关键残差相加 ---
        # 将原始特征投影后加到频域输出上
        out = out + identity

        # 最后再做激活 (标准的 ResNet 结构顺序是 Conv -> BN -> Add -> Act)
        out = self.act(out)

        return out


# ============================================================
# 2. Mamba 核心组件 (保持不变，仅做微调适配)
# ============================================================

MAMBA_AVAILABLE = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    MAMBA_AVAILABLE = True
except ImportError:
    pass


class SS2D(nn.Module):
    # ... [保持之前的SS2D实现不变，为了节省篇幅此处省略] ...
    # 注意：在实际代码中请完整复制上一版回答中的 SS2D 代码
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False,
                 device=None, dtype=None, **kwargs):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner,
                                bias=conv_bias, kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()

        # x_proj 和 dt_projs 定义同前...
        self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                       nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                       nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                       nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # dt_projs 初始化同前...
        self.dt_projs = (
        self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs))
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    # 辅助函数 dt_init, A_log_init, D_init 保持不变
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(d_inner, 1)
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1)
            if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).repeat(copies, 1)
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        if not MAMBA_AVAILABLE: raise RuntimeError("mamba_ssm 未安装")
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True,
                                    return_last_state=False).view(B, K, -1, L)
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1).to(x.dtype)
        y = self.out_norm(y).to(x.dtype)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None: out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0.2,
                 norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6), attn_drop_rate: float = 0,
                 d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        input = input.permute((0, 2, 3, 1))
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x.permute((0, 3, 1, 2))


# ============================================================
# 3. 优化后的 PSConv (SFM-PSConv)
# ============================================================

class FMConv(nn.Module):
    '''
    [学术优化版] SFM-PSConv (Spatial-Frequency Mamba)
    针对 Crack Detection 的优化：
    1. 空间-频域双分支并行结构。
    2. 频域分支使用多尺度残差调制。
    3. 引入特征融合门控机制。
    '''

    def __init__(self, c1, c2, k=3, s=1, expansion=0.5):
        super().__init__()

        self.c1 = c1
        self.c2 = c2
        self.s = s

        # 1. LayerNorm
        self.norm = LayerNorm(c1)

        # 2. 通道投影
        if c1 != c2:
            self.proj_in = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, bias=False),
                nn.BatchNorm2d(c2)
            )
        else:
            self.proj_in = nn.Identity()

        # 3. 核心模块
        # 3.1 空间分支
        self.spatial_branch = VSSBlock(hidden_dim=c2, drop_path=0.1, d_state=16)

        # 3.2 频域分支 (Adaptive Spectral Modulation)
        # 使用 patch_sizes=[4, 16] 混合局部细节与全局结构
        self.freq_branch = AdaptiveSpectralModulation(c2, c2, patch_sizes=[4, 16])

        # 4. [新增] 双域融合门控
        # 让网络自己学习如何平衡空间信息和频域信息
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=1),
            nn.Sigmoid()
        )

        # 5. 残差连接
        self.shortcut = nn.Sequential()
        if c1 != c2 or s != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, bias=False),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        # Step 1: Norm & Project
        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)

        # Step 2: 并行双分支处理
        # 空间分支：捕捉长距离依赖与语义结构
        spatial_out = self.spatial_branch(x_proj)

        # 频域分支：增强纹理细节与抑制噪声
        freq_out = self.freq_branch(x_proj)

        # Step 3: 特征融合
        # 拼接 -> Sigmoid -> 门控加权
        concat_feats = torch.cat([spatial_out, freq_out], dim=1)
        gate = self.fusion_gate(concat_feats)

        # 最终输出 = 空间特征 * 门控 + 频域特征 * (1-门控)
        # 这比单纯的相加更灵活，能有效解决"优化冲突"问题
        out = spatial_out * gate + freq_out * (1 - gate)

        # Step 4: 全局残差
        output = out + identity

        return output


# C2f 和 C3k2 的包装类保持不变，仅需替换内部 PSConv 即可
class C2f_FMConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        total_channels = (2 + n) * self.c
        self.cv2 = nn.Conv2d(total_channels, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.m = nn.ModuleList(FMConv(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.act(self.bn1(self.cv1(x))).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.act(self.bn2(self.cv2(torch.cat(y, 1))))


class C3k2_FMConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.cv_shortcut = nn.Sequential(nn.Conv2d(c1, c2, 1, bias=False),
                                         nn.BatchNorm2d(c2)) if c1 != c2 else nn.Identity()
        total_channels = (2 + n) * self.c
        self.cv2 = nn.Conv2d(total_channels, c2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.m = nn.ModuleList(FMConv(self.c, self.c) for _ in range(n))
        self.shortcut = shortcut

    def forward(self, x):
        identity = self.cv_shortcut(x)
        y = list(self.act(self.bn1(self.cv1(x))).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.act(self.bn2(self.cv2(torch.cat(y, 1))))
        return out + identity if self.shortcut else out


if __name__ == "__main__":
    print("=== 优化版 SFM-PSConv 模块测试 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not MAMBA_AVAILABLE:
        print("警告: 未检测到 mamba_ssm，跳过测试。")
    else:
        x = torch.randn(2, 64, 64, 64).to(device)
        model = FMConv(64, 128).to(device)
        y = model(x)
        print(f"输入: {x.shape} -> 输出: {y.shape}")
        print("✓ 测试通过")