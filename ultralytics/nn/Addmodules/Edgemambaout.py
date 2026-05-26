
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except ImportError:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor


# ===================== 基础工具 =====================
def autopad(k, p=None, d=1):
    if isinstance(k, int):
        k = d * (k - 1) + 1 if d > 1 else k
    else:
        k = tuple(d * (x - 1) + 1 if d > 1 else x for x in k)

    if p is None:
        p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # 更快：var_mean 一次性得到方差和均值
        var, mean = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
        x = (x - mean) * torch.rsqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _to_odd_int(v, min_v=3, max_v=5):
    """把可能的 float / str / int 转成奇数 int，并限制范围"""
    v = int(v)
    if v < min_v:
        v = min_v
    if v % 2 == 0:
        v += 1
    if v > max_v:
        v = max_v if (max_v % 2 == 1) else max_v - 1
    return v


# ================================================================
#  轻量方向扫描器：GatedHVMixer
# ================================================================
class GatedHVMixer(nn.Module):
    """
    轻量版 GatedHVMixer：
    1) 归一化
    2) 横/纵方向差分扫描
    3) depthwise 扫描增强
    4) 通道门控
    """
    def __init__(self, dim, num_heads=4, kv_bias=True):
        super().__init__()
        self.num_heads = num_heads  # 保留接口兼容，但不做重注意力
        self.norm = LayerNorm2d(dim)

        self.scan_h = nn.Conv2d(
            dim, dim, kernel_size=(1, 3), padding=(0, 1),
            groups=dim, bias=False
        )
        self.scan_v = nn.Conv2d(
            dim, dim, kernel_size=(3, 1), padding=(1, 0),
            groups=dim, bias=False
        )

        hidden = max(dim // 16, 8)  # 更轻
        self.edge_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1, bias=True),
            nn.Sigmoid()
        )

        self.row_gate = nn.Parameter(torch.tensor(0.5))
        self.col_gate = nn.Parameter(torch.tensor(0.5))
        self.scan_gate = nn.Parameter(torch.tensor(0.5))

        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        x = self.norm(x)

        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]

        dx = F.pad(dx, (1, 0, 0, 0))
        dy = F.pad(dy, (0, 0, 1, 0))

        scan = self.scan_h(dx) + self.scan_v(dy)
        gate = self.edge_gate(x)

        out = self.row_gate * dx + self.col_gate * dy + self.scan_gate * scan
        out = out * gate
        return self.proj(F.silu(out))


# ================================================================
#  Haar 小波变换：切片极速版
# ================================================================
class HaarDWT(nn.Module):
    """
    更快的 Haar DWT：
    - 不用 conv2d
    - 用 2x2 slicing 直接计算
    - 输出 [B, 4C, H/2, W/2]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 保留 buffer，兼容原 state_dict 结构（虽然 forward 不再依赖它）
        ll = torch.tensor([[1, 1],
                           [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1],
                           [ 1,  1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1,  1],
                           [-1,  1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[ 1, -1],
                           [-1,  1]], dtype=torch.float32) * 0.5
        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("weight", weight.repeat(dim, 1, 1, 1))

    def forward(self, x):
        _, _, h, w = x.shape
        if (h & 1) or (w & 1):
            x = F.pad(x, (0, w & 1, 0, h & 1), mode="reflect")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5

        return torch.cat([ll, lh, hl, hh], dim=1)


class WaveletEdgeEnhance(nn.Module):
    """
    从小波高频子带提取边缘信息
    """
    def __init__(self, dim):
        super().__init__()
        self.dwt = HaarDWT(dim)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        coeffs = self.dwt(x)
        ll, lh, hl, hh = coeffs.chunk(4, dim=1)

        detail = lh.abs() + hl.abs() + hh.abs()
        detail = F.interpolate(detail, size=x.shape[-2:], mode="nearest")
        return self.fuse(detail)


# ================================================================
#  EdgeMambaOut：最终极速版
# ================================================================
class EdgeMambaOut(nn.Module):
    """
    保持类定义不变：
    - 输入输出通道仍然是 dim -> dim
    - 构造参数不变
    - 仅优化内部实现
    """
    def __init__(self, dim, k_size=11, num_heads=4, drop_path=0.):
        super().__init__()

        # 更偏速度：限制核大小到 5
        k_size = _to_odd_int(k_size, min_v=3, max_v=5)

        # ① 条形卷积：局部边缘
        self.strip_h = nn.Conv2d(
            dim, dim, kernel_size=(1, k_size),
            padding=(0, k_size // 2), groups=dim, bias=True
        )
        self.strip_v = nn.Conv2d(
            dim, dim, kernel_size=(k_size, 1),
            padding=(k_size // 2, 0), groups=dim, bias=True
        )
        self._init_edge_kernels()

        # ② 小波高频分支
        self.wavelet = WaveletEdgeEnhance(dim)
        self.wavelet_gate = nn.Parameter(torch.tensor(0.5))

        # ③ 门控方向混合分支
        self.norm = LayerNorm2d(dim)
        self.in_proj = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.mixer = GatedHVMixer(dim, num_heads=num_heads)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

        # ④ 三路融合
        self.proj = nn.Conv2d(dim * 3, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)

        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _init_edge_kernels(self):
        """初始化成差分风格卷积核"""
        with torch.no_grad():
            # 横向条形卷积核
            kh = self.strip_h.weight
            k_h = kh.shape[-1]
            h_vec = torch.arange(-(k_h // 2), k_h // 2 + 1, dtype=kh.dtype, device=kh.device)
            h_vec = h_vec / (h_vec.abs().sum() + 1e-6)
            self.strip_h.weight.copy_(h_vec.view(1, 1, 1, k_h).repeat(kh.shape[0], 1, 1, 1))

            # 纵向条形卷积核
            kv = self.strip_v.weight
            k_v = kv.shape[-2]
            v_vec = torch.arange(-(k_v // 2), k_v // 2 + 1, dtype=kv.dtype, device=kv.device)
            v_vec = v_vec / (v_vec.abs().sum() + 1e-6)
            self.strip_v.weight.copy_(v_vec.view(1, 1, k_v, 1).repeat(kv.shape[0], 1, 1, 1))

            if self.strip_h.bias is not None:
                nn.init.zeros_(self.strip_h.bias)
            if self.strip_v.bias is not None:
                nn.init.zeros_(self.strip_v.bias)

    def forward(self, x):
        # 1) 条形卷积分支
        h_edge = self.strip_h(x)
        v_edge = self.strip_v(x)
        edge_mag = h_edge.abs() + v_edge.abs()

        # 2) 小波高频分支
        wave_edge = self.wavelet(x)

        # 3) 门控混合分支
        xz = self.in_proj(self.norm(x))
        x_proj, z = xz.chunk(2, dim=1)
        x_mix = self.mixer(x_proj)
        m_out = self.out_proj(x_mix * F.silu(z))

        # 4) 三路融合
        fused = torch.cat([
            h_edge,
            v_edge,
            edge_mag + self.wavelet_gate * wave_edge + m_out
        ], dim=1)

        out = self.bn(self.proj(fused))

        # 5) 残差输出
        return x + self.drop_path(self.gamma * out)


# ================================================================
#  YOLO 适配：C3k / C3k2 封装
# ================================================================
class C3k_EdgeMambaOut(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(EdgeMambaOut(c_) for _ in range(n))
        self.shortcut = shortcut

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class C3k2_CMamba(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)

        if c3k:
            self.m = nn.ModuleList(
                C3k_EdgeMambaOut(c_, c_, 2, shortcut, g) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(EdgeMambaOut(c_) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# ===================== 简单自测 =====================
if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 32)

    m1 = EdgeMambaOut(dim=64, k_size=11, num_heads=4, drop_path=0.0)
    y1 = m1(x)
    print(f'EdgeMambaOut: {x.shape} -> {y1.shape}  params={sum(p.numel() for p in m1.parameters()):,}')

    m2 = C3k2_CMamba(64, 128, n=2, c3k=False)
    y2 = m2(x)
    print(f'C3k2_CMamba:  {x.shape} -> {y2.shape}  params={sum(p.numel() for p in m2.parameters()):,}')

    m3 = C3k2_CMamba(64, 128, n=2, c3k=True)
    y3 = m3(x)
    print(f'C3k2_CMamba(c3k): {x.shape} -> {y3.shape}  params={sum(p.numel() for p in m3.parameters()):,}')

    # 简单测速
    import time
    x_test = torch.randn(4, 64, 64, 64)
    m1.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(50):
            _ = m1(x_test)
    t_out = time.time() - t0
    print(f'EdgeMambaOut 50次耗时: {t_out:.3f}s')

