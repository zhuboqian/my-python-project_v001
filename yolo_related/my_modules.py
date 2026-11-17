# my_modules.py
import torch
import torch.nn as nn


class SPDConv(nn.Module):
    """Space-to-Depth + Conv 下采样模块"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.stride = 2
        self.conv = nn.Conv2d(c1 * 4, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C*4, H//2, W//2)
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class IlluminationResidualGate(nn.Module):
    """Feature-illumination-aware residual channel recalibration.

    The block is intentionally identity-preserving at initialization.  A
    sample-wise gate is inferred from global feature mean and standard
    deviation, while a lightweight SE branch predicts channel modulation.
    The residual strength is learned through a bounded scalar.  This lets the
    network use stronger recalibration for illumination-degraded samples
    without forcing the same attention response on every image.

    This is a shape-preserving standalone layer so it can follow a normal C2f
    block without modifying Ultralytics' channel parser.
    """

    def __init__(self, channels, reduction=16, gate_hidden=8, init_strength=0.05):
        super().__init__()
        hidden = max(int(channels) // int(reduction), 4)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.condition_mlp = nn.Sequential(
            nn.Linear(2, max(int(gate_hidden), 4)),
            nn.SiLU(inplace=True),
            nn.Linear(max(int(gate_hidden), 4), 1),
            nn.Sigmoid(),
        )
        # tanh keeps the learned residual bounded.  atanh(0.05) initializes
        # the complete block very close to an identity mapping.
        init = torch.tensor(float(init_strength)).clamp(-0.99, 0.99)
        self.residual_logit = nn.Parameter(torch.atanh(init))

    def forward(self, x):
        # Use the second-moment identity instead of torch.std.  The latter is
        # exported as a large backend-specific reduction graph on Jetson and
        # produced outputs inconsistent with PyTorch.  This form is
        # mathematically equivalent for population variance and keeps the
        # reduction explicitly in FP32 for ONNX/TensorRT deployment.
        x32 = x.float()
        mean32 = x32.mean(dim=(1, 2, 3))
        second_moment = (x32 * x32).mean(dim=(1, 2, 3))
        variance = torch.clamp(second_moment - mean32 * mean32, min=0.0)
        std = torch.sqrt(variance).to(x.dtype)
        mean = mean32.to(x.dtype)
        condition = torch.stack((mean, std), dim=1)
        sample_gate = self.condition_mlp(condition).view(-1, 1, 1, 1)
        # Convert sigmoid channel weights to signed modulation in [-1, 1].
        channel_delta = 2.0 * self.channel_mlp(x) - 1.0
        strength = torch.tanh(self.residual_logit)
        return x + strength * sample_gate * x * channel_delta


class C2fAttn(nn.Module):
    """C2f with channel attention (SE block) — 适用于顺序前向传播
    签名兼容 ultralytics 内置 C2fAttn 的参数注入规则:
      parse_model 会注入 c1, c2, n, ec, nh, 后续可接 shortcut, g, e
    """
    def __init__(self, c1, c2, n=1, ec=128, nh=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.se = SEBlock(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.act(self.bn1(self.cv1(x))).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.act(self.bn2(self.cv2(torch.cat(y, 1))))
        return self.se(out)


class Bottleneck(nn.Module):
    """Standard bottleneck, 复用 ultralytics 的 Bottleneck 逻辑"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, k[0] // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, k[1] // 2, groups=g, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.bn2(self.cv2(self.act(self.bn1(self.cv1(x))))) if self.add else self.bn2(self.cv2(self.act(self.bn1(self.cv1(x)))))

class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module (SimAM).

    Based on neuroscience theory: informative neurons show different firing
    patterns from surrounding neurons. Computes per-neuron importance via
    an energy function over spatial+channel dimensions.

    Zero parameters, negligible FLOPs — just mean, variance, and multiply.
    Reference: Yang et al., "SimAM: A Simple, Parameter-Free Attention Module
    for Convolutional Neural Networks", ICML 2021.
    """
    def __init__(self, channels=None, lambda_e=1e-4):
        super().__init__()
        self.lambda_e = lambda_e

    def forward(self, x):
        # x: [B, C, H, W]
        n = x.shape[2] * x.shape[3] - 1  # spatial dims - 1
        # Channel-spatial mean per neuron
        mu = x.mean(dim=[2, 3], keepdim=True)
        # (x - mu)^2 averaged over H*W per channel
        d = (x - mu).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        # Energy: e_inv = 4*(σ²+λ) / ((x-μ)² + 2σ² + 2λ)
        e_inv = d / (4 * (v + self.lambda_e)) + 0.5
        # Attention = sigmoid(1/E) ≈ sigmoid(e_inv)
        attn = torch.sigmoid(e_inv)
        return x * attn


class SimAMBlock(nn.Module):
    """Standalone SimAM as a separate layer for YAML injection.

    Usage in YAML: [-1, 1, SimAMBlock, []] — applies attention to previous feature map.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.simam = SimAM()

    def forward(self, x):
        return self.simam(x)


class DARMBlock(nn.Module):
    """Degradation-Aware Recalibration Module — 多统计量通道注意力.
    GAP (均值): 感知通道整体激活强度偏移
    GMP (最大值): 感知局部饱和/过曝尖峰
    GVP (方差): 感知纹理复杂度变化（平滑→杂波）
    三者互补: 均值感知偏移、极值感知异常、方差感知纹理变化。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # GVP: Global Variance Pooling via E[x^2] - E[x]^2
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 3, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # GAP: channel-wise mean
        gap = self.gap(x)
        # GMP: channel-wise max
        gmp = self.gmp(x)
        # GVP: channel-wise variance = E[x^2] - E[x]^2
        x_sq = x * x
        var = self.gap(x_sq) - gap * gap
        # Concat all 3 statistics
        stats = torch.cat([gap, gmp, var], dim=1)  # [B, 3C, 1, 1]
        weights = self.fc(stats)  # [B, C, 1, 1]
        return x * weights


class GDARMBlock(nn.Module):
    """Gated-DARM — 多统计量注意力+自适应门控.
    门控机制: 当GMP远大于GAP(局部异常尖峰)或GVP远大于GAP(纹理突变)时,
    自动压低对应统计量的贡献,避免退化噪声被放大.
    """
    def __init__(self, channels, reduction=16, temperature=1.0):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.T = nn.Parameter(torch.tensor(temperature), requires_grad=True)  # learnable
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 3, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gap = self.gap(x)
        gmp = self.gmp(x)
        x_sq = x * x
        gvp = self.gap(x_sq) - gap * gap
        # Adaptive gating: suppress GMP/GVP when they deviate from GAP
        w_gmp = torch.sigmoid((gap - gmp) / torch.abs(self.T) + 0.5)  # ~1 when similar, ~0 when gmp >> gap
        w_gvp = torch.sigmoid((gap - gvp) / torch.abs(self.T) + 0.5)
        stats = torch.cat([gap, w_gmp * gmp, w_gvp * gvp], dim=1)
        weights = self.fc(stats)
        return x * weights


class C2fGDARM(nn.Module):
    """C2f with GDARM (Gated DARM) attention — 签名兼容ultralytics YAML解析."""
    def __init__(self, c1, c2, n=1, ec=128, nh=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.se = GDARMBlock(c2)  # stored as self.se for C2fAttn compat
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.act(self.bn2(out) + self.se(out))


class C2fDARM(nn.Module):
    """C2f with DARM attention — 签名兼容ultralytics YAML解析.
    用法: 在YAML中写成 C2fDARM [c2, n, ec], 与C2f/C2fAttn完全互换.
    """
    def __init__(self, c1, c2, n=1, ec=128, nh=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.darm = DARMBlock(c2)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.act(self.bn2(out) + self.darm(out))
