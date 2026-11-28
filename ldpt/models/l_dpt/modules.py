"""
Building blocks of the L-DPT latent decoder: RoPE self-attention, the decoder block and
the pixel readout. Every class keeps the parameter names of the trained checkpoints.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..dinov3.layers import LayerScale, Mlp
from ..dinov3.layers.attention import rope_apply

__all__ = [
    "ATTN_DTYPES",
    "Attention",
    "Block",
    "LocalMix",
    "PixelReadout",
    "cast_linear_conv_",
]

# attention dtype -> the dtype q / k / v are cast to for SDPA (None: the input dtype)
ATTN_DTYPES = {"fp32": None, "bf16": torch.bfloat16}
# qk_norm (LatentDPT dec_qk_norm): tau_h = min(tau_init exp(QK_TAU_GAIN rho_h), tau_max);
# the gain lets Adam (a step of at most lr on rho) move tau by x9 in 4000 warm-up
# iterations
QK_TAU_GAIN = 30.0
QK_TAU_MAX = 200.0
QK_NORM_EPS = 1e-6


def check_choice(name: str, value, choices) -> None:
    """ValueError naming the valid choices when value is not one of them."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}")


def clip_norm(t: Tensor, r: float) -> Tensor:
    """t / max(1, |t| / r) over the last dim: the norm at most r, the direction kept.
    Bitwise t where |t| <= r: the divisor is exactly 1.0 there (a factor r / |t| rounds
    to 1 - 6e-8 for some r). Shapes: [..., d] -> [..., d]."""
    return t / (t.norm(dim=-1, keepdim=True) / r).clamp_min(1.0)


@torch.no_grad()
def cast_linear_conv_(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Cast the weights and biases of every Linear / Conv2d in module (in place)."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            for p in (m.weight, m.bias):
                if p is not None and p.dtype != dtype:
                    p.data = p.data.to(dtype)
    return module


class Attention(nn.Module):
    """Self-attention with 2D RoPE; q / k / v project dim -> dim // downscale.

    attn_dtype="bf16" casts q / k / v to bf16 for SDPA and the output back to the input
    dtype; "fp32" runs SDPA in the input dtype.
    force_fp32 (LatentDPT attn_fp32, an fp32 island in a bf16 decoder): q / k / v
    projections, RoPE, SDPA and out_proj run in fp32 on x.float() with autocast off,
    whatever the caller's autocast and attn_dtype; the output is fp32 (fp32_forward,
    eager also inside a compiled block).
    qk_bound (LatentDPT dec_qk_bound, a logit bound B > 0): per head, the query and
    the key of each token are clipped to an L2 norm of at most r = sqrt(B sqrt(d))
    before RoPE (q <- q min(1, r / |q|), in fp32), so |q.k| / sqrt(d) <= B for every
    pair (RoPE is a rotation). bf16 operands (SDPA in bf16) round each entry by up to
    2^-8: at most B (1 + 2^-8)^2 = 1.0078 B (201.57 at B = 200).
    No parameters; q / k below r are bitwise unchanged. In qkv, so the fp32 island and
    every SDPA backend see the bounded q / k.
    qk_norm (LatentDPT dec_qk_norm, cosine attention with a learned capped temperature,
    Swin V2): per head, q and k of each token are L2-normalised (fp32, eps 1e-6) and
    scaled by sqrt(tau_h sqrt(d)) before RoPE, so SDPA's own 1 / sqrt(d) gives the
    logit tau_h cos(q, R k) and |logit| <= tau_max for any weights and input (bf16
    operands: at most tau_max (1 + 2^-8)^2, 201.57 at 200). tau_h = min(tau_init
    exp(QK_TAU_GAIN rho_h), qk_tau_max) with rho_h = logit_scale[h] (zeros at init, no
    random number drawn) and tau_init = sqrt(d) / 3, the logit scale of the
    unnormalised attention at nn.Linear's default init; the clamp passes no gradient
    to rho_h. One parameter [heads] (first attn.* key); the loss has no gradient along
    the scale of W_q / W_k. Excludes qk_bound; composes with force_fp32 (in qkv).

    Shapes:
        x: [B, N, dim]; rope: (sin, cos), each [N, head_dim] -> [B, N, dim]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        downscale: int = 1,
        attn_dtype: str = "fp32",
        force_fp32: bool = False,
        qk_bound: Optional[float] = None,
        qk_norm: bool = False,
        qk_tau_max: float = QK_TAU_MAX,
    ) -> None:
        super().__init__()
        check_choice("attn_dtype", attn_dtype, ATTN_DTYPES)
        if dim % downscale or (dim // downscale) % num_heads:
            raise ValueError(
                f"dim // downscale must split into num_heads heads: dim={dim}, "
                f"downscale={downscale}, num_heads={num_heads}"
            )
        # a bool, a string ("None" from --options) or inf would not be a bound
        if qk_bound is not None and not (
            isinstance(qk_bound, (int, float))
            and not isinstance(qk_bound, bool)
            and 0 < qk_bound < math.inf
        ):
            raise ValueError(f"qk_bound must be None or a number > 0, got {qk_bound!r}")
        self.num_heads = num_heads
        self.attn_dtype = ATTN_DTYPES[attn_dtype]
        self.force_fp32 = bool(force_fp32)
        inner_dim = dim // downscale
        self.qk_bound = None if qk_bound is None else float(qk_bound)
        # r = sqrt(B sqrt(d)), d = head dim: 40.0 for B = 200, d = 64
        self.qk_radius = (
            None
            if qk_bound is None
            else math.sqrt(self.qk_bound * math.sqrt(inner_dim // num_heads))
        )
        if not isinstance(qk_norm, bool):
            raise ValueError(f"qk_norm must be True or False, got {qk_norm!r}")
        self.qk_norm = qk_norm
        if self.qk_norm:
            if self.qk_bound is not None:
                raise ValueError("qk_norm bounds every logit: qk_bound must be None")
            self.tau_init = math.sqrt(inner_dim // num_heads) / 3.0
            if not (
                isinstance(qk_tau_max, (int, float))
                and not isinstance(qk_tau_max, bool)
                and self.tau_init <= qk_tau_max < math.inf
            ):
                raise ValueError(
                    f"qk_tau_max must be a number in [tau_init = {self.tau_init:.4g}, "
                    f"inf), got {qk_tau_max!r}"
                )
            self.tau_max = float(qk_tau_max)
            # rho_h, zeros: no random number drawn (every other initial weight as
            # without qk_norm); registered first, the first attn.* key
            self.logit_scale = nn.Parameter(torch.zeros(num_heads))
        self.q_proj = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(inner_dim, dim, bias=proj_bias)

    def temperature(self) -> Tensor:
        """qk_norm: the per-head logit scale tau_h, [heads], fp32 (no gradient to
        logit_scale where the clamp binds)."""
        tau = self.tau_init * torch.exp(QK_TAU_GAIN * self.logit_scale.float())
        return tau.clamp(max=self.tau_max)

    def qkv(
        self,
        x: Tensor,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
        bound: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """q, k, v as forward hands them to SDPA, before the attn_dtype cast: heads
        split, q / k clipped to the qk_bound radius (bound=False: unclipped, for the
        log statistics) or normalised and scaled (qk_norm, whatever bound), RoPE
        applied to q and k.
        Shapes: [B, N, dim] -> 3 x [B, heads, N, d]."""
        B, N, _ = x.shape
        C = self.q_proj.out_features
        q, k, v = [
            proj(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        ]
        clip = self.qk_bound is not None and bound
        # qk_norm: sqrt(tau_h sqrt(d)) on q and on k, |q'| |k'| / sqrt(d) = tau_h
        gain = (
            torch.sqrt(self.temperature() * math.sqrt(C // self.num_heads))[
                None, :, None, None
            ]
            if self.qk_norm
            else None
        )
        # q finished (back in its dtype) before k starts: keeps the peak memory down
        q = self._clip_rope(q, rope, clip, gain)
        k = self._clip_rope(k, rope, clip, gain)
        return q, k, v

    def _clip_rope(
        self,
        t: Tensor,
        rope: Optional[Tuple[Tensor, Tensor]],
        clip: bool,
        gain: Optional[Tensor] = None,
    ) -> Tensor:
        """One of q / k: normalised and scaled by gain (qk_norm, [1, heads, 1, 1]) or
        clipped to the qk_bound radius (clip), rotated by RoPE, back in the Linear's
        dtype (bf16 under autocast). No cast between the normalisation / clip and the
        rotation. Shapes: [B, heads, N, d] -> [B, heads, N, d]."""
        dtype = t.dtype
        if gain is not None:  # fp32; a per-head scalar commutes with the rotation
            t = F.normalize(t.float(), dim=-1, eps=QK_NORM_EPS) * gain
        elif clip:  # .float() is exact for bf16; the norm and the divisor in fp32
            t = clip_norm(t.float(), self.qk_radius)
        if rope is not None:
            # latent tokens only (no prefix tokens); RoPE math in the rope dtype (fp32)
            sin, cos = rope
            t = rope_apply(t.to(sin.dtype), sin, cos)
        # a no-op without the bound, qk_norm and RoPE (.to of a tensor of that dtype)
        return t.to(dtype)

    def forward(
        self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor:
        if self.force_fp32:
            return self.fp32_forward(x, rope=rope)
        B, N, _ = x.shape
        C = self.q_proj.out_features
        q, k, v = self.qkv(x, rope=rope)
        if self.attn_dtype is not None:
            dtype = q.dtype
            q, k, v = [t.to(self.attn_dtype) for t in (q, k, v)]
            x = F.scaled_dot_product_attention(q, k, v).to(dtype)
        else:
            x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)

    @torch.compiler.disable
    def fp32_forward(
        self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor:
        """forward of the force_fp32 island. Eager also inside a compiled block
        (a graph break): AOT autograd (torch 2.8) traces a graph's backward under the
        caller's autocast, which would run the island's gradient matmuls in bf16."""
        B, N, _ = x.shape
        C = self.q_proj.out_features
        with torch.autocast(x.device.type, enabled=False):
            q, k, v = self.qkv(x.float(), rope=rope)
            x = F.scaled_dot_product_attention(q, k, v)
            return self.out_proj(x.transpose(1, 2).reshape(B, N, C))


class LocalMix(nn.Module):
    """Token mixer of a convolutional decoder block (LatentDPT dec_conv_mixer): a 3x3
    depthwise convolution on the token grid (replicate padding, bias), then a Linear.
    The depthwise kernel starts as the identity (centre tap 1, bias 0), so the block
    starts as a per-token linear map. The convolution runs in fp32 with autocast off
    (forward and backward); out_proj runs under the caller's autocast, like
    attn.out_proj. No RoPE, no softmax.

    Shapes:
        x: [B, N, dim], hw: (h, w) with h * w = N (row-major token grid) -> [B, N, dim]
    """

    def __init__(self, dim: int, proj_bias: bool = True) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim, 3, padding=1, padding_mode="replicate", groups=dim, bias=True
        )
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)
        with torch.no_grad():
            self.dw.weight.zero_()
            self.dw.weight[:, 0, 1, 1] = 1.0
            self.dw.bias.zero_()

    def forward(self, x: Tensor, hw: Optional[Tuple[int, int]]) -> Tensor:
        b, n, c = x.shape
        if hw is None or n != hw[0] * hw[1]:
            raise ValueError(f"LocalMix: {n} tokens do not form the grid hw={hw}")
        h, w = hw
        with torch.autocast(x.device.type, enabled=False):
            # [B, N, C] -> [B, C, h, w], fp32 and contiguous in one copy: a channels-last
            # input gives dw.weight a gradient with other strides than the parameter
            # (DDP warns "Grad strides do not match bucket view strides")
            y = x.transpose(1, 2).reshape(b, c, h, w)
            y = self.dw(y.to(torch.float32, memory_format=torch.contiguous_format))
        return self.out_proj(y.flatten(2).transpose(1, 2))


class Block(nn.Module):
    """Decoder block. The attention residual is added to the *normalised* tensor
    (x = norm1(x); x = x + attn(x)), not to the block input as in a standard pre-LN
    block: the released checkpoints depend on it. chunk_tokens (no autograd only) runs
    the FFN on token chunks. attn_fp32: the attention is an fp32 island (Attention
    force_fp32); qk_bound / qk_norm / qk_tau_max: Attention's. convert_mixer_ replaces
    the attention by LocalMix (mixer "conv", LatentDPT dec_conv_mixer): x = norm1(x);
    x = x + ls1(local(x, hw)); the FFN as in an attention block. Such a block needs
    the token grid hw and ignores rope.

    Shapes:
        x: [B, N, dim]; rope: (sin, cos), each [N, head_dim]; hw: (h, w), h * w = N
        -> [B, N, dim]
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        downscale: int = 1,
        attn_dtype: str = "fp32",
        init_values: Optional[float] = None,
        attn_fp32: bool = False,
        qk_bound: Optional[float] = None,
        qk_norm: bool = False,
        qk_tau_max: float = QK_TAU_MAX,
    ) -> None:
        super().__init__()
        self.mixer = "attn"
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            downscale=downscale,
            attn_dtype=attn_dtype,
            force_fp32=attn_fp32,
            qk_bound=qk_bound,
            qk_norm=qk_norm,
            qk_tau_max=qk_tau_max,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(dim, int(dim * ffn_ratio), act_layer=nn.GELU, bias=ffn_bias)
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    @torch.no_grad()
    def convert_mixer_(self) -> None:
        """Replace the attention by LocalMix (same proj_bias), in place. Draws from the
        global RNG (out_proj's default init): LatentDPT converts at the end of its
        __init__ and restores the RNG state afterwards."""
        proj_bias = self.attn.out_proj.bias is not None
        dim = self.attn.q_proj.in_features
        del self.attn
        self.local = LocalMix(dim, proj_bias=proj_bias)
        self.mixer = "conv"

    def forward(
        self,
        x: Tensor,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
        chunk_tokens: Optional[int] = None,
        hw: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        # residual on the normalised tensor (the checkpoints depend on it), not on the
        # block input
        x = self.norm1(x)
        if self.mixer == "conv":
            x = x + self.ls1(self.local(x, hw))
        else:
            x = x + self.ls1(self.attn(x, rope=rope))
        if chunk_tokens is None or torch.is_grad_enabled():
            return x + self.ls2(self.mlp(self.norm2(x)))
        # inference: the FFN on token chunks, added in place (no [B, N, hidden] tensor)
        for a in range(0, x.shape[1], chunk_tokens):
            x_chunk = x[:, a : a + chunk_tokens]
            x_chunk += self.ls2(self.mlp(self.norm2(x_chunk)))
        return x


class PixelReadout(nn.Module):
    """Readout: k x k conv (k=1 is a linear projection) + pixel shuffle.

    Shapes:
        x: [B, in_channels, h, w] -> [B, out_channels, h * P, w * P] (P = patch_size)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        patch_size: int = 1,
    ) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"readout_kernel_size must be odd and >= 1 (3: released weights, "
                f"1: a linear projection), got {kernel_size}"
            )
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            out_channels * patch_size * patch_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x: Tensor) -> Tensor:
        # contiguous NCHW output (the conv ran on a channels-last view): a no-op for
        # one output channel, and it keeps the semseg 1x1 head bit-exact
        return F.pixel_shuffle(self.proj(x), self.patch_size).contiguous()
