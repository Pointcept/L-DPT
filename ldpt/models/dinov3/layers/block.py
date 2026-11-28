# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .attention import SelfAttention, rope_apply
from .ffn_layers import Mlp
from .layer_scale import LayerScale


class SelfAttentionBlock(nn.Module):
    """Pre-LN ViT block of DINOv3 (inference copy: no stochastic depth, no list input).

    Shapes:
        x: [B, N, dim] (prefix tokens first); rope: (sin, cos), each [N - prefix, D_head]
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
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        if drop_path:
            raise NotImplementedError(
                "the vendored DINOv3 block has no stochastic depth (drop_path=0 only)"
            )
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )

    def forward(
        self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tensor:
        x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
        x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
        return x_ffn

    def forward_chunked(
        self,
        x: Tensor,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
        chunk_tokens: int = 65536,
        inplace: bool = False,
    ) -> Tensor:
        """forward() for inference at large N, without its full-size intermediates.

        norm1, qkv and RoPE run on chunks of chunk_tokens tokens into one q / k / v
        buffer; then per chunk of queries: attention over all keys, proj, both
        residuals, norm2 and the FFN. Same per-token math as forward(); matrix products
        over fewer rows may round differently. Inference only (no autograd). inplace
        writes the output into x (the caller must not need x afterwards).
        """
        if torch.is_grad_enabled():
            raise RuntimeError("forward_chunked is inference only (torch.no_grad)")
        B, N, C = x.shape
        H = self.attn.num_heads
        # the prefix tokens (class, storage) get no RoPE: they form their own chunk
        prefix = N - rope[0].shape[-2] if rope is not None else 0
        qkv = None  # [3, B, H, N, C // H]
        for a, b in _chunks(0, prefix) + _chunks(prefix, N, chunk_tokens):
            t = self.attn.qkv(self.norm1(x[:, a:b]))
            t = t.reshape(B, b - a, 3, H, C // H).permute(2, 0, 3, 1, 4)
            if qkv is None:
                qkv = t.new_empty(3, B, H, N, C // H)
            if rope is not None and a >= prefix:
                # as SelfAttention.apply_rope: RoPE in the rope dtype, cast back on copy
                sin, cos = (r[a - prefix : b - prefix] for r in rope)
                for j in (0, 1):  # q, k
                    qkv[j, :, :, a:b] = rope_apply(t[j].to(sin.dtype), sin, cos)
                qkv[2, :, :, a:b] = t[2]
            else:
                qkv[:, :, :, a:b] = t
            del t
        q, k, v = qkv.unbind(0)
        y = None
        for a, b in _chunks(0, N, chunk_tokens):
            o = F.scaled_dot_product_attention(q[:, :, a:b], k, v)
            o = o.transpose(1, 2).reshape(B, b - a, C)
            x_attn = x[:, a:b] + self.ls1(self.attn.proj_drop(self.attn.proj(o)))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
            if y is None:
                same = inplace and x_ffn.dtype == x.dtype
                y = x if same else x_ffn.new_empty(B, N, C)
            y[:, a:b] = x_ffn
            del o, x_attn, x_ffn
        return y


def _chunks(start: int, stop: int, size: Optional[int] = None) -> List[Tuple[int, int]]:
    """[(a, b)] covering range(start, stop) in pieces of at most size (one if None)."""
    size = size or max(stop - start, 1)
    return [(a, min(a + size, stop)) for a in range(start, stop, size)]
