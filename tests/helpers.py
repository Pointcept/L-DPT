"""
Shared helpers for the CPU tests: tiny encoders and random initialisation.
"""

import os

import torch
import torch.nn as nn

import ldpt.models  # noqa: F401  (fills the registries)
from ldpt.models.builder import MODELS
from ldpt.models.dinov3.layers import LayerScale
from ldpt.models.dinov3.layers.attention import LinearKMaskedBias
from ldpt.models.dinov3.model import DinoVisionTransformer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DINOv3TinyTest(DinoVisionTransformer):
    """ViT-7B layout (swiglu, no qkv bias, 4 storage tokens, RoPE rescale 2) at toy size."""

    def __init__(self, qkv_bias=False, ffn_layer="swiglu64"):
        super().__init__(
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=64,
            depth=4,
            num_heads=4,
            ffn_ratio=3,
            qkv_bias=qkv_bias,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer=ffn_layer,
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )


class FeatureStub(nn.Module):
    """Parameter-free encoder returning fixed random features (ViT-7B width by default)."""

    def __init__(self, embed_dim=4096, patch_size=16, seed=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.seed = seed

    def get_intermediate_layers(
        self, x, n, return_class_token=False, chunk_tokens=None
    ):
        """``chunk_tokens`` (LatentDPT's memory option) changes nothing here."""
        B, _, H, W = x.shape
        N = (H // self.patch_size) * (W // self.patch_size)
        g = torch.Generator().manual_seed(self.seed)
        return tuple(
            torch.randn(B, N, self.embed_dim, generator=g).to(x.device) for _ in n
        )


MODELS.register_module("dinov3_tiny_test", force=True, module=DINOv3TinyTest)
MODELS.register_module("feature_stub_test", force=True, module=FeatureStub)


@torch.no_grad()
def init_encoder_(encoder, seed=0):
    """DINOv3 random init that is not near-identity and has no NaN bias masks."""
    torch.manual_seed(seed)
    encoder.init_weights()
    for m in encoder.modules():
        if isinstance(m, LinearKMaskedBias) and m.bias is not None:
            # DINOv3 layout: q and v biases on, k bias masked
            d = m.out_features // 3
            m.bias_mask.copy_(torch.tensor([1.0] * d + [0.0] * d + [1.0] * d))
            m.bias.normal_(std=0.02)
        if isinstance(m, LayerScale):
            m.gamma.uniform_(0.05, 0.5)
    return encoder


@torch.no_grad()
def randomize_decoder_(model, seed=0):
    """Random decoder weights with LayerScale / LayerNorm away from their init."""
    g = torch.Generator().manual_seed(seed)
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            continue
        if name.endswith("gamma"):
            p.copy_(torch.rand(p.shape, generator=g) * 0.45 + 0.05)
        elif "norm" in name and name.endswith("weight"):
            p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
        elif name.endswith("bias"):
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
        else:
            fan_in = p[0].numel() if p.ndim > 1 else p.numel()
            p.copy_(torch.randn(p.shape, generator=g) / fan_in**0.5)
    return model


def strip_module(state_dict):
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
