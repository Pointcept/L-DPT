# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn.init
from torch import Tensor, nn

from .layers import (
    LayerScale,
    Mlp,
    PatchEmbed,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
    rope_is_random,
)
from ..builder import MODELS

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu64": partial(SwiGLUFFN, align_to=64),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, (nn.LayerNorm, LayerScale, PatchEmbed)):
        module.reset_parameters()


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class DinoVisionTransformer(nn.Module):
    """DINOv3 ViT, inference copy for L-DPT: the official backbone code without the
    training-only parts (stochastic depth, multi-crop list forward, local-crop CLS norm).
    Parameter and buffer names are the official ones."""

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device)
            )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks=None
    ) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features(
        self, x: Tensor, masks: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x, masks)
        rope_sincos = None
        for blk in self.blocks:
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
        x_norm = self._norm_tokens(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_storage_tokens": x_norm[:, 1 : self.n_storage_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.n_storage_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _norm_tokens(self, x: Tensor) -> Tensor:
        if self.untie_cls_and_patch_norms:
            x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
            x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            return torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
        return self.norm(x)

    def _iter_tapped(
        self,
        x: Tensor,
        n: Union[int, Sequence[int]] = 1,
        norm: bool = False,
        chunk_tokens: Optional[int] = None,
    ) -> Iterator[Tensor]:
        """All tokens of the tapped blocks, yielded as each is computed; every block
        runs. chunk_tokens (no autograd only): blocks run token-chunked
        (SelfAttentionBlock.forward_chunked)."""
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        total_block_len = len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        missing = set(blocks_to_take) - set(range(total_block_len))
        assert not missing, f"blocks {sorted(missing)} not in 0..{total_block_len - 1}"
        # RoPE is re-sampled per block only when its coordinate augmentation is
        # active (training); otherwise it is deterministic and computed once.
        rope = self.rope_embed
        rope_random = rope is not None and rope_is_random(rope)
        rope_sincos = None if rope is None or rope_random else rope(H=H, W=W)
        chunked = chunk_tokens is not None and not torch.is_grad_enabled()
        for i, blk in enumerate(self.blocks):
            if rope_random:
                rope_sincos = rope(H=H, W=W)
            if chunked:
                # in place, unless a tapped x was handed out un-normalised
                x = blk.forward_chunked(x, rope_sincos, chunk_tokens, inplace=norm)
            else:
                x = blk(x, rope_sincos)
            if i in blocks_to_take:
                # normalise when tapped, so the pre-norm output is not kept alive
                yield self._norm_tokens(x) if norm else x

    def iter_intermediate_layers(
        self,
        x: Tensor,
        *,
        n: Union[int, Sequence[int]] = 1,
        norm: bool = True,
        chunk_tokens: Optional[int] = None,
    ) -> Iterator[Tensor]:
        """Patch tokens of each tapped block, yielded as soon as it is computed (same
        values as get_intermediate_layers), so a caller can consume and free one layer
        before the next. Grad mode / autocast apply per next() call: set them around
        next(), not around a loop body that also runs other code.

        Shapes:
            x: [B, 3, H, W] -> len(n) x [B, (H / P) * (W / P), embed_dim]
        """
        for out in self._iter_tapped(x, n, norm=norm, chunk_tokens=chunk_tokens):
            yield out[:, self.n_storage_tokens + 1 :]
            del out  # not alive while the next blocks run

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
        chunk_tokens: Optional[int] = None,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = list(self._iter_tapped(x, n, norm=norm, chunk_tokens=chunk_tokens))
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(
        self, *args, is_training: bool = False, **kwargs
    ) -> Dict[str, Tensor] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


@MODELS.register_module("dinov3_vitb")
class DINOv3ViTB(DinoVisionTransformer):
    def __init__(
        self,
        patch_size=16,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
            **kwargs,
        )


@MODELS.register_module("dinov3_vitl")
class DINOv3ViTL(DinoVisionTransformer):
    def __init__(
        self,
        patch_size=16,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
            **kwargs,
        )


@MODELS.register_module("dinov3_vith")
class DINOv3ViTH(DinoVisionTransformer):
    def __init__(
        self,
        patch_size=16,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1280,
            depth=32,
            num_heads=20,
            ffn_ratio=6.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
            **kwargs,
        )


@MODELS.register_module("dinov3_vit7b")
class DINOv3ViT7B(DinoVisionTransformer):
    def __init__(
        self,
        patch_size=16,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=4096,
            depth=40,
            num_heads=32,
            ffn_ratio=3,
            qkv_bias=False,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu64",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
            **kwargs,
        )
