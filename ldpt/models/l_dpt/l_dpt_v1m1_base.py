"""
L-DPT v1m1: dense prediction with latent decoding

One class for every L-DPT variant (depth and segmentation); named
configurations are in presets.py. The state-dict keys are those of the trained
checkpoints (rope.periods, neck_blocks.{s} | neck_block, dec_blocks.{s}.{i}.*,
unpatchfy.0.proj, seg_head), so they load strictly. rope.periods is a persistent buffer
only because those checkpoints hold it.

Parts (one module, flat attributes, so the state-dict keys match the checkpoints):
- encoder adapter: frozen DINOv3, tapped layers concatenated, input padded to the patch
  grid, optional bf16 weight storage (pad, encode);
- latent decoder: per-stage linear selection of the encoder features, added to a latent
  grid (zero at stage 0) and refined by RoPE self-attention blocks (select, decode);
  encode_select streams the selection layer by layer (no concatenation; high-res input);
  encode_select_tiled runs encoder and selection on overlapping tiles of a large input
  and blends them onto the whole grid (encoder_tiles.py; infer(image, encoder_tiles));
- pixel readout: k x k conv + pixel shuffle back to the input resolution (decode);
- task heads: relative depth sigmoid(u) (default) or exp(clamp(u, -30, 30))
  (head="exp", the released depth models) of the readout logit u, or 1x1 seg_head
  logits (infer); losses (forward).

Behaviours the trained checkpoints depend on (keep them):
- the attention residual is added to the *normalised* tensor (x = norm1(x); x = x + attn(x)),
  not to the block input as in a standard pre-LN block;
- in training, the decoder RoPE coordinate augmentation (jitter / rescale) is re-sampled
  for every block, not once per forward;
- by default the frozen encoder follows model.train(), so the DINOv3 RoPE rescale
  augmentation is active on the encoder during training (encoder_train_mode).

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import contextlib
import math
import weakref
from functools import partial
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp.autocast_mode import is_autocast_available

from ..builder import MODELS, build_model
from ..dinov3.layers import RopePositionEmbedding, rope_is_random
from ..losses import LOSSES, build_criteria
from .encoder_tiles import EncoderTiles
from .encoder_tiles import resolve as resolve_tiles
from .encoder_tiles import tile_windows
from .modules import (
    ATTN_DTYPES,
    QK_TAU_MAX,
    Block,
    PixelReadout,
    cast_linear_conv_,
    check_choice,
)

__all__ = ["LatentDPT"]

TASKS = ("relative_depth", "semseg")
SELECTIONS = ("stage", "shared")
SELECTION_IMPLS = ("concat", "stream")
PRECISIONS = ("fp32", "bf16")
# legacy attn_impl argument -> attn_dtype
ATTN_IMPL_TO_DTYPE = {"auto": "fp32", "flash": "bf16"}
ENCODER_WEIGHT_DTYPES = {None: None, "bf16": torch.bfloat16}
# decoder blocks of an fp32 island (attn_fp32 / block_fp32)
FP32_SCOPES = (None, "last", "last_stage", "all")
# relative depth head on the readout logit u: sigmoid(u) or exp(u)
HEADS = ("sigmoid", "exp")
LOGIT_CLAMP = 30.0


def _cast_post_hook(module, incompatible_keys, dtype):
    cast_linear_conv_(module, dtype)


class _TrainCompiledForward:
    """A block's forward: the compiled one while the owning model trains, else eager."""

    def __init__(self, eager, owner: "weakref.ref", options: Dict[str, object]):
        self.eager = eager
        self.compiled = torch.compile(eager, **options)
        self.owner = owner

    def __call__(self, *args, **kwargs):
        model = self.owner()
        training = model is not None and model.training
        return (self.compiled if training else self.eager)(*args, **kwargs)


@MODELS.register_module("L-DPT-v1m1")
class LatentDPT(nn.Module):
    """L-DPT: frozen ViT encoder + latent decoder + light readout.

    A zero-initialised latent grid (one token per patch) is decoded in stages. Stage s
    adds a linear selection of the concatenated encoder layers and runs
    dec_num_layers[s] RoPE self-attention blocks; a k x k conv + pixel shuffle reads
    out to pixels. The released depth models are this class with presets.RELEASE_L.
    AGENTS.md ("Model") gives an overview of the options, docs/inference.md the
    inference memory options.

    Building does no device or data work beyond the encoder storage cast: build under
    ``torch.device("meta")`` and ``load_state_dict(..., assign=True)`` (api.load does).
    An option that is off by default is bitwise the model without it when off.

    Args (beyond the decoder sizes dec_dim, dec_num_heads, dec_num_layers,
    dec_attn_downscale, dec_ffn_ratio and the decoder RoPE options):
        encoder: registry cfg of the encoder (``dict(type="dinov3_vit7b")``) or a module
            with ``patch_size``, ``embed_dim`` and ``get_intermediate_layers``.
        encoder_inter_layers: encoder blocks to tap, one per decoder stage.
        freeze_encoder: True (default): the encoder gets no gradient and runs under
            bf16 autocast.
        encoder_train_mode: True (default): the frozen encoder follows model.train(),
            so its RoPE augmentation is active in training (the released checkpoints
            were trained this way); False keeps the encoder in eval mode.
        encoder_weight_dtype: None (default) or "bf16" (the released models; needs
            freeze_encoder): the encoder's Linear / Conv2d weights are stored in bf16
            (it runs under bf16 autocast, so outputs are bit-identical); norms,
            LayerScale, tokens and RoPE stay fp32. Re-applied after every
            load_state_dict.
        task: "relative_depth" (one channel, see head) or "semseg" (out_dim pixel
            features + a 1x1 seg_head with num_classes outputs).
        criterion: relative_depth: one loss cfg, default (None)
            dict(type="relative_depth_loss"); semseg: a list of loss cfgs for
            build_criteria. False: inference only, no loss built (release files,
            api.load): self.criterion is None, a forward in eval returns the
            prediction without a loss even when the target is given, and a training
            forward raises. No loss has state-dict keys, so the same checkpoints load
            either way.
        head: relative depth from the readout logit u: "sigmoid" (default, depth =
            sigmoid(u)) or "exp" (depth = exp(clamp(u, -30, 30)), the released
            depth models). No parameters. A criterion with a head attribute must
            name the same head.
        readout_fp32: False (default) or True (the released depth models): the
            readout (conv + pixel shuffle) runs outside the autocast on the fp32
            decoder output, so the logit is not rounded to bf16.
        selection: "stage" (default: one Linear per stage) or "shared" (one Linear for
            all stages, key 'neck_block').
        readout_kernel_size: 3 (default, the released weights) or 1 (a linear
            projection).
        precision: "fp32" (default) or "bf16" (the released models): selection,
            decoder blocks and readout conv under bf16 autocast; readout output, heads
            and loss in fp32. "fp32" adds no autocast, so an autocast the caller
            entered (trainer enable_amp) still applies.
        attn_dtype: "fp32" (default; the segmentation preset) or "bf16" (the
            released depth models: q / k / v cast to bf16 for SDPA). attn_impl
            ("auto" / "flash") is the legacy name of the same choice.
        attn_fp32: None (default), "last" (the last decoder block), "last_stage" (the
            released depth models) or "all": the attention of those blocks (q / k / v
            projections, RoPE, SDPA, out_proj) runs in fp32 with autocast off.
        block_fp32: None (default), "last" or "last_stage": those whole blocks run in
            fp32 with autocast off. Without an autocast, attn_fp32 and block_fp32 are
            the same math as without them.
        dec_qk_bound: None (default) or a logit bound B > 0 (the released depth
            models: 1000): in every decoder attention q and k are clipped per head
            and token to an L2 norm of at most sqrt(B sqrt(head_dim)) before RoPE, so
            |q.k| / sqrt(head_dim) <= B. No parameters. Active in training and eval:
            a model trained with it needs it at inference. The strings "None" /
            "null" / "" (from --options) turn it off.
        Used by the COCO-Stuff segmentation recipe 1 (semseg-l-dpt-v1m1-1-large) and the
        segmentation preset:
        dec_qk_norm (+ dec_qk_tau_max, default 200.0): cosine attention with a
            learned temperature per head, capped at dec_qk_tau_max (Attention
            qk_norm); one key dec_blocks.{s}.{i}.attn.logit_scale per block; raises
            with dec_qk_bound.
        tap_bn: False (default) or True: a BatchNorm on each tapped encoder layer
            before the selection, in fp32 (keys tap_bn.{i}.*; the trainer's sync_bn
            converts them to SyncBatchNorm).
        Experimental, used by no recipe (details: modules.py and the tests):
        dec_conv_mixer: None (default), "last" or "last_stage": those blocks mix
            tokens with LocalMix (3x3 depthwise conv on the token grid) instead of
            self-attention; keys local.* replace attn.*, every other parameter and the
            RNG stream stay as without it. attn_fp32 does not apply to them,
            block_fp32 does; compile_blocks leaves them eager.
        dec_pos_embed_rope_eval_scale: None (default) or r > 0: in eval mode the
            decoder RoPE coordinates are multiplied by r (training unchanged); r > 1
            spreads the tokens of a large input toward the trained token spacing.

    Inference memory (attributes, may be changed after building; same math):
        selection_impl: "concat" (default: the tapped layers concatenated to
            [B, N, L * C_enc], one Linear) or "stream" (encode_select: each layer is
            projected by its slice of the selection weights as the encoder yields it,
            summed in fp32 and freed; another summation order).
        chunk_tokens: None (default) or a token count: in eval without autograd the
            encoder blocks and the decoder FFN run on chunks of this many tokens;
            per-token math unchanged, matrix products over fewer rows may round
            differently.
    """

    def __init__(
        self,
        encoder: Union[Mapping, nn.Module],
        encoder_inter_layers: Sequence[int],
        freeze_encoder: bool = True,
        encoder_train_mode: bool = True,
        encoder_weight_dtype: Optional[str] = None,
        task: str = "relative_depth",
        out_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        criterion: Union[Mapping, Sequence[Mapping], None, bool] = None,
        dec_dim: int = 1024,
        dec_num_heads: int = 16,
        dec_num_layers: Sequence[int] = (2, 2, 2, 2),
        dec_attn_downscale: int = 1,
        dec_ffn_ratio: float = 4.0,
        dec_pos_embed_rope_base: float = 100,
        dec_pos_embed_rope_normalize_coords: str = "separate",
        dec_pos_embed_rope_shift_coords: Optional[float] = None,
        dec_pos_embed_rope_jitter_coords: Optional[float] = None,
        dec_pos_embed_rope_rescale_coords: Optional[float] = 2,
        dec_pos_embed_rope_eval_scale: Optional[float] = None,
        dec_layer_scale_init: Optional[float] = 1.0e-05,
        dec_qkv_bias: bool = False,
        dec_qk_bound: Optional[float] = None,
        dec_qk_norm: bool = False,
        dec_qk_tau_max: float = QK_TAU_MAX,
        dec_proj_bias: bool = True,
        dec_ffn_bias: bool = True,
        selection: str = "stage",
        readout_kernel_size: int = 3,
        precision: str = "fp32",
        attn_dtype: Optional[str] = None,
        attn_impl: Optional[str] = None,
        selection_impl: str = "concat",
        chunk_tokens: Optional[int] = None,
        attn_fp32: Optional[str] = None,
        block_fp32: Optional[str] = None,
        dec_conv_mixer: Optional[str] = None,
        head: str = "sigmoid",
        readout_fp32: bool = False,
        tap_bn: bool = False,
    ) -> None:
        super().__init__()
        check_choice("task", task, TASKS)
        if criterion is True:
            raise ValueError(
                "criterion=True is not an option: pass a loss cfg, None (the default "
                "loss) or False (inference only, no loss)"
            )
        if not isinstance(tap_bn, bool):
            raise ValueError(f"tap_bn must be True or False, got {tap_bn!r}")
        check_choice("head", head, HEADS)
        if task != "relative_depth" and head != "sigmoid":
            raise ValueError(f"head={head!r} applies to task='relative_depth' only")
        check_choice("attn_fp32", attn_fp32, FP32_SCOPES)
        check_choice("block_fp32", block_fp32, FP32_SCOPES[:-1])
        check_choice("dec_conv_mixer", dec_conv_mixer, FP32_SCOPES[:-1])
        # --options model.dec_qk_bound=None arrives as a string: the bound off (to
        # evaluate a bound run's checkpoint without it)
        if isinstance(dec_qk_bound, str) and dec_qk_bound.strip().lower() in (
            "",
            "none",
            "null",
        ):
            dec_qk_bound = None
        if not isinstance(dec_qk_norm, bool):
            raise ValueError(f"dec_qk_norm must be True or False, got {dec_qk_norm!r}")
        if dec_qk_norm and dec_qk_bound is not None:
            raise ValueError(
                "dec_qk_norm bounds every logit by dec_qk_tau_max: dec_qk_bound must "
                f"be None with it, got {dec_qk_bound!r}"
            )
        if dec_pos_embed_rope_eval_scale is not None and not (
            isinstance(dec_pos_embed_rope_eval_scale, (int, float))
            and not isinstance(dec_pos_embed_rope_eval_scale, bool)
            and 0 < dec_pos_embed_rope_eval_scale < math.inf
        ):
            raise ValueError(
                "dec_pos_embed_rope_eval_scale must be None or a number > 0, got "
                f"{dec_pos_embed_rope_eval_scale!r}"
            )
        check_choice("selection", selection, SELECTIONS)
        check_choice("selection_impl", selection_impl, SELECTION_IMPLS)
        check_choice("precision", precision, PRECISIONS)
        if chunk_tokens is not None and int(chunk_tokens) < 1:
            raise ValueError(f"chunk_tokens must be None or >= 1, got {chunk_tokens}")
        check_choice(
            "encoder_weight_dtype", encoder_weight_dtype, ENCODER_WEIGHT_DTYPES
        )
        attn_dtype = _resolve_attn_dtype(attn_dtype, attn_impl)
        if len(dec_num_layers) != len(encoder_inter_layers):
            raise ValueError(
                f"one encoder tap per decoder stage: encoder_inter_layers "
                f"{list(encoder_inter_layers)} vs dec_num_layers {list(dec_num_layers)}"
            )
        rope_dim = dec_dim // dec_attn_downscale
        if dec_dim % dec_attn_downscale or rope_dim % (4 * dec_num_heads):
            raise ValueError(
                f"2D RoPE needs (dec_dim // dec_attn_downscale) divisible by "
                f"4 * dec_num_heads: dec_dim={dec_dim}, dec_attn_downscale="
                f"{dec_attn_downscale}, dec_num_heads={dec_num_heads}"
            )
        if encoder_weight_dtype is not None and not freeze_encoder:
            raise ValueError(
                "encoder_weight_dtype stores frozen weights in low precision; it needs "
                "freeze_encoder=True (or encoder_weight_dtype=None)"
            )
        out_dim = out_dim or (1 if task == "relative_depth" else 32)
        if task == "relative_depth" and out_dim != 1:
            raise ValueError(
                f"relative_depth predicts one channel: out_dim=1, not {out_dim}"
            )
        if task == "semseg" and num_classes is None:
            raise ValueError("task='semseg' needs num_classes (the seg_head width)")

        # encoder adapter
        self.encoder = _build_encoder(encoder)
        self.encoder_inter_layers = [int(i) for i in encoder_inter_layers]
        n_blocks = getattr(self.encoder, "n_blocks", None)
        if n_blocks is not None and max(self.encoder_inter_layers) >= n_blocks:
            raise ValueError(
                f"encoder_inter_layers {self.encoder_inter_layers} out of range for an "
                f"encoder with {n_blocks} blocks (ViT-7B taps: [9, 19, 29, 39])"
            )
        self.freeze_encoder = freeze_encoder
        self.encoder_train_mode = encoder_train_mode
        self.encoder_weight_dtype = ENCODER_WEIGHT_DTYPES[encoder_weight_dtype]
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if self.encoder_weight_dtype is not None:
            cast_linear_conv_(self.encoder, self.encoder_weight_dtype)
            # re-cast after any load_state_dict (also with assign=True)
            self.encoder.register_load_state_dict_post_hook(
                partial(_cast_post_hook, dtype=self.encoder_weight_dtype)
            )
        self.patch_size = self.encoder.patch_size
        self.enc_dim = self.encoder.embed_dim

        # latent decoder
        self.task = task
        self.selection = selection
        self.selection_impl = selection_impl
        self.chunk_tokens = chunk_tokens
        self.precision = precision
        self.dec_dim = dec_dim
        self.dec_num_layers = tuple(dec_num_layers)
        self.out_dim = out_dim
        self.head = head
        self.readout_fp32 = bool(readout_fp32)
        self.rope = RopePositionEmbedding(
            embed_dim=rope_dim,
            num_heads=dec_num_heads,
            base=dec_pos_embed_rope_base,
            normalize_coords=dec_pos_embed_rope_normalize_coords,
            shift_coords=dec_pos_embed_rope_shift_coords,
            jitter_coords=dec_pos_embed_rope_jitter_coords,
            rescale_coords=dec_pos_embed_rope_rescale_coords,
            eval_scale=(
                None
                if dec_pos_embed_rope_eval_scale is None
                else float(dec_pos_embed_rope_eval_scale)
            ),
        )
        # (stage, block) of the blocks run outside autocast (block_fp32)
        self.fp32_blocks = _scope_blocks(self.dec_num_layers, block_fp32)
        attn_scope = _scope_blocks(self.dec_num_layers, attn_fp32)
        attn_fp32_blocks = attn_scope | self.fp32_blocks
        # (stage, block) of the blocks with the LocalMix token mixer (dec_conv_mixer)
        self.conv_blocks = _scope_blocks(self.dec_num_layers, dec_conv_mixer)
        if attn_scope and attn_scope <= self.conv_blocks:  # it would change nothing
            raise ValueError(
                f"attn_fp32={attn_fp32!r} names only dec_conv_mixer={dec_conv_mixer!r} "
                f"blocks, which have no attention (block_fp32 applies to them)"
            )
        self.dec_blocks = nn.ModuleList(
            nn.ModuleList(
                Block(
                    dec_dim,
                    num_heads=dec_num_heads,
                    ffn_ratio=dec_ffn_ratio,
                    qkv_bias=dec_qkv_bias,
                    proj_bias=dec_proj_bias,
                    ffn_bias=dec_ffn_bias,
                    downscale=dec_attn_downscale,
                    attn_dtype=attn_dtype,
                    init_values=dec_layer_scale_init,
                    attn_fp32=(s, i) in attn_fp32_blocks,
                    qk_bound=dec_qk_bound,
                    qk_norm=dec_qk_norm,
                    qk_tau_max=dec_qk_tau_max,
                )
                for i in range(num_layers)
            )
            for s, num_layers in enumerate(self.dec_num_layers)
        )
        in_dim = len(self.encoder_inter_layers) * self.enc_dim
        if selection == "stage":
            self.neck_blocks = nn.ModuleList(
                nn.Linear(in_dim, dec_dim) for _ in self.dec_num_layers
            )
        else:
            self.neck_block = nn.Linear(in_dim, dec_dim)
        # tap_bn: one BatchNorm per tapped layer, before the selection (no random draw)
        self.tap_bn: Optional[nn.ModuleList] = (
            nn.ModuleList(
                nn.BatchNorm1d(self.enc_dim) for _ in self.encoder_inter_layers
            )
            if tap_bn
            else None
        )

        # pixel readout ("unpatchfy.0" is the trained key)
        self.unpatchfy = nn.Sequential(
            PixelReadout(
                dec_dim,
                self.out_dim,
                kernel_size=readout_kernel_size,
                patch_size=self.patch_size,
            ),
        )

        # task head and loss (criterion=False: inference only, no loss)
        self.criterion: Optional[nn.Module] = None
        if task == "relative_depth":
            if criterion is not False:
                if criterion is None:
                    criterion = dict(type="relative_depth_loss")
                self.criterion = LOSSES.build(criterion)
                # a criterion that reads the head (the L-DPT depth loss) must match it
                criterion_head = getattr(self.criterion, "head", None)
                if criterion_head is not None and criterion_head != head:
                    raise ValueError(
                        f"criterion head={criterion_head!r} differs from the model's "
                        f"head={head!r}: set both (model.head, model.criterion.head)"
                    )
        else:
            self.num_classes = num_classes
            self.seg_head = nn.Conv2d(
                self.out_dim, num_classes, kernel_size=1, bias=True
            )
            if criterion is not False:
                self.criterion = build_criteria(criterion)

        # dec_conv_mixer last: every other parameter got the random numbers it gets
        # without the option, and whatever draws after __init__ sees the same stream
        if self.conv_blocks:
            rng_state = torch.get_rng_state()
            for s, i in sorted(self.conv_blocks):
                self.dec_blocks[s][i].convert_mixer_()
            torch.set_rng_state(rng_state)

    @property
    def num_stages(self) -> int:
        return len(self.dec_num_layers)

    def extra_repr(self) -> str:
        return (
            f"task={self.task!r}, selection={self.selection!r}, "
            f"precision={self.precision!r}, encoder_train_mode={self.encoder_train_mode}, "
            f"selection_impl={self.selection_impl!r}, chunk_tokens={self.chunk_tokens}"
        )

    # ---- encoder adapter

    def train(self, mode: bool = True) -> "LatentDPT":
        super().train(mode)
        if not self.encoder_train_mode:
            self.encoder.eval()
        return self

    def compile_blocks(
        self, encoder: bool = True, decoder: bool = True, **options
    ) -> None:
        """torch.compile the encoder and / or decoder blocks for the training steps.

        While the model is in training mode each block runs its compiled forward (fixed
        crops: one graph serves every block of a kind); in eval mode (evaluation,
        inference, any image size) the eager forward runs. Parameters and state-dict
        keys are unchanged. options go to torch.compile (default dynamic=False).
        dec_conv_mixer blocks stay eager: AOT autograd would trace their backward
        under the caller's autocast (bf16 gradients of the fp32 convolution).
        """
        options = {"dynamic": False, **options}
        blocks = list(self.encoder.blocks) if encoder else []
        if decoder:
            blocks += [
                blk for stage in self.dec_blocks for blk in stage if blk.mixer == "attn"
            ]
        owner = weakref.ref(self)
        for blk in blocks:
            if isinstance(vars(blk).get("forward"), _TrainCompiledForward):
                raise RuntimeError("compile_blocks: the blocks are compiled already")
            blk.forward = _TrainCompiledForward(blk.forward, owner, options)

    def pad(self, image: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Zero-pad bottom / right to the patch grid.

        Shapes:
            image: [B, C, H, W] -> ([B, C, h * P, w * P], (h, w)),
            h = ceil(H / P), w = ceil(W / P), P = patch_size
        """
        _, _, rh, rw = image.shape
        h = int(math.ceil(rh / self.patch_size))
        w = int(math.ceil(rw / self.patch_size))
        H, W = h * self.patch_size, w * self.patch_size
        if H != rh or W != rw:
            image = F.pad(image, (0, W - rw, 0, H - rh), mode="constant", value=0)
        return image, (h, w)

    def encode(self, image: Tensor) -> Tensor:
        """Tapped encoder layers, concatenated (frozen: no grad, bf16 autocast; with
        tap_bn each layer through its BatchNorm first, outside the no-grad).

        Shapes:
            image: [B, 3, h * P, w * P] (normalised; pad() first) -> [B, h * w, L * C_enc]
        """
        self._check_image(image)
        with self._encoder_context(image.device):
            feats = self.encoder.get_intermediate_layers(
                image, **self._encoder_kwargs(), return_class_token=False
            )
        if self.tap_bn is not None:
            feats = [self._tap_norm(f, i) for i, f in enumerate(feats)]
        features = torch.cat(feats, dim=-1)
        del feats
        return features

    def _iter_layers(self, image: Tensor) -> Iterator[Tensor]:
        """Tapped encoder layers one at a time, [B, h * w, C_enc] each."""
        self._check_image(image)
        if hasattr(self.encoder, "iter_intermediate_layers"):
            layers = self.encoder.iter_intermediate_layers(
                image, **self._encoder_kwargs()
            )
        else:  # an encoder without it computes every layer first
            with self._encoder_context(image.device):
                layers = iter(
                    self.encoder.get_intermediate_layers(
                        image, **self._encoder_kwargs(), return_class_token=False
                    )
                )
        while True:
            # the encoder context covers the encoder step, not the caller's loop body
            with self._encoder_context(image.device):
                feat = next(layers, None)
            if feat is None:
                return
            yield feat
            del feat  # not alive while the next layer is computed

    @contextlib.contextmanager
    def _encoder_context(self, device: torch.device) -> Iterator[None]:
        """Frozen encoder: no grad, bf16 autocast."""
        if not self.freeze_encoder:
            yield
            return
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            yield

    def _encoder_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = dict(n=self.encoder_inter_layers)
        if self._eval_chunk_tokens() is not None:
            kwargs["chunk_tokens"] = self._eval_chunk_tokens()
        return kwargs

    def _eval_chunk_tokens(self) -> Optional[int]:
        """chunk_tokens in eval mode; None in training (full-size path)."""
        if self.training or self.chunk_tokens is None:
            return None
        return int(self.chunk_tokens)

    def _check_image(self, image: Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must be [B, 3, H, W], got {list(image.shape)}")
        if image.shape[-2] % self.patch_size or image.shape[-1] % self.patch_size:
            raise ValueError(
                f"image size {list(image.shape[-2:])} is not a multiple of the patch size "
                f"{self.patch_size}: pad it first (model.pad(image)) or call model.infer"
            )

    # ---- latent decoder and readout

    def select(self, features: Tensor) -> List[Tensor]:
        """Per-stage linear selections of the encoder features.

        Shapes:
            features: [B, N, L * C_enc] -> num_stages x [B, N, dec_dim]
        """
        with self._autocast(features.device):
            if self.selection == "shared":
                return [self.neck_block(features)] * self.num_stages
            return [neck(features) for neck in self.neck_blocks]

    def encode_select(self, image: Tensor) -> List[Tensor]:
        """select(encode(image)) without the [B, N, L * C_enc] concatenation.

        Tapped layer i is projected by its column slice of each selection weight as
        soon as the encoder yields it, then freed; the partial products are summed in
        fp32 (bias once) and returned in the dtype select() would return. Same math as
        select(encode(image)); under bf16 autocast each partial product is rounded to
        bf16 before the sum, so the result differs at the rounding level.

        Shapes:
            image: [B, 3, h * P, w * P] (normalised; pad() first)
            -> num_stages x [B, h * w, dec_dim]
        """
        necks = [self.neck_block] if self.selection == "shared" else self.neck_blocks
        sums: List[Optional[Tensor]] = [None] * len(necks)
        i = 0
        # a plain loop: enumerate() would keep layer i alive while layer i + 1 is
        # computed (it caches its last item)
        for feat in self._iter_layers(image):
            if self.tap_bn is not None:  # the generator still holds the raw layer
                feat = self._tap_norm(feat, i)
            for s, neck in enumerate(necks):
                part = self._select_layer(feat, i, neck)
                dtype = part.dtype
                sums[s] = part.float() if sums[s] is None else sums[s].add_(part)
                del part
            del feat
            i += 1
        selections = [t.to(dtype) for t in sums]
        if self.selection == "shared":
            return selections * self.num_stages
        return selections

    def encode_select_tiled(self, image: Tensor, tiles: EncoderTiles) -> List[Tensor]:
        """The per-stage selections of the whole token grid from overlapping encoder
        tiles (encoder_tiles.py): each tile of ``tiles.tile`` x ``tiles.tile`` tokens
        (fewer along a shorter axis) runs alone through the encoder and the selection
        (encode_select or select(encode()), per selection_impl), with the encoder's
        RoPE coordinates times ``tiles.rope_scale``; the selections are summed in fp32
        with the tile weights (tile_windows), divided by the summed weights and
        returned in the dtype select() returns. The grid is tiled even when it fits
        one tile (infer skips this then); one tile with ``rope_scale`` 1.0 gives the
        plain selections bit for bit. Eval mode only (the encoder's RoPE scale
        applies in eval). Memory beyond the encoder of one tile: the fp32 sums,
        num_stages x [B, h * w, dec_dim] (one with selection "shared").

        Shapes:
            image: [B, 3, h * P, w * P] (normalised; pad() first)
            -> num_stages x [B, h * w, dec_dim]
        """
        if self.training:
            raise ValueError(
                "encoder tiles are an inference option: call model.eval() first"
            )
        self._check_image(image)
        P = self.patch_size
        B = image.shape[0]
        h, w = image.shape[-2] // P, image.shape[-1] // P
        rope = getattr(self.encoder, "rope_embed", None)
        if tiles.rope_scale != 1.0 and not hasattr(rope, "eval_scale"):
            raise ValueError(
                f"EncoderTiles(rope_scale={tiles.rope_scale}) needs an encoder with "
                f"rope_embed.eval_scale (DINOv3); {type(self.encoder).__name__} has "
                f"none: use rope_scale=1.0"
            )
        n_sums = 1 if self.selection == "shared" else self.num_stages
        sums: List[Optional[Tensor]] = [None] * n_sums
        weight = torch.zeros(h, w, device=image.device)
        dtype = None
        saved = rope.eval_scale if rope is not None else None
        try:
            if tiles.rope_scale != 1.0:
                rope.eval_scale = float(tiles.rope_scale)
            for y0, x0, th, tw, win in tile_windows((h, w), tiles, image.device):
                crop = image[..., y0 * P : (y0 + th) * P, x0 * P : (x0 + tw) * P]
                selections = self._selections(crop)
                for s in range(n_sums):
                    part = selections[s]
                    dtype = part.dtype
                    if sums[s] is None:
                        sums[s] = torch.zeros(
                            B, h, w, part.shape[-1], device=image.device
                        )
                    sums[s][:, y0 : y0 + th, x0 : x0 + tw].addcmul_(
                        part.float().reshape(B, th, tw, -1), win[None, :, :, None]
                    )
                    del part
                del selections
                weight[y0 : y0 + th, x0 : x0 + tw] += win
        finally:
            if rope is not None:
                rope.eval_scale = saved
        out = []
        for s in range(n_sums):
            total = sums[s]
            sums[s] = None  # one fp32 sum at a time is converted and freed
            out.append(
                total.div_(weight[None, :, :, None]).view(B, h * w, -1).to(dtype)
            )
            del total
        if self.selection == "shared":
            return out * self.num_stages
        return out

    def _selections(self, image: Tensor) -> List[Tensor]:
        """select(encode(image)), or encode_select(image) with selection_impl
        "stream". Shapes: [B, 3, h * P, w * P] -> num_stages x [B, h * w, dec_dim]."""
        if self.selection_impl == "stream":
            return self.encode_select(image)
        features = self.encode(image)
        selections = self.select(features)
        del features  # [B, N, L * C_enc]: the largest tensor after the encoder
        return selections

    def _tap_norm(self, feat: Tensor, i: int) -> Tensor:
        """tap_bn: tapped layer i through its BatchNorm over the channels of every
        token, in fp32 with autocast off. Shapes: [B, N, C_enc] -> same, fp32."""
        off = (
            torch.autocast(feat.device.type, enabled=False)
            if is_autocast_available(feat.device.type)
            else contextlib.nullcontext()
        )
        with off:
            out = self.tap_bn[i](feat.float().reshape(-1, feat.shape[-1]))
        return out.reshape(feat.shape)

    def _select_layer(self, feat: Tensor, i: int, neck: nn.Linear) -> Tensor:
        """Layer i's share of a selection: feat @ W[:, i*C:(i+1)*C].T (+ bias, i=0)."""
        c = self.enc_dim
        weight = neck.weight[:, i * c : (i + 1) * c]
        with self._autocast(feat.device):
            return F.linear(feat, weight, neck.bias if i == 0 else None)

    def decode(self, selections: Sequence[Tensor], patch_hw: Tuple[int, int]) -> Tensor:
        """Latent decoding (additive injection + RoPE blocks) and pixel readout.

        Shapes:
            selections: num_stages x [B, h * w, dec_dim]; patch_hw: (h, w)
            -> [B, out_dim, h * P, w * P] (fp32, before the task head)
        """
        if len(selections) != self.num_stages:
            raise ValueError(
                f"decode needs {self.num_stages} stage selections (model.select), "
                f"got {len(selections)}"
            )
        h, w = patch_hw
        if selections[0].shape[1] != h * w:
            raise ValueError(
                f"patch_hw {tuple(patch_hw)} does not match {selections[0].shape[1]} tokens"
            )
        with self._autocast(selections[0].device):
            # random RoPE augmentation (training) is re-sampled per block, else once
            rope_random = rope_is_random(self.rope)
            rope = None if rope_random else self.rope(H=h, W=w)
            chunk_tokens = self._eval_chunk_tokens()
            # stage 0 starts from the zero-initialised grid: 0 + selection
            query = selections[0].float()
            for s, blocks in enumerate(self.dec_blocks):
                if s > 0:
                    query = query + selections[s]
                for i, blk in enumerate(blocks):
                    # drawn for conv blocks too: the attention blocks see the same
                    # random stream as without dec_conv_mixer
                    if rope_random:
                        rope = self.rope(H=h, W=w)
                    # a conv block (dec_conv_mixer) needs the token grid
                    kw = dict(hw=(h, w)) if (s, i) in self.conv_blocks else {}
                    if (s, i) in self.fp32_blocks:  # block_fp32: autocast off
                        with torch.autocast(query.device.type, enabled=False):
                            query = blk(
                                query.float(),
                                rope=rope,
                                chunk_tokens=chunk_tokens,
                                **kw,
                            )
                    else:
                        query = blk(query, rope=rope, chunk_tokens=chunk_tokens, **kw)
            b, _, c = query.shape
            # [B, N, C] -> [B, C, h, w] as a view, no copy (channels-last strides)
            query = query.reshape(b, h, w, c).permute(0, 3, 1, 2)
            if not self.readout_fp32:
                pred = self.unpatchfy(query)
        if self.readout_fp32:  # the readout on the fp32 stream, autocast off
            with torch.autocast(query.device.type, enabled=False):
                pred = self.unpatchfy(query.float())
        return pred.float()

    def _autocast(self, device: torch.device) -> contextlib.AbstractContextManager:
        if self.precision == "fp32" or not is_autocast_available(device.type):
            return contextlib.nullcontext()
        return torch.autocast(device.type, dtype=torch.bfloat16)

    # ---- task heads and losses

    def infer(
        self, image: Tensor, encoder_tiles: Union[EncoderTiles, None, bool] = None
    ) -> Tensor:
        """Task output without a loss (pads, encodes, decodes, crops back).

        encoder_tiles: None / False (default): one encoder pass over the whole image.
        An EncoderTiles (eval mode): when a side of the token grid exceeds its tile,
        the encoder runs on overlapping tiles (encode_select_tiled) and the decoder on
        the whole grid; a smaller input runs the plain forward, bit for bit.

        Shapes:
            image: [B, 3, H, W] (normalised, any H, W)
            -> relative_depth: [B, 1, H, W], in (0, 1) with head "sigmoid", in
               [exp(-30), exp(30)] with "exp"; semseg: [B, num_classes, H, W] logits
        """
        tiles = resolve_tiles(encoder_tiles)
        _, _, rh, rw = image.shape
        image, patch_hw = self.pad(image)
        if tiles is not None and tiles.applies(patch_hw):
            selections = self.encode_select_tiled(image, tiles)
        else:
            selections = self._selections(image)
        pred = self.decode(selections, patch_hw)
        del selections
        if self.task == "semseg":
            return self.seg_head(pred)[:, :, :rh, :rw]
        if self.head == "exp":
            return torch.exp(pred[:, :, :rh, :rw].clamp(-LOGIT_CLAMP, LOGIT_CLAMP))
        return torch.sigmoid(pred[:, :, :rh, :rw])

    def forward(self, data_dict: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """Training / evaluation step of the trainer.

        Shapes:
            data_dict["image"]: [B, 3, H, W] normalised
            data_dict["depth"] (relative_depth): [B, 1, H', W'], 0 = invalid; the
                prediction is resized bilinearly to H' x W' for the loss
            data_dict["segment"] (semseg): [B, 1, H, W] int, -1 = ignore
            data_dict["name"]: B sample names, passed to a loss that uses them
                (relative_depth_loss with label_types)
            -> dict(loss, ...) when the target is given (and the model has a loss);
               in eval also pred_depth [B, 1, H, W] or pred_segment
               [B, num_classes, H, W]
        """
        key = "segment" if self.task == "semseg" else "depth"
        if self.training and self.criterion is None:
            raise ValueError(
                "this model was built without a loss (criterion=False, inference only, "
                "as api.load builds it): build it from its training config to train"
            )
        if self.training and key not in data_dict:
            raise ValueError(
                f"training needs the target data_dict[{key!r}]; for inference call "
                f"model.eval() or model.infer(image)"
            )
        pred = self.infer(data_dict["image"])
        result_dict = {} if self.training else {f"pred_{key}": pred}
        if key not in data_dict or self.criterion is None:
            return result_dict
        if self.task == "semseg":
            gt_segment = data_dict["segment"].squeeze(1)
            result_dict.update(self.criterion(pred, gt_segment))
            return result_dict
        gt_depth = data_dict["depth"]
        valid_mask = gt_depth > 0
        if pred.shape != gt_depth.shape:
            pred = F.interpolate(
                pred, size=gt_depth.shape[-2:], mode="bilinear", align_corners=False
            )
        # a loss with per-dataset rules (relative_depth_loss label_types) gets the
        # sample names
        kwargs = {}
        if getattr(self.criterion, "uses_names", False):
            kwargs["names"] = data_dict.get("name")
        result_dict.update(self.criterion(pred, gt_depth, valid_mask, **kwargs))
        return result_dict


def _scope_blocks(num_layers: Sequence[int], scope: Optional[str]) -> frozenset:
    """(stage, block) indices of an attn_fp32 / block_fp32 / dec_conv_mixer scope."""
    blocks = [(s, i) for s, n in enumerate(num_layers) for i in range(n)]
    if scope is None or not blocks:
        return frozenset()
    if scope == "last":
        return frozenset(blocks[-1:])
    if scope == "last_stage":
        return frozenset(b for b in blocks if b[0] == blocks[-1][0])
    return frozenset(blocks)


def _resolve_attn_dtype(attn_dtype: Optional[str], attn_impl: Optional[str]) -> str:
    if attn_impl is not None:
        check_choice("attn_impl", attn_impl, ATTN_IMPL_TO_DTYPE)
        legacy = ATTN_IMPL_TO_DTYPE[attn_impl]
        if attn_dtype not in (None, legacy):
            raise ValueError(
                f"attn_dtype={attn_dtype!r} conflicts with the legacy attn_impl="
                f"{attn_impl!r} (= attn_dtype={legacy!r}); pass only attn_dtype"
            )
        attn_dtype = legacy
    attn_dtype = attn_dtype or "fp32"
    check_choice("attn_dtype", attn_dtype, ATTN_DTYPES)
    return attn_dtype


def _build_encoder(encoder: Union[Mapping, nn.Module]) -> nn.Module:
    if isinstance(encoder, Mapping):
        encoder = build_model(dict(encoder))
    elif not isinstance(encoder, nn.Module):
        raise TypeError(
            f"encoder must be a registry cfg (dict(type='dinov3_vit7b')) or an "
            f"nn.Module, got {type(encoder).__name__}"
        )
    missing = [
        a
        for a in ("patch_size", "embed_dim", "get_intermediate_layers")
        if not hasattr(encoder, a)
    ]
    if missing:
        raise TypeError(
            f"encoder {type(encoder).__name__} lacks {missing}: L-DPT needs a ViT with "
            f"patch_size, embed_dim and get_intermediate_layers (DINOv3 interface)"
        )
    return encoder
