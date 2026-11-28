"""
Tiled encoder for large inputs: the frozen encoder runs on overlapping windows of the
token grid, the per-stage selections of every window are blended onto the whole grid,
and the latent decoder decodes that grid in one pass
(``LatentDPT.infer(image, encoder_tiles=EncoderTiles())``; api.predict runs it by
default for depth; a segmentation model runs one encoder pass unless asked).

DINOv3 normalises its patch coordinates to [-1, 1] per axis whatever the token count,
and L-DPT was trained on 768 px crops (48 tokens) with the encoder's coordinate rescale
of 0.5 to 2 active: neighbour spacings of 1 / 48 to 1 / 12. A 100 MP photo (547 x 729
tokens) packs its tokens 5.7 to 7.6 times denser than that. A window of 192 x 192
tokens (3072 px) with its RoPE coordinates times 2 sits at the trained limit (spacing
4 / 192 = 1 / 48, coordinates within [-2, 2]). The decoder was trained with its own coordinates and runs
unchanged on the whole grid.

Geometry, per axis of n tokens (tile_starts, tile_ramp): k = ceil((n - O) / (T - O))
tiles of T tokens at round(i (n - T) / (k - 1)), so neighbours overlap by O tokens or
more; each tile weighs its tokens with a linear ramp over the O tokens next to an edge
that is not the image border (1 elsewhere), the 2-D weight is the product of the two
axes' ramps, and the blend divides the weighted sum by the summed weights. An axis of
at most T tokens is one tile. A grid of at most T x T tokens is not tiled at all
(EncoderTiles.applies): the plain forward runs, bit for bit.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import dataclasses
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

__all__ = ["EncoderTiles", "tile_ramp", "tile_starts", "tile_windows"]


@dataclasses.dataclass(frozen=True)
class EncoderTiles:
    """How LatentDPT.infer tiles the encoder of a large input.

    Args:
        tile: tile side in tokens (192: 3072 px with 16 px patches). A grid with no
            side above it runs the plain forward.
        overlap: the least overlap of neighbouring tiles and the length of the blend
            ramp, in tokens; 1 <= overlap <= tile / 2.
        rope_scale: the encoder's RoPE coordinates are multiplied by it inside a tile
            (``encoder.rope_embed.eval_scale``, restored afterwards); 1.0 leaves them
            as they are.
    """

    tile: int = 192
    overlap: int = 64
    rope_scale: float = 2.0

    def __post_init__(self) -> None:
        for name in ("tile", "overlap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"EncoderTiles.{name} must be an int, got {value!r}")
        if not 1 <= self.overlap <= self.tile // 2:
            raise ValueError(
                f"EncoderTiles: overlap must be in [1, tile / 2] = [1, "
                f"{self.tile // 2}], got {self.overlap} (tile {self.tile})"
            )
        scale = self.rope_scale
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not 0 < scale < math.inf
        ):
            raise ValueError(f"EncoderTiles.rope_scale must be > 0, got {scale!r}")

    def applies(self, patch_hw: Tuple[int, int]) -> bool:
        """True when a side of the (h, w) token grid exceeds the tile."""
        return max(patch_hw) > self.tile


def tile_starts(n: int, tile: int, overlap: int) -> List[int]:
    """Origins (tokens) of the tiles along an axis of ``n`` tokens: [0] when
    ``n <= tile``, else k = ceil((n - overlap) / (tile - overlap)) tiles at
    round(i (n - tile) / (k - 1)), the first at 0 and the last ending at ``n``;
    neighbours overlap by at least ``overlap`` tokens."""
    if n <= tile:
        return [0]
    k = math.ceil((n - overlap) / (tile - overlap))
    return [round(i * (n - tile) / (k - 1)) for i in range(k)]


def tile_ramp(
    n: int,
    lo: bool,
    hi: bool,
    overlap: int,
    device: Union[str, torch.device, None] = None,
) -> Tensor:
    """Blend weights of a tile of ``n`` tokens along one axis: 1, with a linear ramp
    (i + 0.5) / overlap, i = 0 .. overlap - 1, rising over the first ``overlap``
    tokens when ``lo`` (that edge is inside the image) and falling over the last ones
    when ``hi``. Two ramps of an overlap of exactly ``overlap`` tokens sum to 1.

    Shapes:
        -> [n] float32
    """
    w = torch.ones(n, device=device)
    r = (torch.arange(overlap, device=device, dtype=torch.float32) + 0.5) / overlap
    if lo:
        w[:overlap] = r
    if hi:
        w[n - overlap :] = r.flip(0)
    return w


def tile_windows(
    patch_hw: Tuple[int, int],
    tiles: EncoderTiles,
    device: Union[str, torch.device, None] = None,
) -> List[Tuple[int, int, int, int, Tensor]]:
    """The tiles of an (h, w) token grid: (y0, x0, th, tw, weight [th, tw] float32)
    per tile, rows first. th = min(tile, h), tw = min(tile, w); the weight is the
    product of the two axes' tile_ramp."""
    h, w = patch_hw
    th, tw = min(tiles.tile, h), min(tiles.tile, w)
    out = []
    for y0 in tile_starts(h, tiles.tile, tiles.overlap):
        wy = tile_ramp(th, y0 > 0, y0 + th < h, tiles.overlap, device)
        for x0 in tile_starts(w, tiles.tile, tiles.overlap):
            wx = tile_ramp(tw, x0 > 0, x0 + tw < w, tiles.overlap, device)
            out.append((y0, x0, th, tw, wy[:, None] * wx[None, :]))
    return out


def resolve(value: Union[EncoderTiles, None, bool]) -> Optional[EncoderTiles]:
    """An EncoderTiles, or None for None / False (the plain forward); TypeError for
    anything else."""
    if value is None or value is False:
        return None
    if isinstance(value, EncoderTiles):
        return value
    raise TypeError(
        f"encoder_tiles must be an EncoderTiles, None or False, got {value!r}"
    )
