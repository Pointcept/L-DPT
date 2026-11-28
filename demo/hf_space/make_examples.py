"""
Downsampled JPEG examples for the Space from full-resolution photos.

    python demo/hf_space/make_examples.py photo1.jpg photo2.jpg --mp 8 2

writes <stem>_8mp.jpg and <stem>_2mp.jpg into demo/hf_space/examples/ (area
interpolation, EXIF orientation applied, never upsampled: a target at or above the
photo's size is skipped). Two photos with the same stem (trip1/IMG_1.jpg and
trip2/IMG_1.jpg; case-insensitive) are refused before anything is written. The images
are not committed to this repo (.gitignore); upload them with the Space.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Sequence

HERE = Path(__file__).resolve().parent


def target_hw(height: int, width: int, mp: float):
    """(H, W) with H * W ~= mp megapixels and the aspect ratio kept."""
    scale = math.sqrt(mp * 1e6 / (height * width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def make_examples(
    sources: Sequence[os.PathLike],
    out: os.PathLike = HERE / "examples",
    mps: Sequence[float] = (8.0, 2.0),
    quality: int = 92,
) -> List[Path]:
    import cv2

    unique, stems = [], {}
    for src in map(Path, sources):
        key = src.stem.casefold()
        if key in stems:
            if stems[key].resolve() == src.resolve():
                continue  # the same photo twice
            raise SystemExit(
                f"{stems[key]} and {src} would write the same examples "
                f"({src.stem}_<mp>mp.jpg): rename one"
            )
        stems[key] = src
        unique.append(src)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for src in unique:
        image = cv2.imread(os.fspath(src), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read {src}")
        h, w = image.shape[:2]
        for mp in mps:
            if mp * 1e6 >= h * w:
                print(f"{src}: {w}x{h} is not above {mp:g} MP, skipped")
                continue
            th, tw = target_hw(h, w, mp)
            small = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
            dst = out / f"{Path(src).stem}_{mp:g}mp.jpg"
            if not cv2.imwrite(
                os.fspath(dst), small, [cv2.IMWRITE_JPEG_QUALITY, quality]
            ):
                raise SystemExit(f"cannot write {dst}")
            print(f"{dst}: {tw}x{th} ({tw * th / 1e6:.1f} MP)")
            written.append(dst)
    return written


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("sources", nargs="+", help="full-resolution photos")
    parser.add_argument("--out", default=str(HERE / "examples"), help="output folder")
    parser.add_argument(
        "--mp", type=float, nargs="+", default=[8.0, 2.0], help="megapixel targets"
    )
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality")
    args = parser.parse_args(argv)
    make_examples(args.sources, args.out, args.mp, args.quality)


if __name__ == "__main__":
    sys.exit(main())
