"""
Relative depth of one image with a released L-DPT model: a colour PNG and the depth.

    python scripts/demo_depth.py photo.jpg
    python scripts/demo_depth.py photo.jpg --preset l-dpt_large_depth_mixed --size auto

writes photo_depth.png (near blue, far red, on a tone curve fitted to the photo: the
colours of the demo, ldpt/utils/depth_image.py) and photo_depth.npy (HxW float32
relative depth: larger = farther, up to a scale and a shift), and prints both paths.
The weights of the preset (default l-dpt_large_depth_zeroshot) are downloaded from
Hugging Face on first use, or read from --weights or $LDPT_CKPT_LARGE_DEPTH_ZEROSHOT
($LDPT_CKPT_LARGE_DEPTH_MIXED); the DINOv3 weights (the official .pth file or a download
of its Hugging Face repository) come from --encoder-weights or $LDPT_DINOV3. The network
runs at the photo's native resolution (--size native, the default; api.predict's
high-resolution memory options) or with the short side resized (--size auto: 768, the
training resolution; or an int), the depth resized back to the photo. An input with a
side above 3072 px runs the frozen encoder on overlapping 3072 px windows and the
decoder on the whole latent grid in one pass (api.predict's default encoder_tiles);
--no-encoder-tiles runs the encoder once on the whole image.
"""

import argparse
import os
import sys

# PyTorch's allocator with expandable segments reserves less GPU memory beyond the peak it
# allocates at native resolution (set before torch initialises CUDA; a value already in
# the environment wins)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpt.utils.depth_image import COLORMAPS, DISPLAY_CMAP  # noqa: E402  (numpy only)

DEFAULT_PRESET = "l-dpt_large_depth_zeroshot"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="input image (any format OpenCV reads)")
    parser.add_argument(
        "-o",
        "--output",
        help="output PNG (default: <image stem>_depth.png); the .npy gets its stem",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"released model: {DEFAULT_PRESET} (default) or l-dpt_large_depth_mixed",
    )
    parser.add_argument("--weights", help="local weights file of the preset")
    parser.add_argument(
        "--encoder-weights",
        help="official DINOv3 file or Hugging Face (transformers-format) folder "
        "(default: $LDPT_DINOV3)",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    parser.add_argument(
        "--size",
        default="native",
        help="'native' (default): the photo's resolution; 'auto': short side 768 (the "
        "training resolution); or the short side as an int",
    )
    parser.add_argument(
        "--flip", action="store_true", help="also average with the mirrored image"
    )
    parser.add_argument(
        "--no-encoder-tiles",
        dest="encoder_tiles",
        action="store_const",
        const=None,
        default="auto",
        help="run the encoder once on the whole image (default: overlapping 3072 px "
        "windows when a side exceeds 3072 px)",
    )
    parser.add_argument(
        "--cmap", default=DISPLAY_CMAP, choices=list(COLORMAPS), help="colormap"
    )
    args = parser.parse_args(argv)
    if args.size == "native":
        args.size = None
    elif args.size != "auto":
        try:
            args.size = int(args.size)
            assert args.size > 0
        except (ValueError, AssertionError):
            parser.error(f"--size: expected native, auto or an int, got {args.size!r}")
    if args.output is None:
        args.output = os.path.splitext(args.image)[0] + "_depth.png"
    return args


def main(argv=None):
    args = parse_args(argv)
    import cv2
    import numpy as np
    import torch

    from ldpt.models.l_dpt import api
    from ldpt.utils.depth_image import colorize

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = api.load(
        args.preset,
        encoder_weights=args.encoder_weights,
        device=args.device,
        weights=args.weights,
    )
    if model.task != "relative_depth":
        raise SystemExit(
            f"{args.preset} is a {model.task} model: scripts/demo_segment.py writes "
            f"its class map"
        )
    try:
        depth = api.predict(
            model,
            image,
            size=args.size,
            flip=args.flip,
            encoder_tiles=args.encoder_tiles,
        )
    except torch.cuda.OutOfMemoryError:
        if args.size is not None:
            raise
        h, w = image.shape[:2]
        raise SystemExit(
            f"out of GPU memory at native resolution ({w} x {h}): try --size auto"
        ) from None
    npy = os.path.splitext(args.output)[0] + ".npy"
    if not cv2.imwrite(args.output, colorize(depth, args.cmap, bgr=True)):
        raise SystemExit(f"cannot write {args.output}")
    np.save(npy, depth.astype(np.float32, copy=False))
    h, w = depth.shape
    print(f"{args.output}  colour, {w}x{h}")
    print(f"{npy}  float32 depth, {w}x{h}")


if __name__ == "__main__":
    main()
