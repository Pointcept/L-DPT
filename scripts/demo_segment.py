"""
Semantic segmentation of one image with an L-DPT model: a colour PNG and the class ids.

    python scripts/demo_segment.py photo.jpg
    python scripts/demo_segment.py photo.jpg --weights l-dpt_large_semseg_cocostuff.pth

writes photo_seg.png (one fixed colour per class: ldpt/utils/seg_image.py) and
photo_seg.npy (HxW class ids: uint8, int32 for a model of more than 256 classes), and
prints both paths and the largest classes by area. The model is the preset
l-dpt_large_semseg_cocostuff (COCO-Stuff, 171 classes, the names in
ldpt.utils.seg_image.COCOSTUFF_CLASS_NAMES); its weights come from --weights, or from
the file pinned in presets.WEIGHTS (downloaded from Hugging Face on first use, or read
from $LDPT_CKPT_LARGE_SEMSEG_COCOSTUFF); the DINOv3 weights (the official .pth file or a
download of its Hugging Face repository) from --encoder-weights or $LDPT_DINOV3. The
network runs with the short side resized to 768 (--size auto, the default: the
evaluation's resolution; or an int) or at the photo's native resolution (--size
native); the class map is at the photo's resolution.
"""

import argparse
import os
import sys

# PyTorch's allocator with expandable segments reserves less GPU memory beyond the peak it
# allocates at native resolution (set before torch initialises CUDA; a value already in
# the environment wins)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_PRESET = "l-dpt_large_semseg_cocostuff"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="input image (any format OpenCV reads)")
    parser.add_argument(
        "-o",
        "--output",
        help="output PNG (default: <image stem>_seg.png); the .npy gets its stem",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"model preset (default {DEFAULT_PRESET})",
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
        default="auto",
        help="'auto' (default): short side 768, as evaluated; 'native': the photo's "
        "resolution; or the short side as an int",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="also average with the mirrored image (the softmaxes)",
    )
    parser.add_argument(
        "--top", type=int, default=5, help="classes listed by area (default 5)"
    )
    args = parser.parse_args(argv)
    if args.size == "native":
        args.size = None
    elif args.size != "auto":
        try:
            args.size = int(args.size)
            assert args.size > 0
        except (ValueError, AssertionError):
            parser.error(f"--size: expected auto, native or an int, got {args.size!r}")
    if args.top < 0:
        parser.error(f"--top: expected an int >= 0, got {args.top}")
    if args.output is None:
        args.output = os.path.splitext(args.image)[0] + "_seg.png"
    return args


def main(argv=None):
    args = parse_args(argv)
    import cv2
    import numpy as np
    import torch

    from ldpt.models.l_dpt import api
    from ldpt.utils.seg_image import colorize, top_classes

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
    if model.task != "semseg":
        raise SystemExit(
            f"{args.preset} is a {model.task} model: scripts/demo_depth.py writes its "
            f"depth"
        )
    try:
        classes = api.predict(model, image, size=args.size, flip=args.flip)
    except torch.cuda.OutOfMemoryError:
        if args.size is not None:
            raise
        h, w = image.shape[:2]
        raise SystemExit(
            f"out of GPU memory at native resolution ({w} x {h}): try --size auto"
        ) from None
    npy = os.path.splitext(args.output)[0] + ".npy"
    color = colorize(classes, model.num_classes, bgr=True)
    if not cv2.imwrite(args.output, color):
        raise SystemExit(f"cannot write {args.output}")
    dtype = np.uint8 if model.num_classes <= 256 else np.int32
    np.save(npy, classes.astype(dtype))
    h, w = classes.shape
    print(f"{args.output}  colour, {w}x{h}")
    print(f"{npy}  {np.dtype(dtype).name} class ids, {w}x{h}")
    for c, name, share in top_classes(classes, model.class_names, args.top):
        print(f"{100 * share:5.1f} %  {name if name is not None else 'class'} ({c})")


if __name__ == "__main__":
    main()
