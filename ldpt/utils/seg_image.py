"""
Segmentation maps as images, one convention for scripts/demo_segment.py and the API:

- class names: COCOSTUFF_CLASS_NAMES, the 171 COCO-Stuff classes in the train-id order
  of COCOStuffDataset (label_mapping="official": the 80 things, then the 91 stuff
  classes, each in the order of the official labels.txt), which is the order of the
  segmentation model's output channels;
- colour: class c is entry c + 1 of the PASCAL VOC colour map (bit-interleaved RGB;
  entry 0, black, is left for ignored pixels, id -1), the same colour in every image.

    from ldpt.utils.seg_image import COCOSTUFF_CLASS_NAMES, colorize, top_classes

    rgb = colorize(classes)                  # HxW class ids -> HxWx3 uint8 RGB
    for c, name, share in top_classes(classes, COCOSTUFF_CLASS_NAMES):
        print(f"{100 * share:5.1f} %  {name}")

Only numpy is imported here.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "COCOSTUFF_CLASS_NAMES",
    "IGNORE_INDEX",
    "colorize",
    "palette",
    "top_classes",
]

IGNORE_INDEX = -1  # the label of unlabelled pixels in the datasets (drawn black)

COCOSTUFF_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "banner",
    "blanket",
    "branch",
    "bridge",
    "building other",
    "bush",
    "cabinet",
    "cage",
    "cardboard",
    "carpet",
    "ceiling other",
    "ceiling tile",
    "cloth",
    "clothes",
    "clouds",
    "counter",
    "cupboard",
    "curtain",
    "desk stuff",
    "dirt",
    "door stuff",
    "fence",
    "floor marble",
    "floor other",
    "floor stone",
    "floor tile",
    "floor wood",
    "flower",
    "fog",
    "food other",
    "fruit",
    "furniture other",
    "grass",
    "gravel",
    "ground other",
    "hill",
    "house",
    "leaves",
    "light",
    "mat",
    "metal",
    "mirror stuff",
    "moss",
    "mountain",
    "mud",
    "napkin",
    "net",
    "paper",
    "pavement",
    "pillow",
    "plant other",
    "plastic",
    "platform",
    "playingfield",
    "railing",
    "railroad",
    "river",
    "road",
    "rock",
    "roof",
    "rug",
    "salad",
    "sand",
    "sea",
    "shelf",
    "sky other",
    "skyscraper",
    "snow",
    "solid other",
    "stairs",
    "stone",
    "straw",
    "structural other",
    "table",
    "tent",
    "textile other",
    "towel",
    "tree",
    "vegetable",
    "wall brick",
    "wall concrete",
    "wall other",
    "wall panel",
    "wall stone",
    "wall tile",
    "wall wood",
    "water other",
    "waterdrops",
    "window blind",
    "window other",
    "wood",
]


def palette(num_classes: int) -> np.ndarray:
    """The colour of each class: [num_classes, 3] uint8 RGB, class c = entry c + 1 of
    the PASCAL VOC colour map (the bits of c + 1 spread over the high bits of R, G, B),
    distinct for every class."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    ids = np.arange(1, num_classes + 1, dtype=np.int64)
    rgb = np.zeros((num_classes, 3), np.int64)
    for bit in range(8):
        for channel in range(3):
            rgb[:, channel] |= ((ids >> (3 * bit + channel)) & 1) << (7 - bit)
    return rgb.astype(np.uint8)


def colorize(
    classes: np.ndarray,
    num_classes: Optional[int] = None,
    *,
    colors: Optional[np.ndarray] = None,
    bgr: bool = False,
) -> np.ndarray:
    """A class map in colour: [H, W] integer ids -> [H, W, 3] uint8 (RGB, BGR for
    OpenCV with ``bgr``). ``colors`` ([N, 3] uint8) defaults to ``palette(num_classes)``
    (num_classes default: the largest id + 1); ids below 0 (IGNORE_INDEX) are black, an
    id >= N raises."""
    classes = np.asarray(classes)
    if classes.ndim != 2 or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError(
            f"classes must be an HxW integer map, got {classes.dtype} {classes.shape}"
        )
    top = int(classes.max()) if classes.size else -1
    if colors is None:
        colors = palette(max(num_classes or 0, top + 1, 1))
    colors = np.asarray(colors, np.uint8)
    if top >= len(colors):
        raise ValueError(f"class id {top} has no colour ({len(colors)} colours)")
    table = np.concatenate([np.zeros((1, 3), np.uint8), colors])  # 0: ignored
    rgb = table[np.where(classes < 0, 0, classes.astype(np.int64) + 1)]
    return np.ascontiguousarray(rgb[..., ::-1]) if bgr else rgb


def top_classes(
    classes: np.ndarray,
    names: Optional[Sequence[str]] = None,
    k: Optional[int] = 5,
) -> List[Tuple[int, Union[str, None], float]]:
    """The ``k`` largest classes of a class map (None: all present), largest first:
    (class id, name or None, share of the labelled pixels); ties by class id, ignored
    pixels (< 0) not counted."""
    flat = np.asarray(classes).ravel()
    flat = flat[flat >= 0]
    if flat.size == 0:
        return []
    counts = np.bincount(flat.astype(np.int64))
    present = np.flatnonzero(counts)
    order = present[np.argsort(-counts[present], kind="stable")][:k]
    return [
        (
            int(c),
            names[c] if names is not None and c < len(names) else None,
            counts[c] / flat.size,
        )
        for c in order
    ]
