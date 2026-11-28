# PASCAL VOC 2012 (VOCdevkit/VOC2012 layout) at <root>.
# Writes <root>/splits/{train,val}.json ({"image", "segment"} relative to <root>).

import argparse
import json
from pathlib import Path

SPLITS = ["train", "val"]


def build_split(voc_root: Path, split: str):
    split_file = voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(split_file)

    items = []
    with open(split_file, "r") as f:
        for line in f:
            img_id = line.strip()
            if not img_id:
                continue
            image_rel = f"JPEGImages/{img_id}.jpg"
            segment_rel = f"SegmentationClass/{img_id}.png"

            items.append(
                {
                    "image": image_rel,
                    "segment": segment_rel,
                }
            )

    out_path = voc_root / "splits" / f"{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)

    print(f"{split}: {len(items)} samples -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOC 2012 segmentation split JSONs")
    parser.add_argument(
        "--root", required=True, help="VOC2012 folder, e.g. data/voc2012 (JPEGImages/)"
    )
    args = parser.parse_args()
    for s in SPLITS:
        build_split(Path(args.root), s)
