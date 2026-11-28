# COCO-Stuff (train2017 / val2017 images and stuffthingmaps_trainval2017 label maps)
# at <root>. Writes <root>/splits/{train,val}.json ({"image", "segment"}).

import json
import argparse
from pathlib import Path


def build_split_list(root, img_rel_dir, seg_rel_dir):
    root = Path(root)
    img_dir = root / img_rel_dir
    seg_dir = root / seg_rel_dir

    seg_map = {}
    for seg_path in seg_dir.rglob("*.png"):
        seg_map[seg_path.stem] = seg_path

    data_list = []

    for img_path in sorted(img_dir.rglob("*.jpg")):
        stem = img_path.stem
        if stem not in seg_map:
            continue

        seg_path = seg_map[stem]

        img_rel = img_path.relative_to(root)
        seg_rel = seg_path.relative_to(root)

        data_list.append(
            {
                "image": str(img_rel),
                "segment": str(seg_rel),
            }
        )

    return data_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to the COCO-Stuff dataset root, e.g. data/cocostuff",
    )
    parser.add_argument(
        "--out-train",
        type=str,
        default=None,
        help="Output path for the training split JSON (default: <root>/splits/train.json)",
    )
    parser.add_argument(
        "--out-val",
        type=str,
        default=None,
        help="Output path for the validation split JSON (default: <root>/splits/val.json)",
    )
    args = parser.parse_args()
    splits = Path(args.root) / "splits"
    args.out_train = args.out_train or str(splits / "train.json")
    args.out_val = args.out_val or str(splits / "val.json")
    for out in (args.out_train, args.out_val):
        Path(out).parent.mkdir(parents=True, exist_ok=True)

    # Modify these paths based on your directory layout:
    IMG_DIRS = {
        "train": "train2017",
        "val": "val2017",
    }
    SEG_DIRS = {
        "train": "train2017",
        "val": "val2017",
    }

    train_list = build_split_list(args.root, IMG_DIRS["train"], SEG_DIRS["train"])
    val_list = build_split_list(args.root, IMG_DIRS["val"], SEG_DIRS["val"])

    print(f"train samples: {len(train_list)}, val samples: {len(val_list)}")

    with open(args.out_train, "w") as f:
        json.dump(train_list, f, indent=2)

    with open(args.out_val, "w") as f:
        json.dump(val_list, f, indent=2)


if __name__ == "__main__":
    main()
