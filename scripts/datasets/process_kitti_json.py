# KITTI: raw images from https://www.cvlibs.net/datasets/kitti/raw_data.php into
# <root>/image, depth GT as in
# https://github.com/cleinc/bts?tab=readme-ov-file#prepare-kitti-official-ground-truth-depth-maps
# into <root>/depth, and the BTS Eigen file lists in <root>/splits.
# Writes <root>/splits/<split>.json.

import argparse
import json
import os

LISTS = {
    "test": "eigen_test_files_with_gt.txt",
    "train": "eigen_train_files_with_gt.txt",
}


def process_file(data_root, input_txt, split):
    data_list = []
    input_txt = os.path.join(data_root, input_txt)
    output_json = os.path.join(data_root, split + ".json")

    with open(input_txt, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if parts[1] == "None":
            continue
        if parts[0][0] == "/":
            parts[0] = parts[0][1:]
        if parts[1][0] == "/":
            parts[1] = parts[1][1:]
        if len(parts) >= 2:
            item = {
                "image": os.path.join("image", parts[0]),
                "depth": os.path.join("depth", parts[1]),
            }
            data_list.append(item)

    with open(output_json, "w") as f:
        json.dump(data_list, f, indent=4)
    print(f"{split}: {len(data_list)} samples -> {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI (Eigen split) JSONs")
    parser.add_argument("--root", required=True, help="dataset root, e.g. data/kitti")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=sorted(LISTS),
        help="train is only used by the configs of the mixed model",
    )
    args = parser.parse_args()
    for split in args.splits:
        process_file(os.path.join(args.root, "splits"), LISTS[split], split)
