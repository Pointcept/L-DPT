# Structured3D: all Perspective (full) zips from
# https://zju-kjl-jointlab-azure.kujiale.com/Structured3D/README.txt, extracted into
# <root>/Structured3D. Drops mostly-white renders; writes <root>/splits/train.json.

import argparse
import glob
import json
import os
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image
import numpy as np


def check_image(args):
    item, root_dir = args
    image_path = os.path.join(root_dir, item["image"])
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    white = np.sum(np.all(arr > 240, axis=-1))
    total = arr.size // 3
    if white / total < 0.3:
        return item
    return None


def generate_dataset_json(root_dir, output_file="train.json", num_workers=64):
    search_pattern = os.path.join(root_dir, "Structured3D", "**", "depth.png")
    depth_files = glob.glob(search_pattern, recursive=True)

    dataset_list = []
    for depth_file in tqdm(depth_files):
        image_file = depth_file.replace("depth.png", "rgb_rawlight.png")
        entry = {
            "image": os.path.relpath(image_file, root_dir),
            "depth": os.path.relpath(depth_file, root_dir),
        }
        dataset_list.append(entry)

    print(f"Scan complete. Found {len(dataset_list)} pairs.")
    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    check_image, [(item, root_dir) for item in dataset_list]
                ),
                total=len(dataset_list),
            )
        )
    dataset_list = [r for r in results if r is not None]

    print(f"Filter complete. Keep {len(dataset_list)} pairs.")
    if len(dataset_list) > 0:
        print("Sample data (first 2 entries):")
        print(json.dumps(dataset_list[:2], indent=4))

    os.makedirs(os.path.join(root_dir, "splits"), exist_ok=True)
    output_file = os.path.join(root_dir, "splits", output_file)
    with open(output_file, "w") as f:
        json.dump(dataset_list, f, indent=4)
    print(f"Full list saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structured3D split JSON")
    parser.add_argument(
        "--root", required=True, help="dataset root, e.g. data/structured3d"
    )
    parser.add_argument("--output", default="train.json", help="name in <root>/splits/")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    generate_dataset_json(args.root, args.output, args.num_workers)
