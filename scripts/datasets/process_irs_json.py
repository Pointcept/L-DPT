# IRS (https://github.com/HKBU-HPML/IRS), extracted into <root>/IRS_small (its Home/
# Office/ Restaurant/ Store/ folders). Keeps the IRS_small subset; writes
# <root>/splits/train_small.json ({"image", "depth"} relative to <root>).

import argparse
import glob
import json
import os

from tqdm import tqdm


def generate_dataset_json(root_dir, output_file="train_small.json"):
    search_pattern = os.path.join(root_dir, "**", "d_*.exr")
    depth_files = glob.glob(search_pattern, recursive=True)

    dataset_list = []
    for depth_file in tqdm(depth_files):
        if "IRS_small" not in depth_file:
            continue
        image_file = depth_file.replace(".exr", ".png")
        image_file = image_file.replace("d_", "l_")
        entry = {
            "image": os.path.relpath(image_file, root_dir),
            "depth": os.path.relpath(depth_file, root_dir),
        }
        dataset_list.append(entry)

    print(f"Scan complete. Found {len(dataset_list)} pairs.")
    if len(dataset_list) > 0:
        print("Sample data (first 2 entries):")
        print(json.dumps(dataset_list[:2], indent=4))

    os.makedirs(os.path.join(root_dir, "splits"), exist_ok=True)
    output_file = os.path.join(root_dir, "splits", output_file)
    with open(output_file, "w") as f:
        json.dump(dataset_list, f, indent=4)
    print(f"Full list saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRS split JSON")
    parser.add_argument("--root", required=True, help="dataset root, e.g. data/irs")
    parser.add_argument(
        "--output", default="train_small.json", help="name in <root>/splits/"
    )
    args = parser.parse_args()
    generate_dataset_json(args.root, args.output)
