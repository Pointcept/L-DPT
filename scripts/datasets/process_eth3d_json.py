# ETH3D: images (distorted) from https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z
# and depth from https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z,
# extracted into <root>. Writes <root>/splits/train.json (only train has depth).

import argparse
import glob
import json
import os

from tqdm import tqdm


def generate_dataset_json(root_dir, output_file="train.json"):
    search_pattern = os.path.join(root_dir, "**", "ground_truth_depth", "**", "*.JPG")
    depth_files = glob.glob(search_pattern, recursive=True)

    dataset_list = []
    for depth_file in tqdm(depth_files):
        image_file = depth_file.replace("ground_truth_depth", "images")
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
    parser = argparse.ArgumentParser(description="ETH3D split JSON")
    parser.add_argument("--root", required=True, help="dataset root, e.g. data/eth3d")
    parser.add_argument("--output", default="train.json", help="name in <root>/splits/")
    args = parser.parse_args()
    generate_dataset_json(args.root, args.output)
