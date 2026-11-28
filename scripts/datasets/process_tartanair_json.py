# TartanAir, with the download tool of https://github.com/castacks/tartanair_tools:
#   python download_training.py --output-dir <root>/data --rgb --unzip --huggingface
#   python download_training.py --output-dir <root>/data --depth --unzip --huggingface
# Left camera only. Writes <root>/splits/<prefix>_easy.json and <prefix>_hard.json
# ({"image", "depth"} relative to <root>).

import argparse
import json
import os


def generate_dataset_json(root_dir, output_prefix="train", output_dir=None):
    easy_list = []
    hard_list = []

    images_root = os.path.join(root_dir, "data")
    if not os.path.exists(images_root):
        print(f"Error: Image directory not found at {images_root}")
        return

    print(f"Scanning directory: {images_root} ...")
    for root, dirs, files in os.walk(images_root):
        for file in files:
            if ".png" not in file:
                continue

            # Skip right camera images, only keep left
            if "_right.png" in file:
                continue

            image_abs_path = os.path.join(root, file)
            depth_abs_path = image_abs_path.replace("/image_", "/depth_")
            depth_abs_path = depth_abs_path.replace(".png", "_depth.npy")

            rel_image_path = os.path.relpath(image_abs_path, root_dir)
            rel_depth_path = os.path.relpath(depth_abs_path, root_dir)

            entry = {
                "image": rel_image_path,
                "depth": rel_depth_path,
            }

            # Separate by difficulty level
            if "/Easy/" in image_abs_path:
                easy_list.append(entry)
            elif "/Hard/" in image_abs_path:
                hard_list.append(entry)

    output_dir = output_dir or os.path.join(root_dir, "splits")
    os.makedirs(output_dir, exist_ok=True)

    # Save Easy dataset
    print(f"\nEasy samples: {len(easy_list)}")
    if len(easy_list) > 0:
        print("Sample Easy data (first 2 entries):")
        print(json.dumps(easy_list[:2], indent=4))
        easy_output = os.path.join(output_dir, f"{output_prefix}_easy.json")
        with open(easy_output, "w") as f:
            json.dump(easy_list, f, indent=4)
        print(f"Easy list saved to: {easy_output}")

    # Save Hard dataset
    print(f"\nHard samples: {len(hard_list)}")
    if len(hard_list) > 0:
        print("Sample Hard data (first 2 entries):")
        print(json.dumps(hard_list[:2], indent=4))
        hard_output = os.path.join(output_dir, f"{output_prefix}_hard.json")
        with open(hard_output, "w") as f:
            json.dump(hard_list, f, indent=4)
        print(f"Hard list saved to: {hard_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TartanAir split JSONs")
    parser.add_argument(
        "--root", required=True, help="dataset root, e.g. data/tartanair"
    )
    parser.add_argument("--prefix", default="train", help="output name prefix")
    parser.add_argument("--output-dir", help="default: <root>/splits")
    args = parser.parse_args()
    generate_dataset_json(args.root, args.prefix, args.output_dir)
