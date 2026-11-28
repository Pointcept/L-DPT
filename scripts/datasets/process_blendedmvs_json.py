# BlendedMVS / BlendedMVS+ / BlendedMVS++ (low-res set), e.g. from
# https://aistudio.baidu.com/datasetdetail/130299/0, extracted into <root>/files/.
# Writes <root>/splits/train.json: {"image", "depth"} paths relative to <root>.

import argparse
import glob
import json
import os


def generate_dataset_json(root_dir, output_file="train.json"):
    dataset_list = []

    if not os.path.exists(root_dir):
        print(f"Error: Image directory not found at {root_dir}")
        return

    print(f"Scanning directory: {root_dir} ...")
    image_files = glob.glob(os.path.join(root_dir, "files/*/blended_images/*.jpg"))
    for image_file in image_files:
        if "_masked" in image_file:
            continue
        depth_file = image_file.replace("blended_images", "rendered_depth_maps")
        depth_file = depth_file.replace(".jpg", ".pfm")
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
    parser = argparse.ArgumentParser(description="BlendedMVS split JSON")
    parser.add_argument(
        "--root", required=True, help="dataset root, e.g. data/blendedmvs"
    )
    parser.add_argument("--output", default="train.json", help="name in <root>/splits/")
    args = parser.parse_args()
    generate_dataset_json(args.root, args.output)
