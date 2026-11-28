# DIODE from https://diode-dataset.org/, with the file lists <root>/data_list/<split>.csv
# (image, depth, depth mask per row). Writes <root>/splits/<split>.json.

import argparse
import csv
import json
import os


def generate_dataset_json(root_dir, csv_files, output_file="val.json"):
    dataset_list = []
    for csv_file in csv_files:
        with open(os.path.join(root_dir, csv_file), newline="") as f:
            for row in csv.reader(f):
                image_file, depth_file, mask_file = row[:3]
                dataset_list.append(
                    {"image": image_file, "depth": depth_file, "depth_mask": mask_file}
                )

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
    parser = argparse.ArgumentParser(description="DIODE split JSONs")
    parser.add_argument("--root", required=True, help="dataset root, e.g. data/diode")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val_indoors", "val_outdoor"],
        help="data_list/<split>.csv -> splits/<split>.json (the mixed model's configs "
        "also use train_indoors and train_outdoor)",
    )
    args = parser.parse_args()
    for split in args.splits:
        generate_dataset_json(args.root, [f"data_list/{split}.csv"], f"{split}.json")
