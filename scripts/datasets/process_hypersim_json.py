# Hypersim (https://github.com/apple/ml-hypersim), official scene split from
# evermotion_dataset/analysis/metadata_images_split_scene_v1.csv (public frames:
# 59,543 train / 7,386 val / 7,690 test). Expects the ml-hypersim scene layout under
# <root>/scenes/<scene>/images/: scene_cam_XX_final_preview/frame.NNNN.tonemap.jpg and
# scene_cam_XX_geometry_hdf5/frame.NNNN.{depth_meters,semantic}.hdf5 (download: see
# ldpt/datasets/README.md). Writes <root>/splits/{train,val,test}.json; frames with a
# missing file are skipped and counted.

import argparse
import csv
import json
import os

FILES = {
    "image": ("final_preview", "tonemap.jpg"),
    "depth": ("geometry_hdf5", "depth_meters.hdf5"),
    "segment": ("geometry_hdf5", "semantic.hdf5"),
}


def list_dir(root_dir, folder, cache):
    # one listdir per camera folder instead of a stat per file (slow on network disks)
    if folder not in cache:
        try:
            cache[folder] = set(os.listdir(os.path.join(root_dir, folder)))
        except FileNotFoundError:
            cache[folder] = set()
    return cache[folder]


def generate_dataset_json(root_dir, metadata_csv, splits):
    data_lists = {split: [] for split in splits}
    skipped = {split: 0 for split in splits}
    cache = {}
    with open(metadata_csv, newline="") as f:
        for row in csv.DictReader(f):
            split = row["split_partition_name"]
            if row["included_in_public_release"] != "True" or split not in data_lists:
                continue
            scene, camera = row["scene_name"], row["camera_name"]
            frame = int(row["frame_id"])
            item = {}
            for key, (kind, suffix) in FILES.items():
                folder = f"scenes/{scene}/images/scene_{camera}_{kind}"
                name = f"frame.{frame:04d}.{suffix}"
                if name not in list_dir(root_dir, folder, cache):
                    break
                item[key] = f"{folder}/{name}"
            else:
                data_lists[split].append(item)
                continue
            skipped[split] += 1

    os.makedirs(os.path.join(root_dir, "splits"), exist_ok=True)
    for split, data_list in data_lists.items():
        output_file = os.path.join(root_dir, "splits", f"{split}.json")
        with open(output_file, "w") as f:
            json.dump(data_list, f, indent=4)
        print(
            f"{split}: {len(data_list)} samples, {skipped[split]} skipped "
            f"(missing files) -> {output_file}"
        )
    if any(data_lists.values()):
        sample = next(v for v in data_lists.values() if v)[0]
        print("Sample entry:")
        print(json.dumps(sample, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hypersim split JSONs")
    parser.add_argument(
        "--root", required=True, help="dataset root, e.g. data/hypersim"
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="ml-hypersim evermotion_dataset/analysis/metadata_images_split_scene_v1.csv",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="split_partition_name values -> splits/<split>.json",
    )
    args = parser.parse_args()
    generate_dataset_json(args.root, args.metadata_csv, args.splits)
