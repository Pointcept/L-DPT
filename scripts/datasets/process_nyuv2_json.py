# NYUv2: download from
# https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing
# train set as in https://github.com/cleinc/bts/tree/master/pytorch#nyu-depvh-v2,
# test set as in https://github.com/cleinc/bts?tab=readme-ov-file#prepare-nyu-depth-v2-test-set,
# with the BTS file lists in <root>/splits. Writes <root>/splits/{train,test}.json.

import argparse
import json
import os


def process_file(data_root, input_txt, split):
    data_list = []
    input_txt = os.path.join(data_root, input_txt)
    output_json = os.path.join(data_root, split + ".json")

    with open(input_txt, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if parts[0][0] == "/":
            parts[0] = parts[0][1:]
        if parts[1][0] == "/":
            parts[1] = parts[1][1:]
        if len(parts) >= 2:
            item = {
                "image": os.path.join(split, parts[0]),
                "depth": os.path.join(split, parts[1]),
            }
            data_list.append(item)

    with open(output_json, "w") as f:
        json.dump(data_list, f, indent=4)
    print(f"{split}: {len(data_list)} samples -> {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NYUv2 split JSONs")
    parser.add_argument("--root", required=True, help="dataset root, e.g. data/nyuv2")
    args = parser.parse_args()
    splits_dir = os.path.join(args.root, "splits")
    process_file(splits_dir, "nyudepthv2_train_files_with_gt.txt", "train")
    process_file(splits_dir, "nyudepthv2_test_files_with_gt.txt", "test")
