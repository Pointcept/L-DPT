# Virtual KITTI 2 (rgb, depth, classSegmentation tars) extracted into <root>.
# Camera_0; Scene06 -> val, the rest -> train. Writes <root>/splits/{train,val}.json.

import argparse
import json
from pathlib import Path


def collect_vkitti2_samples(root, camera="Camera_0"):
    """
    Scan the VKITTI2 directory and collect aligned (image, depth, semantic) triplets.
    Each entry corresponds to a single frame from a given camera.

    Expected structure:
        SceneXX/
          <domain>/
            frames/
              rgb/Camera_Z/rgb_%05d.(jpg|png)
              depth/Camera_Z/depth_%05d.png
              classSegmentation/Camera_Z/classgt_%05d.png
    """
    root = Path(root)
    samples = []

    for scene_dir in sorted(root.glob("Scene*")):
        if not scene_dir.is_dir():
            continue

        scene_name = scene_dir.name  # e.g. "Scene06"

        for domain_dir in sorted(scene_dir.iterdir()):
            if not domain_dir.is_dir():
                continue

            frames_dir = domain_dir / "frames"
            rgb_dir = frames_dir / "rgb" / camera
            depth_dir = frames_dir / "depth" / camera
            class_dir = (
                frames_dir / "classSegmentation" / camera
            )  # camelCase as required

            if not (rgb_dir.exists() and depth_dir.exists() and class_dir.exists()):
                print(f"[Warning] Missing modal dirs in {domain_dir} for {camera}")
                continue

            rgb_files = sorted(rgb_dir.glob("rgb_*.jpg"))
            if not rgb_files:
                rgb_files = sorted(rgb_dir.glob("rgb_*.png"))

            if not rgb_files:
                print(f"[Warning] No RGB files found in {rgb_dir}")
                continue

            for img_file in rgb_files:
                stem = img_file.stem  # "rgb_00000"
                frame_id = stem.split("rgb_")[-1]

                depth_file = depth_dir / f"depth_{frame_id}.png"
                seg_file = class_dir / f"classgt_{frame_id}.png"

                if not depth_file.exists() or not seg_file.exists():
                    print(f"[Warning] Missing depth or segment for {img_file}")
                    continue

                samples.append(
                    {
                        "scene": scene_name,  # include scene temporarily for splitting
                        "image": img_file.relative_to(root).as_posix(),
                        "depth": depth_file.relative_to(root).as_posix(),
                        "segment": seg_file.relative_to(root).as_posix(),
                    }
                )

    # Sort for stable reproducible output
    samples.sort(key=lambda x: x["image"])
    return samples


def main():
    parser = argparse.ArgumentParser(description="VKITTI2 split JSONs")
    parser.add_argument("--root", required=True, help="dataset root, e.g. data/vkitti2")
    parser.add_argument("--output-dir", help="default: <root>/splits")
    args = parser.parse_args()
    root = args.root

    # Collect all samples
    samples = collect_vkitti2_samples(root, camera="Camera_0")

    # Split: Scene06 → val, rest → train
    train_samples = []
    val_samples = []

    for item in samples:
        if item["scene"] == "Scene06":
            val_samples.append(
                {
                    "image": item["image"],
                    "depth": item["depth"],
                    "segment": item["segment"],
                }
            )
        else:
            train_samples.append(
                {
                    "image": item["image"],
                    "depth": item["depth"],
                    "segment": item["segment"],
                }
            )

    # Save outputs
    out_dir = Path(args.output_dir or Path(root) / "splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.json", "w") as f:
        json.dump(train_samples, f, indent=4)

    with open(out_dir / "val.json", "w") as f:
        json.dump(val_samples, f, indent=4)

    print(f"Total samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples  : {len(val_samples)}")
    print(f"Saved {out_dir / 'train.json'} and {out_dir / 'val.json'}")


if __name__ == "__main__":
    main()
