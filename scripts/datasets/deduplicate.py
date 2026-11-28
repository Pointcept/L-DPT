"""
Sequence-level deduplication of a split JSON (the *_dino_95.json splits of the configs).

Every image gets a global feature (default facebook/dinov2-base: the CLS token of the
last layer, L2-normalised). Within a scene (the image path minus its file name), frames
are taken in frame-number order and a frame is kept when its cosine similarity to the
last kept frame is below --threshold (0.95 for the *_dino_95.json splits); the first
frame of a scene is always kept. Kept entries are written unchanged (Hypersim's need
"segment").

    torchrun --nproc_per_node=8 scripts/datasets/deduplicate.py \
        --input_json data/blendedmvs/splits/train.json \
        --output_json data/blendedmvs/splits/train_dino_95.json \
        --data_root data/blendedmvs

Needs CUDA (NCCL) and the transformers package.
"""

import os
import json
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--model", type=str, default="facebook/dinov2-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


class ImageDataset(Dataset):

    def __init__(self, data, root):
        self.data = data
        self.root = root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        path = os.path.join(self.root, item["image"]) if self.root else item["image"]
        return {"img": Image.open(path).convert("RGB"), "meta": item}


def main():
    args = parse_args()

    # 1. Initialize DDP
    dist.init_process_group("nccl")
    rank, local_rank, world_size = (
        dist.get_rank(),
        int(os.environ["LOCAL_RANK"]),
        dist.get_world_size(),
    )
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2. Data Preparation
    with open(args.input_json, "r") as f:
        raw_data = json.load(f)

    # Parse metadata immediately
    dataset_list = []
    for x in raw_data:
        image_path = x["image"]
        depth_path = x["depth"]
        parts = image_path.split("/")
        dataset_list.append(
            {
                "image": image_path,
                "depth": depth_path,
                "entry": x,
                "scene_id": "-".join(parts[:-1]),
                "image_id": int("".join(re.findall(r"\d", parts[-1]))),
            }
        )

    # frames in (scene, frame number) order
    dataset_list.sort(key=lambda x: (x["scene_id"], x["image_id"]))
    print(len(dataset_list))

    ds = ImageDataset(dataset_list, args.data_root)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,  # Custom collate to handle list of dicts
        sampler=DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=False
        ),
    )

    # 3. Model Setup
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    # 4. Inference Loop
    local_results = []

    for batch in tqdm(loader, disable=(rank != 0), desc="Extracting"):
        imgs = [b["img"] for b in batch]
        with torch.no_grad():
            inputs = processor(images=imgs, return_tensors="pt").to(device)

            # Handle different model architectures
            if "siglip" in args.model:
                feats = model.get_image_features(**inputs)
            elif "dinov2" in args.model:
                feats = model(**inputs).last_hidden_state[:, 0, :]
            else:
                # Fallback for CLIP/others
                feats = (
                    model.get_image_features(**inputs)
                    if hasattr(model, "get_image_features")
                    else model(**inputs).pooler_output
                )

            # Normalize features for cosine similarity later
            feats = F.normalize(feats, p=2, dim=1)

        # Store results in CPU memory to save GPU VRAM and prepare for gathering
        for i, feat in enumerate(feats):
            meta = batch[i]["meta"]
            local_results.append({**meta, "feature": feat.cpu()})

    # 5. Gather Results from all GPUs (No Disk I/O)
    # create a list of lists to hold results from all ranks
    gathered_data = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_data, local_results)

    # 6. Process on Rank 0
    threshold = args.threshold

    if rank == 0:
        # Flatten list of lists
        all_data = [item for sublist in gathered_data for item in sublist]
        print(f"\nTotal samples processed: {len(all_data)}")

        # Group by Scene
        scenes = defaultdict(list)
        for item in all_data:
            scenes[item["scene_id"]].append(item)

        print("Calculating Similarities...")
        data = []
        total_samples = 0
        for sid, items in tqdm(scenes.items()):
            # Sort by image_id to ensure temporal order
            items.sort(key=lambda x: x["image_id"])
            items = list({item["image"]: item for item in items}.values())
            assert len(items) >= 2
            total_samples += len(items)

            feats = torch.stack([x["feature"] for x in items])
            keep_idxs = [0]
            last_kept_idx = 0
            for i in range(1, len(feats)):
                sim = F.cosine_similarity(
                    feats[i].unsqueeze(0), feats[last_kept_idx].unsqueeze(0)
                )
                if sim < threshold:
                    keep_idxs.append(i)
                    last_kept_idx = i
            keep_items = [items[x]["entry"] for x in keep_idxs]
            data.extend(keep_items)

        print("Load", total_samples, "samples")
        print("Keep", len(data), "samples")
        json.dump(data, open(args.output_json, "w"), indent=4)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
