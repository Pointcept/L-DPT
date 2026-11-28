"""
Default Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import json
import math
import numpy as np
from pathlib import Path
from typing import Sequence
from torch.utils.data import Dataset

from .builder import DATASETS, build_dataset
from .transforms import Compose


def parse_depth_range(depth_range):
    """A dataset's depth_range option -> None or (min, max) floats in metres.

    None (also the strings "None" / "null" of --options): the class's DEPTH_MIN /
    DEPTH_MAX. (min, max) with 0 <= min < max; max None (or inf): no upper bound.
    """
    if depth_range is None or (
        isinstance(depth_range, str) and depth_range.lower() in ("none", "null")
    ):
        return None
    if isinstance(depth_range, str) or len(depth_range) != 2:
        raise ValueError(f"depth_range must be None or (min, max), got {depth_range!r}")
    lo, hi = depth_range
    lo = float(lo)
    hi = math.inf if hi is None or str(hi).lower() in ("none", "null") else float(hi)
    if not (0.0 <= lo < hi):
        raise ValueError(f"depth_range needs 0 <= min < max, got {depth_range!r}")
    return lo, hi


@DATASETS.register_module()
class DefaultDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        transform=None,
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = Path(data_root)
        self.split = split

        self.data_list = self.get_data_list()
        self.transform = Compose(transform)

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        elif isinstance(self.split, Sequence):
            split_list = self.split
        else:
            raise NotImplementedError(
                "Data split should be a string or a sequence of strings."
            )

        data_list = []
        for split in split_list:
            split_path = self.data_root / split
            if split_path.is_file():
                assert split.endswith(".json")
                with open(split_path) as f:
                    data_list += json.load(f)
            elif split_path.is_dir():
                data_list += glob.glob(str(split_path / "*"))
            else:
                # e.g. a *_dino_95.json split not generated yet: fail instead of
                # silently training without this dataset
                raise FileNotFoundError(f"split {split_path} not found")
        return data_list

    def get_data(self, idx):
        pass

    def get_data_name(self, idx):
        pass

    def crop_protocol(self):
        """Evaluation crop applied to GT depth, e.g. "kb+garg", "eigen-nyu"."""
        crops = []
        for flag in ("kb_crop", "garg_crop", "eigen_crop"):
            value = getattr(self, flag, False)
            if value:
                name = flag[: -len("_crop")]
                crops.append(name if value is True else f"{name}-{value}")
        return "+".join(crops) if crops else "none"

    def range_bounds(self):
        """(min, max) of the GT range and of the evaluator's clip of the aligned
        prediction (keys depth_min / depth_max): depth_range, else the class's
        DEPTH_MIN / DEPTH_MAX; max inf: no upper bound."""
        depth_range = getattr(self, "depth_range", None)
        if depth_range is not None:
            return depth_range
        return self.DEPTH_MIN, self.DEPTH_MAX

    def range_valid(self, depth):
        """Bool mask of the GT depth inside the range; NaN is never inside.

        depth_range None: DEPTH_MIN <= d <= DEPTH_MAX (the class range, as in
        training). (min, max): min < d < max, Marigold's evaluation mask
        (BaseDepthDataset._get_valid_mask); max inf: d < inf.
        """
        lo, hi = self.range_bounds()
        if getattr(self, "depth_range", None) is None:
            return (depth >= lo) & (depth <= hi)
        return (depth > lo) & (depth < hi)

    def range_protocol(self):
        """GT range for logging: "[a,b]" (closed, the class range), "(a,b)" (open,
        depth_range), "mask,clip[a,b]" when GT validity is the dataset's mask alone
        (DIODE mask_only) and the range only clips the aligned prediction."""
        lo, hi = self.range_bounds()
        if getattr(self, "mask_only", False):
            return f"mask,clip[{lo:g},{hi:g}]"
        if getattr(self, "depth_range", None) is None:
            return f"[{lo:g},{hi:g}]"
        return f"({lo:g},{hi:g})"

    def info(self):
        info = "Totally {} samples in {} / {} set.".format(
            self.__len__(), self.data_root.name, str(self.split)
        )
        return info

    def __getitem__(self, idx):
        data_dict = self.transform(self.get_data(idx))
        return data_dict

    def __len__(self):
        return len(self.data_list)


@DATASETS.register_module()
class ConcatDataset(Dataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.datasets = [build_dataset(dataset) for dataset in datasets]
        self.data_list = self.get_data_list()

    def get_data_list(self):
        data_list = []
        for i in range(len(self.datasets)):
            data_list.extend(
                zip(
                    np.ones(len(self.datasets[i]), dtype=int) * i,
                    np.arange(len(self.datasets[i])),
                )
            )
        return data_list

    def get_data(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx][data_idx]

    def get_data_name(self, idx):
        dataset_idx, data_idx = self.data_list[idx % len(self.data_list)]
        return self.datasets[dataset_idx].get_data_name(data_idx)

    def info(self):
        info = "Totally {} samples from {} datasets, including:".format(
            self.__len__(), len(self.datasets)
        )
        for i in range(len(self.datasets)):
            info += "\n\t|- " + self.datasets[i].info()
        return info

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list)
