import cv2
import h5py
import numpy as np
from pathlib import Path

from .keys import *
from .meta import HYPERSIM_CLASS_NAMES
from .default import DefaultDataset
from .builder import DATASETS
from .utils import distance2depth


@DATASETS.register_module()
class HypersimDataset(DefaultDataset):
    DEPTH_MIN = 0.01
    DEPTH_MAX = 100.0
    CLASS_NAMES = HYPERSIM_CLASS_NAMES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_mapper = self._create_label_mapping()

    def _create_label_mapping(self):
        # Initialize mapping with -1 (ignore label)
        # range -1, 0, ..., 39, 40 -> total 42
        mapping = np.full(42, -1, dtype=np.int32)
        for i in range(38):
            # 0 ~ 37 -> -1 ~ 36
            mapping[i] = i - 1
        for i in range(38, 41):
            # 38 ~ 40 -> ignore
            mapping[i] = -1
        return mapping

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]
        segment_path = self.data_root / self.data_list[idx][SEGMENT]

        # process image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # process depth & disparity
        distance = np.array(h5py.File(depth_path, "r")["dataset"])
        depth = distance2depth(distance, focal=886.81)

        disparity = np.zeros_like(depth)
        finite = np.isfinite(depth) & (depth > 0)
        disparity[finite] = 1.0 / np.maximum(depth[finite], self.DEPTH_MIN)

        depth[np.isnan(depth) | (depth > self.DEPTH_MAX) | (depth < self.DEPTH_MIN)] = 0

        # process segment
        segment = np.array(h5py.File(segment_path, "r")["dataset"]).astype(np.int32)
        segment = self.label_mapper[segment]

        data_dict = {
            IMAGE: image,
            SEGMENT: segment,
            DISPARITY: disparity,
            DEPTH: depth,
            DEPTH_MIN: self.DEPTH_MIN,
            DEPTH_MAX: self.DEPTH_MAX,
            NAME: self.get_data_name(idx),
        }
        return data_dict

    def get_data_name(self, idx):
        image_path = Path(self.data_list[idx][IMAGE])
        scene_id = image_path.parts[1]
        camera_id = "cam_" + image_path.parts[-2].split("_")[2]
        frame_id = "frame_" + image_path.name.split(".")[1]
        return f"hypersim/{scene_id}-{camera_id}-{frame_id}"

    def info(self):
        info = "Totally {} samples in Hypersim / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
