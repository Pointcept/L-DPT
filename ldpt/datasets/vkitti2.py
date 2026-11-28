import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .default import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class VKITTI2Dataset(DefaultDataset):
    DEPTH_MIN = 0.01
    DEPTH_MAX = 80.0

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth / 100.0

        disparity = np.zeros_like(depth)
        finite = np.isfinite(depth) & (depth > 0) & (depth < 655.35)
        disparity[finite] = 1.0 / np.maximum(depth[finite], self.DEPTH_MIN)

        depth[np.isnan(depth) | (depth > self.DEPTH_MAX) | (depth < self.DEPTH_MIN)] = 0

        data_dict = {
            IMAGE: image,
            DEPTH: depth,
            DISPARITY: disparity,
            DEPTH_MIN: self.DEPTH_MIN,
            DEPTH_MAX: self.DEPTH_MAX,
            NAME: self.get_data_name(idx),
        }
        return data_dict

    def get_data_name(self, idx):
        image_path = Path(self.data_list[idx][IMAGE])
        scene_id = image_path.parts[0]
        condition = image_path.parts[1].replace("-", "_")
        camera_id = image_path.parts[4]
        frame_id = "Frame_" + image_path.stem.split("_")[1]
        return f"VKITTI2/{scene_id}-{condition}-{camera_id}-{frame_id}"

    def info(self):
        info = "Totally {} samples in VKITTI2 / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
