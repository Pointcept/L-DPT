import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .default import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class TartanAirDataset(DefaultDataset):
    DEPTH_MIN = 0.01
    DEPTH_MAX = 500.0

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = np.load(depth_path)

        disparity = np.zeros_like(depth)
        finite = np.isfinite(depth) & (depth > 0)
        disparity[finite] = 1.0 / np.maximum(depth[finite], self.DEPTH_MIN)

        depth[np.isnan(depth) | (depth > self.DEPTH_MAX) | (depth < self.DEPTH_MIN)] = 0

        data_dict = {
            IMAGE: image,
            DISPARITY: disparity,
            DEPTH: depth,
            DEPTH_MIN: self.DEPTH_MIN,
            DEPTH_MAX: self.DEPTH_MAX,
            NAME: self.get_data_name(idx),
        }
        return data_dict

    def get_data_name(self, idx):
        image_path = Path(self.data_list[idx][IMAGE])
        scene_id = "-".join(image_path.parts[1:4]).lower()
        image_id = image_path.parts[-1].lower()
        return f"tartanair/{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in TartanAir / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
