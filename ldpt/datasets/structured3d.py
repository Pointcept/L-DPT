import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .default import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class Structured3DDataset(DefaultDataset):
    DEPTH_MIN = 1e-6
    DEPTH_MAX = 65.0  # 65536 / 1000 = 65.536 which should be ignored

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth / 1000.0

        disparity = np.zeros_like(depth)
        finite = np.isfinite(depth) & (depth > 0) & (depth < 655.35)
        disparity[finite] = 1.0 / np.maximum(depth[finite], self.DEPTH_MIN)

        depth[
            np.isnan(depth)
            | np.isinf(depth)
            | (depth > self.DEPTH_MAX)
            | (depth < self.DEPTH_MIN)
        ] = 0

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
        scene_id = "-".join([image_path.parts[1], image_path.parts[3]]).lower()
        image_id = image_path.parts[-2].lower()
        return f"structure3d/{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in Structured3D / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
