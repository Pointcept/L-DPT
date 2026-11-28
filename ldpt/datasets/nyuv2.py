import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .default import DefaultDataset, parse_depth_range
from .builder import DATASETS


@DATASETS.register_module()
class NYUv2Dataset(DefaultDataset):
    """NYU Depth v2 (BTS split, 480x640).

    Evaluation crops only zero out GT depth (invalid), the image is untouched.
    - eigen_crop=True / "nyu": standard NYU Eigen crop (Eigen et al. 2014),
      rows 45:471, cols 41:601 of the 480x640 frame (77.7% of the pixels); with
      depth_range=(1e-3, 10) it is Marigold's NYUv2 evaluation mask.
    - eigen_crop="legacy": KITTI's Eigen ratios on the NYUv2 frame, rows 159:438,
      cols 23:616 (53.9% of the pixels); the crop of the mixed model's NYUv2
      training labels.
    - garg_crop=True: KITTI's Garg crop ratios.
    - depth_range: None (default): GT kept where DEPTH_MIN <= d <= DEPTH_MAX (1e-6 to
      10 m, as trained); (min, max): GT kept where min < d < max (Marigold's mask;
      the raw GT holds many pixels at exactly 10.0 m, which it drops). Either range
      is also the evaluator's clip of the aligned prediction.
    """

    DEPTH_MIN = 1e-6
    DEPTH_MAX = 10.0
    # (top, bottom, left, right) as ratios of height / width
    EIGEN_CROP = dict(
        nyu=(45 / 480, 471 / 480, 41 / 640, 601 / 640),
        legacy=(0.3324324, 0.91351351, 0.0359477, 0.96405229),
    )

    def __init__(
        self, *args, eigen_crop=False, garg_crop=False, depth_range=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.depth_range = parse_depth_range(depth_range)
        if eigen_crop is True:
            eigen_crop = "nyu"
        assert (
            not eigen_crop or eigen_crop in self.EIGEN_CROP
        ), f"eigen_crop should be False, True or one of {list(self.EIGEN_CROP)}"
        self.eigen_crop = eigen_crop
        self.garg_crop = garg_crop

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth / 1000.0

        depth[~self.range_valid(depth)] = 0

        if self.garg_crop:
            # Apply Garg crop: https://arxiv.org/abs/1603.04992
            height, width = depth.shape
            mask = np.zeros_like(depth, dtype=bool)
            mask[
                int(0.40810811 * height) : int(0.99189189 * height),
                int(0.03594771 * width) : int(0.96405229 * width),
            ] = True
            depth[~mask] = 0

        if self.eigen_crop:
            # Apply Eigen crop: https://arxiv.org/abs/1406.2283
            top, bottom, left, right = self.EIGEN_CROP[self.eigen_crop]
            height, width = depth.shape
            mask = np.zeros_like(depth, dtype=bool)
            mask[
                int(top * height) : int(bottom * height),
                int(left * width) : int(right * width),
            ] = True
            depth[~mask] = 0

        data_dict = {
            IMAGE: image,
            DEPTH: depth,
            DEPTH_MIN: self.range_bounds()[0],
            DEPTH_MAX: self.range_bounds()[1],
            NAME: self.get_data_name(idx),
        }
        return data_dict

    def get_data_name(self, idx):
        image_path = Path(self.data_list[idx][IMAGE])
        scene_id = image_path.parts[2].lower()
        return f"nyuv2/{scene_id}"

    def info(self):
        info = "Totally {} samples in NYUv2 / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
