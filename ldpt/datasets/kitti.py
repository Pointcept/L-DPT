import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .default import DefaultDataset, parse_depth_range
from .builder import DATASETS


@DATASETS.register_module()
class KITTIDataset(DefaultDataset):
    """KITTI (Eigen split, BTS lists; GT = the annotated depth maps / 256).

    - kb_crop: image and GT cropped to the 352 x 1216 benchmark window (bottom,
      centred), before the masks below.
    - garg_crop / eigen_crop: GT masked to the Garg (rows 40.8-99.2 %) / Eigen (rows
      33.2-91.4 %) window, cols 3.6-96.4 % of the (cropped) frame. kb_crop +
      eigen_crop + depth_range=(1e-5, 80) is Marigold's KITTI evaluation mask
      (valid_mask_crop="eigen"); the mixed model's KITTI training labels use
      kb_crop + garg_crop.
    - depth_range: None (default): GT kept where DEPTH_MIN <= d <= DEPTH_MAX (1e-6 to
      80 m, as trained); (min, max): GT kept where min < d < max. Either range is
      also the evaluator's clip of the aligned prediction.
    """

    DEPTH_MIN = 1e-6
    DEPTH_MAX = 80.0

    def __init__(
        self,
        *args,
        kb_crop=False,
        eigen_crop=False,
        garg_crop=False,
        depth_range=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.depth_range = parse_depth_range(depth_range)
        self.kb_crop = kb_crop
        self.eigen_crop = eigen_crop
        self.garg_crop = garg_crop

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth / 256.0
        depth[~self.range_valid(depth)] = 0

        if self.kb_crop:
            # Apply KITTI benchmark crop
            height, width = depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[
                top_margin : top_margin + 352, left_margin : left_margin + 1216, :
            ]
            depth = depth[
                top_margin : top_margin + 352, left_margin : left_margin + 1216
            ]

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
            height, width = depth.shape
            mask = np.zeros_like(depth, dtype=bool)
            mask[
                int(0.3324324 * height) : int(0.91351351 * height),
                int(0.0359477 * width) : int(0.96405229 * width),
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
        split = Path(self.split).stem
        scene_id = image_path.parts[2].lower()
        image_id = image_path.parts[-1].lower()
        return f"kitti/{split}-{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in KITTI / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
