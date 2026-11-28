import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .default import DefaultDataset, parse_depth_range
from .builder import DATASETS


@DATASETS.register_module()
class DIODEDataset(DefaultDataset):
    """DIODE (768 x 1024; GT = the depth map where DIODE's depth mask is set).

    - depth_range: None (default): GT kept where the mask is set and DEPTH_MIN <= d
      <= DEPTH_MAX (the class range: indoor 10 m, outdoor 80 m, as trained);
      (min, max): GT kept where min < d < max. Either range is also the evaluator's
      clip of the aligned prediction.
    - mask_only: GT validity is DIODE's depth mask alone (and d > 0, which every
      masked pixel has); the range only clips the prediction. With
      depth_range=(0.6, 350) and no crop this is Marigold's DIODE protocol
      (the sensor's range; masked GT below 0.6 m is scored).
    - eigen_crop / garg_crop: GT masked to the NYU Eigen window as ratios / the Garg
      window (the mixed model's DIODE training labels: indoor Eigen, outdoor Garg).
    """

    DEPTH_MIN = 1e-6
    DEPTH_MAX = 1000.0

    def __init__(
        self,
        *args,
        eigen_crop=False,
        garg_crop=False,
        depth_range=None,
        mask_only=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.eigen_crop = eigen_crop
        self.garg_crop = garg_crop
        self.depth_range = parse_depth_range(depth_range)
        self.mask_only = bool(mask_only)

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]
        mask_path = str(depth_path).replace("_depth.npy", "_depth_mask.npy")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = np.load(depth_path).squeeze()
        depth_mask = np.load(mask_path).astype(bool)
        if self.mask_only:
            valid = depth_mask & np.isfinite(depth) & (depth > 0)
        else:
            valid = depth_mask & self.range_valid(depth)
        depth[~valid] = 0

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
            # original only supports 640x480 images
            # convert to ratio
            height, width = depth.shape
            mask = np.zeros_like(depth, dtype=bool)
            mask[
                int(0.09375 * height) : int(0.98125 * height),
                int(0.0640625 * width) : int(0.9390625 * width),
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
        scene_id = "-".join(image_path.parts[1:4]).lower()
        image_id = image_path.parts[-1].lower()
        return f"DIODE/{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in DIODE / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info


@DATASETS.register_module()
class DIODEIndoorDataset(DIODEDataset):
    DEPTH_MIN = 1e-6
    DEPTH_MAX = 10.0

    def __init__(self, *args, unified=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.unified = unified

    def get_data_name(self, idx):
        image_path = Path(self.data_list[idx][IMAGE])
        scene_id = "-".join(image_path.parts[1:4]).lower()
        image_id = image_path.parts[-1].lower()
        dataset_name = "DIODE" if self.unified else "DIODE-INDOOR"
        return f"{dataset_name}/{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in {} / {} set.".format(
            self.__len__(),
            "DIODE" if self.unified else "DIODE-INDOOR",
            str(self.split),
        )
        return info


@DATASETS.register_module()
class DIODEOutdoorDataset(DIODEDataset):
    DEPTH_MIN = 1e-6
    DEPTH_MAX = 80.0

    def __init__(self, *args, unified=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.unified = unified

    def get_data_name(self, idx):
        image_path = Path(self.data_list[idx][IMAGE])
        scene_id = "-".join(image_path.parts[1:4]).lower()
        image_id = image_path.parts[-1].lower()
        dataset_name = "DIODE" if self.unified else "DIODE-OUTDOOR"
        return f"{dataset_name}/{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in {} / {} set.".format(
            self.__len__(),
            "DIODE" if self.unified else "DIODE-OUTDOOR",
            str(self.split),
        )
        return info
