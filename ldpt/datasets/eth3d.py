import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .default import DefaultDataset, parse_depth_range
from .builder import DATASETS


@DATASETS.register_module()
class ETH3DDataset(DefaultDataset):
    """ETH3D high-resolution multi-view (the DSLR training scenes, which have depth);
    GT = the float32 depth maps, inf = no depth.

    depth_range: None (default): GT kept where DEPTH_MIN <= d <= DEPTH_MAX (0.1 to
    200 m); (min, max): GT kept where min < d < max, max None: no upper bound
    ((1e-5, None) is Marigold's ETH3D mask). Either range is also the evaluator's
    clip of the aligned prediction.
    """

    DEPTH_MIN = 0.1
    DEPTH_MAX = 200.0

    def __init__(self, *args, depth_range=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_range = parse_depth_range(depth_range)

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = np.fromfile(depth_path, dtype=np.float32).reshape(image.shape[:2])
        depth[np.isinf(depth) | ~self.range_valid(depth)] = 0

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
        scene_id = "-".join(image_path.parts[:-1]).lower()
        image_id = image_path.parts[-1].lower()
        return f"eth3d/{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in ETH3D / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
