import cv2
import numpy as np
from pathlib import Path

from .keys import *
from .meta import COCOSTUFF_CLASS_NAMES
from .default import DefaultDataset
from .builder import DATASETS

# PNG values of the official 0-based stuffthingmaps (value = labels.txt id - 1): 80
# things in 0..89 (with gaps) and 91 stuff classes in 91..181; 255 is unlabeled.
# fmt: off
THING_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 83, 84, 85, 86, 87, 88, 89,
]
# label_mapping="legacy": things read as 1-based (person, value 0, ignored; value
# 91 = banner, the first stuff class, listed as the last thing) and stuff 92..182
# (value 182 never occurs, so train id 170 is empty)
LEGACY_THING_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 66, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 83, 84, 85, 86, 87, 88, 89, 91,
]
# fmt: on
LABEL_IDS = dict(
    official=THING_IDS + list(range(91, 182)),
    legacy=LEGACY_THING_IDS + list(range(92, 183)),
)


@DATASETS.register_module()
class COCOStuffDataset(DefaultDataset):
    # train-id order of label_mapping="official" (person first, stuff in PNG order);
    # under "legacy" train id i holds the class after CLASS_NAMES[i] (a "legacy" run
    # logs it under CLASS_NAMES[i])
    CLASS_NAMES = COCOSTUFF_CLASS_NAMES

    def __init__(self, *args, label_mapping="official", **kwargs):
        """COCO-Stuff 164k on the official stuffthingmaps PNGs.

        Args:
            label_mapping: "official" (default): the standard 171 train ids
                (mmseg clsID_to_trID: things 0..89 -> 0..79, stuff 91..181 ->
                80..170, all else ignored). "legacy": the thing ids read as 1-based
                (person ignored, ids shifted by one, train id 170 empty); only for
                checkpoints that were trained with it.
        """
        self.label_mapper = self.create_label_mapping(label_mapping)
        self.label_mapping = label_mapping
        super().__init__(*args, **kwargs)

    @staticmethod
    def create_label_mapping(label_mapping="official"):
        """PNG value (0..255) -> train id (-1: ignored)."""
        if label_mapping not in LABEL_IDS:
            raise ValueError(
                f"label_mapping must be one of {list(LABEL_IDS)}, got {label_mapping!r}"
            )
        mapping = np.full(256, -1, dtype=np.int32)
        for train_id, raw_id in enumerate(LABEL_IDS[label_mapping]):
            mapping[raw_id] = train_id
        return mapping

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        segment_path = self.data_root / self.data_list[idx][SEGMENT]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        segment = cv2.imread(segment_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        if segment.ndim == 3:
            segment = segment[:, :, 0]
        segment = self.label_mapper[segment]

        data_dict = {
            IMAGE: image,
            SEGMENT: segment,
            NAME: self.get_data_name(idx),
        }
        return data_dict

    def get_data_name(self, idx):
        image_path = Path(self.data_list[idx][IMAGE])
        return f"cocostuff/{image_path.stem}"

    def info(self):
        info = "Totally {} samples in COCO-Stuff / {} set (label_mapping {}).".format(
            self.__len__(), str(self.split), self.label_mapping
        )
        return info
