import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from .keys import *
from .meta import VOC2012_CLASS_NAMES
from .default import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class VOC2012Dataset(DefaultDataset):
    CLASS_NAMES = VOC2012_CLASS_NAMES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_mapper = self._create_label_mapping()

    def _create_label_mapping(self):
        """
        VOC2012 has 21 classes
        """
        # Initialize mapping with -1 (ignore label)
        mapping = np.full(256, -1, dtype=np.int32)

        # Map valid classes (0 to 20) to themselves
        # 0 is background, 1-20 are object classes
        for i in range(21):
            mapping[i] = i

        # Note: 255 (and any other index > 20) remains -1
        return mapping

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        segment_path = self.data_root / self.data_list[idx][SEGMENT]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        segment = Image.open(segment_path)
        segment = np.array(segment).astype(np.int32)

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
        return f"voc2012/{image_path.stem}"

    def info(self):
        info = "Totally {} samples in VOC2012 / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
