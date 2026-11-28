import cv2
import numpy as np
from pathlib import Path
import OpenEXR
import Imath

from .keys import *
from .default import DefaultDataset
from .builder import DATASETS


def exr2hdr(exrpath):
    File = OpenEXR.InputFile(str(exrpath))
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()["dataWindow"]
    CNum = len(File.header()["channels"].keys())
    if CNum > 1:
        Channels = ["R", "G", "B"]
        CNum = 3
    else:
        Channels = ["G"]
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [
        np.frombuffer(File.channel(c, PixType), dtype=np.float32) for c in Channels
    ]
    if CNum == 1:
        hdr = np.zeros((Size[1], Size[0]), dtype=np.float32)
        hdr = np.reshape(Pixels[0], (Size[1], Size[0]))
    else:
        hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
        hdr[:, :, 1] = np.reshape(Pixels[1], (Size[1], Size[0]))
        hdr[:, :, 2] = np.reshape(Pixels[2], (Size[1], Size[0]))
    hdr = np.maximum(hdr, 1e-6)
    File.close()
    return hdr


@DATASETS.register_module()
class IRSDataset(DefaultDataset):
    DEPTH_MIN = 0.01
    DEPTH_MAX = 20.0

    def get_data(self, idx):
        image_path = self.data_root / self.data_list[idx][IMAGE]
        depth_path = self.data_root / self.data_list[idx][DEPTH]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        disparity = exr2hdr(depth_path) / 48
        depth = 1.0 / disparity
        # convert disparity to depth wirh baseline = 0.1 and focal length = 480
        depth[
            np.isnan(depth)
            | np.isinf(depth)
            | (depth > self.DEPTH_MAX)
            | (depth < self.DEPTH_MIN)
        ] = 0

        disparity = np.clip(disparity, max=1.0 / self.DEPTH_MIN)

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
        scene_id = "-".join(image_path.parts[1:-1]).lower()
        image_id = image_path.parts[-1].lower()
        return f"irs/{scene_id}-{image_id}"

    def info(self):
        info = "Totally {} samples in IRS / {} set.".format(
            self.__len__(), str(self.split)
        )
        return info
