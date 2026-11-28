"""
Preprocessing & Augmentations

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from ldpt.utils.registry import Registry
from .keys import *

TRANSFORMS = Registry("transforms")


def get_target_keys(data_dict):
    default_target_keys = [IMAGE, DEPTH, DISPARITY, SEGMENT, INSTANCE]
    if TARGET_KEYS in data_dict:
        return data_dict[TARGET_KEYS]
    else:
        return default_target_keys


class TVAugWrapper(object):
    def __call__(self, data_dict):
        # collect keys pointing to tv_tensors
        keys = []
        tensors = []
        target_keys = get_target_keys(data_dict)
        for key in data_dict.keys():
            if key in target_keys and (
                isinstance(data_dict[key], tv_tensors.Image)
                or isinstance(data_dict[key], tv_tensors.Mask)
            ):
                keys.append(key)
                tensors.append(data_dict[key])

        # apply transform
        tensors = self.transform(tensors)

        # update data_dict
        for key, tensor in zip(keys, tensors):
            data_dict[key] = tensor
        return data_dict


class TVImageAugWrapper(object):
    def __call__(self, data_dict):
        # collect keys pointing to tv_tensors
        keys = []
        tensors = []
        target_keys = get_target_keys(data_dict)
        for key in data_dict.keys():
            # only apply to Image tv_tensors
            if key in target_keys and (isinstance(data_dict[key], tv_tensors.Image)):
                keys.append(key)
                tensors.append(data_dict[key])

        # apply transform
        tensors = self.transform(tensors)

        # update data_dict
        for key, tensor in zip(keys, tensors):
            data_dict[key] = tensor
        return data_dict


def depth_edge_free(depth, kernel=5, ltol=0.01):
    """Source pixels that may take part in an interpolation (docs/depth_loss.md 4.6).

    True where the kernel x kernel window is fully valid (depth > 0, inside the image)
    and holds no depth edge. Edge (MoGe's rule, utils3d ``depth_map_edge(kernel_size=5,
    ltol=0.01)``): with q = 1 / depth and L = (q - mean_window q) / q, the window's
    maximum of L exceeds ltol and its minimum is below -ltol. A plane has L = 0.

    Shapes: depth [1, H, W] float -> [1, H, W] bool
    """
    valid = (depth > 0)[None].float()
    r = kernel // 2
    all_valid = 1.0 - F.max_pool2d(
        F.pad(1.0 - valid, (r, r, r, r), value=1.0), kernel, 1
    )
    disp = torch.where(depth > 0, 1.0 / depth.clamp_min(1e-30), 0.0)[None]
    mean = F.avg_pool2d(disp, kernel, 1, r)
    lap = torch.where(all_valid > 0, (disp - mean) / disp.clamp_min(1e-30), 0.0)
    lap_max = F.max_pool2d(lap, kernel, 1, r)
    lap_min = -F.max_pool2d(-lap, kernel, 1, r)
    edge = (lap_max > ltol) & (lap_min < -ltol)
    return ((all_valid > 0) & ~edge)[0]


def resize_depth_edge_aware(depth, size, kernel=5, ltol=0.01):
    """Resize a depth map without staircases (docs/depth_loss.md 4.6; MoGe's rule,
    moge/train/dataloader.py#L325-L330).

    Bilinear in inverse depth where all four source neighbours are edge free
    (depth_edge_free), else the nearest source pixel (pixel centres aligned,
    "nearest-exact"); invalid (0 or negative) stays invalid as 0. align_corners=False,
    no antialiasing. A sparse map (LiDAR: no fully valid 5 x 5 window) is resized by
    nearest-exact everywhere.

    Shapes: depth [1, H, W] float, size (h, w) -> [1, h, w]
    """
    size = [int(size[0]), int(size[1])]
    depth = torch.as_tensor(depth)
    ok = depth_edge_free(depth, kernel, ltol)
    disp = torch.where(depth > 0, 1.0 / depth.clamp_min(1e-30), 0.0)
    w_ok = F.interpolate(
        ok[None].to(depth.dtype), size=size, mode="bilinear", align_corners=False
    )[0]
    d_bil = F.interpolate(disp[None], size=size, mode="bilinear", align_corners=False)[
        0
    ]
    near = F.interpolate(depth.clamp_min(0.0)[None], size=size, mode="nearest-exact")[0]
    use = w_ok >= 1.0 - 1e-6
    return torch.where(use, 1.0 / d_bil.clamp_min(1e-30), near)


DEPTH_RESAMPLES = ("nearest", "edge_aware")


@TRANSFORMS.register_module()
class Resize(TVAugWrapper):
    """torchvision Resize of the target Images and Masks.

    depth_resample: "nearest" (default: the depth Mask goes through torchvision,
    which resizes a Mask by nearest) or "edge_aware" (the depth recipes of the
    released models, docs/depth_loss.md 4.6: the ``depth`` Mask is resized by
    resize_depth_edge_aware to the image's new size; every other Mask as with
    "nearest").
    """

    def __init__(self, depth_resample="nearest", **kwargs):
        if depth_resample not in DEPTH_RESAMPLES:
            raise ValueError(
                f"depth_resample must be one of {list(DEPTH_RESAMPLES)}, "
                f"got {depth_resample!r}"
            )
        self.depth_resample = depth_resample

        interpolation_wrapper = dict(
            nearest=T.InterpolationMode.NEAREST,
            nearest_exact=T.InterpolationMode.NEAREST_EXACT,
            bilinear=T.InterpolationMode.BILINEAR,
            bicubic=T.InterpolationMode.BICUBIC,
            box=T.InterpolationMode.BOX,
            hamming=T.InterpolationMode.HAMMING,
            lanczos=T.InterpolationMode.LANCZOS,
        )

        if "interpolation" in kwargs:
            kwargs["interpolation"] = interpolation_wrapper[kwargs["interpolation"]]
        self.transform = T.Resize(**kwargs)

    def __call__(self, data_dict):
        target_keys = get_target_keys(data_dict)
        depth = data_dict.get(DEPTH)
        if (
            self.depth_resample == "nearest"
            or DEPTH not in target_keys
            or not isinstance(depth, tv_tensors.Mask)
        ):
            return super().__call__(data_dict)
        # every target but the depth Mask through torchvision, as TVAugWrapper does
        keys = [
            k
            for k, v in data_dict.items()
            if k in target_keys
            and k != DEPTH
            and isinstance(v, (tv_tensors.Image, tv_tensors.Mask))
        ]
        tensors = self.transform([data_dict[k] for k in keys]) if keys else []
        for key, tensor in zip(keys, tensors):
            data_dict[key] = tensor
        # the depth Mask to the image's new size
        images = [t for t in tensors if isinstance(t, tv_tensors.Image)]
        if images:
            size = tuple(images[0].shape[-2:])
        else:  # no image resized: the size torchvision gives the depth map itself
            size = tuple(self.transform(depth).shape[-2:])
        data_dict[DEPTH] = tv_tensors.Mask(resize_depth_edge_aware(depth, size))
        return data_dict


@TRANSFORMS.register_module()
class RandomCrop(TVAugWrapper):
    def __init__(self, **kwargs):
        if "fill" not in kwargs:
            kwargs["fill"] = {tv_tensors.Image: 0, tv_tensors.Mask: -1}
        else:
            map = {
                "tv_tensors.Image": tv_tensors.Image,
                "tv_tensors.Mask": tv_tensors.Mask,
            }
            fill = {}
            for k, v in kwargs["fill"].items():
                fill[map[k]] = v
            kwargs["fill"] = fill
        self.transform = T.RandomCrop(**kwargs)


@TRANSFORMS.register_module()
class GaussianBlur(TVImageAugWrapper):
    def __init__(self, p=0.5, **kwargs):
        self.transform = T.GaussianBlur(**kwargs)
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            return super().__call__(data_dict)
        else:
            return data_dict


@TRANSFORMS.register_module()
class GaussianNoise(TVImageAugWrapper):
    def __init__(self, p=0.5, **kwargs):
        self.transform = T.GaussianNoise(**kwargs)
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            return super().__call__(data_dict)
        else:
            return data_dict


@TRANSFORMS.register_module()
class ColorJitter(TVImageAugWrapper):
    def __init__(self, p=0.5, **kwargs):
        self.transform = T.ColorJitter(**kwargs)
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            return super().__call__(data_dict)
        else:
            return data_dict


@TRANSFORMS.register_module()
class AutoContrast(TVImageAugWrapper):
    def __init__(self, p=0.5):
        self.transform = T.RandomAutocontrast(p=1.0)
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            return super().__call__(data_dict)
        else:
            return data_dict


@TRANSFORMS.register_module()
class AutoEqualise(TVImageAugWrapper):
    def __init__(self, p=0.5):
        self.transform = T.RandomEqualize(p=1.0)
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            return super().__call__(data_dict)
        else:
            return data_dict


@TRANSFORMS.register_module()
class NormalizeImage(TVImageAugWrapper):
    def __init__(self, **kwargs):
        self.transform = T.Normalize(**kwargs)


@TRANSFORMS.register_module()
class RandomChoice(object):
    def __init__(self, transforms, p=None):
        self.transforms = Compose(transforms)
        self.p = p

        if p is not None:
            assert len(p) == len(transforms)
            assert abs(sum(p) - 1.0) < 1e-6

    def __call__(self, data_dict):
        if self.p is None:
            idx = np.random.randint(len(self.transforms.transforms))
        else:
            idx = np.random.choice(len(self.transforms.transforms), p=self.p)

        transform = self.transforms.transforms[idx]
        return transform(data_dict)


@TRANSFORMS.register_module()
class MultiAugment(object):
    """Apply multiple augmentations to the same data and stack results.

    This transform is designed for Test-Time Augmentation (TTA). It creates
    multiple augmented versions of the input data by applying each transform
    in the list. The results are stacked along a new first dimension.

    For example, if input image has shape [H, W, C], and we have N transforms,
    the output will have shape [N, H, W, C]. After collate_fn with batch_size B,
    the final batch shape will be [B, N, H, W, C], which can be reshaped to
    [B*N, H, W, C] for model inference and then reshaped back for ensemble.

    Args:
        transforms (list): List of transform configs to apply sequentially.
            Each transform will be applied to a deep copy of the input data.

    Example:
        >>> # Config example for TTA with horizontal flip and different scales
        >>> dict(
        ...     type="MultiAugment",
        ...     transforms=[
        ...         [],  # original, no augmentation
        ...         [dict(type="RandomFlip", p=1.0, direction="horizontal")],
        ...         [dict(type="Resize", size=(512, 512))],
        ...     ]
        ... )
    """

    def __init__(self, transforms):
        # Each element in transforms is a list of transform configs
        self.transforms_list = [Compose(t) for t in transforms]
        self.num_augments = len(self.transforms_list)

    def __call__(self, data_dict):
        augmented_dicts = []
        for transform_compose in self.transforms_list:
            # Deep copy the data dict to avoid modifying the original
            data_dict_copy = copy.deepcopy(data_dict)
            # Apply transforms
            augmented_dict = transform_compose(data_dict_copy)
            augmented_dicts.append(augmented_dict)

        # Stack all augmented versions along a new first dimension
        result_dict = {}

        # Get keys from the first augmented dict (all should have same keys)
        keys = augmented_dicts[0].keys()

        for key in keys:
            values = [aug_dict[key] for aug_dict in augmented_dicts]

            # Stack tensors along a new first dimension
            if isinstance(values[0], torch.Tensor):
                # Stack tensors: [H, W, C] * N -> [N, H, W, C]
                result_dict[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], (int, float, np.number)):
                # For scalar values, stack into a tensor
                result_dict[key] = torch.tensor(values)
            elif isinstance(values[0], str):
                # For strings, keep as list (e.g., sample names)
                result_dict[key] = values
            elif isinstance(values[0], list):
                # For lists, keep as nested list
                result_dict[key] = values
            else:
                # For other types, keep the first value or store as list
                # This handles cases like metadata that should be the same across augments
                if all(v == values[0] for v in values):
                    result_dict[key] = values[0]
                else:
                    result_dict[key] = values

        return result_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data_dict):
        for key in data_dict.keys():
            if key == IMAGE:
                data_dict[key] = tv_tensors.Image(
                    torch.from_numpy(
                        np.ascontiguousarray(np.transpose(data_dict[key], (2, 0, 1)))
                    )
                ).to(torch.float32)
            elif key == DEPTH:
                data_dict[key] = tv_tensors.Mask(
                    torch.from_numpy(
                        np.ascontiguousarray(data_dict[key][np.newaxis, ...])
                    )
                ).to(torch.float32)
            elif key == DISPARITY:
                data_dict[key] = tv_tensors.Mask(
                    torch.from_numpy(
                        np.ascontiguousarray(data_dict[key][np.newaxis, ...])
                    )
                ).to(torch.float32)
            elif key == SEGMENT:
                data_dict[key] = tv_tensors.Mask(
                    torch.from_numpy(
                        np.ascontiguousarray(data_dict[key][np.newaxis, ...])
                    )
                ).to(torch.int64)
            elif isinstance(data_dict[key], float):
                data_dict[key] = torch.tensor(data_dict[key], dtype=torch.float32)
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(TVAugWrapper):
    def __init__(self, p=0.5, direction="horizontal"):
        if direction == "horizontal":
            self.transform = T.RandomHorizontalFlip(p=p)
        elif direction == "vertical":
            self.transform = T.RandomVerticalFlip(p=p)
        else:
            raise ValueError(f"Invalid flip direction: {direction}")


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_image", depth="origin_depth")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, data_dict):
        for key, value in self.kwargs.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        return data


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
