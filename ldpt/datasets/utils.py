import numpy as np


def distance2depth(distance, focal):
    height, width = distance.shape[:2]
    plane_x = (
        np.linspace((-0.5 * width) + 0.5, (0.5 * width) - 0.5, width)
        .reshape(1, width)
        .repeat(height, 0)
        .astype(np.float32)[:, :, None]
    )
    plane_y = (
        np.linspace((-0.5 * height) + 0.5, (0.5 * height) - 0.5, height)
        .reshape(height, 1)
        .repeat(width, 1)
        .astype(np.float32)[:, :, None]
    )
    plane_z = np.full([height, width, 1], focal, np.float32)
    plane = np.concatenate([plane_x, plane_y, plane_z], 2)
    depth = distance / np.linalg.norm(plane, ord=2, axis=2) * focal
    return depth
