import os
import wandb
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # figures are only saved; no display on training nodes
import matplotlib.pyplot as plt
import torch.nn.functional as F

from .default import HookBase
from .builder import HOOKS
from ldpt.models.utils.depth import compute_scale_and_shift
from ldpt.utils.device import get_device
from ldpt.utils.comm import is_main_process


@HOOKS.register_module()
class DepthVisualizer(HookBase):
    def __init__(
        self,
        data_idx=0,
        mean=None,
        std=None,
        relative=False,
        mask_invalid=True,
    ):
        super().__init__()
        self.data_idx = data_idx
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        self.relative = relative
        self.mask_invalid = mask_invalid
        self.dpi = 100

    def before_train(self):
        if is_main_process():
            os.makedirs(os.path.join(self.trainer.cfg.save_path, "viz"), exist_ok=True)

    def after_epoch(self):
        if not is_main_process():
            return
        # load data
        data_dict = self.trainer.val_loader.dataset[self.data_idx]
        name = data_dict["name"]
        image = np.ascontiguousarray(
            np.transpose(data_dict["image"].detach().cpu().numpy(), (1, 2, 0))
        )
        image = image * self.std + self.mean

        # inference
        data_dict = self.trainer.val_loader.collate_fn([data_dict])
        device = get_device()
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor):
                data_dict[key] = data_dict[key].to(
                    device, non_blocking=(device.type == "cuda")
                )

        self.trainer.model.eval()
        gt_depth = data_dict.pop("depth")
        with torch.inference_mode():
            pred_depth = self.trainer.model(data_dict)["pred_depth"]
        if pred_depth.shape != gt_depth.shape:
            pred_depth = F.interpolate(
                pred_depth,
                size=gt_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if len(pred_depth.shape) == 4:
            pred_depth = pred_depth.squeeze(1)
        if len(gt_depth.shape) == 4:
            gt_depth = gt_depth.squeeze(1)

        valid_mask = gt_depth > 0
        if self.relative:
            scale, shift, _ = compute_scale_and_shift(
                pred_depth,
                gt_depth,
                valid_mask,
            )
            pred_depth = pred_depth * scale[0] + shift[0]  # scale and shift to GT
        pred_depth = pred_depth.clamp(min=1e-6)
        if "depth_min" in data_dict:
            pred_depth = pred_depth.clamp(min=data_dict["depth_min"].view(-1, 1, 1))
        if "depth_max" in data_dict:
            pred_depth = pred_depth.clamp(max=data_dict["depth_max"].view(-1, 1, 1))
        if self.mask_invalid:
            pred_depth[~valid_mask] = torch.nan  # mask invalid pixels
            gt_depth[~valid_mask] = torch.nan  # mask invalid pixels
        pred_depth = pred_depth.squeeze().cpu().numpy()
        gt_depth = gt_depth.squeeze().cpu().numpy()

        # visualization
        h, w = image.shape[:2]
        cmap = plt.colormaps.get_cmap("Spectral_r")
        depth_max = np.nanmax(gt_depth)

        fig, axes = plt.subplots(
            1, 3, figsize=(w / self.dpi * 3, h / self.dpi + 0.1), dpi=self.dpi
        )

        axes[0].imshow(image)
        axes[0].set_title(name)
        axes[0].axis("off")

        axes[1].imshow(cmap(pred_depth / depth_max)[:, :, :3])
        axes[1].set_title("Predicted Depth")
        axes[1].axis("off")

        axes[2].imshow(cmap(gt_depth / depth_max)[:, :, :3])
        axes[2].set_title("GT Depth")
        axes[2].axis("off")

        # write to logger
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_figure(
                "val_viz/metric_depth", fig, current_epoch, close=False
            )
            if self.trainer.cfg.enable_wandb:
                wandb.log(
                    {
                        "Epoch": current_epoch,
                        "val_viz/rgb": wandb.Image(fig),
                    },
                    step=wandb.run.step,
                )

        fig.savefig(
            os.path.join(self.trainer.cfg.save_path, "viz", f"{current_epoch}.png")
        )
        plt.close(fig)
