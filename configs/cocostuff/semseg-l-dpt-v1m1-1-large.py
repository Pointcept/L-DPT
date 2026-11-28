# L-DPT Large, COCO-Stuff semantic segmentation (171 classes), recipe 1: the latent
# decoder with cosine (QK-norm) attention, a BatchNorm on each tapped encoder layer and
# 64-channel pixel features, trained with MuonAdamW. Stage 1 of the COCO-Stuff model;
# stage 2 is semseg-l-dpt-v1m1-1a-large-anneal.py, a one-epoch anneal from this run's
# epoch 9 with seeds 0, 1 and 2, averaged with its start (docs/training.md).
#   ./train.sh -g 8 -d cocostuff -c semseg-l-dpt-v1m1-1-large \
#       -n semseg-l-dpt-v1m1-1-large
# Model: frozen DINOv3 ViT-7B/16 (stored in bf16), taps [9, 19, 29, 39], each followed
# by a BatchNorm (tap_bn, synchronised over the GPUs); latent decoder 2048 / 16 heads /
# (2, 2, 2, 2) blocks, attention at dim 1024 (dec_attn_downscale 2) with QK-norm
# (dec_qk_norm: a learned temperature per head, capped at 200), bf16; 64-channel pixel
# features and a 1x1 classifier; cross-entropy + Lovasz loss.
# 25 epochs at batch 32. MuonAdamW (lr 5e-4, weight decay 0.1): orthogonalised Nesterov
# momentum on the decoder's hidden 2-D weights (the attention, MLP and selection
# matrices, scaled to the RMS of an AdamW step), AdamW on the rest; no weight decay on
# norms, tap BatchNorms, biases and LayerScale; OneCycleLR (5 % warm-up, cosine).
# Stage 2 reads model/epoch_9.pth: the schedule stays 25 epochs because OneCycleLR
# depends on the total number of steps, and the run may be stopped once epoch_9.pth is
# written. Evaluation: short side 768, whole image, single scale, mIoU over the 171
# classes. This file resolves key by key to the configuration of the run it reproduces
# (tests/test_configs.py pins it); only launch settings are left out.
# Speed (not part of the recipe): -o "compile_blocks=True
#   sdpa_backends=[cudnn,flash,efficient,math]".
_base_ = ["../_base_/default_runtime.py"]

encoder_scale = "vit7b"
checkpoint = "checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"

seed = 0
num_workers = 32  # total over all GPUs
batch_size = 32  # total over all GPUs
sync_bn = True
clip_grad = 1.0


patch_size = 16
log_interval = 10

num_classes = 171
ignore_index = -1

model = dict(
    type="L-DPT-v1m1",
    task="semseg",
    encoder=dict(type="dinov3_vit7b"),
    encoder_inter_layers=[9, 19, 29, 39],
    freeze_encoder=True,
    encoder_weight_dtype="bf16",  # frozen encoder stored in bf16 (bit-exact)
    dec_dim=2048,
    out_dim=64,
    dec_num_heads=16,
    dec_num_layers=(2, 2, 2, 2),
    dec_attn_downscale=2,
    dec_ffn_ratio=4.0,
    dec_pos_embed_rope_base=100,
    dec_pos_embed_rope_normalize_coords="separate",
    dec_pos_embed_rope_shift_coords=None,
    dec_pos_embed_rope_jitter_coords=None,
    dec_pos_embed_rope_rescale_coords=2,
    dec_layer_scale_init=1.0e-05,
    dec_qkv_bias=False,
    dec_proj_bias=True,
    dec_ffn_bias=True,
    readout_kernel_size=3,
    precision="bf16",  # decoder under bf16 autocast; heads and loss in fp32
    attn_dtype="fp32",
    dec_qk_norm=True,  # cosine attention, learned temperature per head
    dec_qk_tau_max=200.0,  # the temperature cap
    tap_bn=True,  # a BatchNorm on each tapped encoder layer (SyncBatchNorm: sync_bn)
    num_classes=num_classes,
    criterion=[
        dict(type="ce_loss", loss_weight=1.0, ignore_index=ignore_index),
        dict(
            type="lovasz_loss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=ignore_index,
        ),
    ],
)

epoch = 25

base_lr = 5e-4
base_wd = 0.1

# group 1: the frozen encoder (none of its parameters trains); group 2: no weight decay
# (first match wins, so the encoder's own norms and biases stay in group 1). MuonAdamW
# splits every group into an AdamW part and a matrix part (2-D weights).
param_dicts = [
    dict(keyword="encoder", lr=base_lr / 10),
    dict(keyword=["norm", "tap_bn", "bias", "gamma"], weight_decay=0.0),
]

optimizer = dict(type="MuonAdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr, base_lr / 10, base_lr],  # peaks of group 0 (the rest), 1, 2
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

image_size = 768
scale_list = (
    [i / 20 + 0.5 for i in range(10)] + [1] + [i / 10 + 1 for i in range(1, 11)]
)  # 0.5 ~ 1.0: 10, 1.0 ~ 2.0: 10, 1.0: 1
train_transform = [
    dict(type="ToTensor"),
    dict(
        type="ColorJitter", p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    dict(type="GaussianBlur", p=0.3, kernel_size=(5, 5), sigma=(0.1, 2.0)),
    dict(type="GaussianNoise", p=0.3, mean=0.0, sigma=0.1),
    dict(type="AutoContrast", p=0.2),
    dict(type="AutoEqualise", p=0.2),
    dict(
        type="NormalizeImage",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    dict(
        type="RandomChoice",
        transforms=[
            dict(
                type="Resize",
                size=int(scale * image_size),
                interpolation="bicubic",
            )
            for scale in scale_list
        ],
    ),
    dict(
        type="RandomCrop",
        size=(image_size, image_size),
        pad_if_needed=True,
        padding_mode="constant",
    ),
    dict(type="RandomFlip", p=0.5, direction="horizontal"),
    dict(type="Collect", keys=["image", "segment", "name"]),
]

val_transform = [
    dict(type="ToTensor"),
    dict(
        type="Update",
        target_keys=["image"],  # only process image, keep segment unchanged
    ),
    dict(
        type="Resize",
        size=image_size,
        interpolation="bicubic",
    ),
    dict(
        type="NormalizeImage",
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    dict(type="Collect", keys=["image", "segment", "name"]),
]

# label_mapping="official": the standard 171 train ids of the official 0-based
# stuffthingmaps ("legacy": another mapping, see COCOStuffDataset)
data = dict(
    train=dict(
        type="COCOStuffDataset",
        data_root="data/cocostuff",
        split="splits/train.json",
        label_mapping="official",
        transform=train_transform,
    ),
    val=dict(
        type="COCOStuffDataset",
        data_root="data/cocostuff",
        split="splits/val.json",
        label_mapping="official",
        transform=val_transform,
    ),
)

# hook
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.",
        replacement="module.encoder.",
    ),  # the DINOv3 file into model.encoder; decoder and head start from their init
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=10),
    dict(
        type="SemSegEvaluator",
        num_classes=num_classes,
        ignore_index=ignore_index,
        # tta=[["flip"]],  # average with the flipped image; off in the recipe
        # best_metric="miou",  # default
    ),
    dict(
        type="CheckpointSaver",
        save_freq=None,
        skip_keywords=["encoder"],
        keep_epoch_weights=True,  # model/epoch_<N>.pth, decoder only (3.1 GB each)
    ),
]
