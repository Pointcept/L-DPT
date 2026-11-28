# L-DPT Large, COCO-Stuff semantic segmentation (171 classes), stage 2 of recipe 1: a
# one-epoch cosine anneal of semseg-l-dpt-v1m1-1-large.py from its epoch 9, run with
# seeds 0, 1 and 2, then the uniform weight average of the start and the three ends.
#   ./train.sh -g 8 -d cocostuff -c semseg-l-dpt-v1m1-1-large \
#       -n semseg-l-dpt-v1m1-1-large                           # stage 1 (epoch 9)
#   ./train.sh -g 8 -d cocostuff -c semseg-l-dpt-v1m1-1a-large-anneal \
#       -n semseg-l-dpt-v1m1-1a-large-anneal                   # seed 0
#   for seed in 1 2; do
#     ./train.sh -g 8 -d cocostuff -c semseg-l-dpt-v1m1-1a-large-anneal \
#         -n semseg-l-dpt-v1m1-1a-large-anneal-seed$seed -o "seed=$seed"
#   done
#   RUN=exp/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal
#   python scripts/average_weights.py \
#       --inputs exp/cocostuff/semseg-l-dpt-v1m1-1-large/model/epoch_9.pth \
#                $RUN/model/epoch_1.pth $RUN-seed1/model/epoch_1.pth \
#                $RUN-seed2/model/epoch_1.pth --output $RUN/model/average.pth
#   python scripts/evaluate.py \
#       --config-file configs/cocostuff/semseg-l-dpt-v1m1-1a-large-anneal.py \
#       --checkpoints $RUN/model/average.pth --num-gpus 8 --output $RUN/eval.json
# This file is semseg-l-dpt-v1m1-1-large.py with:
# - checkpoint: [DINOv3, stage 1's epoch_9.pth], loaded by hook 0 (two files);
#   allow_missing=False: a start file that leaves any model key at its init raises;
# - epoch 1 (3,696 iterations at batch 32); MuonAdamW as stage 1 at the peak lr 4e-4
#   (encoder group 4e-5), OneCycleLR as stage 1 (div 10, final div 1000, cosine) except
#   pct_start 0.03 (the decoder is trained; epoch_9.pth holds no optimizer state).
# It resolves key by key to the configuration of the run it reproduces
# (tests/test_configs.py pins it); only launch settings are left out.
_base_ = ["../_base_/default_runtime.py"]

encoder_scale = "vit7b"
# the DINOv3 file into the encoder, stage 1's epoch 9 (decoder only) into the decoder
checkpoint = [
    "checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "exp/cocostuff/semseg-l-dpt-v1m1-1-large/model/epoch_9.pth",
]

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

epoch = 1

base_lr = 4e-4
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
    pct_start=0.03,  # the decoder is trained; epoch_9.pth holds no optimizer state
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
        keywords=["module.", "module."],
        replacement=["module.encoder.", "module."],
        allow_missing=False,  # every model key must come from the two files
    ),  # checkpoint[0] into model.encoder, checkpoint[1] into the decoder
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
