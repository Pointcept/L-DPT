# L-DPT Large (mixed), relative depth, stage 2 (preset "l-dpt_large_depth_mixed"): a
# one-epoch cosine anneal of depth-l-dpt-v1m1-1-large-mixed.py from its epoch 4, then
# the uniform weight average of the start and the end of the anneal.
#   ./train.sh -g 8 -d hybrid -c depth-l-dpt-v1m1-1-large-mixed \
#       -n depth-l-dpt-v1m1-1-large-mixed                      # stage 1 (epoch 4)
#   ./train.sh -g 8 -d hybrid -c depth-l-dpt-v1m1-1a-large-mixed-anneal \
#       -n depth-l-dpt-v1m1-1a-large-mixed-anneal
#   python scripts/average_weights.py \
#       --inputs exp/hybrid/depth-l-dpt-v1m1-1-large-mixed/model/epoch_4.pth \
#                exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/epoch_1.pth \
#       --output exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/average.pth
#   python -m ldpt.models.l_dpt.convert \
#       exp/hybrid/depth-l-dpt-v1m1-1a-large-mixed-anneal/model/average.pth \
#       l-dpt_large_depth_mixed.pth --preset l-dpt_large_depth_mixed
# Stage 1 only needs to reach epoch 4 (stop it after model/epoch_4.pth is written, or
# keep it running: later epochs are not used). This file is
# depth-l-dpt-v1m1-1-large-mixed.py with:
# - checkpoint: [DINOv3, stage 1's epoch_4.pth], loaded by hook 0 (two files);
# - epoch 1 (9,057 iterations at batch 64); AdamW and OneCycleLR as stage 1 (peak lr
#   1e-4, encoder group 1e-5, div 10, final div 1000, cosine) except pct_start 0.03;
# - no CriterionWeightWarmup (the decoder is trained: aux_scale 1 from the start).
# Its training keys (all but data.val and hook 4, the evaluation) resolve key by key
# to the configuration the released weights were trained with (tests/test_configs.py
# pins it); only launch settings are left out.
_base_ = ["../_base_/default_runtime.py"]

encoder_scale = "vit7b"
# the DINOv3 file into the encoder, stage 1's epoch 4 (decoder only) into the decoder
checkpoint = [
    "checkpoints/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    "exp/hybrid/depth-l-dpt-v1m1-1-large-mixed/model/epoch_4.pth",
]

seed = 0
num_workers = 64  # total over all GPUs
batch_size = 64  # total over all GPUs
sync_bn = True
clip_grad = 1.0

patch_size = 16
log_interval = 10

model = dict(
    type="L-DPT-v1m1",
    task="relative_depth",
    encoder=dict(type="dinov3_vit7b"),
    encoder_inter_layers=[9, 19, 29, 39],
    freeze_encoder=True,
    encoder_weight_dtype="bf16",  # frozen encoder stored in bf16 (bit-exact)
    dec_dim=1024,
    out_dim=1,
    dec_num_heads=16,
    dec_num_layers=(4, 4, 4, 4),
    dec_attn_downscale=1,
    dec_ffn_ratio=4.0,
    dec_pos_embed_rope_base=100,
    dec_pos_embed_rope_normalize_coords="min",  # = "separate" on square crops
    dec_pos_embed_rope_shift_coords=None,
    dec_pos_embed_rope_jitter_coords=1.2,
    dec_pos_embed_rope_rescale_coords=8,
    dec_layer_scale_init=1.0e-05,
    dec_qkv_bias=False,
    dec_proj_bias=True,
    dec_ffn_bias=True,
    readout_kernel_size=3,
    precision="bf16",  # decoder under bf16 autocast; heads and loss in fp32
    attn_dtype="bf16",
    criterion=dict(
        type="relative_depth_loss",
        mage_weight=4.0,
        mage_num_scales=4,
        male_weight=0.0,
        male_num_scales=4,
        align="roe",
        head="exp",
        min_valid=1000,
        roe_points=2304,
        roe_trunc=1.0,
        weight_cap=10.0,
        refit_floor=0.01,
        label_types=dict(
            A=["hypersim", "VKITTI2", "structure3d", "tartanair", "irs"],
            B=["blendedmvs", "blendedmvs+", "blendedmvs++"],
            C=["nyuv2", "kitti", "DIODE-INDOOR", "DIODE-OUTDOOR"],
        ),
        type_trim=dict(A=0.0, B=0.05, C=0.05),
        local_levels=dict(A=[4, 16, 64], B=[4, 16], C=[4]),
        local_weight=1.0,
        local_cell=768.0,  # a nominal focal length in pixels, scaled with the resize
        local_pred_ratio=0.5,
        grad_types=["A"],
        aux_scale=1.0,
        log_logit_stats=True,
    ),
    head="exp",
    readout_fp32=True,
    attn_fp32="last_stage",
    dec_qk_bound=1000,
)

epoch = 1

base_lr = 1e-4
base_wd = 0.1

param_dicts = [dict(keyword="encoder", lr=base_lr / 10)]

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.03,  # the decoder is trained; epoch_4.pth holds no optimizer state
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
                depth_resample="edge_aware",
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
    dict(type="Collect", keys=["image", "depth", "name"]),
]

val_transform = [
    dict(type="ToTensor"),
    dict(
        type="Update",
        target_keys=["image"],  # only process image, keep depth unchanged
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
    dict(type="Collect", keys=["image", "depth", "depth_min", "depth_max", "name"]),
]

# data_root: data/<name> (ldpt/datasets/README.md); a comment marks the sets whose
# split JSON paths expect the data in a sub-folder of data_root
data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="HypersimDataset",
                data_root="data/hypersim",  # ml-hypersim evermotion_dataset (scenes/)
                split=["splits/train_dino_95.json", "splits/val_dino_95.json"],
                transform=train_transform,
            ),
            dict(
                type="VKITTI2Dataset",
                data_root="data/vkitti2",
                split=["splits/train_dino_95.json", "splits/val_dino_95.json"],
                transform=train_transform,
            ),
            dict(
                type="Structured3DDataset",
                data_root="data/structured3d",  # extract into Structured3D/
                split="splits/train.json",
                transform=train_transform,
            ),
            dict(
                type="TartanAirDataset",
                data_root="data/tartanair",  # extract into data/
                split=["splits/easy_dino_95.json", "splits/hard_dino_95.json"],
                transform=train_transform,
            ),
            dict(
                type="BlendedMVSDataset",
                data_root="data/blendedmvs",  # extract into files/
                split=["splits/train_dino_95.json"],
                transform=train_transform,
            ),
            dict(
                type="BlendedMVSPDataset",
                data_root="data/blendedmvs+",  # extract into files/
                split=["splits/train_dino_95.json"],
                transform=train_transform,
            ),
            dict(
                type="BlendedMVSPPDataset",
                data_root="data/blendedmvs++",  # extract into files/
                split=["splits/train_dino_95.json"],
                transform=train_transform,
            ),
            dict(
                type="IRSDataset",
                data_root="data/irs",  # extract into IRS_small/ (the split keeps that subset)
                split=["splits/train_dino_95.json"],
                transform=train_transform,
            ),
            dict(
                type="NYUv2Dataset",
                data_root="data/nyuv2",
                split="splits/train.json",
                transform=train_transform,
                eigen_crop="legacy",  # KITTI's Eigen ratios; True: standard crop
            ),
            dict(
                type="KITTIDataset",
                data_root="data/kitti",
                split="splits/train.json",
                transform=train_transform,
                kb_crop=True,
                garg_crop=True,
            ),
            dict(
                type="DIODEIndoorDataset",
                data_root="data/diode",
                split="splits/train_indoors.json",
                transform=train_transform,
                eigen_crop=True,
            ),
            dict(
                type="DIODEOutdoorDataset",
                data_root="data/diode",
                split="splits/train_outdoor.json",
                transform=train_transform,
                garg_crop=True,
            ),
        ],
    ),
    # evaluation: Marigold's images and masks on every set (docs/evaluation.md; GT kept
    # where min < depth < max, the aligned prediction clipped to [min, max])
    val=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="NYUv2Dataset",
                data_root="data/nyuv2",
                split="splits/test.json",
                transform=val_transform,
                eigen_crop=True,  # the standard Eigen crop, rows 45:471, cols 41:601
                depth_range=(1e-3, 10.0),
            ),
            dict(
                type="KITTIDataset",
                data_root="data/kitti",
                split="splits/test.json",
                transform=val_transform,
                kb_crop=True,
                eigen_crop=True,  # the Eigen mask on the 352 x 1216 crop
                depth_range=(1e-5, 80.0),
            ),
            dict(
                type="ETH3DDataset",
                data_root="data/eth3d",
                split="splits/train.json",  # the ETH3D split with ground truth
                transform=val_transform,
                depth_range=(1e-5, None),  # no upper bound
            ),
            dict(
                type="DIODEIndoorDataset",
                data_root="data/diode",
                split="splits/val_indoors.json",
                transform=val_transform,
                depth_range=(0.6, 350.0),  # clips the prediction
                mask_only=True,  # GT = DIODE's depth mask; no crop
            ),
            dict(
                type="DIODEOutdoorDataset",
                data_root="data/diode",
                split="splits/val_outdoor.json",
                transform=val_transform,
                depth_range=(0.6, 350.0),
                mask_only=True,
            ),
        ],
    ),
)

# hooks, addressed by index in overrides (4: evaluator, 6: saver); no weight warm-up
hooks = [
    dict(
        type="CheckpointLoader",
        keywords=["module.", "module."],
        replacement=["module.encoder.", "module."],
    ),  # checkpoint[0] into model.encoder, checkpoint[1] into the decoder
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=10),
    dict(
        type="DepthEvaluator",
        relative=True,
        tta=[["flip"]],
        align_solver="ransac",  # seeded depth RANSAC, then least squares on inliers
        ransac_seed=0,
    ),
    dict(type="DepthVisualizer", data_idx=233, relative=True, mask_invalid=False),
    dict(
        type="CheckpointSaver",
        save_freq=None,
        skip_keywords=["encoder"],
        keep_epoch_weights=True,  # model/epoch_<N>.pth, decoder weights only
    ),
]
