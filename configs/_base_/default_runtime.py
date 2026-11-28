checkpoint = None  # path to model checkpoint
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_workers = 16  # total over all GPUs
batch_size = 16  # total over all GPUs
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100  # total epoch
clip_grad = None  # disable with None, enable with a float
log_interval = 10  # evaluator progress log interval (iterations)

sync_bn = False  # use sync batch norm
enable_amp = False  # enable automatic mixed precision training
amp_dtype = "float16"  # "float16" or "bfloat16"
find_unused_parameters = False  # find unused parameters in model
# speed of the training steps, per machine (train.sh -o / --options; unset = off;
# evaluation runs eager with PyTorch's attention choice):
# compile_blocks = True  # or dict(encoder=..., decoder=...): torch.compile the blocks
# sdpa_backends = ["cudnn", "flash", "efficient", "math"]  # SDPA backend order
# allow_tf32 = True  # TF32 float32 matmuls

enable_wandb = False  # True: also log to Weights & Biases (TensorBoard always)
wandb_project = "l-dpt"  # W&B project name
wandb_key = None  # W&B API key; None: the key of `wandb login`
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter", log_interval=1),
    dict(type="CheckpointSaver", save_freq=None),
]

# Trainer
train = dict(type="DefaultTrainer")
