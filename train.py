"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from ldpt.engines.default import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from ldpt.engines.launch import launch
from ldpt.engines.trainer import build_trainer


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer_args = dict(**cfg.train)
    trainer_args["cfg"] = cfg
    trainer = build_trainer(trainer_args)
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
