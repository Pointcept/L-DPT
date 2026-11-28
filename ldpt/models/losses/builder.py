"""
Criteria Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from ldpt.utils.registry import Registry

LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.losses = []
        self.names = []
        for loss_cfg in self.cfg:
            self.losses.append(LOSSES.build(cfg=loss_cfg))
            self.names.append(loss_cfg["type"])

    def __call__(self, pred, target):
        result_dict = dict()
        loss = pred.sum() * 0.0  # to create a zero loss with correct device
        for n, c in zip(self.names, self.losses):
            loss_ = c(pred, target)
            loss = loss + loss_
            result_dict.update({n: loss_ / c.loss_weight})
        result_dict["loss"] = loss
        return result_dict


def build_criteria(cfg):
    return Criteria(cfg)
