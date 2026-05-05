# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_weight_decay(optimizer, epoch, args):
    """Cosine weight decay schedule (no warmup)."""
    wd = args.weight_decay_end + 0.5 * (args.weight_decay - args.weight_decay_end) * \
        (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if param_group.get("apply_wd", True):
            param_group["weight_decay"] = wd
    return wd


def adjust_dino_momentum(epoch, args):
    """Cosine momentum schedule for EMA teacher (no warmup)."""
    return 1. - (1. - args.momentum_teacher) * 0.5 * \
        (1. + math.cos(math.pi * epoch / args.epochs))

def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m