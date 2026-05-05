# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
from PIL import Image

import torchvision.transforms as transforms
from timm.data import create_transform
import torch.distributed as dist
import torchvision.datasets as datasets

from util_mae import misc

def build_imagenet(args):
    tmp_dir = os.environ.get('TMPDIR', '/tmp')
    train_dir = os.path.join(tmp_dir, 'train')
    val_dir = os.path.join(tmp_dir, 'val')

    if misc.get_local_rank() == 0:
        misc.extract_imagenet(args.data_path, tmp_dir)
    print("Extraction complete!")
    if args.distributed:
        dist.barrier()

    dataset_train = datasets.ImageFolder(train_dir, transform=augmentation_imagenet(True, args))
    dataset_val = datasets.ImageFolder(val_dir, transform=augmentation_imagenet(False, args))

    return dataset_train, dataset_val

def augmentation_imagenet(is_train, args):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_cifar100(args):
    dataset_train = datasets.CIFAR100(
        root=args.data_path, train=True, download=True,
        transform=augmentation_cifar100(True, args),
    )
    dataset_val = datasets.CIFAR100(
        root=args.data_path, train=False, download=True,
        transform=augmentation_cifar100(False, args),
    )
    return dataset_train, dataset_val

def augmentation_cifar100(is_train, args):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    if is_train:
        # For CIFAR, RandAugment (args.aa = 'rand-m9-mstd0.5-inc1') 
        # is often more robust than standard AutoAugment.
        transform = create_transform(
            input_size=args.input_size, # Usually 224
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa, 
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            mean=mean,
            std=std,
        )
        return transform

    # Evaluation transform
    t = []
    # If upsampling from 32 to 224, CenterCrop is often unnecessary 
    # if you resize directly to the input size.
    if args.input_size > 32:
        t.append(transforms.Resize((args.input_size, args.input_size), interpolation=Image.BICUBIC))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
