# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from functools import partial

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torch.distributed as dist

import timm
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup

import util_mae.misc as misc
from util_mae.misc import NativeScalerWithGradNormCount as NativeScaler
from util_mae.misc import extract_imagenet
import util_moco.builder, util_moco.loader

import models.models_mae as models_mae
import models.models_moco as models_moco
import models.models_lejepa as models_lejepa
import models.models_dino as models_dino


from models.augmentation import augmentation_mae, augmentation1_moco, augmentation2_moco, augmentation_ce, AugmentationLeJEPA, AugmentationDINO
from engine_pretrain import train_one_epoch_mae, train_one_epoch_moco, train_one_epoch_ce, train_one_epoch_lejepa, train_one_epoch_dino


def get_args_parser():
    parser = argparse.ArgumentParser('Pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--method', default='mae', type=str,
                        help='Which loss to use')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    #MAE parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    #MoCo parameters
    parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
    parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--moco-m', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-m-cos', action='store_true',
                        help='gradually increase moco momentum to 1 with a '
                            'half-cycle cosine schedule')
    parser.add_argument('--moco-t', default=0.2, type=float,
                        help='softmax temperature (default: 1.0)')

    # CE parameters
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='Number of classification classes')

    # LeJEPA parameters
    parser.add_argument('--lejepa-lamb', default=0.05, type=float,
                        help='LeJEPA loss weight for SIGReg regularization (default: 0.1)')
    parser.add_argument('--lejepa-proj-dim', default=128, type=int,
                        help='Projection head output dimension for LeJEPA (default: 128)')

    # DINO parameters
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                        help='Base EMA parameter for teacher update.')
    parser.add_argument('--freeze_last_layer', default=3, type=int,
                        help='Number of epochs to keep the output layer fixed.')
    parser.add_argument('--local_crops_number', type=int, default=8,
                        help='Number of small local views to generate.')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help='Final value of the weight decay for DINO cosine schedule.')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Maximal parameter gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can help optimization for larger ViT architectures. None for disabling.')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default="/cluster/scratch/dammeier/imagenet/", type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    assert(torch.cuda.is_bf16_supported())

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    tmp_dir = os.environ.get('TMPDIR', '/tmp')
    train_dir = os.path.join(tmp_dir, 'train')

    if misc.get_local_rank() == 0:  # one extractor per node (TMPDIR is node-local)
        extract_imagenet(args.data_path, tmp_dir)
    if args.distributed:
        dist.barrier()
    print("Extraction complete!")

    # simple augmentation
    if args.method == 'mae':
        dataset_train = datasets.ImageFolder(train_dir, transform=augmentation_mae(args))
    elif args.method == 'moco':
        dataset_train= datasets.ImageFolder(
            train_dir, util_moco.loader.TwoCropsTransform(augmentation1_moco(args), (augmentation2_moco(args))))
    elif args.method == 'ce':
        dataset_train = datasets.ImageFolder(train_dir, transform=augmentation_ce(args))
    elif args.method == 'lejepa':
        transform = AugmentationLeJEPA(
            (0.3, 1.),
            (0.05, 0.3),
            args.local_crops_number,
        )
        dataset_train = datasets.ImageFolder(train_dir, transform=transform)
    elif args.method == 'dino':
        transform = AugmentationDINO(
            (0.25, 1.),
            (0.05, 0.25),
            args.local_crops_number,
        )
        dataset_train = datasets.ImageFolder(train_dir, transform=transform)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    if args.method == 'mae':
        model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    elif args.method == 'moco':
        model = util_moco.builder.MoCo_ViT(
            partial(models_moco.__dict__[args.model], stop_grad_conv1=True),  # stop_grad_conv1 just set to true as it give better performance
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    elif args.method == 'ce':
        model = timm.create_model(args.model, pretrained=False, num_classes=args.nb_classes, drop_path_rate=0.1) #use 'vit_base_patch16_224'
        mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
            mode='batch', label_smoothing=0.1, num_classes=args.nb_classes)
    elif args.method == 'lejepa':
        model = models_lejepa.VisionTransformerLeJEPA(proj_dim=args.lejepa_proj_dim)
        sigreg = models_lejepa.SIGReg().to(device)
    elif args.method == 'dino':
        model = models_dino.__dict__[args.model](
            patch_size=16, drop_path_rate=0.1,  # stochastic depth
        )
        teacher = models_dino.__dict__[args.model](patch_size=16)

        embed_dim = model.embed_dim
        model = models_dino.MultiCropWrapper(model, models_dino.DINOHead(
            embed_dim, 65536, use_bn=False, norm_last_layer=True,
        ))
        teacher = models_dino.MultiCropWrapper(
            teacher, models_dino.DINOHead(embed_dim, 65536, False),
        )
        teacher.to(device)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        if args.method in ['moco', 'lejepa']:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #if args.method == 'dino':
            #teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
            #teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[misc.get_local_rank()], find_unused_parameters=False) #not sure what to set it to
        model_without_ddp = model.module
            
    
    if args.method == 'dino':
        teacher.load_state_dict(model_without_ddp.state_dict())
        for p in teacher.parameters():
            p.requires_grad = False

        dino_loss = models_dino.DINOLoss(
            65536,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            0.04, #warmup teacher temp
            0.07, #teacher temp
            50, #warmup teacher temp epochs
            args.epochs,
        ).to(device)

    # following timm: set wd as 0 for bias and norm layers
    if args.method == 'mae':
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    elif args.method == 'moco':
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    elif args.method in ['ce', 'lejepa', 'dino']:
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    print(optimizer)
    #loss_scaler = NativeScaler()

    misc.load_model(
        args=args, model_without_ddp=model_without_ddp,
        teacher_without_ddp=teacher if args.method == 'dino' else None,
        dino_loss=dino_loss if args.method == 'dino' else None,
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        
        if args.method == 'mae':
            train_stats = train_one_epoch_mae(
                model, data_loader_train,
                optimizer, device, epoch,
                log_writer=log_writer,
                args=args
            )
        elif args.method == 'moco':
            train_stats = train_one_epoch_moco(
                model, data_loader_train,
                optimizer, device, epoch,
                log_writer=log_writer,
                args=args
            )
        elif args.method == 'ce':
            train_stats = train_one_epoch_ce(
                model, data_loader_train,
                optimizer, device, epoch,
                mixup_fn=mixup_fn,
                log_writer=log_writer,
                args=args
            )
        elif args.method == 'lejepa':
            train_stats = train_one_epoch_lejepa(
                model, sigreg, data_loader_train,
                optimizer, device, epoch,
                log_writer=log_writer,
                args=args
            )
        elif args.method == 'dino':
            train_stats = train_one_epoch_dino(
                model, teacher, dino_loss,
                data_loader_train, optimizer,
                device, epoch,
                log_writer=log_writer,
                args=args
            )

        if args.output_dir and (epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                epoch=epoch,
                teacher_without_ddp=teacher if args.method == 'dino' else None,
                dino_center=dino_loss.center if args.method == 'dino' else None,
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
