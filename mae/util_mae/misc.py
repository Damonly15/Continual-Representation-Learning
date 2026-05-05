# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import glob
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import subprocess
import scipy.io

import torch
import torch.distributed as dist
from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return get_rank() % torch.cuda.device_count()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(minutes=60))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, model=None, cancel_last_layer_grads=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            self._scaler.unscale_(optimizer)
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)
            if cancel_last_layer_grads and model is not None:
                for n, p in model.named_parameters():
                    if "last_layer" in n and p.grad is not None:
                        p.grad = None
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer=None, teacher_without_ddp=None, dino_center=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if optimizer is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                #'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            if teacher_without_ddp is not None:
                to_save['teacher'] = teacher_without_ddp.state_dict()
            if dino_center is not None:
                to_save['dino_center'] = dino_center

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer=None, teacher_without_ddp=None, dino_loss=None):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        #if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
        #    optimizer.load_state_dict(checkpoint['optimizer'])
        #    args.start_epoch = checkpoint['epoch'] + 1
        #    if 'scaler' in checkpoint:
        #        loss_scaler.load_state_dict(checkpoint['scaler'])
        #    print("With optim & sched!")

        if getattr(args, 'method', None) == 'ce':
            model_without_ddp.head.reset_parameters()

        if teacher_without_ddp is not None and 'teacher' in checkpoint:
            teacher_without_ddp.load_state_dict(checkpoint['teacher'])

        if dino_loss is not None and 'dino_center' in checkpoint:
            dino_loss.center.copy_(checkpoint['dino_center'])

def load_checkpoint_model(args, checkpoint):

    if 'mae' in args.finetune or 'ce' in args.finetune:
        # saved by main_pretrain.py: {'model': MAE/ViT.state_dict(), ...}
        checkpoint_model = checkpoint['model']
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    elif 'moco' in args.finetune:
        # saved by main_pretrain.py: {'model': MoCo_ViT.state_dict(), ...}
        state_dict = checkpoint['model']
        checkpoint_model = {
            k[len("base_encoder."):]: v
            for k, v in state_dict.items()
            if k.startswith('base_encoder.') and not k.startswith('base_encoder.head.')
        }
    elif 'lejepa' in args.finetune:
        # saved by main_pretrain.py: {'model': VisionTransformerLeJEPA.state_dict(), ...}
        state_dict = checkpoint['model']
        checkpoint_model = {
            k[len("backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith('backbone.') and not k.startswith('backbone.head.')
        }
    elif 'dino' in args.finetune:
        # saved by main_pretrain.py: {'model': MultiCropWrapper.state_dict(), ...}
        state_dict = checkpoint['model']
        checkpoint_model = {
            k[len("backbone."):]: v
            for k, v in state_dict.items()
            if k.startswith('backbone.')
        }
    return checkpoint_model
    

def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def extract_imagenet(data_path, tmp_dir):
    train_tar = os.path.abspath(os.path.join(data_path, "ILSVRC2012_img_train.tar"))
    val_tar = os.path.abspath(os.path.join(data_path, "ILSVRC2012_img_val.tar"))

    for name, tar_path in [('train', train_tar), ('val', val_tar)]:
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Expected tar not found: {tar_path}")

        target_dir = os.path.join(tmp_dir, name)
        os.makedirs(target_dir, exist_ok=True)

        subprocess.run(["tar", "-xf", tar_path, "-C", target_dir], check=True)

        # Extract inner class tars and remove them from local SSD
        for class_tar in sorted(glob.glob(os.path.join(target_dir, "*.tar"))):
            class_dir = class_tar[:-4]  # strip .tar
            os.makedirs(class_dir, exist_ok=True)
            subprocess.run(["tar", "-xf", class_tar, "-C", class_dir], check=True)
            os.remove(class_tar)

    print("Finished extraction.")

def analyze_features(tmp_dir: str, device: torch.device) -> dict:
    """Load features from tmp_dir, move to GPU, and compute intra/inter-class variance
    and hyperspherical uniformity (mean and variance across classes).

    Must be called only from the main process.

    Returns a dict with keys:
        within_variance  : scalar — mean intra-class variance across classes
        between_variance : scalar — mean squared distance of class means from global mean
        hu_mean          : scalar — mean hyperspherical uniformity across classes
        hu_var           : scalar — variance of hyperspherical uniformity across classes
    """
    # ------------------------------------------------------------------ load
    import numpy as np
    # Free any cached GPU memory from training before loading features.
    torch.cuda.empty_cache()

    features = torch.cat(
        [torch.from_numpy(np.load(os.path.join(tmp_dir, f"feat_rank_{r}.npy")).copy())
         for r in range(dist.get_world_size())], dim=0)               # (N, D) on CPU
    targets = torch.cat(
        [torch.from_numpy(np.load(os.path.join(tmp_dir, f"targets_rank_{r}.npy")).copy())
         for r in range(dist.get_world_size())], dim=0)               # (N,) on CPU

    classes = targets.unique()
    nb_classes = len(classes)
    D = features.shape[1]

    # ------------------------------------------------------------------ per-class stats

    cls_means       = torch.zeros(nb_classes, D, device=device)
    within_variance = torch.zeros(nb_classes, device=device)

    for i, c in enumerate(classes):
        mask = targets == c
        assert (mask.sum() > 0)
        cls_feats = features[mask].to(device)   # (N_c, D)

        cls_mean = cls_feats.mean(dim=0)        # (D,)
        cls_feats -= cls_mean                   # in-place: no extra copy
        within_variance[i] = cls_feats.pow(2).sum(dim=1).mean()
        del cls_feats                           # free GPU memory before next class

        cls_means[i] = cls_mean

    global_mean = cls_means.mean(dim=0)         # (D,)

    within_variance = within_variance.mean()
    between_variance = (cls_means - global_mean).pow(2).sum(dim=1).mean()

    # hyperspherical uniformity across class means
    # normalize mean-centered class means: (μ_c - μ̄) / ||μ_c - μ̄||
    centered = cls_means - global_mean                                       # (C, D)
    normed = torch.nn.functional.normalize(centered, dim=1)                 # (C, D)

    # pairwise squared Euclidean distances: ||u - v||^2 = 2 - 2 cos(theta)
    sim = normed @ normed.T                                                  # (C, C)
    dist_sq = (2 - 2 * sim).clamp(min=1e-8)
    log_inv_dist = -0.5 * dist_sq.log()                                     # (C, C)
    # per-class mean over its pairs (row mean excluding diagonal)
    hu_per_class = (log_inv_dist.sum(dim=1) - log_inv_dist.diagonal()) / (nb_classes - 1)
    # global HU: mean over all upper-triangle pairs
    idx = torch.triu_indices(nb_classes, nb_classes, offset=1)
    hu_mean = log_inv_dist[idx[0], idx[1]].mean()
    hu_var  = hu_per_class.var()

    return dict(
        within_variance=within_variance.item(),
        between_variance=between_variance.item(),
        hu_mean=hu_mean.item(),
        hu_var=hu_var.item(),
    )