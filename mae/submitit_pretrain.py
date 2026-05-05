# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import main_pretrain as trainer
import submitit

#--resume /cluster/home/dammeier/CRL/checkpoints/mae_pretrain_task0_vit_base.pth
#python submitit_pretrain.py --method mae --norm_pix_loss --blr 1.5e-4 --data_path /cluster/scratch/dammeier/imagenet_task0/
#python submitit_pretrain.py --model vit_base --method moco --epochs 300 --batch_size 128 --accum_iter 1 --blr 1.5e-4 --weight_decay 0.1 --moco-m-cos --data_path /cluster/scratch/dammeier/imagenet_task0/ --nodes 4
#python submitit_pretrain.py --model vit_base_patch16_224 --method ce --epochs 300 --warmup_epochs 10 --batch_size 128 --accum_iter 1  --blr 2.5e-4  --min_lr 1e-6 --nb_classes 250 --clip_grad 1.0 --data_path /cluster/scratch/dammeier/imagenet_task0/
#python submitit_pretrain.py --model vit_base_patch16_224 --method lejepa --epochs 100 --warmup_epochs 5 --batch_size 64 --accum_iter 1  --lr 5e-4  --min_lr 1e-6 --local_crops_number 8 --data_path /cluster/scratch/dammeier/imagenet_task0/
#python submitit_pretrain.py --model vit_base --method dino --weight_decay 0.04 --epochs 400 --batch_size 32 --accum_iter 1 --warmup_epochs 10 --blr 7.5e-4 --min_lr 2e-6 --clip_grad 0.3 --local_crops_number 10 --data_path /cluster/scratch/dammeier/imagenet_task0/ --nodes 4

def parse_args():
    trainer_parser = trainer.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE pretrain", parents=[trainer_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=1400, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--gpu_mem", default="20G", help="Request 20G GPU memory per GPU.")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    
    #scratch_env = os.getenv("SCRATCH")
    
    #if scratch_env is not None:
    #    p = Path(scratch_env) / "experiments"
    #else:
    #    raise RuntimeError("No shared folder available")
    p = Path(f"/cluster/home/{user}/CRL/experiments")
        
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_pretrain as trainer

        self._setup_gpu_args()
        trainer.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    kwargs = {}
    if args.comment:
        kwargs['slurm_comment'] = args.comment
    kwargs['slurm_constraint'] = 'ib'

    executor.update_parameters(
        slurm_mem_per_cpu='4G',
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=5,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_additional_parameters={
            'tmp': '200G',
            'gres': f'gpumem:{args.gpu_mem}',
            'exclude': 'eu-g3-[062-079,083-084],eu-lo-g3-[049-061]', #exclude rtx 6000 and rtx titan
        },
        slurm_srun_args=['--gres', f'gpumem:{args.gpu_mem}'],
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="pretrain")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    # print("Submitted job_id:", job.job_id)
    print(job.job_id)


if __name__ == "__main__":
    main()
