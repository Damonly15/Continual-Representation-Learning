import subprocess


JOBS = [
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/mae_pretrain_task0_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/mae_pretrain_task1_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/mae_pretrain_task2_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/moco_pretrain_task0_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/moco_pretrain_task1_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/moco_pretrain_task2_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/ce_pretrain_task0_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/ce_pretrain_task1_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/ce_pretrain_task2_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/lejepa_pretrain_task0_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/lejepa_pretrain_task1_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
    "python submitit_finetune.py --cls_token --blr 5e-4 --batch_size 256 --plotting_dir /cluster/home/dammeier/CRL/plotting --finetune /cluster/home/dammeier/CRL/checkpoints/lejepa_pretrain_task2_vit_base.pth --data_path /cluster/scratch/dammeier/cifar100/ --nb_classes 100",
]

if __name__ == "__main__":
    for i, cmd in enumerate(JOBS):
        print(f"[{i+1}/{len(JOBS)}] Submitting: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}")
        else:
            print(f"  Job ID: {result.stdout.strip()}")
