import os
import subprocess
import random
import json

DATA_PATH = "/cluster/scratch/dammeier/imagenet/"
SCRATCH_PATH = "/cluster/scratch/dammeier/"
TMP_DIR = os.environ.get('TMPDIR', '/tmp')
NUM_TASKS = 4
SEED = 0


def split_into_tasks(data_path, scratch_path, tmp_dir, num_tasks=4, seed=0):
    train_tar = os.path.join(data_path, "ILSVRC2012_img_train.tar")
    val_tar = os.path.join(data_path, "ILSVRC2012_img_val.tar")

    # Extract outer tars to get the 1000 inner synset tars
    for name, tar_path in [('train', train_tar), ('val', val_tar)]:
        target_dir = os.path.join(tmp_dir, name)
        os.makedirs(target_dir, exist_ok=True)
        print(f"Extracting {name} tar...")
        subprocess.run(["tar", "-xf", tar_path, "-C", target_dir], check=True)

    train_dir = os.path.join(tmp_dir, 'train')
    val_dir = os.path.join(tmp_dir, 'val')

    synsets = sorted([f[:-4] for f in os.listdir(train_dir) if f.endswith('.tar')])
    assert len(synsets) == 1000, f"Expected 1000 synset tars, found {len(synsets)}"

    random.seed(seed)
    shuffled = synsets[:]
    random.shuffle(shuffled)

    task_size = len(shuffled) // num_tasks
    class_mapping = {}

    for task_id in range(num_tasks):
        # Do NOT re-sort: preserve the shuffled order so label 0..249 is random
        task_synsets = shuffled[task_id * task_size: (task_id + 1) * task_size]
        class_mapping[task_id] = {local_label: synset
                                   for local_label, synset in enumerate(task_synsets)}

        task_dir = os.path.join(scratch_path, f"imagenet_task{task_id}")
        os.makedirs(task_dir, exist_ok=True)

        for split_name, split_dir in [('train', train_dir), ('val', val_dir)]:
            synset_tars = [f"{s}.tar" for s in task_synsets]
            out_tar = os.path.join(task_dir, f"ILSVRC2012_img_{split_name}.tar")
            print(f"Task {task_id}: packing {split_name} tar ({len(synset_tars)} synsets)...")
            subprocess.run(["tar", "-cf", out_tar] + synset_tars, cwd=split_dir, check=True)

        print(f"Task {task_id} done -> {task_dir}")

    mapping_path = os.path.join(scratch_path, "imagenet_class_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=2)
    print(f"Class mapping saved to {mapping_path}")


if __name__ == "__main__":
    split_into_tasks(DATA_PATH, SCRATCH_PATH, TMP_DIR, NUM_TASKS, SEED)

    print("\nDone. Task folders:")
    for i in range(NUM_TASKS):
        print(f"  /cluster/scratch/dammeier/imagenet_task{i}/")
