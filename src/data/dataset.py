import os
import random


def create_splits(preprocessed_dir, train_ratio=0.8, val_ratio=0.15, seed=42):
    """
    Create and persist train/val/test splits under <preprocessed_dir>/splits/,
    ensuring that all scans from the same patient (stem before the final '-XXX')
    stay together, and that the split proportions are by total scan count.
    Returns three lists of file paths.
    """
    splits_dir = os.path.join(preprocessed_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    # Gather all .pt files
    all_files = sorted(
        os.path.join(preprocessed_dir, f)
        for f in os.listdir(preprocessed_dir)
        if f.endswith('.pt')
    )
    if not all_files:
        raise ValueError(f"No .pt files found in {preprocessed_dir}")

    # Group by patient_id (drop last '-XXX')
    groups = {}
    for f in all_files:
        pid = os.path.basename(f).rsplit('-', 1)[0]
        groups.setdefault(pid, []).append(f)

    # Total scan count and targets
    total_scans = len(all_files)
    train_target = total_scans * train_ratio
    val_target = total_scans * val_ratio

    # Shuffle patients
    rnd = random.Random(seed)
    patient_ids = list(groups.keys())
    rnd.shuffle(patient_ids)

    # Greedy assignment
    train_pids, val_pids, test_pids = [], [], []
    train_count = val_count = 0
    for pid in patient_ids:
        count = len(groups[pid])
        # assign to train if it keeps us within train_target
        if train_count + count <= train_target:
            train_pids.append(pid)
            train_count += count
        # else assign to val if within val_target
        elif val_count + count <= val_target:
            val_pids.append(pid)
            val_count += count
        else:
            test_pids.append(pid)

    # Build file lists
    train_files = [f for pid in train_pids for f in groups[pid]]
    val_files = [f for pid in val_pids for f in groups[pid]]
    test_files = [f for pid in test_pids for f in groups[pid]]

    # Persist splits
    def _write(path, lst):
        with open(path, 'w') as fh:
            for fn in lst:
                fh.write(fn + '\n')

    _write(os.path.join(splits_dir, 'train.txt'), train_files)
    _write(os.path.join(splits_dir, 'val.txt'), val_files)
    _write(os.path.join(splits_dir, 'test.txt'), test_files)

    return train_files, val_files, test_files


def get_splits(preprocessed_dir):
    """
    Read persisted splits from <preprocessed_dir>/splits/ and return lists.
    """
    splits_dir = os.path.join(preprocessed_dir, "splits")
    train_txt = os.path.join(splits_dir, "train.txt")
    val_txt = os.path.join(splits_dir, "val.txt")
    test_txt = os.path.join(splits_dir, "test.txt")

    if not os.path.isdir(splits_dir) or not all(os.path.exists(p)
                                                for p in [train_txt, val_txt, test_txt]):
        raise RuntimeError(f"Splits not found in {splits_dir}. Please run create_splits first.")

    def _read(path):
        with open(path) as fh:
            return [l.strip() for l in fh if l.strip()]

    train_files = _read(train_txt)
    val_files = _read(val_txt)
    test_files = _read(test_txt)
    return train_files, val_files, test_files
