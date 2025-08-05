import os
import shutil

import torch
import torchio as tio
from synapseclient import Synapse

from data.dataset import create_splits


def download_raw(synapse_ids, dest):
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if token is None:
        raise RuntimeError("Set SYNAPSE_AUTH_TOKEN")
    syn = Synapse(cache_root_dir=dest)
    syn.login(authToken=token)
    os.makedirs(dest, exist_ok=True)
    for sid in synapse_ids:
        print(f"[download] {sid}")
        ent = syn.get(sid, downloadLocation=dest)
        shutil.unpack_archive(ent.path, dest)
        os.remove(ent.path)


def build_subjects(raw_dir):
    subjects = []
    # Expect structure raw_dir/<patient>/<scan_folder>
    for patient in os.listdir(raw_dir):
        patient_dir = os.path.join(raw_dir, patient)
        if not os.path.isdir(patient_dir):
            continue
        for entry in os.listdir(patient_dir):
            ep = os.path.join(patient_dir, entry)
            # define modality file names
            mods = ['t2w', 't2f', 't1n', 't1c']
            paths = {m: os.path.join(ep, f"{entry}-{m}.nii.gz") for m in mods}
            # include only if all modalities present
            if all(os.path.exists(p) for p in paths.values()):
                subjects.append(
                    tio.Subject(
                        **{m: tio.ScalarImage(paths[m]) for m in ['t2w', 't2f', 't1n', 't1c']}
                    )
                )
    return subjects


def check_and_prepare(run_dir, cfg, raw_data_path):
    subjects_file = os.path.join(run_dir, "subjects_list.pt")
    raw_dir = raw_data_path
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        download_raw(cfg["data"]["synapse_ids"], raw_dir)

    # Build and save subjects list once
    if not os.path.exists(subjects_file):
        subs = build_subjects(raw_dir)
        torch.save(subs, subjects_file)
        print(f"Saved {len(subs)} subjects to {subjects_file}.")
    else:
        subs = torch.load(subjects_file, weights_only=False)

    # Write each Subject to its own .pt named by its sample ID
    subj_dir = os.path.join(run_dir, 'patch_subjects')
    os.makedirs(subj_dir, exist_ok=True)

    for subj in subs:
        sample_id = os.path.basename(subj['t1n'].path).replace('-t1n.nii.gz', '')
        out_path = os.path.join(subj_dir, f"{sample_id}.pt")
        if not os.path.exists(out_path):
            torch.save(subj, out_path)

    # Now split by patient (unchanged)
    create_splits(
        subj_dir,
        train_ratio=cfg['data']['train_ratio'],
        val_ratio=cfg['data']['val_ratio'],
        seed=cfg['data']['random_seed']
    )
    print(f"Created train/val/test splits under {subj_dir}/splits/")
