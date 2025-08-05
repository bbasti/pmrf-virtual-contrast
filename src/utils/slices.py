from pathlib import Path
from typing import Dict
from tqdm import tqdm
import torch
import torchio as tio

from utils.config_io import get_run_dir
from utils.slices_helpers import save_plane_slices, transform_raw_volume
from data.dataset import get_splits

PLANES = ('axial', 'coronal', 'sagittal')


def get_slice_transform(resample_to: str = None) -> tio.Compose:
    """
    Build a standard ToCanonical + (optional Resample) + ZNormalization pipeline.
    """
    transforms = [tio.ToCanonical()]
    if resample_to:
        transforms.append(tio.Resample(resample_to))
    transforms.append(tio.ZNormalization())
    return tio.Compose(transforms)


def ensure_output_dirs(output_dirs: Dict[str, Dict[str, Path]]):
    """
    Given a mapping output_dirs[key][plane] = Path, create all directories.
    """
    for plane_dirs in output_dirs.values():
        for d in plane_dirs.values():
            d.mkdir(parents=True, exist_ok=True)


def generate_slices_real(run_id: str, n_slices: int = 9, step: int = 10):
    """
    Generate FID slices (T1C and T1N) from training volumes.
    """
    run_dir = get_run_dir(run_id)
    subj_dir = run_dir / 'patch_subjects'
    train_files, _, _ = get_splits(str(subj_dir))

    transform = get_slice_transform()

    base_out = run_dir / 'fid_slices'
    output_dirs = {
        't1c': {pl: base_out / 't1c' / pl for pl in PLANES},
        't1n': {pl: base_out / 't1n' / pl for pl in PLANES},
    }
    ensure_output_dirs(output_dirs)

    for pt_path in tqdm(train_files, desc="Generating real slices"):
        subj = torch.load(pt_path, weights_only=False)
        subj_t = transform(subj)
        for key in ('t1c', 't1n'):
            vol = subj_t[key][tio.DATA].squeeze().numpy()
            base = Path(subj[key].path).stem.replace(f'-{key}.nii', '')
            for pl in PLANES:
                save_plane_slices(vol, pl, base, str(output_dirs[key][pl]), n_slices, step)


def generate_slices_test(run_id: str, n_slices: int = 9, step: int = 10):
    """
    Generate FID PNG slices from test volumes (t1c, t1n, t2w, t2f).
    """
    run_dir = get_run_dir(run_id)
    subj_dir = run_dir / 'patch_subjects'
    _, _, test_files = get_splits(str(subj_dir))

    transform = get_slice_transform()

    base_out = run_dir / 'fid_slices_test'
    output_dirs = {
        key: {pl: base_out / key / pl for pl in PLANES}
        for key in ('t1c', 't1n', 't2w', 't2f')
    }
    ensure_output_dirs(output_dirs)

    for pt_path in tqdm(test_files, desc="Generating test slices"):
        subj = torch.load(pt_path, weights_only=False)
        subj_t = transform(subj)
        for key in ('t1c', 't1n', 't2w', 't2f'):
            vol = subj_t[key][tio.DATA].squeeze().numpy()
            original = Path(subj[key].path).stem
            base = original.replace(f'-{key}.nii', '').replace(f'-{key}.nii.gz', '')
            for pl in PLANES:
                save_plane_slices(vol, pl, base, str(output_dirs[key][pl]), n_slices, step)


def _save_infer_volume_slices(model, n_slices, nii_path, output_dirs, step, transform):
    stem = nii_path.with_suffix('').with_suffix('').name
    if stem.endswith(f'_{model}'):
        base = stem[: -len(f'_{model}')]
    else:
        base = stem

    vol = transform_raw_volume(nii_path, transform)
    for pl in PLANES:
        save_plane_slices(vol, pl, base, str(output_dirs[model][pl]), n_slices, step)


def generate_slices_infer(run_id: str, infer_run: str, n_slices: int = 9, step: int = 10):
    """
    Generate inference slices from volumes produced by PM and RF.
    """
    run_dir = get_run_dir(run_id)
    rf_dir = run_dir / 'output_inference_rf' / infer_run
    vols_dir_rf = rf_dir / 'output_volumes'
    if not vols_dir_rf.exists():
        raise FileNotFoundError(f"RF volumes directory not found: {vols_dir_rf}")

    pm_run_file = rf_dir / 'used_pm_run.txt'
    if not pm_run_file.exists():
        raise FileNotFoundError(f"PM run info not found: {pm_run_file}")
    pm_run = pm_run_file.read_text().strip()

    pm_dir = run_dir / 'output_inference_pm' / pm_run
    vols_dir_pm = pm_dir / 'output_volumes'
    if not vols_dir_pm.exists():
        raise FileNotFoundError(f"PM volumes directory not found: {vols_dir_pm}")

    transform = get_slice_transform()

    # Prepare output dirs
    slice_base = rf_dir / 'slices'
    output_dirs = {
        'pm': {pl: slice_base / 'pm' / pl for pl in PLANES},
        'rf': {pl: slice_base / 'rf' / pl for pl in PLANES},
    }
    ensure_output_dirs(output_dirs)

    for model, vols_dir in (('pm', vols_dir_pm), ('rf', vols_dir_rf)):
        for nii_path in tqdm(sorted(vols_dir.glob(f'*_{model}.nii.gz')), desc=f'Generating infer slices [{model}]'):
            _save_infer_volume_slices(model, n_slices, nii_path, output_dirs, step, transform)


def generate_slices_infer_t1n(run_id: str, infer_run: str, n_slices: int = 9, step: int = 10):
    """
    Generate inference T1N-only slices from PM and RF volumes.
    """
    run_dir = get_run_dir(run_id)

    rf_dir = run_dir / 'output_inference_rf_t1n' / infer_run
    vols_dir_rf = rf_dir / 'output_volumes'
    if not vols_dir_rf.exists():
        raise FileNotFoundError(f"RF-T1N volumes directory not found: {vols_dir_rf}")

    pm_run_file = rf_dir / 'used_pm_run.txt'
    if not pm_run_file.exists():
        raise FileNotFoundError(f"PM run info not found: {pm_run_file}")
    pm_run = pm_run_file.read_text().strip()

    pm_dir = run_dir / 'output_inference_pm_t1n' / pm_run
    vols_dir_pm = pm_dir / 'output_volumes'
    if not vols_dir_pm.exists():
        raise FileNotFoundError(f"PM-T1N volumes directory not found: {vols_dir_pm}")

    transform = get_slice_transform()

    slice_base = rf_dir / 'slices'
    output_dirs = {
        'pm_t1n': {pl: slice_base / 'pm_t1n' / pl for pl in PLANES},
        'rf_t1n': {pl: slice_base / 'rf_t1n' / pl for pl in PLANES},
    }
    ensure_output_dirs(output_dirs)

    for model, vols_dir in (('pm_t1n', vols_dir_pm), ('rf_t1n', vols_dir_rf)):
        for nii_path in tqdm(sorted(vols_dir.glob('*.nii*')), desc=f'Generating T1N infer slices [{model}]'):
            _save_infer_volume_slices(model, n_slices, nii_path, output_dirs, step, transform)
