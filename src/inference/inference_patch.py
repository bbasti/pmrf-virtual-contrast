import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import nibabel as nib
import pandas as pd
import torchio as tio
from tqdm import tqdm

from data.dataset import get_splits
from utils.metrics import compute_volume_metrics
from models.posterior_mean import PosteriorMeanModel3D
from models.rectified_flow import RectifiedFlowModel3D


def _get_inference_params(cfg):
    """
    Extract patch-based inference parameters and common transform.

    Returns:
        transform, patch_size, overlap, batch_size
    """
    patch_size = int(cfg['inference']['patch_size'])
    overlap = int(cfg['inference'].get('overlap', patch_size // 2))
    batch_size = int(cfg['inference'].get('batch_size', 1))
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample('t1n'),
        tio.ZNormalization(),
        tio.CropOrPad(patch_size, only_pad=True),
    ])
    return transform, patch_size, overlap, batch_size


def _load_test_subjects(run_dir: Path):
    """
    Load test subjects (torchio.Subject) from patch_subjects.
    """
    subj_dir = run_dir / 'patch_subjects'
    _, _, test_files = get_splits(str(subj_dir))
    return [torch.load(str(p), weights_only=False) for p in test_files]


def _save_metrics_summary(records: list, out_dir: Path):
    """
    Write mean/std summary of metrics to CSV.
    """
    pd.DataFrame.from_records(records).describe().loc[['mean', 'std']].to_csv(out_dir / 'metrics_summary.csv')


def _prepare_dirs(
        run_dir: Path,
        inference_subdir: str,
        model: str,
        overlap: int,
        steps: int = None,
        pm_run: str = None
):
    """
    Prepare output directories for inference.
    """
    out_base = run_dir / inference_subdir
    out_base.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if steps is not None:
        out_dir = out_base / f"ov_{overlap}_steps_{steps}_{model}_{timestamp}"
    else:
        out_dir = out_base / f"ov_{overlap}_{model}_{timestamp}"
    out_dir.mkdir()
    if pm_run is not None:
        (out_dir / 'used_pm_run.txt').write_text(pm_run + '\n')
    volume_out_dir = out_dir / 'output_volumes'
    volume_out_dir.mkdir()
    metrics_csv = out_dir / 'metrics.csv'
    return out_dir, volume_out_dir, metrics_csv


def _run_pm_inference(
        cfg: dict,
        run_id: str,
        in_channels: int,
        pm_training_subdir: str,
        inference_subdir: str,
        model_name: str
):
    """
    Generic PM inference (multi- or single-channel).
    """
    run_dir = Path(os.environ.get('RUNS_BASE_DIR', 'runs')) / run_id

    transform, patch_size, overlap, batch_size = _get_inference_params(cfg)

    out_dir, volume_out_dir, metrics_csv = _prepare_dirs(
        run_dir, inference_subdir, model_name, overlap
    )

    test_subjects = _load_test_subjects(run_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pm = PosteriorMeanModel3D(in_channels=in_channels, out_channels=1).to(device)
    pm = torch.compile(pm)
    ckpt = run_dir / pm_training_subdir / cfg['posterior_mean']['save_path']
    pm.load_state_dict(torch.load(str(ckpt), map_location=device))
    pm.eval()

    records = []
    for subj in tqdm(test_subjects, desc=f"PM inference ({model_name})"):
        subj_t = transform(subj)
        sampler = tio.inference.GridSampler(subj_t, patch_size, overlap)
        loader = tio.SubjectsLoader(sampler, batch_size=batch_size)
        agg = tio.inference.GridAggregator(sampler, overlap_mode='hann')

        with torch.no_grad():
            for batch in loader:
                if in_channels == 1:
                    x = batch['t1n'][tio.DATA].to(device)
                else:
                    t1 = batch['t1n'][tio.DATA]
                    t2w = batch['t2w'][tio.DATA]
                    t2f = batch['t2f'][tio.DATA]
                    x = torch.cat([t1, t2w, t2f], dim=1).to(device)
                pred = pm(x)
                agg.add_batch(pred, batch[tio.LOCATION])

        vol = agg.get_output_tensor().squeeze().cpu().numpy()
        gt = subj_t['t1c'][tio.DATA].squeeze().numpy()
        mse, psnr, ssim = compute_volume_metrics(vol, gt)

        base = Path(subj['t1n'].path).stem.replace('-t1n.nii', '')
        nib.save(nib.Nifti1Image(vol.astype(np.float32), np.eye(4)),
                 volume_out_dir / f"{base}_{model_name}.nii.gz")
        rec = {'subject': base,
               f'mse_{model_name}': mse,
               f'psnr_{model_name}': psnr,
               f'ssim_{model_name}': ssim}
        records.append(rec)
        pd.DataFrame([rec]).to_csv(
            metrics_csv, mode='a', header=not metrics_csv.exists(), index=False
        )

    _save_metrics_summary(records, out_dir)
    print(f"{inference_subdir} inference saved to {out_dir}")


def run_infer_patch_pm(cfg: dict, run_id: str):
    """
    Inference for 3-channel PM.
    """
    _run_pm_inference(
        cfg, run_id,
        in_channels=3,
        pm_training_subdir='output_training_pm',
        inference_subdir='output_inference_pm',
        model_name='pm'
    )


def run_infer_patch_pm_t1n(cfg: dict, run_id: str):
    """
    Inference for T1N-only PM.
    """
    _run_pm_inference(
        cfg, run_id,
        in_channels=1,
        pm_training_subdir='output_training_pm_t1n',
        inference_subdir='output_inference_pm_t1n',
        model_name='pm_t1n'
    )


def _run_rf_inference(
        cfg: dict,
        run_id: str,
        steps: int,
        pm_run: str,
        pm_inference_subdir: str,
        pm_suffix: str,
        rf_training_subdir: str,
        inference_subdir: str,
        model_name: str
):
    """
    Generic RF inference.
    """
    run_dir = Path(os.environ.get('RUNS_BASE_DIR', 'runs')) / run_id

    transform, patch_size, overlap, batch_size = _get_inference_params(cfg)

    out_dir, volume_out_dir, metrics_csv = _prepare_dirs(
        run_dir, inference_subdir, model_name, overlap, steps, pm_run
    )

    test_subjects = _load_test_subjects(run_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rf = RectifiedFlowModel3D(in_channels=1, out_channels=1).to(device)
    rf = torch.compile(rf)
    ckpt_rf = run_dir / rf_training_subdir / cfg['rectified_flow']['save_path']
    rf.load_state_dict(torch.load(str(ckpt_rf), map_location=device))
    rf.eval()

    sigma_s = float(cfg['inference']['sigma_s'])
    dt = 1.0 / steps if steps > 0 else 0.0

    pm_vol_dir = run_dir / pm_inference_subdir / pm_run / 'output_volumes'
    if not pm_vol_dir.exists():
        raise FileNotFoundError(f"PM outputs not found: {pm_vol_dir}")

    records = []
    for subj in tqdm(test_subjects, desc=f"RF inference ({model_name})"):
        subj_t = transform(subj)
        base = Path(subj['t1n'].path).stem.replace('-t1n.nii', '')
        pm_path = pm_vol_dir / f"{base}_{pm_suffix}.nii.gz"
        subj_t['pm_pred'] = tio.ScalarImage(str(pm_path))

        sampler = tio.inference.GridSampler(subj_t, patch_size, overlap)
        loader = tio.SubjectsLoader(sampler, batch_size=batch_size)
        agg = tio.inference.GridAggregator(sampler, overlap_mode='hann')

        with torch.no_grad():
            for batch in loader:
                Z = batch['pm_pred'][tio.DATA].to(device)
                Z = Z + sigma_s * torch.randn_like(Z)
                for i in range(steps):
                    t = torch.full((Z.size(0),), i / steps, device=device)
                    Z = Z + rf(Z, t) * dt
                agg.add_batch(Z, batch[tio.LOCATION])

        rf_vol = agg.get_output_tensor().squeeze().cpu().numpy()
        gt = subj_t['t1c'][tio.DATA].squeeze().numpy()
        mse_rf, psnr_rf, ssim_rf = compute_volume_metrics(rf_vol, gt)

        base = Path(subj['t1n'].path).stem.replace('-t1n.nii', '')
        nib.save(nib.Nifti1Image(rf_vol.astype(np.float32), np.eye(4)),
                 volume_out_dir / f"{base}_{model_name}.nii.gz")

        rec = {'subject': base, f'mse_{model_name}': mse_rf, f'psnr_{model_name}': psnr_rf,
               f'ssim_{model_name}': ssim_rf}
        records.append(rec)
        pd.DataFrame([rec]).to_csv(metrics_csv, mode='a', header=not metrics_csv.exists(), index=False)

    _save_metrics_summary(records, out_dir)
    print(f"{inference_subdir} inference saved to {out_dir}")


def run_infer_patch_rf(cfg: dict, run_id: str, steps: int, pm_run: str):
    """
    RF inference using 3-channel PM.
    """
    _run_rf_inference(
        cfg, run_id, steps, pm_run,
        pm_inference_subdir='output_inference_pm',
        pm_suffix='pm',
        rf_training_subdir='output_training_rf',
        inference_subdir='output_inference_rf',
        model_name='rf'
    )


def run_infer_patch_rf_t1n(cfg: dict, run_id: str, steps: int, pm_run: str):
    """
    RF inference using T1N-only PM.
    """
    _run_rf_inference(
        cfg, run_id, steps, pm_run,
        pm_inference_subdir='output_inference_pm_t1n',
        pm_suffix='pm_t1n',
        rf_training_subdir='output_training_rf_t1n',
        inference_subdir='output_inference_rf_t1n',
        model_name='rf_t1n'
    )
