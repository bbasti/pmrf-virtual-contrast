import os
from pathlib import Path
from datetime import datetime

import torch
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchio as tio

from models.rectified_flow import RectifiedFlowModel3D
from data.dataset import get_splits
from utils.metrics import compute_volume_metrics
from train.loss_functions import _get_input_and_target


def _prepare_inference_dirs(run_id: str, subdir: str, suffix: str):
    """
    Create run_dir, out_dir, volume_out, metrics_csv.
    Returns (run_dir, out_dir, volume_out, metrics_csv).
    """
    runs_base = os.environ.get("RUNS_BASE_DIR", "runs")
    run_dir = Path(runs_base) / run_id
    out_base = run_dir / subdir
    out_base.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"{suffix}_{timestamp}"
    out_dir.mkdir()
    volume_out = out_dir / "output_volumes"
    volume_out.mkdir()
    metrics_csv = out_dir / "metrics.csv"
    return run_dir, out_dir, volume_out, metrics_csv


def _load_test_subjects(run_dir: Path):
    """
    Load all test subjects from run_dir/patch_subjects.
    """
    subj_dir = run_dir / "patch_subjects"
    _, _, test_files = get_splits(str(subj_dir))
    return [torch.load(str(p), weights_only=False) for p in test_files]


def _get_inference_params(cfg: dict):
    """
    Extract patch_size, overlap, batch_size and build common transform.
    """
    patch_size = int(cfg["inference"]["patch_size"])
    overlap = int(cfg["inference"].get("overlap", patch_size // 2))
    batch_size = int(cfg["inference"].get("batch_size", 1))
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample("t1n"),
        tio.ZNormalization(),
        tio.CropOrPad(patch_size, only_pad=True),
    ])
    return transform, patch_size, overlap, batch_size


def _load_rf_model(run_dir: Path, baseline: str, device: torch.device):
    """
    Instantiate and load RF model for given baseline.
    """
    in_ch = {
        "flow_from_x_t1n": 1,
        "cond_x": 4,
        "cond_yhat_pm": 2
    }[baseline]
    rf = RectifiedFlowModel3D(in_channels=in_ch, out_channels=1).to(device)
    rf = torch.compile(rf)
    ckpt = run_dir / "output_training_baselines" / f"{baseline}.pt"
    rf.load_state_dict(torch.load(str(ckpt), map_location=device))
    rf.eval()
    return rf


def _attach_pm_volumes(subjects, run_dir: Path, pm_run: str, pm_subdir: str):
    """
    For cond_yhat_pm, attach pm_pred channel to each subject.
    """
    pm_vol_dir = run_dir / pm_subdir / pm_run / "output_volumes"
    if not pm_vol_dir.exists():
        raise FileNotFoundError(f"PM volumes not found: {pm_vol_dir}")
    for subj in subjects:
        base = Path(subj["t1n"].path).stem.replace("-t1n.nii.gz", "").replace("-t1n.nii", "")
        pm_path = pm_vol_dir / f"{base}_pm.nii.gz"
        subj["pm_pred"] = tio.ScalarImage(str(pm_path))


def _write_metrics_csv(metrics_csv: Path, rec: dict):
    """
    Append a single record to CSV, with header if file did not exist.
    """
    df = pd.DataFrame([rec])
    df.to_csv(metrics_csv, mode="a", header=not metrics_csv.exists(), index=False)


def _write_summary(records: list, out_dir: Path):
    """
    Write summary (mean, std) of records to metrics_summary.csv
    """
    df = pd.DataFrame.from_records(records)
    summary = df.describe().loc[["mean", "std"]]
    summary.to_csv(out_dir / "metrics_summary.csv")


BASELINE_DISPATCH = {
    "flow_from_x_t1n": (
        # init z0 from T1n + noise
        lambda batch, sigma_s, d: batch["t1n"][tio.DATA].to(d) + sigma_s * torch.randn_like(batch["t1n"][tio.DATA]).to(
            d),
        # input is z only
        lambda z, batch, device: z
    ),
    "cond_x": (
        # z0 ~ N(0,I) shaped like t1c
        lambda batch, sigma_s, d: torch.randn_like(batch["t1c"][tio.DATA]).to(d),
        # input is concat([z, x]) where x = concat(T1n,T2w,T2f)
        lambda z, batch, device: torch.cat([
            z,
            *_get_input_and_target(batch, device)[0:1]  # x from helper
        ], dim=1)
    ),
    "cond_yhat_pm": (
        # same unconditional init
        lambda batch, sigma_s, d: torch.randn_like(batch["t1c"][tio.DATA]).to(d),
        # input is concat([z, pm_pred])
        lambda z, batch, device: torch.cat([
            z,
            batch["pm_pred"][tio.DATA].to(device)
        ], dim=1)
    )
}


def run_infer_patch_baseline(
        cfg: dict,
        run_id: str,
        baseline: str,
        steps: int = 100,
        pm_run: str = None
):
    """
    Patch-wise inference for the three PMRF baselines:
      - flow_from_x_t1n
      - cond_x
      - cond_yhat_pm
    """
    if baseline not in BASELINE_DISPATCH:
        raise ValueError(f"Unknown baseline '{baseline}'")

    transform, patch_size, overlap, batch_size = _get_inference_params(cfg)

    # Prepare directories
    suffix = f"{baseline}_ov{overlap}_steps_{steps}"
    run_dir, out_dir, volume_out, metrics_csv = _prepare_inference_dirs(
        run_id, "output_inference_baseline", suffix
    )

    # Load subjects, transform, params
    subjects = _load_test_subjects(run_dir)

    # Attach PM volumes if needed
    if baseline == "cond_yhat_pm":
        if pm_run is None:
            raise RuntimeError("`pm_run` must be supplied for cond_yhat_pm")
        _attach_pm_volumes(subjects, run_dir, pm_run, "output_inference_pm")

    # Load RF model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rf = _load_rf_model(run_dir, baseline, device)

    # Integration hyperparams
    sigma_s = float(cfg["inference"]["sigma_s"])
    dt = 1.0 / steps if steps > 0 else 0.0

    # Get baseline-specific functions
    init_z_fn, build_input_fn = BASELINE_DISPATCH[baseline]

    records = []
    for subj in tqdm(subjects, desc=f"Inference [{baseline}]"):
        subj_t = transform(subj)
        sampler = tio.inference.GridSampler(subj_t, patch_size, overlap)
        loader = tio.SubjectsLoader(sampler, batch_size=batch_size)
        aggregator = tio.inference.GridAggregator(sampler, overlap_mode="hann")

        with torch.no_grad():
            for batch in loader:
                # initialize z
                z = init_z_fn(batch, sigma_s, device)
                # integrate flow
                for i in range(steps):
                    t = torch.full((z.size(0),), i / steps, device=device)
                    inp = build_input_fn(z, batch, device)
                    z = z + rf(inp, t) * dt
                aggregator.add_batch(z, batch[tio.LOCATION])

        # reconstruct and metrics
        rf_vol = aggregator.get_output_tensor().squeeze().cpu().numpy()
        gt = subj_t["t1c"][tio.DATA].squeeze().numpy()
        mse, psnr, ssim = compute_volume_metrics(rf_vol, gt)

        # save volume
        base = Path(subj["t1n"].path).stem.replace("-t1n.nii.gz", "").replace("-t1n.nii", "")
        nib.save(
            nib.Nifti1Image(rf_vol.astype(np.float32), np.eye(4)),
            volume_out / f"{base}_{baseline}.nii.gz"
        )

        # record
        rec = {"subject": base, "mse": mse, "psnr": psnr, "ssim": ssim}
        records.append(rec)
        _write_metrics_csv(metrics_csv, rec)

    # summary
    _write_summary(records, out_dir)
    print(f"[{baseline}] inference complete → {out_dir}")
