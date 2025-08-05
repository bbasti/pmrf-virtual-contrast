import os
import re
from pathlib import Path
import pandas as pd
from typing import Callable, Dict, List, Tuple

from utils.config_io import load_config


def _parse_summary(csv_path: Path, model: str) -> dict:
    df = pd.read_csv(csv_path, index_col=0)
    mean_row = df.loc['mean'] if 'mean' in df.index else pd.Series(dtype=float)
    std_row = df.loc['std'] if 'std' in df.index else pd.Series(dtype=float)
    if model == 'pm':
        mse_k, psnr_k, ssim_k = 'mse_pm', 'psnr_pm', 'ssim_pm'
    elif model == 'pmrf':
        mse_k, psnr_k, ssim_k = 'mse_rf', 'psnr_rf', 'ssim_rf'
    else:
        mse_k, psnr_k, ssim_k = 'mse', 'psnr', 'ssim'
    return {
        'mse': mean_row.get(mse_k, None),
        'psnr': mean_row.get(psnr_k, None),
        'ssim': mean_row.get(ssim_k, None),
        'mse_std': std_row.get(mse_k, None),
        'psnr_std': std_row.get(psnr_k, None),
        'ssim_std': std_row.get(ssim_k, None),
    }


def _parse_fidelity(file_path: Path) -> dict:
    text = file_path.read_text()
    metrics = {}
    key_map = {
        'inception_score_mean': 'isc_mean',
        'inception_score_std': 'isc_std',
        'frechet_inception_distance': 'fid',
        'kernel_inception_distance_mean': 'kid_mean',
        'kernel_inception_distance_std': 'kid_std',
        'precision': 'precision',
        'recall': 'recall',
        'f_score': 'f_score',
    }
    for line in text.splitlines():
        parts = line.strip().split(':', 1)
        if len(parts) != 2:
            continue
        raw_key, raw_val = parts[0].strip(), parts[1].strip()
        if raw_key in key_map:
            try:
                metrics[key_map[raw_key]] = float(raw_val)
            except ValueError:
                metrics[key_map[raw_key]] = None
    return metrics


def _parse_plane_perceptual(eval_dir: Path, prefix: str) -> dict:
    result = {}
    planes = ['axial', 'coronal', 'sagittal']
    for pl in planes:
        fname = f"{prefix}_{pl}.txt"
        fpath = eval_dir / fname
        if fpath.exists():
            m = _parse_fidelity(fpath)
        else:
            m = {k: None for k in
                 ['isc_mean', 'isc_std', 'fid', 'kid_mean', 'kid_std', 'precision', 'recall', 'f_score']}
        for k, v in m.items():
            result[f"{k}_{pl}"] = v
    return result


def _extract_ov_steps(name: str, kind: str) -> Tuple[int, int, str]:
    ov = None
    steps = None
    model_name = None
    if kind in ('pm', 'pmrf'):
        m_ov = re.search(r"ov_(\d+)", name)
        if m_ov:
            ov = int(m_ov.group(1))
        if kind == 'pmrf':
            m_st = re.search(r"steps_(\d+)", name)
            if m_st:
                steps = int(m_st.group(1))
    elif kind == 'baseline':
        m = re.match(r"^(.*?)_ov(\d+)_steps?_(\d+)_", name)
        if m:
            model_name = m.group(1)
            ov = int(m.group(2))
            steps = int(m.group(3))
    return ov, steps, model_name


# ---------------------------
# Aggregation helpers
# ---------------------------

def _build_pm_to_rf_map(base: Path) -> Dict[str, str]:
    """
    Map PM run names to RF run names using 'used_pm_run.txt'.
    """
    mapping = {}
    rf_base = base / 'output_inference_rf'
    if not rf_base.exists():
        return mapping
    for rf_sub in rf_base.iterdir():
        if not rf_sub.is_dir():
            continue
        txt = rf_sub / 'used_pm_run.txt'
        if txt.exists():
            pm_name = txt.read_text().strip()
            mapping[pm_name] = rf_sub.name
    return mapping


def _aggregate_folder(
        base: Path,
        subdir: str,
        kind: str,
        model_key_fn: Callable[[str], str],
        summary_kind_fn: Callable[[str], str],
        plane_prefix_fn: Callable[[str], str]
) -> List[dict]:
    """
    Generic aggregation for a folder of inference runs.
    """
    records = []
    folder = base / subdir
    if not folder.exists():
        return records
    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        ov, st, model_name = _extract_ov_steps(sub.name, kind)
        model_key = model_key_fn(sub.name) or model_name
        summary_kind = summary_kind_fn(sub.name) or model_key
        rec = {'model': model_key, 'overlap': ov, 'steps': st}
        summary = sub / 'metrics_summary.csv'
        if summary.exists():
            rec.update(_parse_summary(summary, summary_kind))
        perf = sub / 'perceptual_evaluation'
        if perf.exists():
            prefix = plane_prefix_fn(sub.name)
            rec.update(_parse_plane_perceptual(perf, prefix))
        records.append(rec)
    return records


def _aggregate_t1n_section(
        base: Path,
        subdir: str,
        model_key: str,
        summary_suffix: str,
        plane_prefix: str,
        default_ov: int
) -> List[dict]:
    """
    Aggregation for T1N-only PM or RF sections.
    """
    records = []
    folder = base / subdir
    if not folder.exists():
        return records
    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        # overlap same for all
        ov = default_ov
        # steps may be encoded differently: try regex
        m = re.search(r"steps?_(\d+)", sub.name)
        st = int(m.group(1)) if m else None

        rec = {'model': model_key, 'overlap': ov, 'steps': st}

        # summary metrics (mse, psnr, ssim)
        summary = sub / 'metrics_summary.csv'
        if summary.exists():
            df = pd.read_csv(summary, index_col=0)
            mean = df.loc['mean'] if 'mean' in df.index else pd.Series(dtype=float)
            std = df.loc['std'] if 'std' in df.index else pd.Series(dtype=float)
            for k in ['mse', 'psnr', 'ssim']:
                rec[k] = mean.get(f"{k}_{summary_suffix}")
                rec[f"{k}_std"] = std.get(f"{k}_{summary_suffix}")

        # perceptual (FID/KID/etc.)
        if model_key == 'pm_t1n':
            # fidelity for pm_t1n lives under the RF-T1N inference folder
            rf_base = base / 'output_inference_rf_t1n'
            pm_to_rf = {}
            if rf_base.exists():
                for rf_sub in rf_base.iterdir():
                    used = rf_sub / 'used_pm_run.txt'
                    if used.exists():
                        pm_run = used.read_text().strip()
                        pm_to_rf[pm_run] = rf_sub.name
            rf_run = pm_to_rf.get(sub.name)
            if rf_run:
                perf = rf_base / rf_run / 'perceptual_evaluation'
                if perf.exists():
                    rec.update(_parse_plane_perceptual(perf, plane_prefix))
        else:
            perf = sub / 'perceptual_evaluation'
            if perf.exists():
                rec.update(_parse_plane_perceptual(perf, plane_prefix))

        records.append(rec)
    return records


def aggregate_evaluations(run_id: str, output_csv: str):
    """
    Aggregate all inference metrics for `run_id` into one CSV.
    """
    base = Path(os.environ.get('RUNS_BASE_DIR', 'runs')) / run_id
    cfg = load_config(run_id)
    default_ov = int(cfg['inference'].get('overlap', int(cfg['inference']['patch_size']) // 2))
    records = []

    # 1) PM patch inference + attach its perceptual from RF
    pm_to_rf = _build_pm_to_rf_map(base)
    pm_folder = base / 'output_inference_pm'
    if pm_folder.exists():
        for sub in pm_folder.iterdir():
            if not sub.is_dir():
                continue
            ov, st, _ = _extract_ov_steps(sub.name, 'pm')
            rec = {'model': 'pm', 'overlap': ov, 'steps': st}
            summary = sub / 'metrics_summary.csv'
            if summary.exists():
                rec.update(_parse_summary(summary, 'pm'))
            # attach perceptual from matching RF
            rf_name = pm_to_rf.get(sub.name)
            if rf_name:
                perf = base / 'output_inference_rf' / rf_name / 'perceptual_evaluation'
                if perf.exists():
                    rec.update(_parse_plane_perceptual(perf, 'fidelity_pm'))
            records.append(rec)

    # 2) PMRF patch inference
    records += _aggregate_folder(
        base,
        'output_inference_rf',
        kind='pmrf',
        model_key_fn=lambda _: 'pmrf',
        summary_kind_fn=lambda _: 'pmrf',
        plane_prefix_fn=lambda _: 'fidelity_rf'
    )

    # 3) Baseline inference
    records += _aggregate_folder(
        base,
        'output_inference_baseline',
        kind='baseline',
        model_key_fn=lambda name: _extract_ov_steps(name, 'baseline')[2],
        summary_kind_fn=lambda name: _extract_ov_steps(name, 'baseline')[2],
        plane_prefix_fn=lambda name: f"fidelity_{name}"
    )

    # 4) T1N-only PM inference
    records += _aggregate_t1n_section(
        base,
        'output_inference_pm_t1n',
        model_key='pm_t1n',
        summary_suffix='pm_t1n',
        plane_prefix='fidelity_pm_t1n',
        default_ov=default_ov
    )

    # 5) T1N-only RF inference
    records += _aggregate_t1n_section(
        base,
        'output_inference_rf_t1n',
        model_key='rf_t1n',
        summary_suffix='rf_t1n',
        plane_prefix='fidelity_rf_t1n',
        default_ov=default_ov
    )

    # write output
    df = pd.DataFrame(records)
    csv_path = base / output_csv
    df.to_csv(csv_path, index=False)
    print(f"Aggregated results written to {csv_path}")
