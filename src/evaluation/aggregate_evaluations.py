import os
import re
from pathlib import Path
import pandas as pd
from typing import Callable, List, Optional, Tuple

from utils.config_io import load_config


def _summary_keys(spec: Optional[str]) -> Tuple[str, str, str]:
    """
    Build (mse_key, psnr_key, ssim_key) for a metrics_summary.csv,
    given a spec:
      - None        -> plain columns: mse, psnr, ssim
      - "pm"        -> *_pm
      - "pmrf"      -> *_rf
      - any string  -> treated as suffix: f"{k}_{spec}"
    """
    if spec is None:
        return "mse", "psnr", "ssim"
    if spec == "pm":
        return "mse_pm", "psnr_pm", "ssim_pm"
    if spec == "pmrf":
        return "mse_rf", "psnr_rf", "ssim_rf"
    return f"mse_{spec}", f"psnr_{spec}", f"ssim_{spec}"


def _parse_summary(csv_path: Path, spec: Optional[str]) -> dict:
    df = pd.read_csv(csv_path, index_col=0)
    mean_row = df.loc["mean"] if "mean" in df.index else pd.Series(dtype=float)
    std_row = df.loc["std"] if "std" in df.index else pd.Series(dtype=float)
    mse_k, psnr_k, ssim_k = _summary_keys(spec)
    return {
        "mse": mean_row.get(mse_k, None),
        "psnr": mean_row.get(psnr_k, None),
        "ssim": mean_row.get(ssim_k, None),
        "mse_std": std_row.get(mse_k, None),
        "psnr_std": std_row.get(psnr_k, None),
        "ssim_std": std_row.get(ssim_k, None),
    }


def _parse_fidelity(file_path: Path) -> dict:
    text = file_path.read_text()
    metrics = {}
    key_map = {
        "inception_score_mean": "isc_mean",
        "inception_score_std": "isc_std",
        "frechet_inception_distance": "fid",
        "kernel_inception_distance_mean": "kid_mean",
        "kernel_inception_distance_std": "kid_std",
        "precision": "precision",
        "recall": "recall",
        "f_score": "f_score",
    }
    for line in text.splitlines():
        parts = line.strip().split(":", 1)
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
    planes = ["axial", "coronal", "sagittal"]
    missing = {k: None for k in ["isc_mean", "isc_std", "fid", "kid_mean", "kid_std", "precision", "recall", "f_score"]}
    for pl in planes:
        fpath = eval_dir / f"{prefix}_{pl}.txt"
        m = _parse_fidelity(fpath) if fpath.exists() else missing
        for k, v in m.items():
            result[f"{k}_{pl}"] = v
    return result


def _find_matched_perf_dir_exact(
        base: Path,
        rf_subdir: str,
        used_pm_run_dirname: str,
        fidelity_prefix: str,
) -> Optional[Path]:
    """
    Search under `rf_subdir` for runs whose `used_pm_run.txt` content exactly equals
    `used_pm_run_dirname` and which contain the three required fidelity files:
      perceptual_evaluation/{fidelity_prefix}_axial.txt
      perceptual_evaluation/{fidelity_prefix}_coronal.txt
      perceptual_evaluation/{fidelity_prefix}_sagittal.txt

    Return the most recent matching perceptual_evaluation directory, or None.
    """
    rf_base = base / rf_subdir
    if not rf_base.exists():
        return None

    candidates: List[Path] = []
    required_names = [f"{fidelity_prefix}_axial.txt", f"{fidelity_prefix}_coronal.txt",
                      f"{fidelity_prefix}_sagittal.txt"]

    for rf_sub in rf_base.iterdir():
        if not rf_sub.is_dir():
            continue
        used = rf_sub / "used_pm_run.txt"
        perf = rf_sub / "perceptual_evaluation"
        if not used.exists() or not perf.exists():
            continue
        if used.read_text().strip() != used_pm_run_dirname:
            continue
        if all((perf / r).exists() for r in required_names):
            candidates.append(perf)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _extract_ov_steps(name: str, kind: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    ov = None
    steps = None
    model_name = None
    if kind in ("pm", "pmrf"):
        m_ov = re.search(r"ov_(\d+)", name)
        if m_ov:
            ov = int(m_ov.group(1))
        if kind == "pmrf":
            m_st = re.search(r"steps_(\d+)", name)
            if m_st:
                steps = int(m_st.group(1))
    elif kind == "baseline":
        m = re.match(r"^(.*?)_ov(\d+)_steps?_(\d+)_", name)
        if m:
            model_name = m.group(1)
            ov = int(m.group(2))
            steps = int(m.group(3))
    return ov, steps, model_name


def _aggregate_folder(
        base: Path,
        subdir: str,
        kind: str,
        model_key_fn: Callable[[str], str],
        summary_spec_fn: Callable[[str], Optional[str]],
        plane_prefix_fn: Callable[[str], str],
) -> List[dict]:
    """
    Generic aggregation for a folder of inference runs.
    `summary_spec_fn` returns the spec for _parse_summary (see _summary_keys docstring).
    """
    records: List[dict] = []
    folder = base / subdir
    if not folder.exists():
        return records

    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        ov, st, model_name = _extract_ov_steps(sub.name, kind)
        model_key = model_key_fn(sub.name) or model_name
        summary_spec = summary_spec_fn(sub.name)

        rec = {"model": model_key, "overlap": ov, "steps": st}

        summary = sub / "metrics_summary.csv"
        if summary.exists():
            rec.update(_parse_summary(summary, summary_spec))

        perf = sub / "perceptual_evaluation"
        if perf.exists():
            rec.update(_parse_plane_perceptual(perf, plane_prefix_fn(sub.name)))

        records.append(rec)
    return records


def _aggregate_t1n_section(
        base: Path,
        subdir: str,
        model_key: str,
        summary_spec: Optional[str],
        plane_prefix: str,
        default_ov: int,
) -> List[dict]:
    """
    Aggregation for T1N-only sections
    """
    records: List[dict] = []
    folder = base / subdir
    if not folder.exists():
        return records

    for sub in folder.iterdir():
        if not sub.is_dir():
            continue
        m = re.search(r"steps?_(\d+)", sub.name)
        st = int(m.group(1)) if m else None

        rec = {"model": model_key, "overlap": default_ov, "steps": st}

        summary = sub / "metrics_summary.csv"
        if summary.exists():
            rec.update(_parse_summary(summary, summary_spec))

        perf = sub / "perceptual_evaluation"
        if perf.exists():
            rec.update(_parse_plane_perceptual(perf, plane_prefix))

        records.append(rec)
    return records


def aggregate_evaluations(run_id: str, output_csv: str):
    """
    Aggregate all inference metrics for `run_id` into one CSV.
    """
    base = Path(os.environ.get("RUNS_BASE_DIR", "runs")) / run_id
    cfg = load_config(run_id)
    default_ov = int(cfg["inference"].get("overlap", int(cfg["inference"]["patch_size"]) // 2))
    records: List[dict] = []

    # 1) PM patch inference + perceptual from matched RF (exact)
    pm_folder = base / "output_inference_pm"
    if pm_folder.exists():
        for sub in pm_folder.iterdir():
            if not sub.is_dir():
                continue
            ov, st, _ = _extract_ov_steps(sub.name, "pm")
            rec = {"model": "pm", "overlap": ov, "steps": st}

            summary = sub / "metrics_summary.csv"
            if summary.exists():
                rec.update(_parse_summary(summary, "pm"))

            perf_dir = _find_matched_perf_dir_exact(
                base=base,
                rf_subdir="output_inference_rf",
                used_pm_run_dirname=sub.name,
                fidelity_prefix="fidelity_pm",
            )
            if perf_dir:
                rec.update(_parse_plane_perceptual(perf_dir, "fidelity_pm"))

            records.append(rec)

    # 2) PMRF patch inference (direct)
    records += _aggregate_folder(
        base=base,
        subdir="output_inference_rf",
        kind="pmrf",
        model_key_fn=lambda _: "pmrf",
        summary_spec_fn=lambda _: "pmrf",  # -> *_rf
        plane_prefix_fn=lambda _: "fidelity_rf",
    )

    # 3) Baseline inference (direct)
    records += _aggregate_folder(
        base=base,
        subdir="output_inference_baseline",
        kind="baseline",
        model_key_fn=lambda name: _extract_ov_steps(name, "baseline")[2],
        summary_spec_fn=lambda _: None,  # plain mse/psnr/ssim
        plane_prefix_fn=lambda name: f"fidelity_{name}",
    )

    # 4) T1N-only PM inference: summary from its folder, perceptual from matched RF_T1N (exact)
    pm_t1n_folder = base / "output_inference_pm_t1n"
    if pm_t1n_folder.exists():
        for sub in pm_t1n_folder.iterdir():
            if not sub.is_dir():
                continue
            m = re.search(r"steps?_(\d+)", sub.name)
            st = int(m.group(1)) if m else None

            rec = {"model": "pm_t1n", "overlap": default_ov, "steps": st}

            summary = sub / "metrics_summary.csv"
            if summary.exists():
                rec.update(_parse_summary(summary, "pm_t1n"))

            perf_dir = _find_matched_perf_dir_exact(
                base=base,
                rf_subdir="output_inference_rf_t1n",
                used_pm_run_dirname=sub.name,
                fidelity_prefix="fidelity_pm_t1n",
            )
            if perf_dir:
                rec.update(_parse_plane_perceptual(perf_dir, "fidelity_pm_t1n"))

            records.append(rec)

    # 5) T1N-only RF inference (direct)
    records += _aggregate_t1n_section(
        base=base,
        subdir="output_inference_rf_t1n",
        model_key="rf_t1n",
        summary_spec="rf_t1n",
        plane_prefix="fidelity_rf_t1n",
        default_ov=default_ov,
    )

    # write output
    df = pd.DataFrame(records)
    csv_path = base / output_csv
    df.to_csv(csv_path, index=False)
    print(f"Aggregated results written to {csv_path}")
