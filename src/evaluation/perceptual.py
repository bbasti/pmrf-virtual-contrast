import os
import subprocess
import shlex
from pathlib import Path
from typing import Dict, List

from utils.config_io import get_run_dir


def _run_fidelity(
        real_base: Path,
        synth_roots: Dict[str, Path],
        out_dir: Path,
        gpu: str,
        samples_resize_and_crop: str,
        planes: List[str]
):
    """
    Run fidelity metrics (ISC/FID/KID/PRC) between real and synthetic slices.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for pl in planes:
        real_dir = real_base / pl
        if not real_dir.exists():
            raise FileNotFoundError(f"Real slices not found for plane {pl}: {real_dir}")
        for model, synth_root in synth_roots.items():
            synth_dir = synth_root / pl
            if not synth_dir.exists():
                raise FileNotFoundError(f"Synthetic slices not found for {model} plane {pl}: {synth_dir}")
            outfile = out_dir / f'fidelity_{model}_{pl}.txt'
            cmd = [
                'fidelity',
                '--gpu', gpu,
                '--samples-resize-and-crop', samples_resize_and_crop,
                '--input1', str(real_dir),
                '--input2', str(synth_dir),
                # WORKAROUND: Uncomment this line if working with very small datasets (between 12 and 111 test subjects)
                # '--kid-subset-size', "100",
                '--isc', '--fid', '--kid', '--prc',
            ]
            print(f"\n→ Running fidelity for {model}, plane={pl}:")
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running fidelity for {model}/{pl}: {e.stderr}")
                continue
            with open(outfile, 'w') as f:
                f.write(proc.stdout)
            print(f"Output saved to {outfile}")
    print("\nAll fidelity runs completed.")


def run_perceptual_evaluation_baseline(
        run_id: str,
        infer_run: str,
        gpu: str,
        samples_resize_and_crop: str
):
    """
    Evaluate baseline fidelity on precomputed baseline slices.

    Baseline slices are expected under:
      runs/<run_id>/fid_slices_baseline/<infer_run>/<plane>/*.png
    """
    run_dir = get_run_dir(run_id)
    real_base = run_dir / 'fid_slices' / 't1c'
    synth_root = run_dir / 'fid_slices_baseline' / infer_run
    out_dir = run_dir / 'output_inference_baseline' / infer_run / 'perceptual_evaluation'
    planes = ['axial', 'coronal', 'sagittal']
    _run_fidelity(real_base, {infer_run: synth_root}, out_dir, gpu, samples_resize_and_crop, planes)


def run_perceptual_evaluation(
        run_id: str,
        infer_run: str,
        gpu: str,
        samples_resize_and_crop: str
):
    """
    Evaluate fidelity on patch-based PM vs RF slices.

    Synthetic slices under:
      runs/<run_id>/output_inference_rf/<infer_run>/slices/pm/<plane>
      runs/<run_id>/output_inference_rf/<infer_run>/slices/rf/<plane>
    """
    run_dir = get_run_dir(run_id)
    real_base = run_dir / 'fid_slices' / 't1c'
    pm_root = run_dir / 'output_inference_rf' / infer_run / 'slices' / 'pm'
    rf_root = run_dir / 'output_inference_rf' / infer_run / 'slices' / 'rf'
    out_dir = run_dir / 'output_inference_rf' / infer_run / 'perceptual_evaluation'
    planes = ['axial', 'coronal', 'sagittal']
    _run_fidelity(real_base, {'pm': pm_root, 'rf': rf_root}, out_dir, gpu, samples_resize_and_crop, planes)


def run_perceptual_evaluation_t1n(
        run_id: str,
        infer_run: str,
        gpu: str,
        samples_resize_and_crop: str
):
    """
    Evaluate fidelity on RF-T1N only slices.

    Synthetic slices under:
      runs/<run_id>/output_inference_rf_t1n/<infer_run>/slices/pm_t1n/<plane>
      runs/<run_id>/output_inference_rf_t1n/<infer_run>/slices/rf_t1n/<plane>
    """
    run_dir = get_run_dir(run_id)
    real_base = run_dir / 'fid_slices' / 't1c'
    pm_root = run_dir / 'output_inference_rf_t1n' / infer_run / 'slices' / 'pm_t1n'
    rf_root = run_dir / 'output_inference_rf_t1n' / infer_run / 'slices' / 'rf_t1n'
    out_dir = run_dir / 'output_inference_rf_t1n' / infer_run / 'perceptual_evaluation'
    planes = ['axial', 'coronal', 'sagittal']
    _run_fidelity(real_base, {'pm_t1n': pm_root, 'rf_t1n': rf_root}, out_dir, gpu, samples_resize_and_crop, planes)
