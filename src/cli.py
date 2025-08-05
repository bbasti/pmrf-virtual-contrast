import typer

from utils.config_io import load_config, get_run_dir
from data.preprocess import check_and_prepare

# Training
from train.train_pm_patch import train_pm_patch, train_pm_t1n_patch
from train.train_rf_patch import train_rf_patch, train_rf_t1n_patch
from train.train_baseline_patch import (
    train_flow_from_x_t1n,
    train_flow_cond_x,
    train_flow_cond_yhat_pm,
)

# Inference
from inference.inference_patch import (
    run_infer_patch_pm,
    run_infer_patch_rf,
    run_infer_patch_pm_t1n,
    run_infer_patch_rf_t1n,
)
from inference.inference_baseline import run_infer_patch_baseline

# Slices
from utils.slices import (
    generate_slices_real,
    generate_slices_test,
    generate_slices_infer,
    generate_slices_infer_t1n,
)
from utils.slices_baseline import generate_slices_baseline

# Perceptual evaluation
from evaluation.perceptual import (
    run_perceptual_evaluation,
    run_perceptual_evaluation_t1n,
    run_perceptual_evaluation_baseline,
)
# Aggregation
from evaluation.aggregate_evaluations import aggregate_evaluations

app = typer.Typer()


# -- Preprocessing --
@app.command("preprocess")
def preprocess(run_id: str, raw_data: str):
    """Download and preprocess raw data then split into subjects."""
    cfg = load_config(run_id)
    run_dir = get_run_dir(run_id)
    check_and_prepare(run_dir, cfg, raw_data)


# -- Training commands --
@app.command("train-pm")
def train_pm(run_id: str):
    """Train full-channel posterior-mean model on patches."""
    cfg = load_config(run_id)
    train_pm_patch(cfg, run_id)


@app.command("train-pm-t1n")
def train_pm_t1n(run_id: str):
    """Train T1N-only posterior-mean model on patches."""
    cfg = load_config(run_id)
    train_pm_t1n_patch(cfg, run_id)


@app.command("train-rf")
def train_rf(run_id: str):
    """Train full-channel rectified-flow model on patches."""
    cfg = load_config(run_id)
    train_rf_patch(cfg, run_id)


@app.command("train-rf-t1n")
def train_rf_t1n(run_id: str):
    """Train T1N-only rectified-flow model on patches."""
    cfg = load_config(run_id)
    train_rf_t1n_patch(cfg, run_id)


@app.command("train-baseline")
def train_baseline(
        run_id: str,
        baseline: str = typer.Argument(..., help="One of: flow_from_x_t1n, cond_x, cond_yhat_pm"),
):
    """Train a baseline rectified-flow variant on patches."""
    cfg = load_config(run_id)
    if baseline == "flow_from_x_t1n":
        train_flow_from_x_t1n(cfg, run_id)
    elif baseline == "cond_x":
        train_flow_cond_x(cfg, run_id)
    elif baseline == "cond_yhat_pm":
        train_flow_cond_yhat_pm(cfg, run_id)
    else:
        typer.echo(f"Unknown baseline: {baseline}")
        raise typer.Exit(code=1)


# -- Inference commands --
@app.command("infer-pm")
def infer_patch_pm(run_id: str):
    """Run patch-based inference for full-channel PM model."""
    cfg = load_config(run_id)
    run_infer_patch_pm(cfg, run_id)


@app.command("infer-pm-t1n")
def infer_patch_pm_t1n(run_id: str):
    """Run patch-based inference for T1N-only PM model."""
    cfg = load_config(run_id)
    run_infer_patch_pm_t1n(cfg, run_id)


@app.command("infer-rf")
def infer_patch_rf(
        run_id: str,
        steps: int = typer.Argument(..., help="Euler integration steps"),
        pm_run: str = typer.Argument(..., help="Name of PM inference run"),
):
    """Run patch-based RF inference with specified steps and PM outputs."""
    cfg = load_config(run_id)
    run_infer_patch_rf(cfg, run_id, steps, pm_run)


@app.command("infer-rf-t1n")
def infer_patch_rf_t1n(
        run_id: str,
        steps: int = typer.Argument(..., help="Euler integration steps"),
        pm_run: str = typer.Argument(..., help="Name of T1N PM inference run"),
):
    """Run patch-based RF T1N-only inference."""
    cfg = load_config(run_id)
    run_infer_patch_rf_t1n(cfg, run_id, steps, pm_run)


@app.command("infer-baseline")
def infer_baseline(
        run_id: str,
        baseline: str = typer.Argument(..., help="One of: flow_from_x_t1n, cond_x, cond_yhat_pm"),
        steps: int = typer.Argument(50, help="Euler integration steps"),
        pm_run: str = typer.Option(None, help="Needed for 'cond_yhat_pm'"),
):
    """Run patch-based inference for a baseline model."""
    cfg = load_config(run_id)
    run_infer_patch_baseline(cfg, run_id, baseline, steps, pm_run)


# -- Slice generation --
@app.command("slices-real")
def slices_real(run_id: str, n_slices: int = 9, step: int = 10):
    """Generate real FID slices (T1C and T1N) from training volumes."""
    generate_slices_real(run_id, n_slices, step)


@app.command("slices-test")
def slices_test(run_id: str, n_slices: int = 9, step: int = 10):
    """Generate real FID slices (all modalities) from test volumes."""
    generate_slices_test(run_id, n_slices, step)


@app.command("slices-infer")
def slices_infer(run_id: str, infer_run: str, n_slices: int = 9, step: int = 10):
    """Generate inference FID slices from PM and RF volumes."""
    generate_slices_infer(run_id, infer_run, n_slices, step)


@app.command("slices-infer-t1n")
def slices_infer_t1n(run_id: str, infer_run: str, n_slices: int = 9, step: int = 10):
    """Generate inference FID slices from T1N-only outputs."""
    generate_slices_infer_t1n(run_id, infer_run, n_slices, step)


@app.command("slices-baseline")
def slices_baseline(run_id: str, infer_run: str, n_slices: int = 9, step: int = 10):
    """Generate FID slices from baseline inference volumes."""
    generate_slices_baseline(run_id, infer_run, n_slices, step)


# -- Perceptual evaluation --
@app.command("eval-perceptual")
def eval_perceptual(
        run_id: str,
        infer_run: str,
        gpu: str,
        samples_resize_and_crop: str = "512",
):
    """Run ISC/FID/KID/PRC evaluation between real and synthetic slices."""
    run_perceptual_evaluation(run_id, infer_run, gpu, samples_resize_and_crop)


@app.command("eval-perceptual-t1n")
def eval_perceptual_t1n(
        run_id: str, infer_run: str, gpu: str, samples_resize_and_crop: str = "512"
):
    """Run ISC/FID/KID/PRC evaluation on RF-T1N baseline slices."""
    run_perceptual_evaluation_t1n(run_id, infer_run, gpu, samples_resize_and_crop)


@app.command("eval-perceptual-baseline")
def eval_perceptual_baseline(
        run_id: str, infer_run: str, gpu: str, samples_resize_and_crop: str = "512"
):
    """Run ISC/FID/KID/PRC evaluation on baseline slices."""
    run_perceptual_evaluation_baseline(run_id, infer_run, gpu, samples_resize_and_crop)


# -- Aggregation --
@app.command("aggregate-evaluations")
def aggregate_evaluations_cmd(
        run_id: str,
        output_csv: str = typer.Argument(..., help="Path to write aggregated CSV"),
):
    """Aggregate all inference and perceptual metrics into a single CSV."""
    aggregate_evaluations(run_id, output_csv)


if __name__ == "__main__":
    app()
