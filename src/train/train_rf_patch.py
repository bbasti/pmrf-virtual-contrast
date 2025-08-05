import os
import gc
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from train.train_utils import (
    run_training_loop,
    evaluate_model,
    build_subjects_dataset,
    build_patch_loader,
)
from train.loss_functions import compute_loss_rf, compute_loss_rf_t1n
from models.posterior_mean import PosteriorMeanModel3D
from models.rectified_flow import RectifiedFlowModel3D


def _train_rf_common(
        cfg: dict,
        run_id: str,
        pm_in_channels: int,
        pm_subdir: str,
        compute_loss_fn,
        output_subdir: str,
        project_name: str,
):
    """
    Generic helper for patch-based rectified-flow training.

    - pm_in_channels: number of channels for loading the frozen PM
    - pm_subdir: directory under run_dir where PM checkpoint is stored
    - compute_loss_fn: either compute_loss_rf or compute_loss_rf_t1n
    - output_subdir: where to save RF outputs
    - project_name: wandb project name
    """
    # Paths & Dataset
    runs_base = os.environ.get("RUNS_BASE_DIR", "runs")
    run_dir = os.path.join(runs_base, run_id)
    train_ds, val_ds, test_ds = build_subjects_dataset(run_dir)

    # Data loaders
    batch_size = cfg['rectified_flow']['batch_size']
    samples_per_volume = cfg['data']['samples_per_volume']
    patch_size = cfg['data']['patch_size']
    train_loader = build_patch_loader(
        train_ds, samples_per_volume, patch_size, batch_size, training=True
    )
    val_loader = build_patch_loader(
        val_ds, samples_per_volume, patch_size, batch_size, training=False
    )

    # Device & hyperparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma_s = float(cfg['rectified_flow']['sigma_s'])
    print(f"Using device: {device}")

    # Load frozen posterior mean
    pm = PosteriorMeanModel3D(in_channels=pm_in_channels, out_channels=1).to(device)
    pm = torch.compile(pm)
    pm_ckpt = os.path.join(run_dir, pm_subdir, cfg['posterior_mean']['save_path'])
    pm.load_state_dict(torch.load(pm_ckpt, map_location=device))
    pm.eval()

    # Initialize RF model
    rf = RectifiedFlowModel3D(in_channels=1, out_channels=1).to(device)
    optimizer = AdamW(rf.parameters(), lr=float(cfg['rectified_flow']['lr']))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg['rectified_flow']['num_epochs']))
    rf = torch.compile(rf)

    # Prepare output
    out_dir = os.path.join(run_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, cfg['rectified_flow']['save_path'])

    # Train
    run_training_loop(
        rf,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        lambda m, batch: compute_loss_fn(m, batch, pm, sigma_s, device),
        cfg['rectified_flow']['patience'],
        cfg['rectified_flow']['num_epochs'],
        save_path_best=save_path,
        project_name=project_name,
        data_info={
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'batch_size': batch_size,
        },
    )

    # Test
    test_loader = build_patch_loader(
        test_ds, samples_per_volume, patch_size, batch_size, training=False
    )
    test_loss = evaluate_model(
        rf,
        test_loader,
        lambda m, batch: compute_loss_fn(m, batch, pm, sigma_s, device)
    )
    print(f"Test loss ({project_name}): {test_loss:.4f}")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


def train_rf_patch(cfg: dict, run_id: str):
    """
    Train rectified-flow model fine-tuning a full-channel PM.
    """
    _train_rf_common(
        cfg,
        run_id,
        pm_in_channels=3,
        pm_subdir='output_training_pm',
        compute_loss_fn=compute_loss_rf,
        output_subdir='output_training_rf',
        project_name='volumetric-rf-patch',
    )


def train_rf_t1n_patch(cfg: dict, run_id: str):
    """
    Train rectified-flow model fine-tuning a T1N-only PM.
    """
    _train_rf_common(
        cfg,
        run_id,
        pm_in_channels=1,
        pm_subdir='output_training_pm_t1n',
        compute_loss_fn=compute_loss_rf_t1n,
        output_subdir='output_training_rf_t1n',
        project_name='volumetric-rf-t1n',
    )
