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
    prepare_patch_data_loaders,
)
from train.loss_functions import compute_loss_pm, compute_loss_pm_t1n
from models.posterior_mean import PosteriorMeanModel3D


def _train_patch_common(cfg: dict,
                        run_id: str,
                        in_channels: int,
                        compute_loss_fn,
                        output_subdir: str,
                        project_name: str):
    """
    Internal helper to train any posterior-mean patch model.
    """
    # Setup directories and device
    runs_base = os.environ.get("RUNS_BASE_DIR", "runs")
    run_dir = os.path.join(runs_base, run_id)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Building subjects dataset...")
    train_ds, val_ds, test_ds = build_subjects_dataset(run_dir)
    train_loader, val_loader = prepare_patch_data_loaders(cfg, train_ds, val_ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, optimizer, scheduler
    model = PosteriorMeanModel3D(in_channels=in_channels, out_channels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["posterior_mean"]["lr"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg["posterior_mean"]["num_epochs"]))

    # Prepare output paths
    out_dir = os.path.join(run_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, cfg["posterior_mean"]["save_path"])

    # Compile and train
    print("Preparing model compilation...")
    model = torch.compile(model)
    print("Running training loop...")
    run_training_loop(
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        lambda m, batch: compute_loss_fn(m, batch, device),
        cfg["posterior_mean"]["patience"],
        cfg["posterior_mean"]["num_epochs"],
        save_path_best=best_path,
        project_name=project_name,
        data_info={
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'batch_size': cfg['posterior_mean']['batch_size'],
        },
    )

    # Evaluate on test set
    test_loader = build_patch_loader(
        test_ds,
        cfg['data']['samples_per_volume'],
        cfg['data']['patch_size'],
        cfg['posterior_mean']['batch_size'],
        training=False
    )
    test_loss = evaluate_model(
        model,
        test_loader,
        lambda m, batch: compute_loss_fn(m, batch, device)
    )
    print(f"Test loss ({project_name}): {test_loss:.4f}")

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()


def train_pm_patch(cfg: dict, run_id: str):
    """
    Train posterior-mean model with full (3‑channel) input (T1N, T2W, T2F).
    """
    _train_patch_common(
        cfg,
        run_id,
        in_channels=3,
        compute_loss_fn=compute_loss_pm,
        output_subdir="output_training_pm",
        project_name="volumetric-pm-patch",
    )


def train_pm_t1n_patch(cfg: dict, run_id: str):
    """
    Train posterior-mean model using only T1N as input (1‑channel).
    """
    _train_patch_common(
        cfg,
        run_id,
        in_channels=1,
        compute_loss_fn=compute_loss_pm_t1n,
        output_subdir="output_training_pm_t1n",
        project_name="volumetric-pm-t1n",
    )
