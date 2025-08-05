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
from train.loss_functions import (
    compute_loss_rf_flow_from_x_t1n,
    compute_loss_rf_cond_x,
    compute_loss_rf_cond_yhat_pm,
)
from models.rectified_flow import RectifiedFlowModel3D
from models.posterior_mean import PosteriorMeanModel3D


def _train_flow_common(
        cfg: dict,
        run_id: str,
        rf_builder,
        loss_fn,
        output_subdir: str,
        checkpoint_name: str,
        project_name: str,
):
    """
    Generic helper for rectified-flow baselines training.
    """
    runs_base = os.environ.get("RUNS_BASE_DIR", "runs")
    run_dir = os.path.join(runs_base, run_id)

    # Build datasets and loaders
    train_ds, val_ds, test_ds = build_subjects_dataset(run_dir)
    batch_size = cfg['rectified_flow']['batch_size']
    samples_per_vol = cfg['data']['samples_per_volume']
    patch_size = cfg['data']['patch_size']
    train_loader = build_patch_loader(train_ds, samples_per_vol, patch_size, batch_size, training=True)
    val_loader = build_patch_loader(val_ds, samples_per_vol, patch_size, batch_size, training=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build RF model and extras (e.g., sigma, frozen PM)
    built = rf_builder(device, run_dir)
    rf, *extras = built if isinstance(built, tuple) else (built,)
    optimizer = AdamW(rf.parameters(), lr=float(cfg['rectified_flow']['lr']))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(cfg['rectified_flow']['num_epochs']))
    rf = torch.compile(rf)

    # Output directory
    out_dir = os.path.join(run_dir, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, checkpoint_name)

    # Run training loop
    run_training_loop(
        rf,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        lambda m, batch: loss_fn(m, batch, *extras, device),
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

    # Evaluation
    test_loader = build_patch_loader(test_ds, samples_per_vol, patch_size, batch_size, training=False)
    test_loss = evaluate_model(rf, test_loader, lambda m, batch: loss_fn(m, batch, *extras, device))
    print(f"Test loss ({project_name}): {test_loss:.4f}")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()


def train_flow_from_x_t1n(cfg: dict, run_id: str):
    """
    Unconditional flow starting from T1n + noise.
    """

    def builder(device, run_dir):
        rf = RectifiedFlowModel3D(in_channels=1, out_channels=1).to(device)
        sigma_s = float(cfg['rectified_flow']['sigma_s'])
        return rf, sigma_s

    _train_flow_common(
        cfg,
        run_id,
        rf_builder=builder,
        loss_fn=compute_loss_rf_flow_from_x_t1n,
        output_subdir='output_training_baselines',
        checkpoint_name='flow_from_x_t1n.pt',
        project_name='volumetric-rf-flow_from_x_t1n',
    )


def train_flow_cond_x(cfg: dict, run_id: str):
    """
    Conditional flow appending full x at each step.
    """

    def builder(device, run_dir):
        rf = RectifiedFlowModel3D(in_channels=4, out_channels=1).to(device)
        return rf

    _train_flow_common(
        cfg,
        run_id,
        rf_builder=builder,
        loss_fn=compute_loss_rf_cond_x,
        output_subdir='output_training_baselines',
        checkpoint_name='cond_x.pt',
        project_name='volumetric-rf-cond_x',
    )


def train_flow_cond_yhat_pm(cfg: dict, run_id: str):
    """
    Conditional flow using PM estimate as conditioning.
    """

    def builder(device, run_dir):
        # load frozen posterior mean
        pm = PosteriorMeanModel3D(in_channels=3, out_channels=1).to(device)
        pm = torch.compile(pm)
        ckpt = os.path.join(run_dir, 'output_training_pm', cfg['posterior_mean']['save_path'])
        pm.load_state_dict(torch.load(ckpt, map_location=device))
        pm.eval()
        rf = RectifiedFlowModel3D(in_channels=2, out_channels=1).to(device)
        return rf, pm

    _train_flow_common(
        cfg,
        run_id,
        rf_builder=builder,
        loss_fn=compute_loss_rf_cond_yhat_pm,
        output_subdir='output_training_baselines',
        checkpoint_name='cond_yhat_pm.pt',
        project_name='volumetric-rf-cond_yhat_pm',
    )
