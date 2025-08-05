import os
import time
import torch
import wandb
from tqdm import tqdm
import torchio as tio

from data.dataset import get_splits


def build_subjects_dataset(run_dir, patch_size=64):
    # Load all subjects
    subj_dir = os.path.join(run_dir, 'patch_subjects')
    train_files, val_files, test_files = get_splits(subj_dir)

    # Load each .pt back to a Subject
    train_subs = [torch.load(p, weights_only=False) for p in train_files]
    val_subs = [torch.load(p, weights_only=False) for p in val_files]
    test_subs = [torch.load(p, weights_only=False) for p in test_files]

    # Compose transforms
    transforms = [tio.ToCanonical(),
                  tio.Resample('t1n'),
                  tio.ZNormalization(),
                  tio.CropOrPad(
                      patch_size,
                      only_pad=True,
                  )]
    transform = tio.Compose(transforms)

    train_ds = tio.SubjectsDataset(train_subs, transform=transform)
    val_ds = tio.SubjectsDataset(val_subs, transform=transform)
    test_ds = tio.SubjectsDataset(test_subs, transform=transform)

    return train_ds, val_ds, test_ds


def build_patch_loader(subjects_ds, samples_per_volume=16, patch_size=32, batch_size=64, training=False):
    print(
        f"Building patch loader with samples per volume {samples_per_volume}, patch size {patch_size}, batch size {batch_size}, training {training}")

    patches_queue = tio.Queue(
        subjects_dataset=subjects_ds,
        max_length=batch_size * samples_per_volume,
        samples_per_volume=samples_per_volume,
        sampler=tio.data.UniformSampler(patch_size),
        shuffle_subjects=training,
        shuffle_patches=training,
        start_background=training,
        num_workers=16
    )

    return tio.SubjectsLoader(
        patches_queue,
        batch_size=batch_size,
        num_workers=0
    )


def prepare_patch_data_loaders(cfg, train_ds, val_ds):
    train_loader = build_patch_loader(train_ds,
                                      cfg['data']['samples_per_volume'],
                                      cfg['data']['patch_size'],
                                      cfg['posterior_mean']['batch_size'],
                                      training=True)
    val_loader = build_patch_loader(val_ds,
                                    cfg['data']['samples_per_volume'],
                                    cfg['data']['patch_size'],
                                    cfg['posterior_mean']['batch_size'],
                                    training=False)
    return train_loader, val_loader


def run_training_loop(model, optimizer, scheduler, train_loader, val_loader,
                      compute_loss_fn, patience, num_epochs, save_path_best, project_name, data_info):
    """
    Generic training loop with early stopping, checkpointing, and detailed logging.
    Saves both the best and latest model every epoch.
    Logs metrics and epoch timings to CLI (with tqdm) and to wandb.
    """
    wandb_dir = os.path.abspath(os.path.dirname(save_path_best))

    # Initialize WandB
    wandb.init(project=project_name,
               dir=wandb_dir,
               config={"max_epochs": num_epochs, "patience": patience, "data_info": data_info})
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Epoch loop with tqdm for elapsed/remaining time
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", unit="epoch"):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = compute_loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = sum(train_losses) / len(train_losses)

        # Step LR scheduler
        scheduler.step()

        # Validation Phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(compute_loss_fn(model, batch).item())
        val_loss = sum(val_losses) / len(val_losses)

        # Compute epoch duration
        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            latest_path = save_path_best.replace('.pt', '_latest.pt')
            torch.save(model.state_dict(), latest_path)
            wandb.save(latest_path)
            print(f"Latest model checkpoint saved to {latest_path}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path_best)
            wandb.save(save_path_best)
            print(f"New best model saved to {save_path_best}")
        else:
            epochs_without_improvement += 1

        # Log to WandB
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time_s': epoch_duration,
            'epochs_no_improvement': epochs_without_improvement,
        }, step=epoch)

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"after {patience} epochs without improvement."
            )
            break

    wandb.finish()
    return best_val_loss


def evaluate_model(model, test_loader, compute_loss_fn):
    """
    Evaluate the model on the test set and compute the average loss.
    """
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in test_loader:
            test_losses.append(compute_loss_fn(model, batch).item())
    test_loss = sum(test_losses) / len(test_losses)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss
