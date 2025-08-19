# Posterior-Mean Rectified Flow for Realistic & Accurate Virtual Contrast MRI

[![arXiv](https://img.shields.io/badge/arXiv-2508.12640-b31b1b.svg)](https://arxiv.org/abs/2508.12640)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2508.12640-blue.svg)](https://doi.org/10.48550/arXiv.2508.12640)

A CLI toolkit for training, inference, slice generation, and evaluation of volumetric posterior-mean rectified-flow and
baseline models for virtual contrast enhancement on 3D MRI data.

---

##  General Introduction

> This repository accompanies the following paper:
>
> **“Synthesizing Accurate and Realistic T1-weighted Contrast-Enhanced MR Images using Posterior-Mean Rectified Flow”**  
> *Bastian Brandstötter & Erich Kobler*  
> [arXiv](https://arxiv.org/abs/2508.12640) · [DOI: 10.48550/arXiv.2508.12640](https://doi.org/10.48550/arXiv.2508.12640)

The **PMRF Pipeline** provides end-to-end support for:

* **Preprocessing**: Downloading raw data from Synapse and organizing into training, validation, and test splits.
* **Training**:

    * **Posterior-Mean (PM)** models (full-channel or T1N-only).
    * **Rectified-Flow (RF)** models fine-tuned on PM outputs (full-channel or T1N-only).
    * **Baseline RF** variants (unconditional, conditional on inputs or on PM estimates).
* **Inference**: Patch-based reconstruction of full volumes using trained PM/RF models.
* **Slice Generation & Evaluation**: Extract PNG slices and compute perceptual metrics (FID/KID/etc.).

---

## ⚙️ Environment Setup & Installation

### 1. Create & Activate Conda Environment

```bash
conda create -n pmrf_env python=3.9 -y
conda activate pmrf_env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
# Install this package in editable mode
pip install -e .
```

#### 🚨 Torch-Fidelity Notice

The perceptual evaluation relies on **torch-fidelity ≥ 0.4.0** for the `--samples_resize_and_crop` flag. If you
encounter errors during fidelity runs, install the latest version from the master branch:

```bash
pip install -e git+https://github.com/toshas/torch-fidelity.git@master#egg=torch-fidelity
```

---

## 🔧 Environment Variables

* `SYNAPSE_AUTH_TOKEN`
  Required to download raw archives from Synapse.

* `RUNS_BASE_DIR`  (optional, default: `runs`)
  Base directory under which all run artifacts are stored.

---

## ⚙️ Configuration

Before running any commands, copy and customize the provided base configuration:

```bash
cp config/base.yaml <RUNS_BASE_DIR>/<run_id>/config.yaml
```

Edit `<RUNS_BASE_DIR>/<run_id>/config.yaml` to adjust parameters. Below are the entries from `base.yaml` and their
meanings:

### `data` section

* `random_seed`: Integer seed for reproducible train/val/test splits.
* `synapse_ids`: List of Synapse IDs to download raw datasets.
* `patch_size`: Size (in voxels) of cubic patches for training/inference.
* `samples_per_volume`: Number of patches sampled per volume per epoch.
* `train_ratio`: Fraction of subjects used for training split.
* `val_ratio`: Fraction of subjects used for validation split.

### `posterior_mean` section

* `save_path`: Filename to save the best PM model (e.g., `posterior_mean_model_3d.pth`).
* `num_epochs`: Maximum epochs for PM training.
* `lr`: Learning rate for PM optimizer.
* `batch_size`: Batch size for PM patch training.
* `patience`: Early-stopping patience (epochs without val-loss improvement).

### `rectified_flow` section

* `save_path`: Filename to save the best RF model (e.g., `pmrf_model_3d.pth`).
* `sigma_s`: Noise scale used in flow loss and inference.
* `num_epochs`: Maximum epochs for RF training.
* `lr`: Learning rate for RF optimizer.
* `batch_size`: Batch size for RF patch training.
* `patience`: Early-stopping patience.

### `inference` section

* `batch_size`: Batch size for patch-wise inference.
* `overlap`: Overlap (in voxels) between patches during aggregation.
* `patch_size`: Patch size used at inference (should match training).
* `sigma_s`: Noise scale applied at the start of RF inference.

---

## 🚀 How to Use

All commands are exposed via the `pmrf_pipeline` entry point:

```bash
pmrf_pipeline <command> [arguments]
```

### 📁 Main Commands

#### 1. Preprocessing

```bash
pmrf_pipeline preprocess <run_id> <raw_data_path>
```

Downloads (if needed) and preprocesses raw NIfTI data, builds subjects list, and splits into train/val/test under
`<RUNS_BASE_DIR>/<run_id>/patch_subjects/splits/`.

#### 2. Training

| Command                              | Description                                                                          |
|--------------------------------------|--------------------------------------------------------------------------------------|
| `train-pm <run_id>`                  | Train full-channel PM (T1N+T2W+T2F → T1C).                                           |
| `train-pm-t1n <run_id>`              | Train T1N-only PM model.                                                             |
| `train-rf <run_id>`                  | Train full-channel RF, fine-tuning frozen PM.                                        |
| `train-rf-t1n <run_id>`              | Train T1N-only RF, fine-tuning frozen T1N-only PM.                                   |
| `train-baseline <run_id> <baseline>` | Train baseline RF variant. `<baseline>` ∈ `{flow_from_x_t1n, cond_x, cond_yhat_pm}`. |

#### 3. Inference

| Command                                                   | Description                                                                      |
|-----------------------------------------------------------|----------------------------------------------------------------------------------|
| `infer-pm <run_id>`                                       | Patch-based inference for full-channel PM.                                       |
| `infer-pm-t1n <run_id>`                                   | Patch-based inference for T1N-only PM.                                           |
| `infer-rf <run_id> <steps> <pm_run>`                      | RF inference with `<steps>` integration steps, using PM outputs from `<pm_run>`. |
| `infer-rf-t1n <run_id> <steps> <pm_run>`                  | RF T1N-only inference with `<steps>` steps, using T1N PM run `<pm_run>`.         |
| `infer-baseline <run_id> <baseline> <steps> [--pm-run R]` | Inference for baseline RF variant `<baseline>`.                                  |

#### 4. Slices & Perceptual Evaluation

##### Slice Generation

| Command                                 | Description                                       |
|-----------------------------------------|---------------------------------------------------|
| `slices-real <run_id>`                  | Save PNG slices (T1C & T1N) from train volumes.   |
| `slices-test <run_id>`                  | Save slices for all modalities (test set).        |
| `slices-infer <run_id> <infer_run>`     | Save PM & RF inference slices for `<infer_run>`.  |
| `slices-infer-t1n <run_id> <infer_run>` | Save T1N-only inference slices for `<infer_run>`. |
| `slices-baseline <run_id> <infer_run>`  | Save baseline inference slices for `<infer_run>`. |

##### Perceptual Metrics

| Command                                                                  | Description                                            |
|--------------------------------------------------------------------------|--------------------------------------------------------|
| `eval-perceptual <run_id> <infer_run> <gpu> [--samples_resize_and_crop]` | Compute ISC/FID/KID/PRC between real and PM/RF slices. |
| `eval-perceptual-t1n <run_id> <infer_run> <gpu>`                         | Fidelity eval on PM/RF T1N-only slices.                |
| `eval-perceptual-baseline <run_id> <infer_run> <gpu>`                    | Fidelity eval on baseline slices.                      |

## 📈 Results Aggregation

```bash
pmrf_pipeline aggregate-evaluations <run_id> <output_csv>
```

Collects and merges all inference and perceptual metrics into a single CSV at:

```
<RUNS_BASE_DIR>/<run_id>/<output_csv>
```

---

## 📚 License and Citation

If you use this repository, please cite:

Brandstötter, B., & Kobler, E. (2025). *Synthesizing Accurate and Realistic T1-weighted Contrast-Enhanced MR Images using Posterior-Mean Rectified Flow.* arXiv:2508.12640. https://doi.org/10.48550/arXiv.2508.12640

**BibTeX:**
```bibtex
@misc{brandstotter2025pmrf_mri,
  title         = {Synthesizing Accurate and Realistic T1-weighted Contrast-Enhanced MR Images using Posterior-Mean Rectified Flow},
  author        = {Brandst{\"o}tter, Bastian and Kobler, Erich},
  year          = {2025},
  eprint        = {2508.12640},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2508.12640},
  url           = {https://arxiv.org/abs/2508.12640}
}
```

**Code License:** MIT License (see `LICENSE`).