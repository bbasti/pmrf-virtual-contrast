import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_data_range(volume: np.ndarray) -> float:
    """
    Compute dynamic range for volume, avoiding zero range.
    """
    data_range = float(volume.max() - volume.min())
    return data_range if data_range > 0 else 1.0


def compute_volume_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    Compute MSE, PSNR, and mean SSIM for 3D volumes.
    """
    # Mean Squared Error
    mse = float(np.mean((pred - gt) ** 2))

    # Dynamic Range
    data_range = compute_data_range(gt)

    psnr_val = float(psnr(gt, pred, data_range=data_range))
    ssim_vals = [ssim(gt[i], pred[i], data_range=data_range)
                 for i in range(pred.shape[0])]

    return mse, psnr_val, float(np.mean(ssim_vals))
