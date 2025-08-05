import os

import numpy as np
from PIL import Image
import torchio as tio
import nibabel as nib
import torch


def transform_raw_volume(nii_path, transform):
    raw_vol = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
    img = tio.ScalarImage(tensor=torch.from_numpy(raw_vol[None]).float(), affine=np.eye(4))
    img = transform(img)
    vol = img[tio.DATA].squeeze().numpy()
    return vol


def save_plane_slices(
        vol: np.ndarray,
        plane: str,
        base: str,
        out_dir: str,
        n_slices: int = 9,
        step: int = 10
):
    """
    Extract `n_slices` slices around the central slice of `vol` along `plane`, stepping
    `step` voxels up and down, centered on the middle slice (total odd count).
    For example, with n_slices=9 and step=10, offsets are -40,-30,-20,-10,0,10,20,30,40.
    Filenames: {base}_{plane}_{idx:02d}.png
    """
    if plane == 'axial':
        axis = 2
    elif plane == 'coronal':
        axis = 1
    elif plane == 'sagittal':
        axis = 0
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    size = vol.shape[axis]
    mid = size // 2
    half = n_slices // 2
    offsets = [i * step for i in range(-half, half + 1)]

    os.makedirs(out_dir, exist_ok=True)
    for idx, off in enumerate(offsets):
        sl = int(np.clip(mid + off, 0, size - 1))
        if plane == 'axial':
            img2d = vol[:, :, sl]
        elif plane == 'coronal':
            img2d = vol[:, sl, :]
        else:
            img2d = vol[sl, :, :]

        vmin, vmax = float(img2d.min()), float(img2d.max())
        scale = 255.0 / (vmax - vmin + 1e-8)
        arr8 = ((img2d - vmin) * scale).clip(0, 255).astype(np.uint8)

        fname = f"{base}_{plane}_{idx:02d}.png"
        Image.fromarray(arr8).save(os.path.join(out_dir, fname))
