from tqdm import tqdm

from utils.config_io import get_run_dir
from utils.slices_helpers import save_plane_slices, transform_raw_volume
from utils.slices import PLANES, get_slice_transform


def generate_slices_baseline(run_id: str, baseline_run: str, n_slices: int = 9, step: int = 10):
    """
    Generate FID slices from baseline inference volumes.
    """
    run_dir = get_run_dir(run_id)
    inf_dir = run_dir / 'output_inference_baseline' / baseline_run
    vols_dir = inf_dir / 'output_volumes'
    if not vols_dir.exists():
        raise FileNotFoundError(f"Baseline volumes directory not found: {vols_dir}")

    transform = get_slice_transform()

    base_out = run_dir / 'fid_slices_baseline' / baseline_run
    output_dirs = {pl: base_out / pl for pl in PLANES}
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for vol_path in tqdm(sorted(vols_dir.glob('*.nii*')), desc=f'Baseline slices [{baseline_run}]'):
        vol = transform_raw_volume(vol_path, transform)
        base = vol_path.with_suffix('').with_suffix('').name
        for pl in PLANES:
            save_plane_slices(vol, pl, base, str(output_dirs[pl]), n_slices, step)
