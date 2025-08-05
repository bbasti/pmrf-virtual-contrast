import os
import yaml
from pathlib import Path

# Environment variable for runs base directory
RUNS_BASE_DIR = 'RUNS_BASE_DIR'
DEFAULT_RUNS_DIR = 'runs'


def get_run_dir(run_id: str) -> Path:
    """
    Resolve the base directory for runs from env var or default 'runs'.
    """
    base = os.environ.get(RUNS_BASE_DIR, DEFAULT_RUNS_DIR)
    return Path(base) / run_id


def load_config(run_id: str) -> dict:
    """
    Load YAML configuration for the given run_id.
    """
    run_dir = get_run_dir(run_id)
    cfg_path = run_dir / 'config.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)
