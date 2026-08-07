"""Project configuration constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "novels.parquet"

# Stage 1/2 walk a ~36 GB corpus doing encoding detection and regex chapter splitting.
# Both were single-process on the original single-GPU box. Cap the default well below
# the 128 available cores: every worker holds a whole novel in memory and competes for
# the same disk. Raise it explicitly when the page cache is warm.
DEFAULT_WORKER_CAP = 32


def resolve_worker_count(max_workers: int | None = None) -> int:
    """Resolve a process-pool size, defaulting to a capped share of the machine."""

    if max_workers is not None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        return max_workers
    return max(1, min(DEFAULT_WORKER_CAP, os.cpu_count() or 1))

