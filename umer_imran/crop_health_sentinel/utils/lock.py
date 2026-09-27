from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import portalocker

from crop_health_sentinel.errors import ConcurrentRunError


@contextmanager
def publication_lock(output_dir: str | Path):
    """Acquire a non-blocking exclusive cross-process lock for one output path."""
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with portalocker.Lock(directory / ".crophealth.publish.lock", mode="a", timeout=0, flags=portalocker.LOCK_EX | portalocker.LOCK_NB):
            yield
    except portalocker.exceptions.LockException as exc:
        raise ConcurrentRunError(f"Output directory is busy: {directory}") from exc
