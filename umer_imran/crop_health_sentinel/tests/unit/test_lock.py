from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from crop_health_sentinel.errors import ConcurrentRunError
from crop_health_sentinel.utils.lock import publication_lock


def _hold_lock(directory: str, acquired, release) -> None:
    with publication_lock(directory):
        acquired.set()
        release.wait(20)


def test_publication_lock_rejects_another_process(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    process = context.Process(target=_hold_lock, args=(str(tmp_path / "report"), acquired, release))
    process.start()
    try:
        assert acquired.wait(15)
        with pytest.raises(ConcurrentRunError):
            with publication_lock(tmp_path / "report"):
                pass
    finally:
        release.set()
        process.join(20)
        if process.is_alive():
            process.terminate()
    assert process.exitcode == 0
