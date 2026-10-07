"""Low-overhead, milestone-based progress reporting."""
from __future__ import annotations

import logging


class ProgressReporter:
    """Emit a compact progress bar only when a pipeline milestone is reached.

    Percentages are weighted stage estimates, not elapsed-time predictions.
    Keeping this event-driven avoids polling, background threads, and measurable
    work inside the image-processing loops.
    """

    def __init__(self, log: logging.Logger, width: int = 20) -> None:
        self._log = log
        self._width = width
        self._percent = -1

    @property
    def percent(self) -> int:
        return max(0, self._percent)

    def update(self, percent: int, stage: str) -> None:
        percent = max(self._percent, min(100, max(0, int(percent))))
        if percent == self._percent:
            return
        self._percent = percent
        filled = round(self._width * percent / 100)
        bar = "#" * filled + "-" * (self._width - filled)
        self._log.info("Progress [%s] %3d%% - %s", bar, percent, stage)
