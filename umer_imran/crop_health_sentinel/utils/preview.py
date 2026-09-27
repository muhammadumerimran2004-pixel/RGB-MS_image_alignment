from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

_PREVIEW_LOCK = threading.Lock()


def write_preview(path: str | Path, values, title: str, dpi: int = 120) -> Path:
    """Render one non-interactive diagnostic preview without pyplot global state."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(values, dtype=np.float32).copy(); finite = data[np.isfinite(data)]
    low, high = (0.0, 1.0) if finite.size == 0 else tuple(np.percentile(finite, [2, 98]))
    if low == high: high = low + 1.0
    with _PREVIEW_LOCK:
        figure = Figure(dpi=dpi); FigureCanvasAgg(figure)
        try:
            axis = figure.subplots(); image = axis.imshow(data, vmin=low, vmax=high, cmap="viridis")
            axis.set_title(title); axis.set_axis_off(); figure.colorbar(image, ax=axis)
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
        finally:
            figure.clear()
    return path
