"""Pick a pyramid level for a target downsample.

Ported from lazyslide's ``WSIViewer._select_level_for_downsample`` /
``_resolve_image_level`` (``lazyslide/plotting/_wsi_viewer.py``), which
mesoslide also needs but had no shared place to get it from -- every
``wsidata`` reader backend (including this package's own) exposes
``SlideProperties.n_level``/``level_downsample``, but nothing upstream
searches them for the level closest to a target downsample.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from wsidata import SlideProperties

__all__ = ["select_level_for_downsample", "resolve_display_level"]


def select_level_for_downsample(
    level_downsample: Sequence[float], target_downsample: float, n_level: int
) -> tuple[int, float]:
    """Coarsest pyramid level whose downsample is <= ``target_downsample``.

    Never picks a level coarser than the target (never under-resolves);
    among the remaining levels, picks the one closest to it, so a target
    finer than every level falls back to level 0.
    """
    ds = np.asarray(level_downsample, dtype=float)
    if ds.size == 0 or not np.isfinite(ds).any() or target_downsample <= 1:
        return 0, float(ds[0]) if ds.size else 1.0
    gap = ds - float(target_downsample)
    gap[gap > 0] = np.inf  # forbid levels coarser than the target
    if not np.isfinite(gap).any():
        return 0, float(ds[0])
    level = max(0, min(int(np.argmin(np.abs(gap))), n_level - 1))
    return level, float(ds[level])


def resolve_display_level(
    props: "SlideProperties", axes_px_w: float, axes_px_h: float, oversample: float = 1.5
) -> tuple[int, float]:
    """Best level for filling an axes of the given device-pixel size.

    ``axes_px_w``/``axes_px_h`` are the displayed axes size in device pixels
    (e.g. from ``ax.get_position()`` x figure size x dpi). Reading this many
    times ``oversample`` more pixels than the axes occupies keeps the image
    crisp without reading full resolution into a small figure.
    """
    h0, w0 = props.shape
    target = 1.0
    if axes_px_w and axes_px_h:
        target = max(
            1.0,
            w0 / max(axes_px_w * oversample, 1),
            h0 / max(axes_px_h * oversample, 1),
        )
    return select_level_for_downsample(props.level_downsample, target, props.n_level)
