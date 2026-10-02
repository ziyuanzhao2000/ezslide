"""ezslide's channel-generic replacement for wsidata's ``to_datatree``.

``wsidata.reader.to_datatree`` (``wsidata/reader/_datatree.py``) hardcodes a
3-channel ``uint8`` RGB image: ``dtype=np.uint8``, a ``(3, H, W)`` chunk grid
and ``c_coords=["r", "g", "b"]``. That is right for an H&E/brightfield scan
and wrong for anything else — a multiplexed immunofluorescence image
(``CYX``, an arbitrary channel count, often ``uint16``) or a mono image
(``YX``, no channel axis at all).

This module resolves channel count, dtype and channel names per reader
instead of assuming them, mirroring the ``HnE``/``mIF``/``mono`` branching in
``mesoslide._wsi.read_wsi`` but auto-detecting from the reader rather than
requiring an explicit ``type`` argument, since ``wsidata.open_wsi()`` has no
way to forward one through to ``to_datatree``.

The layout otherwise mirrors wsidata 0.12: one ``da.map_blocks`` per level
with the reader in the task arguments (so the graph pickles), nodes
``scale{i}/image`` with dims ``c, y, x``, the same transforms, and
``attrs = asdict(reader.properties)``.

``patch_to_datatree()`` installs this in place of wsidata's version.
"""

from __future__ import annotations

from dataclasses import asdict
from math import ceil

import dask.array as da
import numpy as np
import xarray as xr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale

from ..array.channel import n_channels

__all__ = ["to_datatree", "patch_to_datatree"]


def _probe_shape(reader):
    """(n_channels, dtype, channel_names) for ``reader``, as cheaply as possible.

    An ezslide ``ZarrSlideReader`` exposes all of this for free, with no
    pixel I/O at all: ``series.axes``/``series.levels[0].shape``/``.dtype``
    fall through to the wrapped tifffile object
    (``TiffLevel.__getattr__``), and ``series.channel_names`` already reads
    the OME ``Channel`` elements or the cellSens tag tree
    (``ezslide.formats.tiff.TiffSeries.channel_names``,
    ``ezslide.formats.vsi.VsiSeries.channel_names``).

    For any other registered ``ReaderBase`` (say, wsidata's own
    ``cucim``/``openslide``/``tiffslide`` readers), fall back to one small
    real read at the coarsest pyramid level — cheap, and exactly the shape
    ``get_region`` is already contractually required to return (``(Y, X, C)``
    or ``(Y, X)`` for mono).
    """
    series = getattr(reader, "series", None)
    axes = getattr(series, "axes", None) if series is not None else None
    if series is not None and axes:
        level0 = series.levels[0]
        dtype = np.dtype(level0.dtype)
        channels = n_channels(level0, axes)
        channel_names = getattr(series, "channel_names", None)
        return channels, dtype, channel_names

    n_level = reader.properties.n_level
    probe = np.asarray(reader.get_region(0, 0, 8, 8, level=n_level - 1))
    if probe.ndim == 2:
        return 1, probe.dtype, None
    return int(probe.shape[-1]), probe.dtype, None


def _channel_coords(n_channels, channel_names, dtype):
    """Channel labels for ``Image2DModel.parse``'s ``c_coords=``.

    Real names win whenever the file supplies them and the count agrees —
    the same source ezslide's own OME writer already trusts for round-tripped
    channel names. Falling back to count alone: 3 channels with no names is
    presumed RGB (``["r", "g", "b"]``), preserving wsidata's original and by
    far the most common behavior unchanged; anything else gets generic
    ``["c0", "c1", ...]``, single-element for mono.
    """
    if channel_names is not None and len(channel_names) == n_channels:
        return list(channel_names)
    if n_channels == 3:
        return ["r", "g", "b"]
    return [f"c{i}" for i in range(n_channels)]


def _read_block(reader, level, block_info=None):
    """One ``(C, h, w)`` block of ``level``, at the block's array location.

    ezslide readers read in level-local pixels, so the block is exact. Other
    readers take level-0 offsets and map them with ``int(x / ds)``; ``ceil``
    lands on the block origin, as in wsidata's ``_read_block``.
    """
    _, (y0, y1), (x0, x1) = block_info[None]["array-location"]
    if hasattr(reader, "_get_level_region"):
        region = reader._get_level_region(level, y0, x0, y1 - y0, x1 - x0)
    else:
        ds = reader.properties.level_downsample[level]
        region = reader.get_region(
            ceil(x0 * ds), ceil(y0 * ds), x1 - x0, y1 - y0, level=level
        )
    region = np.asarray(region)
    if region.ndim == 2:
        region = region[:, :, None]
    return region.transpose(2, 0, 1)


def to_datatree(reader, chunks=(1024, 1024)):
    """Drop-in, channel-generic replacement for wsidata's ``to_datatree``.

    Builds the same multiscale ``DataTree`` of ``Image2DModel``-wrapped
    levels, but with channel count, dtype and ``c_coords`` resolved from the
    reader instead of assumed to be 3-channel ``uint8`` RGB — this is what
    lets a multiplexed IF image (``CYX``, any channel count) or a mono image
    (``YX``) load through the same ``wsidata.open_wsi(..., attach_images=True)``
    path as a standard RGB slide.
    """
    channels, dtype, channel_names = _probe_shape(reader)
    c_coords = _channel_coords(channels, channel_names, dtype)

    levels = {}
    for level, (height, width) in enumerate(reader.properties.level_shape):
        data = da.map_blocks(
            _read_block,
            reader,
            level,
            chunks=da.core.normalize_chunks(
                (channels, *chunks), (channels, height, width)
            ),
            dtype=dtype,
            meta=np.empty((0, 0, 0), dtype=dtype),
        )
        ds = reader.properties.level_downsample[level]
        transform = Identity() if ds == 1 else Scale([ds, ds], axes=("y", "x"))
        image = Image2DModel.parse(
            xr.DataArray(
                data,
                dims=("c", "y", "x"),
                coords={"y": np.arange(height), "x": np.arange(width)},
            ),
            c_coords=c_coords,
            transformations={"global": transform},
        )
        levels[f"scale{level}"] = xr.Dataset({"image": image})

    tree = xr.DataTree.from_dict(levels)
    tree.attrs = asdict(reader.properties)
    return tree


def patch_to_datatree():
    """Monkeypatch wsidata's ``to_datatree`` with the channel-generic version above.

    Two names need patching, not one: ``wsidata.io._wsi`` imports the name at
    module load time (``from ..reader import READERS, to_datatree``) and
    calls the resulting local binding directly inside ``open_wsi()``.
    Patching ``wsidata.reader.to_datatree`` alone would leave that already-bound
    name untouched, so ``open_wsi`` would keep calling the original. Both call
    sites are patched here, so ``open_wsi()`` itself is fixed, and so is
    anyone importing ``to_datatree`` from ``wsidata.reader`` directly.

    Safe to call more than once — it only ever rebinds both names to this
    same function object.
    """
    import wsidata.io._wsi as _wsi_module
    import wsidata.reader as _reader_module

    _wsi_module.to_datatree = to_datatree
    _reader_module.to_datatree = to_datatree
