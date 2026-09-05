"""ezslide's channel-generic replacement for wsidata's ``to_datatree``.

``wsidata.reader._reader_datatree_zarr_v3.to_datatree`` (the code path wsidata
selects whenever ``zarr>=3`` is installed) hardcodes a 3-channel ``uint8`` RGB
image everywhere: ``SlideZarrStore.channels = 3``, a fixed ``"<u1"`` dtype in
its zarr-v3 array metadata, and ``c_coords=["r", "g", "b"]`` in
``to_datatree()`` itself. That is exactly right for an H&E/brightfield scan
and silently wrong for anything else — a multiplexed immunofluorescence image
(``CYX``, an arbitrary channel count, often ``uint16``) or a mono image
(``YX``, no channel axis at all) gets reshaped over the wrong strides, which
is silent data corruption rather than a crash.

This module resolves channel count, dtype and channel names per reader
instead of assuming them, mirroring the ``HnE``/``mIF``/``mono`` branching in
``mesoslide._wsi.read_wsi`` but auto-detecting from the reader rather than
requiring an explicit ``type`` argument, since ``wsidata.open_wsi()`` has no
way to forward one through to ``to_datatree``.

``patch_to_datatree()`` installs this in place of wsidata's version. The
``zarr<3`` code path (``_reader_datatree_zarr_v2.py``) is not patched: this
project's ``pyproject.toml`` pins ``zarr>=3.0``, so that path is unreachable
in any environment satisfying ezslide's own dependency floor.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale
from wsidata.reader._reader_datatree_zarr_v3 import SlideZarrStore
from zarr.core.buffer import default_buffer_prototype

from ..array.channel import channel_axis_of, n_channels

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


class EzslideZarrStore(SlideZarrStore):
    """``SlideZarrStore`` with channel count and dtype resolved per reader.

    Everything else — key parsing, zarr-v3 listing, ``.zgroup``, byte-range
    handling — is inherited unchanged from wsidata's implementation; only the
    two spots that assumed 3 channels of ``uint8`` are overridden.
    """

    def __init__(self, reader, *, chunks=(1024, 1024)):
        super().__init__(reader, chunks=chunks)
        channels, dtype, channel_names = _probe_shape(reader)
        self.channels = channels
        self.dtype = dtype
        self.channel_names = channel_names

    def _array_meta(self, level):
        meta = super()._array_meta(level)
        meta["dtype"] = self.dtype.str
        return meta

    async def get(self, key, prototype, byte_range=None):
        """Override chunk reads to fetch level-local pixels directly.
        """
        parsed = self._parse_chunk_key(key)
        if parsed is None:
            return await super().get(key, prototype, byte_range)

        await self._ensure_open()
        level, row, col = parsed
        if level < 0 or level >= len(self._level_shape):
            return None
        h, w = self._level_shape[level]
        ch_h, ch_w = self.chunks
        chunk_w = min(ch_w, w - col * ch_w)
        chunk_h = min(ch_h, h - row * ch_h)
        if chunk_w <= 0 or chunk_h <= 0:
            return None

        arr = await asyncio.to_thread(
            self._reader._get_level_region,
            level,
            row * ch_h,
            col * ch_w,
            chunk_h,
            chunk_w,
        )
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = np.ascontiguousarray(arr)
        return arr.tobytes()


async def _fetch_chunk_as_array(store, level, row, col):
    """Adapted from ``wsidata.reader._reader_datatree_zarr_v3._fetch_chunk_as_array``,
    generalized from a hardcoded ``uint8``/3-channel buffer to ``store.dtype``/``store.channels``.
    """
    proto = default_buffer_prototype()
    key = f"{level}/{row}.{col}"
    buf = await store.get(key, proto)
    h, w = store._level_shape[level]
    ch_h, ch_w = store.chunks
    chunk_w = min(ch_w, w - col * ch_w)
    chunk_h = min(ch_h, h - row * ch_h)
    channels, dtype = store.channels, store.dtype
    if buf is None:
        return np.zeros((chunk_h, chunk_w, channels), dtype=dtype)
    arr = np.frombuffer(bytes(buf), dtype=dtype)
    return arr.reshape((chunk_h, chunk_w, channels))


def level_to_xarray(store, level):
    """Adapted from ``wsidata.reader._reader_datatree_zarr_v3.level_to_xarray``,
    generalized the same way as :func:`_fetch_chunk_as_array`.
    """
    h, w = store._level_shape[level]
    ch_h, ch_w = store.chunks
    channels, dtype = store.channels, store.dtype

    rows, cols = math.ceil(h / ch_h), math.ceil(w / ch_w)
    blocks = []
    for r in range(rows):
        row_blocks = [
            da.from_delayed(
                delayed(
                    lambda s, L, R, C: asyncio.run(_fetch_chunk_as_array(s, L, R, C))
                )(store, level, r, c),
                shape=(min(ch_h, h - r * ch_h), min(ch_w, w - c * ch_w), channels),
                dtype=dtype,
            )
            for c in range(cols)
        ]
        blocks.append(da.concatenate(row_blocks, axis=1))
    arr = da.concatenate(blocks, axis=0)
    coords = {"y": np.arange(h), "x": np.arange(w), "c": np.arange(channels)}
    return xr.DataArray(arr, dims=("y", "x", "c"), coords=coords)


def to_datatree(reader, chunks=(1024, 1024)):
    """Drop-in, channel-generic replacement for wsidata's zarr-v3 ``to_datatree``.

    Builds the same multiscale ``DataTree`` of ``Image2DModel``-wrapped
    levels, but with channel count, dtype and ``c_coords`` resolved from the
    reader instead of assumed to be 3-channel ``uint8`` RGB — this is what
    lets a multiplexed IF image (``CYX``, any channel count) or a mono image
    (``YX``) load through the same ``wsidata.open_wsi(..., attach_images=True)``
    path as a standard RGB slide.
    """
    store = EzslideZarrStore(reader, chunks=chunks)
    c_coords = _channel_coords(store.channels, store.channel_names, store.dtype)

    images = {}
    for level in range(reader.properties.n_level):
        img = level_to_xarray(store, level)
        scale_factor = reader.properties.level_downsample[level]
        transform = (
            Identity()
            if scale_factor == 1
            else Scale([scale_factor, scale_factor], axes=("y", "x"))
        )
        scale_image = Image2DModel.parse(
            img, transformations={"global": transform}, c_coords=c_coords
        )
        images[f"scale{level}"] = xr.Dataset({"image": scale_image})

    slide_image = xr.DataTree.from_dict(images)
    slide_image.attrs = asdict(reader.properties)
    return slide_image


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
