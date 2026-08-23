"""
array.rechunk — read an array on one chunk grid, emit it on another.

Writing a slide to a new file almost always means regridding: the source stores
512x512 tiles, or 3-row strips spanning the full width, and the output wants
something else. The naive answer — slice each output tile straight out of the
source — is correct but can be quietly expensive, because a source chunk that
straddles several output tiles is decoded once per tile that touches it.

How expensive, measured on real files:

===================  ============  ==========  ===========
source chunks        output tile   cached      uncached
===================  ============  ==========  ===========
``(512, 512)``       512           1.0x        1.0x
``(512, 512)``       256           1.0x        4.0x
``(512, 512)``       128           1.0x        16.0x
``(3, 66833)``       512           1.0x        8.0x
===================  ============  ==========  ===========

The cached column is why ezslide has never had a problem here: the global chunk
pool in :mod:`ezslide.array.cache` absorbs every re-read. But that is a
guarantee nobody wrote down, resting on a pool shared with every other open
slide, and it degrades as ``(source/output)**2`` once the working set stops
fitting.

This module makes the regrid explicit instead. :func:`plan_rechunk` says what a
given tiling will cost before you commit to it, and :func:`iter_rechunked`
delivers the tiles with each source chunk read once and a bounded, stated
buffer — whatever the cache happens to be doing.

    >>> plan = plan_rechunk((512, 512), (256, 256), (20000, 30000), 'uint8')
    >>> print(plan)
    banded: read 512x30000 blocks (23.0 MB), 1.0x source reads

    >>> for tile in iter_rechunked(level, (256, 256)):
    ...     ...

Rechunking to a *zarr* store rather than a tile stream is a different problem —
one with an N-dimensional intermediate — and ``rechunker`` is the right tool for
that. Here the sink is a TIFF tile generator consumed in strict raster order, so
a row-band is the whole answer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

__all__ = ['RechunkPlan', 'plan_rechunk', 'iter_rechunked', 'DEFAULT_MAX_MEM']

#: Default ceiling on the read buffer. Large enough for a full-width band of a
#: whole-slide image at 512-tall tiles (a 112973 px wide RGB slide needs
#: 173 MB), small enough to be unremarkable on a laptop.
DEFAULT_MAX_MEM = 512 * 1024 * 1024

#: Above this, ``iter_rechunked`` warns rather than silently doing the work.
_WARN_AMPLIFICATION = 1.5

#: Never suggest a tile larger than this; for a striped source the arithmetic
#: would otherwise propose one as wide as the whole image.
_MAX_SUGGESTED_TILE = 4096


def _infer_axes(shape, axes=None):
    """Positional fallback matching ``formats.tiff.infer_axes``."""
    if axes is not None:
        return axes
    n = len(shape)
    if n == 2:
        return 'YX'
    return '?' * (n - 2) + 'YX'


def _axis_reads(length, block, chunk):
    """Source chunks touched along one axis when read in ``block``-sized steps.

    Counted exactly rather than approximated: a block boundary that falls
    mid-chunk makes that chunk belong to two blocks, and how often that happens
    depends on the arithmetic of all three numbers.
    """
    if length <= 0 or block <= 0 or chunk <= 0:
        return 0
    total = 0
    for start in range(0, length, block):
        end = min(start + block, length)
        total += (end - 1) // chunk - start // chunk + 1
    return total


def _ceil_div(a, b):
    return -(-a // b)


@dataclass
class RechunkPlan:
    """What a particular source-to-output regrid will cost.

    ``amplification`` is the ratio of source chunks actually decoded to the
    minimum possible (each one once). 1.0 is optimal; 4.0 means every source
    chunk is decoded four times.
    """

    source_chunks: tuple          # (h, w) of the source grid
    out_chunks: tuple             # (h, w) requested
    block: tuple                  # (h, w) actually read at once
    shape: tuple                  # (height, width) of the plane
    samples: int
    itemsize: int
    strategy: str                 # 'direct' | 'banded'
    amplification: float
    max_mem: int

    @property
    def block_bytes(self) -> int:
        """Exact peak buffer for one read, in bytes."""
        return self.block[0] * self.block[1] * self.samples * self.itemsize

    @property
    def fits_budget(self) -> bool:
        return self.block_bytes <= self.max_mem

    @property
    def full_width(self) -> bool:
        return self.block[1] >= self.shape[1]

    def suggestion(self):
        """A tile size that would make this regrid free, or None.

        Only offered when the saving is worth acting on and the answer is a
        sane tile. A striped source would otherwise "suggest" a tile as wide as
        the image, which is advice nobody can use.
        """
        if self.amplification <= _WARN_AMPLIFICATION:
            return None
        sh, sw = self.source_chunks
        hint = (max(sh, self.out_chunks[0] // sh * sh or sh),
                max(sw, self.out_chunks[1] // sw * sw or sw))
        if max(hint) > _MAX_SUGGESTED_TILE:
            return None
        return hint

    def __str__(self) -> str:
        mb = self.block_bytes / 1e6
        text = (f'{self.strategy}: read {self.block[0]}x{self.block[1]} blocks '
                f'({mb:.1f} MB), {self.amplification:.1f}x source reads')
        hint = self.suggestion()
        if hint is not None:
            text += f'; an output tile of {hint[0]}x{hint[1]} would be 1.0x'
        return text


def plan_rechunk(source_chunks, out_chunks, shape, dtype, axes=None,
                 max_mem=DEFAULT_MAX_MEM) -> RechunkPlan:
    """Decide how to read ``shape`` on ``source_chunks`` to emit ``out_chunks``.

    Parameters
    ----------
    source_chunks, shape : tuple
        The source array's chunk shape and shape, in its own axis order.
    out_chunks : (h, w)
        Requested output tile, in Y/X order.
    dtype : dtype-like
    axes : str, optional
        Axis codes for ``shape``; defaults to a trailing-YX guess.
    max_mem : int
        Ceiling on the read buffer, in bytes.

    Notes
    -----
    The decision, in order:

    1. If the output tile is a whole multiple of the source chunk on both axes,
       every source chunk falls inside exactly one output tile — read tile by
       tile (``'direct'``), one tile of memory, 1.0x by construction.
    2. Otherwise prefer a **full-width band** whose height is a multiple of the
       output tile height and at least the source chunk height. Full width is
       what lets a band hold several complete output tile rows while still
       emitting them in the raster order a TIFF writer requires.
    3. If that band exceeds ``max_mem``, drop to **column blocks** one output
       tile row tall. Raster order still holds, at the cost of re-reading source
       chunks that straddle a tile-row boundary.
    4. If even a single output tile exceeds the budget, there is nothing to
       optimize — read tile by tile and report the amplification honestly.
    """
    axes = _infer_axes(shape, axes)
    y_ax, x_ax = axes.index('Y'), axes.index('X')
    height, width = shape[y_ax], shape[x_ax]
    samples = shape[axes.index('S')] if 'S' in axes else 1
    itemsize = np.dtype(dtype).itemsize

    sh = max(1, int(source_chunks[y_ax]))
    sw = max(1, int(source_chunks[x_ax]))
    th, tw = int(out_chunks[0]), int(out_chunks[1])

    def cost(block):
        bh, bw = block
        reads_y = _axis_reads(height, bh, sh)
        reads_x = _axis_reads(width, bw, sw)
        ideal_y, ideal_x = _ceil_div(height, sh), _ceil_div(width, sw)
        if not (ideal_y and ideal_x):
            return 1.0
        return (reads_y / ideal_y) * (reads_x / ideal_x)

    def make(block, strategy):
        return RechunkPlan(
            source_chunks=(sh, sw), out_chunks=(th, tw), block=tuple(block),
            shape=(height, width), samples=samples, itemsize=itemsize,
            strategy=strategy, amplification=round(cost(block), 4),
            max_mem=int(max_mem),
        )

    # 1. aligned: the output tile is built from whole source chunks
    if th % sh == 0 and tw % sw == 0:
        return make((th, tw), 'direct')

    # 2. full-width band, tall enough to cover a whole source chunk row
    band_h = th * _ceil_div(sh, th)
    band = make((band_h, width), 'banded')
    if band.block_bytes <= max_mem:
        return band

    # 3. column blocks, exactly one output tile row tall
    per_tile_row = th * width * samples * itemsize
    if per_tile_row > max_mem:
        tiles_across = max(1, max_mem // (th * tw * samples * itemsize))
        block_w = min(width, tiles_across * tw)
        blocked = make((th, block_w), 'banded')
        # 4. not even one tile fits; nothing to gain from buffering
        if blocked.block_bytes > max_mem and tiles_across <= 1:
            return make((th, tw), 'direct')
        return blocked
    return make((th, width), 'banded')


def _plane_index(axes, y_ax, x_ax, page_axes, page, y_slice, x_slice, ndim):
    index = [slice(None)] * ndim
    for axis, value in zip(page_axes, page):
        index[axis] = value
    index[y_ax] = y_slice
    index[x_ax] = x_slice
    return tuple(index)


def _source_array(level, use_cache):
    """The array to read from — optionally one that bypasses the chunk pool.

    A streaming rechunk reads every source chunk once and never looks at it
    again, so depositing each one in the process-wide pool only evicts whatever
    else is cached. Bypassing is sound *because* the plan guarantees single
    reads; it would be actively harmful for a strategy that relies on re-reads.

    Falls back to the level itself whenever a private store cannot be opened —
    a ``LazyTiffLevel`` is computed from its parent and has no series of its
    own.
    """
    if use_cache:
        return level
    # __dict__, never getattr: LazyTiffLevel.__getattr__ forwards misses to its
    # parent, so getattr(lazy_level, '_level') hands back *level 0's* series
    # and the bypass would silently read level-0 pixels at this level's
    # coordinates.
    series = level.__dict__.get('_level')
    if series is None:
        return level
    try:
        import zarr
        store = zarr.open(series.aszarr(), mode='r')
        array = store['0'] if isinstance(store, zarr.Group) else store
    except Exception:  # noqa: BLE001 - bypass is an optimization, never required
        return level
    # A private store that does not describe this level is a wrong-pixels bug
    # waiting to happen; no performance flag is worth that.
    if tuple(array.shape) != tuple(getattr(level, 'data', level).shape):
        return level
    return array


def iter_rechunked(level, out_chunks, axes=None, max_mem=DEFAULT_MAX_MEM,
                   plan=None, use_cache=True, warn=True):
    """Yield ``level`` as contiguous ``out_chunks`` tiles, in raster order.

    Tiles come back page-major then row-major, with the sample axis kept
    *inside* each tile for interleaved layouts — the order and shape a TIFF
    writer consumes. Edge tiles are clipped to the image, not padded.

    Parameters
    ----------
    level : array-like
        Anything indexable with a tuple, exposing ``shape``/``chunks``; an
        ezslide ``TiffLevel`` or a zarr array both work.
    out_chunks : (h, w)
    axes : str, optional
        Axis codes; taken from ``level.axes`` when absent.
    max_mem : int
        Ceiling on the read buffer.
    plan : RechunkPlan, optional
        Precomputed; otherwise derived from the arguments.
    use_cache : bool
        False reads through a private uncached store where one is available.
    warn : bool
        Warn once when the plan cannot avoid re-reading source chunks.
    """
    data = getattr(level, 'data', level)
    axes = _infer_axes(data.shape, axes if axes is not None
                       else getattr(level, 'axes', None))
    shape = tuple(data.shape)
    y_ax, x_ax = axes.index('Y'), axes.index('X')
    height, width = shape[y_ax], shape[x_ax]

    if plan is None:
        chunks = getattr(data, 'chunks', None) or shape
        plan = plan_rechunk(chunks, out_chunks, shape, data.dtype,
                            axes=axes, max_mem=max_mem)
    if warn and plan.amplification > _WARN_AMPLIFICATION:
        warnings.warn(
            f'rechunking {shape} from {plan.source_chunks} to '
            f'{plan.out_chunks} re-reads source chunks — {plan}',
            stacklevel=2,
        )

    # Axes that are neither Y/X nor a contiguous sample axis become pages, and
    # a TIFF writer wants those iterated outermost.
    contig = 'S' in axes and axes.index('S') > x_ax
    page_axes = [i for i, a in enumerate(axes)
                 if a not in 'YX' and not (contig and a == 'S')]
    page_shape = [shape[i] for i in page_axes]

    src = _source_array(level, use_cache)
    th, tw = plan.out_chunks
    bh, bw = plan.block
    ndim = len(shape)

    for page in np.ndindex(*page_shape):
        if plan.strategy == 'direct':
            for y in range(0, height, th):
                for x in range(0, width, tw):
                    idx = _plane_index(axes, y_ax, x_ax, page_axes, page,
                                       slice(y, min(y + th, height)),
                                       slice(x, min(x + tw, width)), ndim)
                    yield np.ascontiguousarray(np.asarray(src[idx]))
            continue

        for y0 in range(0, height, bh):
            y1 = min(y0 + bh, height)
            for x0 in range(0, width, bw):
                x1 = min(x0 + bw, width)
                idx = _plane_index(axes, y_ax, x_ax, page_axes, page,
                                   slice(y0, y1), slice(x0, x1), ndim)
                # One read; every output tile inside it is a view, not a fetch.
                buf = np.asarray(src[idx])
                for y in range(0, y1 - y0, th):
                    for x in range(0, x1 - x0, tw):
                        sub = [slice(None)] * buf.ndim
                        # buf has the page axes dropped where they were scalars
                        by = y_ax - sum(1 for a in page_axes if a < y_ax)
                        bx = x_ax - sum(1 for a in page_axes if a < x_ax)
                        sub[by] = slice(y, min(y + th, y1 - y0))
                        sub[bx] = slice(x, min(x + tw, x1 - x0))
                        yield np.ascontiguousarray(buf[tuple(sub)])
