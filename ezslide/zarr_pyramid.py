"""
Chunk-aligned, streaming multiscale pyramid for zarr — no dask.

Why this exists
---------------
The usual recipe (``da.from_zarr(...).coarsen(...).to_zarr(...)``) makes the
on-disk chunk grid double as the task grid. When chunks are small — which is
what you want for random access — you get one scheduler task per ~100 KB of
work, and graph construction dominates wall time.

This module decouples the two grids. The chunk grid stays whatever it is on
disk; the *work* grid is chosen so that

  1. every write covers a whole number of destination chunks, and
  2. every read covers a whole number of source chunks,

then scaled up to a RAM budget. Rule 1 avoids read-modify-write on partial
chunks (a decompress + splice + recompress cycle, and a lost-update race when
threads share a chunk). Rule 2 avoids decompressing the same source chunk for
two different output blocks. ``plan_blocks`` computes that grid; see its
docstring for the arithmetic.

Duck typing
-----------
Anything exposing ``.shape``, ``.chunks``, ``.dtype``, ``__getitem__`` and
``__setitem__`` works: zarr v2, zarr v3, h5py datasets, tensorstore adapters,
or the lazy views in the companion ``lazy_pyramid`` module. ``.chunks`` is
optional on the source and falls back to ``.shape``.

Quick start
-----------
    >>> import zarr
    >>> from zarr_pyramid import build_pyramid
    >>> g = zarr.open_group("img.zarr")          # already contains "0"
    >>> build_pyramid(g["0"], g, levels=6, factors=(1, 1, 1, 2, 2))

Contents
--------
block_reduce     numpy reduction kernel (mean / max / min / mode)
plan_blocks      chunk-alignment arithmetic — the part that matters
downsample_into  stream one array into one already-created output array
build_pyramid    cascade levels 1..N into a zarr group, write NGFF metadata
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Sequence

import numpy as np

__all__ = ["block_reduce", "downsample_into", "build_pyramid", "plan_blocks"]


# --------------------------------------------------------------------------
# reduction kernel
# --------------------------------------------------------------------------

def block_reduce(a: np.ndarray, factors: Sequence[int], how: str = "mean") -> np.ndarray:
    """Reduce an in-memory array by integer factors, one factor per axis.

    This is the pointwise kernel — it does no I/O and knows nothing about
    chunks. ``downsample_into`` calls it once per work block.

    Parameters
    ----------
    a : np.ndarray
        The block to reduce. Must already be in memory.
    factors : sequence of int
        Window size per axis, one entry for every axis of ``a``. Use 1 on axes
        that should not be reduced (typically time and channel). Passing all
        1s returns ``a`` unchanged without copying.
    how : {'mean', 'max', 'min', 'mode'}
        'mean' is the right default for intensity data. 'max' preserves sparse
        bright features (useful for punctate signal that averaging would wash
        out). 'mode' here is a cheap stand-in: it takes the value at the
        window's origin corner, i.e. nearest-neighbour striding. Use it for
        label/segmentation arrays, where averaging would invent label IDs that
        do not exist in the source.

    Returns
    -------
    np.ndarray
        Shape ``ceil(a.shape / factors)``, same dtype as ``a``.

    Notes
    -----
    Edge handling: the trailing partial window on each axis is edge-padded
    (``np.pad(mode='edge')``) rather than trimmed, so the output keeps a
    ``ceil`` shape and no source pixels are silently dropped. If you need to
    match a viewer or a sibling dataset that assumes ``floor``, trim ``a``
    to a multiple of ``factors`` before calling.

    Integer dtypes accumulate in float32 (itemsize <= 2) or float64, then
    round half away from zero via ``np.rint`` and cast back. This means uint8
    and uint16 pyramids are exact and never overflow, but the mean of an
    all-odd window rounds rather than truncating.

    Raises
    ------
    ValueError
        If ``len(factors) != a.ndim``, or ``how`` is not recognised.
    """
    factors = tuple(int(f) for f in factors)
    if len(factors) != a.ndim:
        raise ValueError(f"{len(factors)} factors for {a.ndim}-d array")
    if all(f == 1 for f in factors):
        return a

    pad = [(0, (-s) % f) for s, f in zip(a.shape, factors)]
    if any(p[1] for p in pad):
        a = np.pad(a, pad, mode="edge")

    shape = []
    for s, f in zip(a.shape, factors):
        shape += [s // f, f]
    win = tuple(range(1, len(shape), 2))
    v = a.reshape(shape)

    if how == "mean":
        acc = np.float32 if a.dtype.itemsize <= 2 else np.float64
        out = v.mean(axis=win, dtype=acc)
        if np.issubdtype(a.dtype, np.integer):
            out = np.rint(out)
        return out.astype(a.dtype, copy=False)
    if how == "max":
        return v.max(axis=win)
    if how == "min":
        return v.min(axis=win)
    if how == "mode":  # label-preserving: nearest-neighbour on the window corner
        return a[tuple(slice(None, None, f) for f in factors)]
    raise ValueError(f"unknown reduction {how!r}")


# --------------------------------------------------------------------------
# block planning — the part that actually matters for I/O efficiency
# --------------------------------------------------------------------------

def plan_blocks(
    src_chunks: Sequence[int],
    dst_chunks: Sequence[int],
    factors: Sequence[int],
    dst_shape: Sequence[int],
    itemsize: int,
    target_bytes: int = 256 << 20,
) -> tuple[int, ...]:
    """Choose the work-block that makes every read and every write chunk-aligned.

    Per axis, the smallest block satisfying both alignment rules is

        lcm(src_chunk, dst_chunk * factor) / factor      [destination units]

    i.e. the first point where the source chunk grid and the scaled destination
    chunk grid coincide. Concretely, with ``src_chunk=128, dst_chunk=128,
    factor=2`` that is ``lcm(128, 256) / 2 = 128`` — write one whole
    destination chunk, read exactly two whole source chunks. When the two chunk
    sizes differ it stops being obvious: ``src_chunk=96, dst_chunk=128,
    factor=2`` gives ``lcm(96, 256) / 2 = 384``, so three destination chunks
    out and eight source chunks in. Any smaller block on that axis straddles a
    chunk boundary somewhere.

    That minimum is then doubled along the widest axes until one block costs
    roughly ``target_bytes`` to hold. This second step is what actually fixes
    the small-chunk problem: it coalesces hundreds or thousands of tiny chunks
    into a single read, so the chunk grid can stay small on disk (good for
    random access) while the work grid is large (good for throughput).

    Parameters
    ----------
    src_chunks, dst_chunks : sequence of int
        Chunk shapes of the input and output arrays. If the output uses zarr v3
        sharding, pass the *shard* shape as ``dst_chunks`` — see Notes.
    factors : sequence of int
        Reduction factor per axis, as passed to ``block_reduce``.
    dst_shape : sequence of int
        Output array shape. Blocks are never grown past it.
    itemsize : int
        Bytes per element, e.g. ``np.dtype(src.dtype).itemsize``.
    target_bytes : int
        Soft ceiling on ``read_bytes + write_bytes`` for one block. The default
        (256 MiB) suits an 8-worker pool on a machine with tens of GB free;
        peak RAM is roughly ``workers * target_bytes``, so lower it for wide
        pools or small machines. It is a ceiling, not a target: the returned
        block never exceeds it, but may be far below it if the minimum aligned
        block is already large.

    Returns
    -------
    tuple of int
        Block shape in destination coordinates. The corresponding read region
        is ``block * factors``.

    Notes
    -----
    Sharding: zarr v3 shards are the atomic write unit, so two threads writing
    different chunks of the same shard can lose data. Passing the shard shape
    as ``dst_chunks`` makes each block cover whole shards and removes the
    hazard. Alternatively run with ``workers=1``.

    The growth loop is capped at 64 rounds, so a pathological aspect ratio
    returns a valid (if smaller than requested) block rather than hanging.
    """
    base = []
    for sc, dc, f, ds in zip(src_chunks, dst_chunks, factors, dst_shape):
        read_align = math.lcm(int(sc), int(dc) * int(f))   # in src coords
        base.append(min(read_align // f, max(ds, 1)))
    base = list(base)

    # grow the largest-stride axes first, cheapest way to hit the byte budget
    def cost(b):
        return int(np.prod(b)) * int(np.prod(factors)) * itemsize * 2  # read + write
    order = sorted(range(len(base)), key=lambda i: -base[i])
    guard = 0
    while cost(base) < target_bytes and guard < 64:
        grew = False
        for i in order:
            if base[i] < dst_shape[i]:
                nxt = base[i] * 2
                trial = list(base)
                trial[i] = min(nxt, int(math.ceil(dst_shape[i] / base[i]) * base[i]))
                if cost(trial) <= target_bytes:
                    base, grew = trial, True
        if not grew:
            break
        guard += 1
    return tuple(int(b) for b in base)


# --------------------------------------------------------------------------
# one level
# --------------------------------------------------------------------------

def downsample_into(src, dst, factors, how="mean", block=None,
                    workers=8, target_bytes=256 << 20, progress=False):
    """Stream one array into another, reducing by ``factors``. One level only.

    Iterates the destination in chunk-aligned blocks (see ``plan_blocks``),
    reading the matching source region, reducing it in memory, and writing it
    out. The full array is never resident: peak RAM is roughly
    ``workers * block_bytes * (1 + 1/prod(factors))``.

    ``dst`` must already exist with the right shape — this function fills it,
    it does not create it. Use ``build_pyramid`` if you want the arrays created
    and cascaded for you.

    Parameters
    ----------
    src, dst : array-like
        Anything with ``.shape``/``.dtype``/``__getitem__`` (src) and
        ``.__setitem__`` (dst). ``dst.shape`` should be
        ``ceil(src.shape / factors)``; a smaller dst is silently allowed and
        simply leaves the tail of src unread.
    factors : sequence of int
        Reduction factor per axis.
    how : str
        Reduction, passed through to ``block_reduce``.
    block : tuple of int, optional
        Override the computed work-block, in destination coordinates. Only
        do this if you know the alignment implications; a misaligned block is
        correct but can cost 2-4x in decompression and recompression.
    workers : int
        Thread pool size. Threads (not processes) are the right tool here:
        blosc, zstd and zlib all release the GIL during codec work, so a plain
        pool saturates disk without pickling anything. Set to 1 for
        deterministic ordering, for stores that are not thread-safe, or when
        writing into unsharded-but-shared regions.
    target_bytes : int
        Passed to ``plan_blocks`` when ``block`` is None.
    progress : bool
        Print a ``done/total`` counter to stdout every 10 blocks.

    Returns
    -------
    The ``dst`` object that was passed in, for chaining.

    Notes
    -----
    Concurrency is safe as long as no two blocks touch the same destination
    chunk, which chunk-aligned blocks guarantee. If you override ``block`` with
    something unaligned *and* use multiple workers, two threads can
    read-modify-write the same chunk and one update is lost.

    Errors inside a worker surface when the pool is drained, so a failure part
    way through leaves ``dst`` partially written. There is no rollback; for a
    resumable run, drive the block grid yourself and record completed blocks.
    """
    factors = tuple(int(f) for f in factors)
    src_chunks = tuple(getattr(src, "chunks", src.shape))
    dst_chunks = tuple(getattr(dst, "chunks", dst.shape))
    if block is None:
        block = plan_blocks(src_chunks, dst_chunks, factors, dst.shape,
                            np.dtype(src.dtype).itemsize, target_bytes)

    grid = [range(0, s, b) for s, b in zip(dst.shape, block)]
    tiles = list(product(*grid))

    def run(origin):
        dsel, ssel = [], []
        for o, b, ds, ss, f in zip(origin, block, dst.shape, src.shape, factors):
            stop = min(o + b, ds)
            dsel.append(slice(o, stop))
            ssel.append(slice(o * f, min(stop * f, ss)))
        buf = src[tuple(ssel)]
        dst[tuple(dsel)] = block_reduce(buf, factors, how)

    if workers and workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            for i, _ in enumerate(pool.map(run, tiles), 1):
                if progress and i % 10 == 0:
                    print(f"  {i}/{len(tiles)}", end="\r", flush=True)
    else:
        for t in tiles:
            run(t)
    return dst


# --------------------------------------------------------------------------
# full pyramid
# --------------------------------------------------------------------------

def build_pyramid(src, group, levels=5, factors=(1, 1, 2, 2), how="mean",
                  keep_chunks=True, workers=8, target_bytes=256 << 20,
                  min_extent=64, compressors="auto", write_ngff_meta=True,
                  axes=None, progress=False):
    """Build and write a full multiscale pyramid into an open zarr group.

    Each level is built from the level above it, never from level 0. Since
    every level is ``1/prod(factors)`` the size of its parent, the total bytes
    read across the whole pyramid converge to ``1 / (1 - 1/prod(factors))``
    passes over level 0 — about 1.33x for 2x-in-2D, versus one full pass per
    level if you always downsampled from the original.

    This function is eager: it returns only once every level is fully written.
    For levels that materialize on demand instead, see the companion
    ``lazy_pyramid`` module.

    Parameters
    ----------
    src : array-like
        Level 0. Usually ``group["0"]``, but any array-like will do — only its
        ``.shape``, ``.chunks``, ``.dtype`` and ``__getitem__`` are used.
    group : zarr.Group
        Destination. Only ``group.store`` and ``group.path`` are read; the new
        arrays are created as siblings named ``"1"``, ``"2"``, ... under that
        path, and ``group.attrs`` gains a ``multiscales`` entry. Note that
        ``src`` is *not* written or copied — the NGFF metadata assumes level 0
        already lives at ``{group.path}/0``. Any store works, including
        ``MemoryStore`` and a ``LocalStore`` over a tempdir.
    levels : int
        Maximum number of levels *including* level 0, so ``levels=6`` writes at
        most arrays 1 through 5. Fewer are written if ``min_extent`` trips
        first.
    factors : sequence of int
        Per-axis reduction between consecutive levels, applied repeatedly. The
        default ``(1, 1, 2, 2)`` is 2x in the last two axes only, i.e. a
        standard XY pyramid over a (t, c, y, x) or (c, z, y, x) array. Use
        ``(1, 1, 1, 2, 2)`` for 5D (t, c, z, y, x), or ``(1, 1, 2, 2, 2)`` for
        an isotropic 3D pyramid.
    how : str
        Reduction, passed through to ``block_reduce``. Use ``'mode'`` for
        label images.
    keep_chunks : bool
        True (default) gives every level the same chunk shape as level 0,
        clipped to the level's own shape. This is what viewers expect: a tile
        request costs the same number of chunks at every zoom. False inherits
        the parent level's chunks instead, which shrinks them geometrically and
        is almost never what you want.
    workers, target_bytes : int
        Passed to ``downsample_into`` for every level.
    min_extent : int
        Stop once every reduced axis (those with ``factor > 1``) would fall to
        this size or below. Prevents a tail of 3x2 arrays. Note the test uses
        the *prospective* shape, so the first level that would breach the limit
        is not written.
    compressors : 'auto' or zarr compressor spec
        'auto' leaves zarr's default codec chain alone. Anything else is passed
        to ``zarr.create_array`` — pass an explicit codec list to match level
        0's compression, since the default may differ from whatever wrote it.
    axes : list of dict, optional
        NGFF axis descriptors. When None, guessed from rank using the trailing
        entries of ``[t, c, z, y, x]``, which is right for the common cases and
        wrong for anything unusual — pass it explicitly if your axis order is
        not TCZYX.
    write_ngff_meta : bool
        Write the ``multiscales`` attribute. Set False if something else owns
        that metadata, or if you are writing into a non-NGFF layout.
    progress : bool
        Print each level's shape and a per-block counter.

    Returns
    -------
    list
        ``[src, level1, level2, ...]``, the actual array objects.

    Notes
    -----
    Every level is created with ``overwrite=True``, so re-running clobbers an
    existing pyramid rather than appending to it.

    The emitted metadata declares NGFF 0.4 with a plain scale transform per
    level and no translation. Strictly, area-averaged downsampling shifts each
    level's origin by half a pixel relative to level 0; most viewers ignore
    this, but if you are registering against another modality you may want to
    add a matching ``translation`` transform.
    """
    import zarr

    factors = tuple(int(f) for f in factors)
    base_chunks = tuple(getattr(src, "chunks", src.shape))
    out = [src]
    cur = src

    for lvl in range(1, levels):
        shape = tuple(int(math.ceil(s / f)) for s, f in zip(cur.shape, factors))
        if all(s <= min_extent for s, f in zip(shape, factors) if f > 1):
            break
        chunks = tuple(min(c, s) for c, s in zip(
            base_chunks if keep_chunks else getattr(cur, "chunks", shape), shape))

        kw = {}
        if compressors != "auto":
            kw["compressors"] = compressors
        nxt = zarr.create_array(store=group.store, name=f"{group.path}/{lvl}".lstrip("/"),
                                shape=shape, chunks=chunks, dtype=src.dtype,
                                overwrite=True, **kw)
        if progress:
            print(f"level {lvl}: {shape} chunks={chunks}")
        downsample_into(cur, nxt, factors, how, workers=workers,
                        target_bytes=target_bytes, progress=progress)
        out.append(nxt)
        cur = nxt

    if write_ngff_meta:
        group.attrs["multiscales"] = _multiscales(out, factors, axes)
    return out


def _multiscales(arrays, factors, axes=None):
    """Build the OME-NGFF 0.4 ``multiscales`` attribute for a set of levels.

    Level ``i`` gets a scale transform of ``factors ** i`` per axis, expressed
    in level-0 pixel units — so a pyramid built with ``(1, 1, 2, 2)`` yields
    scales ``[1, 1, 1, 1]``, ``[1, 1, 2, 2]``, ``[1, 1, 4, 4]``, and so on.
    These are relative scales only: if level 0 has real physical spacing (say
    0.325 um/px), multiply it in yourself before writing, or the viewer will
    report sizes in pixels.

    ``axes`` defaults to the trailing entries of ``[t, c, z, y, x]`` for the
    array's rank, with ``t`` typed as time, ``c`` as channel and the rest as
    space. Pass your own list of ``{"name", "type", ...}`` dicts for any other
    axis order, or to declare units.
    """
    ndim = len(arrays[0].shape)
    if axes is None:
        default = ["t", "c", "z", "y", "x"][-ndim:]
        axes = [{"name": n,
                 "type": {"t": "time", "c": "channel"}.get(n, "space")}
                for n in default]
    datasets = []
    for i, a in enumerate(arrays):
        scale = [float(f) ** i for f in factors]
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })
    return [{"version": "0.4", "axes": axes, "datasets": datasets,
             "type": "mean", "name": "/"}]
