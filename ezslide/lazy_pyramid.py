"""
Lazy pyramid levels for zarr — nothing is computed until you index into a level.

Where ``zarr_pyramid.build_pyramid`` is eager (it returns once every level is
written), this module gives you level objects that look like arrays but hold no
data. Reading a region pulls exactly the source region that region needs.

    LazyLevel     Pure virtual view. Zero storage, recomputes on every read.
    CachedLevel   Same view, plus a write-through cache: each chunk-aligned
                  block is materialized into a backing store the first time it
                  is touched, and served from there afterwards.
    lazy_pyramid  Assembles a stack of levels, optionally materializing the
                  deep ones eagerly.
    make_store    Small helper for picking RAM vs tempdir vs a real path.

Which mode to use
-----------------
The cost profile is inverted between the two ends of a pyramid, and that is
the whole basis for choosing:

    level 1   ~25% of the dataset's bytes, but only 4 source chunks per
              output chunk. Expensive to store, cheap to compute on demand.
    level 6   ~0.02% of the bytes, but ~4000 source chunks per output chunk.
              Nearly free to store, ruinous to recompute per tile.

So: keep shallow levels lazy (with a cache if you will revisit regions), and
materialize deep levels eagerly. ``lazy_pyramid(..., materialize_below=3)``
does exactly that. Pure ``cache=None`` is only sensible for a one-shot read or
for the top level or two — a 512x512 tile of level 6 requires a 32768x32768
read from level 0.

Duck typing
-----------
Levels expose ``.shape``, ``.dtype``, ``.ndim``, ``.chunks``, ``__getitem__``
and ``__array__``, which is enough for ``np.asarray`` and for most viewers
that accept an array-like per pyramid level. They are not zarr arrays: there is
no ``__setitem__``, no ``.attrs``, and no store behind ``LazyLevel`` at all.

Example
-------
    >>> import zarr
    >>> from lazy_pyramid import lazy_pyramid
    >>> src = zarr.open_array("img.zarr/0", mode="r")
    >>> pyr = lazy_pyramid(src, levels=6, factors=(1, 1, 1, 2, 2), cache="tmp")
    >>> pyr[3].shape          # known immediately, nothing read
    >>> pyr[3][0, 1, 10, 4:20, 4:20]   # reads only what this tile needs
"""

from __future__ import annotations

import math
import tempfile
import threading
from itertools import product

import numpy as np

from .zarr_pyramid import block_reduce, plan_blocks, downsample_into

__all__ = ["LazyLevel", "CachedLevel", "lazy_pyramid", "make_store"]


def _normalize(key, shape):
    """Turn a numpy-style index into a plain bounding box.

    Returns ``(spans, drop)`` where ``spans`` is one ``(start, stop)`` pair per
    axis of ``shape``, and ``drop`` lists the axes that were indexed with a
    scalar and must be squeezed out of the result afterwards.

    Handles integers (including negative), slices, ``Ellipsis``, and implicit
    trailing axes. Deliberately does *not* handle:

    * strided slices — ``level[::4]`` would silently return every 4th element
      of the reduced grid, which is a different array from what most callers
      mean by striding a pyramid level. Slice first, stride the result.
    * fancy/boolean indexing and ``None`` — a lazy level maps a contiguous
      output box to a contiguous input box, and neither of those does.

    Raises
    ------
    IndexError
        Scalar index outside the axis.
    NotImplementedError
        Slice with a step other than 1 or None.
    TypeError
        Any other index type.
    """
    if not isinstance(key, tuple):
        key = (key,)
    if Ellipsis in key:
        i = key.index(Ellipsis)
        key = key[:i] + (slice(None),) * (len(shape) - len(key) + 1) + key[i + 1:]
    key = key + (slice(None),) * (len(shape) - len(key))
    spans, drop = [], []
    for ax, (k, s) in enumerate(zip(key, shape)):
        if isinstance(k, (int, np.integer)):
            k = int(k) + (s if k < 0 else 0)
            if not 0 <= k < s:
                raise IndexError(f"index {k} out of range for axis {ax} of size {s}")
            spans.append((k, k + 1))
            drop.append(ax)
        elif isinstance(k, slice):
            if k.step not in (None, 1):
                raise NotImplementedError("strided slices not supported; slice then stride")
            spans.append(k.indices(s)[:2])
        else:
            raise TypeError(f"unsupported index {type(k)} on axis {ax}")
    return spans, drop


class LazyLevel:
    """A virtual downsampled view of ``base``. No storage, no precomputation.

    Indexing a region maps it back to the corresponding region of ``base``,
    reads that, and reduces it. Output element ``i`` on an axis covers source
    elements ``[i*f, (i+1)*f)``, so the mapping is exact and a read never
    touches more of the source than it has to.

    Levels compose: ``LazyLevel(LazyLevel(zarr_array, F), F)`` is a valid
    2-step pyramid level, and reads chain down to the real array. Be aware that
    the source region grows as ``prod(factors)`` per link, so a deep stack of
    pure lazy levels turns a small tile request into an enormous read. Use
    ``CachedLevel`` or eager materialization past two or three links.

    Parameters
    ----------
    base : array-like
        The level above this one. Needs ``.shape``, ``.dtype`` and
        ``__getitem__``; ``.chunks`` is used if present to derive an implied
        chunk shape for this level.
    factors : sequence of int
        Reduction from ``base`` to this level, one entry per axis.
    how : str
        Reduction, passed to ``block_reduce``. Use ``'mode'`` for labels.
    level : int, optional
        Pyramid depth. Carried for ``repr`` and for naming cache arrays; it has
        no effect on behaviour.
    chunks : tuple of int, optional
        Chunk shape for this level, clipped to its shape. When omitted it is
        derived as ``base.chunks // factors``, which halves at every level and
        is rarely what you want — a tile request then costs more chunks the
        further you zoom out. ``lazy_pyramid`` passes level 0's chunks here by
        default so the shape stays constant down the pyramid.

    Attributes
    ----------
    shape : tuple of int
        ``ceil(base.shape / factors)``.
    chunks : tuple of int
        The ``chunks`` argument clipped to ``shape``, or if none was given,
        ``base.chunks // factors`` floored at 1. Advisory for ``LazyLevel``
        (nothing is stored) but viewers read it to pick tile sizes, and
        ``CachedLevel`` uses it as the backing array's real chunk shape and
        feeds it to ``plan_blocks`` as ``dst_chunks``.
    nbytes : int
        What this level *would* cost if materialized. Useful for deciding
        whether to cache it.

    Notes
    -----
    Reads are not memoized. Requesting the same tile twice does the work twice.
    That is the point of the class — swap in ``CachedLevel`` when it isn't.
    """

    def __init__(self, base, factors, how="mean", level=None, chunks=None):
        self.base = base
        self.factors = tuple(int(f) for f in factors)
        self.how = how
        self.level = level
        self.shape = tuple(int(math.ceil(s / f)) for s, f in zip(base.shape, self.factors))
        self.dtype = np.dtype(base.dtype)
        self.ndim = len(self.shape)
        if chunks is None:
            bc = tuple(getattr(base, "chunks", base.shape))
            chunks = tuple(max(1, c // f) for c, f in zip(bc, self.factors))
        self.chunks = tuple(max(1, min(int(c), s)) for c, s in zip(chunks, self.shape))

    @property
    def nbytes(self):
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def _compute(self, spans):
        """Produce the box described by ``spans`` (one (start, stop) per axis).

        The single hook subclasses override: ``CachedLevel`` replaces this with
        a cache-fill plus a read from the backing array, leaving index parsing
        and scalar-axis squeezing in ``__getitem__`` untouched.
        """
        src = tuple(slice(a * f, min(b * f, s))
                    for (a, b), f, s in zip(spans, self.factors, self.base.shape))
        return block_reduce(self.base[src], self.factors, self.how)

    def __getitem__(self, key):
        """Read a region, computing it on the spot.

        Accepts integers, contiguous slices, ``Ellipsis`` and implicit trailing
        axes; see ``_normalize`` for what is rejected and why. Scalar-indexed
        axes are squeezed out, matching numpy. Always returns a real
        ``np.ndarray`` — there is no lazy result type, so ``level[...]`` on a
        big level will try to materialize the whole thing.
        """
        spans, drop = _normalize(key, self.shape)
        out = self._compute(spans)
        if drop:
            out = out[tuple(0 if ax in drop else slice(None) for ax in range(self.ndim))]
        return out

    def __array__(self, dtype=None):
        """Materialize the entire level as a numpy array.

        Makes ``np.asarray(level)`` and implicit coercion work. This reads the
        whole of ``base`` — check ``.nbytes`` first on anything large.
        """
        a = self[...]
        return a.astype(dtype) if dtype else a

    def __repr__(self):
        return (f"<LazyLevel level={self.level} shape={self.shape} "
                f"{self.dtype} factors={self.factors} virtual>")


class CachedLevel(LazyLevel):
    """A ``LazyLevel`` that remembers what it has already computed.

    Backed by a real zarr array in ``store``, filled block by block on demand.
    A read determines which cache blocks it overlaps, computes any that are
    missing, then serves the request from the backing array. Blocks are
    chunk-aligned via ``plan_blocks``, so filling the cache never decompresses
    a source chunk twice and never partially writes a destination chunk.

    Granularity matters: the cache block is typically much larger than the
    requested tile, so the first read of a region is slower than a pure
    ``LazyLevel`` read (it computes a whole block) and every subsequent read
    nearby is far faster. Lower ``target_bytes`` if first-touch latency
    matters more to you than throughput.

    Parameters
    ----------
    base, factors, how, level
        As ``LazyLevel``.
    store : zarr Store
        Where cached blocks live. See ``make_store``. The backing array is
        created at ``lazy/{level}`` with ``overwrite=True``, so two
        ``CachedLevel`` objects sharing a store and a level number will
        clobber each other.
    chunks : tuple of int, optional
        Chunk shape for the backing array, forwarded to ``LazyLevel``. This
        does affect behaviour beyond storage layout: it is passed to
        ``plan_blocks`` as ``dst_chunks``, so it participates in choosing the
        cache-fill block size.
    block : tuple of int, optional
        Override the cache-fill granularity, in this level's coordinates.
    target_bytes : int
        Budget per fill block, passed to ``plan_blocks``. Default 64 MiB —
        lower than the eager writer's, since a cache fill sits between the
        caller and their tile.

    Attributes
    ----------
    resident_fraction : float
        Share of the level's blocks currently cached, 0.0 to 1.0.

    Notes
    -----
    The cache only grows; there is no eviction policy. ``evict()`` clears it
    wholesale. If you intend to pan indefinitely around a level far larger than
    your store can hold, either put the store on disk, cap it with your own LRU
    over block indices, or just materialize that level eagerly.

    Thread safety: the ``_done`` bookkeeping is lock-guarded, but the lock is
    released while a block is computed, so two threads racing on the same
    missing block will both compute and both write it. That is wasteful, not
    incorrect — the writes are identical and chunk-aligned. Reads of already
    resident blocks are lock-free.
    """

    def __init__(self, base, factors, how="mean", level=None, store=None,
                 chunks=None, block=None, target_bytes=64 << 20):
        super().__init__(base, factors, how, level, chunks=chunks)
        import zarr

        self._arr = zarr.create_array(store=store, name=f"lazy/{level}",
                                      shape=self.shape, chunks=self.chunks,
                                      dtype=self.dtype, overwrite=True)
        src_chunks = tuple(getattr(base, "chunks", base.shape))
        self.block = block or plan_blocks(src_chunks, self.chunks, self.factors,
                                          self.shape, self.dtype.itemsize, target_bytes)
        self._done = set()
        self._lock = threading.Lock()

    def _fill(self, spans):
        """Compute and store every cache block overlapping ``spans``.

        Idempotent and safe to call on a fully resident region — it iterates
        the block grid, skips anything already marked done, and returns
        without touching the backing array in that case.
        """
        idx = [range(lo // b, ((hi - 1) // b) + 1)
               for (lo, hi), b in zip(spans, self.block)]
        for cell in product(*idx):
            with self._lock:
                if cell in self._done:
                    continue
            dsel, ssel = [], []
            for c, b, ds, ss, f in zip(cell, self.block, self.shape,
                                       self.base.shape, self.factors):
                lo, hi = c * b, min((c + 1) * b, ds)
                dsel.append(slice(lo, hi))
                ssel.append(slice(lo * f, min(hi * f, ss)))
            buf = block_reduce(self.base[tuple(ssel)], self.factors, self.how)
            self._arr[tuple(dsel)] = buf
            with self._lock:
                self._done.add(cell)

    def _compute(self, spans):
        """Fill any missing blocks, then serve the region from the cache."""
        self._fill(spans)
        return self._arr[tuple(slice(a, b) for a, b in spans)]

    @property
    def resident_fraction(self):
        """Fraction of this level's cache blocks that have been computed.

        Blocks, not bytes — edge blocks are partial, so this slightly
        overstates memory use on arrays whose shape is not a multiple of the
        block shape.
        """
        total = int(np.prod([math.ceil(s / b) for s, b in zip(self.shape, self.block)]))
        return len(self._done) / total

    def evict(self):
        """Forget every cached block, forcing recomputation on the next read.

        The backing array itself is not deleted or resized — the stale bytes
        stay allocated and are simply overwritten as blocks are refilled. To
        actually reclaim the space, drop the ``CachedLevel`` and its store.
        """
        with self._lock:
            self._done.clear()

    def __repr__(self):
        return (f"<CachedLevel level={self.level} shape={self.shape} {self.dtype} "
                f"block={self.block} resident={self.resident_fraction:.0%}>")


def make_store(kind="memory", path=None):
    """Pick a backing store for cached pyramid levels.

    Parameters
    ----------
    kind : str
        ``'memory'``   a ``MemoryStore``. Fastest, but the cache is bounded by
                       RAM and dies with the process. Good for deep levels,
                       which are tiny.
        ``'tmp'``      a ``LocalStore`` over a fresh ``mkdtemp`` directory.
                       The right choice when levels are too big for RAM but you
                       do not want them written into the real dataset. Nothing
                       cleans this directory up — see Notes.
        anything else  treated as a filesystem path you own and manage.
    path : str, optional
        Only used with ``kind='tmp'``, to point the tempdir somewhere specific
        (a fast local SSD rather than a network mount, say).

    Returns
    -------
    zarr.storage.Store

    Notes
    -----
    ``'tmp'`` uses ``mkdtemp``, not ``TemporaryDirectory``, so the directory
    outlives the process and is not removed on exit. That is deliberate — a
    cache that survives a crashed session is often what you want — but it means
    you are responsible for deleting it. The prefix is ``zpyr-`` to make the
    leftovers identifiable.
    """
    import zarr.storage as zs
    if kind == "memory":
        return zs.MemoryStore()
    if kind == "tmp":
        return zs.LocalStore(path or tempfile.mkdtemp(prefix="zpyr-"))
    return zs.LocalStore(kind)


def lazy_pyramid(base, levels=6, factors=(1, 1, 2, 2), how="mean",
                 cache="memory", store=None, materialize_below=None,
                 keep_chunks=True, chunks=None,
                 min_extent=32, target_bytes=64 << 20, workers=8):
    """Assemble a stack of pyramid levels that materialize on demand.

    Returns immediately regardless of dataset size — unless
    ``materialize_below`` is set, in which case the eager levels are built
    before returning. Each level is chained to the one above it, so reads
    cascade down to ``base``.

    Parameters
    ----------
    base : array-like
        Level 0, typically an open zarr array. Never modified or copied.
    levels : int
        Maximum depth including level 0, so ``levels=6`` yields at most 6
        entries. ``min_extent`` may stop it sooner.
    factors : sequence of int
        Per-axis reduction between consecutive levels. ``(1, 1, 2, 2)`` for
        4D (t/c, y, x)-style data, ``(1, 1, 1, 2, 2)`` for 5D TCZYX,
        ``(1, 1, 2, 2, 2)`` for isotropic 3D.
    how : str
        Reduction, passed to ``block_reduce``. Use ``'mode'`` for labels.
    cache : {None, 'memory', 'tmp'} or str
        ``None`` gives pure ``LazyLevel`` objects with no storage anywhere —
        every read recomputes from scratch. Anything else gives
        ``CachedLevel`` objects sharing one store, created by ``make_store``.
    store : zarr Store, optional
        Use this store instead of creating one. Overrides ``cache`` as the
        storage target, though ``cache=None`` still selects the uncached class.
        Pass a ``LocalStore`` here to keep a cache across runs.
    keep_chunks : bool
        True (default) gives every level the same chunk shape as ``base``,
        clipped to each level's own shape — matching
        ``zarr_pyramid.build_pyramid``. This keeps a tile request costing the
        same number of chunks at every zoom, and keeps lazy and eagerly
        materialized levels in the same pyramid consistent with each other.
        False falls back to per-level ``parent.chunks // factors``, which
        shrinks geometrically and is almost never what you want.
    chunks : tuple of int, optional
        Chunk shape to propagate to every level instead of ``base.chunks``.
        Ignored when ``keep_chunks=False``.
    materialize_below : int, optional
        Levels at or beyond this index are built eagerly and completely, with
        ``downsample_into``; shallower levels stay lazy. This is the
        recommended mode. Deep levels are a rounding error in storage but the
        worst case for on-demand computation (thousands of source chunks per
        tile), while shallow levels are the reverse. ``materialize_below=3``
        is a reasonable default for a 2x XY pyramid. ``None`` keeps everything
        lazy.
    min_extent : int
        Stop once every reduced axis would fall to this size or below.
    target_bytes : int
        Block budget, used both for cache fills and for eager levels.
    workers : int
        Thread count for eagerly materialized levels only. Lazy reads are
        single-threaded, driven by whoever is indexing.

    Returns
    -------
    list
        ``[base, level1, ...]``. Entries are a mix of ``LazyLevel``,
        ``CachedLevel`` and real ``zarr.Array`` objects depending on the mode;
        all of them support ``.shape``, ``.dtype`` and ``__getitem__``, so
        callers generally do not need to care which is which.

    Notes
    -----
    No OME-NGFF metadata is written, since lazy levels have no canonical
    on-disk home. If you need a spec-compliant dataset, use
    ``zarr_pyramid.build_pyramid`` instead, or write the ``multiscales``
    attribute yourself once the levels are materialized.

    Backing arrays are named ``lazy/{level}`` and created with
    ``overwrite=True``. Two pyramids sharing an explicit ``store`` will
    collide.

    Examples
    --------
    Everything lazy, cached in RAM:

    >>> pyr = lazy_pyramid(src, levels=6, factors=(1, 1, 1, 2, 2))

    Recommended: shallow levels lazy on a disk cache, deep levels prebuilt:

    >>> pyr = lazy_pyramid(src, levels=8, factors=(1, 1, 1, 2, 2),
    ...                    cache="tmp", materialize_below=3)

    Nothing stored at all, for a single pass over one level:

    >>> pyr = lazy_pyramid(src, levels=3, factors=(1, 1, 1, 2, 2), cache=None)
    """
    import zarr

    factors = tuple(int(f) for f in factors)
    if cache is not None and store is None:
        store = make_store(cache)

    ref_chunks = tuple(chunks) if chunks is not None else \
        tuple(getattr(base, "chunks", base.shape))

    out = [base]
    cur = base
    for lvl in range(1, levels):
        shape = tuple(int(math.ceil(s / f)) for s, f in zip(cur.shape, factors))
        if all(s <= min_extent for s, f in zip(shape, factors) if f > 1):
            break

        if keep_chunks:
            lvl_chunks = tuple(max(1, min(int(c), s)) for c, s in zip(ref_chunks, shape))
        else:
            lvl_chunks = None

        eager = materialize_below is not None and lvl >= materialize_below
        if eager:
            ec = lvl_chunks or tuple(max(1, min(int(c), s)) for c, s in
                                     zip(getattr(cur, "chunks", shape), shape))
            arr = zarr.create_array(store=store, name=f"lazy/{lvl}", shape=shape,
                                    chunks=ec, dtype=base.dtype, overwrite=True)
            downsample_into(cur, arr, factors, how, workers=workers,
                            target_bytes=target_bytes)
            nxt = arr
        elif cache is None:
            nxt = LazyLevel(cur, factors, how, level=lvl, chunks=lvl_chunks)
        else:
            nxt = CachedLevel(cur, factors, how, level=lvl, store=store,
                              chunks=lvl_chunks, target_bytes=target_bytes)
        out.append(nxt)
        cur = nxt
    return out
