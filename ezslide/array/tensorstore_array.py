"""
Tensorstore views over the zarr arrays this package opens.

Why
---
``zarr.open(...)[sel]`` always returns numpy: every index materializes, right
there, on the calling thread. Tensorstore keeps the same indexing syntax but
returns a *view* — an index transform over the source, holding no pixels — and
reads only when the bytes are actually asked for. A caller can slice, compose
slices, hand a region to something else, and decide much later (or never) to
pay for it. Reads, when they do happen, run on tensorstore's own thread pool
rather than serially.

Two ways in
-----------
``as_tensorstore`` prefers tensorstore's native ``zarr``/``zarr3`` driver,
which reads the store itself, in C++, with its own chunk cache. That needs a
kvstore tensorstore can address — a directory on disk today — which covers the
on-disk pyramid caches (``lazy_pyramid``, ``make_store('tmp')``) but not a
TIFF.

A ``ZarrTiffStore`` is a Python object with no kvstore behind it: it
synthesizes zarr metadata and decodes TIFF tiles through imagecodecs on
demand, and there is no way to hand a Python store to a tensorstore driver.
For those, ``as_tensorstore`` builds a ``ts.virtual_chunked`` instead — a real
TensorStore whose chunks are filled by calling back into the zarr array, and
therefore through ``array.cache``. The bytes travel the same path they
did before; what is new is that indexing is lazy and the result is a
tensorstore, uniformly, whatever the array turned out to be backed by.

Eager mode
----------
Plenty of callers want pixels rather than views — ``PIL.Image.fromarray``,
the reduction kernel in ``array.reduce.block_reduce``, ``tifffile``'s writer.
``TensorStoreArray.eagerfy()`` flips one flag so that from then on ``arr[sel]``
returns ``np.array(arr[sel])``, i.e. exactly what zarr used to return, without
the caller needing to know which kind of array it is holding.

Contents
--------
TensorStoreArray   array-like wrapper, lazy or eager, over a ts.TensorStore
as_tensorstore     wrap an open zarr array, native driver where possible
"""

from __future__ import annotations

import warnings

import numpy as np
import tensorstore as ts

__all__ = ["TensorStoreArray", "as_tensorstore", "tensorstore_context"]


#: Budget for tensorstore's own chunk cache, shared by every view this module
#: opens. It sits one level above ``array.cache.CACHE``: that pool holds
#: encoded chunk bytes keyed by store key, this one holds decoded chunks keyed
#: by position, so a repeat read of a region skips both the decode and the trip
#: back into Python. Worth having — without it every read re-enters the zarr
#: array once per chunk, and zarr's sync round-trip (~0.25 ms) then dominates a
#: warm tile read.
CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB

_context = None


def tensorstore_context(max_bytes=None):
    """The process-wide ``ts.Context``, created on first use.

    Parameters
    ----------
    max_bytes : int, optional
        Build a *separate* context with this cache budget instead of returning
        the shared one — for a reader that should not evict everyone else's
        chunks, or ``0`` to disable caching for it.

    Notes
    -----
    Only read-only views get a cache by default (see ``as_tensorstore``):
    a cached view of an array someone else may write through zarr would serve
    stale chunks, and this package writes pyramid levels exactly that way.
    """
    global _context
    if max_bytes is not None:
        return ts.Context({"cache_pool": {"total_bytes_limit": int(max_bytes)}})
    if _context is None:
        _context = ts.Context(
            {"cache_pool": {"total_bytes_limit": int(CACHE_MAX_BYTES)}})
    return _context


def _normalize_key(key, shape):
    """Rewrite a numpy-style index into one tensorstore will accept.

    Tensorstore is stricter than numpy and zarr in two ways that matter here:
    a slice must lie inside the domain (``a[96:128]`` on a length-100 axis is
    an error, not a clip), and a negative index is out of range rather than
    counted from the end. Both appear in ordinary code — ``iter_tiles``
    produces an over-hanging stop for every edge tile — so translate rather
    than making callers care.

    Integers and positive-step slices are resolved against ``shape``;
    ``Ellipsis`` is expanded; ``None`` passes through and consumes no axis.
    Anything else (index arrays, negative steps) is handed to tensorstore
    unchanged, along with its axis, and follows tensorstore's rules.
    """
    if not isinstance(key, tuple):
        key = (key,)
    if sum(k is Ellipsis for k in key) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    consumed = sum(1 for k in key if k is not Ellipsis and k is not None)
    out, ax = [], 0
    for k in key:
        if k is Ellipsis:
            fill = max(0, len(shape) - consumed)
            out.extend([slice(None)] * fill)
            ax += fill
            continue
        if k is None:
            out.append(None)
            continue
        if ax >= len(shape):
            raise IndexError(
                f"too many indices for array with {len(shape)} dimensions")
        if isinstance(k, (int, np.integer)):
            size = shape[ax]
            i = int(k) + (size if k < 0 else 0)
            if not 0 <= i < size:
                raise IndexError(
                    f"index {int(k)} out of range for axis {ax} of size {size}")
            out.append(i)
            ax += 1
        elif isinstance(k, slice):
            if k.step is None or k.step > 0:
                start, stop, step = k.indices(shape[ax])
                out.append(slice(start, stop) if step == 1
                           else slice(start, stop, step))
            else:
                out.append(k)
            ax += 1
        else:
            out.append(k)
            ax += 1
    return tuple(out)


def _origin_zero(store):
    """Rebase a view's domain to start at zero on every axis.

    Slicing a TensorStore keeps the original coordinates: ``t[100:200]`` has
    domain ``[100, 200)``, so ``t[100:200][0:10]`` is an error rather than the
    first ten rows of the slice. zarr and numpy both re-origin, and callers
    here compose slices (``level[y0:y1][:, x0:x1]``), so normalize instead of
    exporting tensorstore's convention through the wrapper.
    """
    try:
        if any(int(o) != 0 for o in store.origin):
            return store[ts.d[:].translate_to[0]]
    except Exception:                              # unbounded or rank-0 domain
        pass
    return store


class TensorStoreArray:
    """Array-like view over a ``tensorstore.TensorStore``.

    Duck-types the parts of a zarr array the rest of this package uses —
    ``shape``, ``dtype``, ``ndim``, ``size``, ``nbytes``, ``chunks``,
    ``__getitem__``, ``__setitem__``, ``__array__`` — so it drops into
    ``iter_tiles``, ``array.pyramid``, ``array.reduce`` and napari-style
    multiscale consumers in place of one.

    Lazy (the default)
        ``arr[sel]`` returns another ``TensorStoreArray`` wrapping the sliced
        view. Nothing is read; nothing is decoded. Slicing composes, and the
        read happens at ``np.asarray(...)`` — or never.

    Eager (after ``eagerfy()``)
        ``arr[sel]`` returns ``np.array(view)``: the read is issued on the
        spot and you get a plain ``np.ndarray``, which is what zarr did.

    Parameters
    ----------
    store : ts.TensorStore
        The view this wraps.
    eager : bool
        Start in eager mode. Prefer ``eagerfy()`` at the call site that needs
        pixels, so the mode is visible where it matters.
    source : zarr.Array, optional
        The array a ``virtual_chunked`` view reads through. Kept so writes to
        a read-only view can still fall back to the original store, which is
        how ``TiffPage``/``TiffLevel`` behaved before.
    driver : str, optional
        Label for ``repr`` — ``'zarr3'``, ``'virtual_chunked'``, ...

    Notes
    -----
    Mode is a property of the wrapper, not of the underlying TensorStore, and
    ``eagerfy()`` mutates in place: every holder of *this* object sees the
    change, which is the point when a level is shared. Use ``eager_view()``
    when you want a second handle with its own mode instead.

    Sub-views returned by a lazy ``__getitem__`` inherit nothing but the data:
    they start lazy, and they carry no ``source``, so writing through a slice
    of a read-only array raises rather than quietly reaching around.
    """

    def __init__(self, store, *, eager=False, source=None, driver=None):
        self._ts = _origin_zero(store)
        self._eager = bool(eager)
        self._source = source
        self._driver = driver

    # -- metadata: all free, nothing is read -------------------------------

    @property
    def tensorstore(self):
        """The wrapped ``ts.TensorStore``, for tensorstore-native work."""
        return self._ts

    @property
    def source(self):
        """The zarr array behind a ``virtual_chunked`` view, if any."""
        return self._source

    @property
    def shape(self):
        return tuple(int(s) for s in self._ts.shape)

    @property
    def dtype(self):
        return np.dtype(self._ts.dtype.numpy_dtype)

    @property
    def ndim(self):
        return int(self._ts.rank)

    @property
    def size(self):
        return int(np.prod(self.shape)) if self.ndim else 1

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def chunks(self):
        """Read-chunk shape, as zarr spells it.

        Falls back to the full shape when the view has no chunk grid to
        report (a transformed view can leave an axis unconstrained, in which
        case that axis reports its own extent).
        """
        try:
            grid = self._ts.chunk_layout.read_chunk.shape
        except Exception:
            return self.shape
        if grid is None:
            return self.shape
        return tuple(int(c) if c else s for c, s in zip(grid, self.shape))

    @property
    def eager(self):
        return self._eager

    # -- mode --------------------------------------------------------------

    def eagerfy(self):
        """Make ``__getitem__`` materialize from now on. Returns ``self``."""
        self._eager = True
        return self

    def lazify(self):
        """Make ``__getitem__`` return views again. Returns ``self``."""
        self._eager = False
        return self

    def eager_view(self):
        """A second wrapper over the same data, eager, leaving this one alone."""
        return TensorStoreArray(self._ts, eager=True, source=self._source,
                                driver=self._driver)

    def lazy_view(self):
        """A second wrapper over the same data, lazy, leaving this one alone."""
        return TensorStoreArray(self._ts, eager=False, source=self._source,
                                driver=self._driver)

    # -- indexing ----------------------------------------------------------

    def __getitem__(self, key):
        view = self._ts[_normalize_key(key, self.shape)]
        if self._eager:
            return np.array(view)
        return TensorStoreArray(view, driver=self._driver)

    def __setitem__(self, key, value):
        key = _normalize_key(key, self.shape)
        if self._ts.writable:
            self._ts[key] = value
        elif self._source is not None:
            self._source[key] = value        # let the store raise, as before
        else:
            raise ValueError("array is read-only")

    def __array__(self, dtype=None, copy=None):
        out = np.asarray(self._ts)
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def read(self):
        """Materialize the whole view as a ``np.ndarray``."""
        return self._ts.read().result()

    def __len__(self):
        if not self.ndim:
            raise TypeError("len() of 0-d array")
        return self.shape[0]

    def __repr__(self):
        return (f"<TensorStoreArray shape={self.shape} {self.dtype} "
                f"chunks={self.chunks} "
                f"{'eager' if self._eager else 'lazy'}"
                f"{f' driver={self._driver}' if self._driver else ''}>")


def _driver_spec(array):
    """Tensorstore spec for opening ``array`` natively, or ``None``.

    Only stores tensorstore can address itself qualify. That is a
    ``LocalStore`` today: the chunks are files under a directory, which is
    exactly what the ``file`` kvstore reads. A ``ZarrTiffStore``, a
    ``CacheStore``, a ``MemoryStore`` (zarr's, which lives in this process'
    Python heap, not tensorstore's) cannot be reached from C++ at all.
    """
    try:
        import zarr.storage as zs
        store_path = array.store_path
    except Exception:
        return None
    if not isinstance(store_path.store, zs.LocalStore):
        return None
    driver = "zarr3" if array.metadata.zarr_format == 3 else "zarr"
    return {"driver": driver,
            "kvstore": {"driver": "file", "path": str(store_path.store.root)},
            "path": store_path.path}


def _virtual_chunked(array, context=None):
    """A TensorStore whose chunks are filled by reading ``array``.

    The fallback for stores tensorstore has no driver for. Chunk boundaries
    are taken from the zarr array so a fill maps to whole source chunks —
    whole TIFF tiles — and the global chunk cache underneath sees the same
    access pattern it did when zarr was driving.

    Reads are issued from tensorstore's thread pool, so ``array`` must
    tolerate concurrent ``__getitem__``; ``ZarrTiffStore`` does, and so does
    the ``CacheStore``/``PooledCacheStore`` pair, whose ledger is
    lock-guarded.
    """
    shape = tuple(int(s) for s in array.shape)
    raw = tuple(getattr(array, "chunks", None) or shape)
    chunks = tuple(max(1, min(int(c), s)) for c, s in zip(raw, shape))

    def read_fn(domain, out, params):
        out[...] = array[domain.index_exp]

    write_fn = None
    if not getattr(array, "read_only", True):
        def write_fn(domain, chunk, params):        # noqa: F811 - paired hook
            array[domain.index_exp] = chunk

    return ts.virtual_chunked(
        read_fn, write_fn, context=context,
        dtype=np.dtype(array.dtype), shape=shape,
        chunk_layout=ts.ChunkLayout(read_chunk_shape=chunks))


def as_tensorstore(array, *, eager=False, native=True, context=None):
    """Wrap an open zarr array as a lazy :class:`TensorStoreArray`.

    Parameters
    ----------
    array : zarr.Array
        Already open — it supplies shape, dtype, chunk grid, and is the read
        path when no native driver applies.
    eager : bool
        Start the wrapper in eager mode.
    native : bool
        Try tensorstore's own zarr driver first. Set False to force the
        ``virtual_chunked`` path, e.g. to keep reads flowing through a zarr
        store that caches them.
    context : ts.Context, optional
        Overrides the default choice of chunk cache. By default a read-only
        array shares the process-wide cache from ``tensorstore_context`` and a
        writable one gets none, since a cached view cannot see writes made
        through the zarr array behind it.

    Returns
    -------
    TensorStoreArray

    Notes
    -----
    Falling back is not a failure: for TIFF-backed stores it is the only
    option, and it is what every array in ``formats.tiff`` uses. A native
    open that raises *after* the store looked addressable does warn, since
    that one is unexpected and costs the C++ read path.
    """
    read_only = bool(getattr(array, "read_only", False))
    if context is None and read_only:
        context = tensorstore_context()

    if native:
        spec = _driver_spec(array)
        if spec is not None:
            try:
                store = ts.open(spec, open=True, read=True, write=not read_only,
                                context=context).result()
                return TensorStoreArray(store, eager=eager,
                                        driver=spec["driver"])
            except Exception as exc:                # pragma: no cover
                warnings.warn(f"tensorstore {spec['driver']} driver could not "
                              f"open {spec['path']!r} ({exc}); falling back to "
                              f"a virtual_chunked view over zarr")
    return TensorStoreArray(_virtual_chunked(array, context), eager=eager,
                            source=array, driver="virtual_chunked")
