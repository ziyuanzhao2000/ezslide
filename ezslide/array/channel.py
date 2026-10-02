"""One channel of a level, presented as a ``(Y, X)`` array-like.

The two kinds of level in a pyramid disagree about what indexing means.
:class:`~ezslide.array.tensorstore_array.TensorStoreArray` — a level that is in
the file — composes a view and reads nothing. :class:`~ezslide.array.pyramid.LazyLevel`
— a level ezslide derived by pyramidalizing — computes the region on the spot.
So slicing a channel out of a level up front is either free or a full read of
the level, depending on which one you happen to be holding, and a consumer that
wants one tile wants neither.

``ChannelView`` defers the channel index into the tile read. Both kinds of level
then behave the same way and both read exactly one tile, which is what makes a
pyramid with synthesized levels usable by a viewer that asks for regions.

``InterleavedView`` is the same trick for the other question a channel axis
raises: keep every channel, but move the axis last, so a planar level can be
handed to a viewer that only calls ``(Y, X, C)`` an RGB image.
"""

from __future__ import annotations

import numpy as np

__all__ = ['ChannelView', 'InterleavedView', 'channel_axis_of', 'n_channels',
           'n_planar_channels', 'level_region_index', 'to_channel_last',
           'slice_level_region']


def channel_axis_of(axes, prefer=None):
    """The axis index that separates channels, or ``None`` if there is one plane.

    ``C`` if the file records one, else the interleaved sample axis ``S`` — an
    RGB image is one OME channel of three samples, but split apart it is three
    planes. Failing both, the first unknown axis: ``infer_axes`` labels axes it
    had to guess ``?`` and never emits a ``C``, and ``tifffile`` labels an
    axis ``Q`` when its own "shaped" metadata (the ``tifffile.imwrite``
    default) records a shape but no axes — so a multi-channel file with no
    recorded axes string would otherwise look like a single plane.

    ``I`` (tifffile's label for a plain multi-page TIFF with no metadata) is
    not treated as a channel axis.
    """
    if prefer is not None:
        return prefer
    for label in ('C', 'S', 'Q', '?'):
        if label in axes:
            return axes.index(label)
    return None


def n_channels(data, axes, channel_axis=None):
    """How many channels ``data`` holds under ``axes``."""
    axis = channel_axis_of(axes, channel_axis)
    return 1 if axis is None else int(data.shape[axis])


def n_planar_channels(data, axes):
    """Channels stored on a planar axis (``C``/``Q``/``?``) of ``data``.

    1 when the channel axis is the interleaved sample axis ``S`` (an RGB
    image is one channel of three samples) or there is no channel axis.
    """
    axis = channel_axis_of(axes)
    if axis is None or axes[axis] == 'S':
        return 1
    return int(data.shape[axis])


def level_region_index(lv, y0, height, x0, width):
    """Index tuple for a ``(y0:y0+height, x0:x0+width)`` region of ``lv``.

    Built from ``lv.y_ax``/``lv.x_ax`` rather than assuming axis 0 is Y and
    axis 1 is X -- true for channel-last "YXS"/"YXC" files, but wrong for a
    channel-first "SYX"/"CYX" file, where a positional slice lands on the
    channel axis instead of Y and leaves X unsliced.
    """
    idx = [slice(None)] * len(lv.shape)
    idx[lv.y_ax] = slice(y0, y0 + height)
    idx[lv.x_ax] = slice(x0, x0 + width)
    return tuple(idx)


def to_channel_last(lv, arr):
    """Move ``arr``'s channel axis (per ``lv.axes``) to the end, if needed."""
    c_ax = channel_axis_of(lv.axes)
    if c_ax is not None and c_ax != len(lv.axes) - 1:
        arr = np.moveaxis(arr, c_ax, -1)
    return arr


def slice_level_region(lv, y0, height, x0, width):
    """Read a ``(height, width)`` region out of a pyramid level, channel-last."""
    arr = lv[level_region_index(lv, y0, height, x0, width)]
    arr = arr.compute() if hasattr(arr, "compute") else np.asarray(arr)
    return to_channel_last(lv, arr)


class ChannelView:
    """One channel of a level, as a ``(Y, X)`` array-like.

    Exposes ``shape``, ``dtype``, ``ndim``, ``size``, ``chunks``, ``__getitem__``
    and ``__array__`` — enough for ``np.asarray`` and for viewers that accept an
    array-like per pyramid level, napari's ``LayerDataProtocol`` among them.

    Parameters
    ----------
    data : array-like
        A level's array: a ``TensorStoreArray``, a ``LazyLevel``, a zarr array,
        or a plain ndarray.
    axes : str
        The axes string for ``data`` — ``'CYX'``, ``'YXS'``, ``'YX'``. Pass
        ``ezslide.formats.tiff.infer_axes(data, series.axes)`` if the file may
        not record one.
    channel : int
        Which channel to present.
    channel_axis : int, optional
        Override the axis to index. By default ``C``, else ``S``, else the
        first unknown axis (see :func:`channel_axis_of`).

    Notes
    -----
    A level with no channel axis at all — a ``YX`` plane, or a cellSens series
    that is already one channel — passes its index straight through, so the
    same wrapper covers both layouts a multi-channel slide can have.
    """

    def __init__(self, data, axes, channel, channel_axis=None):
        self._data = data
        self._channel = channel
        self._ndim = len(data.shape)
        self._c_ax = channel_axis_of(axes, channel_axis)
        if self._c_ax is not None:
            available = int(data.shape[self._c_ax])
            if not 0 <= channel < available:
                raise IndexError(
                    f'channel {channel} out of range for a level with '
                    f'{available} channel(s)'
                )
        elif channel != 0:
            raise IndexError(
                f'channel {channel} requested from a single-channel level'
            )
        self._y, self._x = axes.index('Y'), axes.index('X')
        self.shape = (int(data.shape[self._y]), int(data.shape[self._x]))
        self.dtype = np.dtype(data.dtype)
        self.ndim = 2
        self.size = int(np.prod(self.shape))
        chunks = getattr(data, 'chunks', None)
        self.chunks = ((int(chunks[self._y]), int(chunks[self._x])) if chunks
                       else (512, 512))

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def channel(self):
        return self._channel

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # A ``YX`` view has nothing for an Ellipsis to expand to beyond the two
        # axes the key is padded to, so it is just a full slice here.
        key = tuple(slice(None) if k is Ellipsis else k for k in key)
        key = key + (slice(None),) * (2 - len(key))
        full = [slice(None)] * self._ndim
        full[self._y], full[self._x] = key[0], key[1]
        if self._c_ax is not None:
            full[self._c_ax] = self._channel
        return np.asarray(self._data[tuple(full)])

    def __array__(self, dtype=None, copy=None):
        out = self[:, :]
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return np.array(out, copy=True) if copy else out

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return (f'<ChannelView channel={self._channel} '
                f'shape={self.shape} {self.dtype}>')


class InterleavedView:
    """A planar level as an interleaved ``(Y, X, C)`` array-like.

    The mirror image of :class:`ChannelView`: instead of picking one channel
    out of a level, this keeps them all and moves the channel axis last, which
    is the layout a viewer means by *RGB*. A slide written planar — three uint8
    planes on a ``C`` axis, as tifffile emits when it rewrites an NDPI — holds
    the same pixels as one written interleaved on an ``S`` axis, but only the
    interleaved one can be handed to napari's ``rgb=True`` as it stands.

    Like ``ChannelView`` it defers the work into the tile read, so a
    :class:`~ezslide.array.tensorstore_array.TensorStoreArray` level and a
    :class:`~ezslide.array.pyramid.LazyLevel` cost the same to read a region
    from, and neither is ever materialized whole. ``np.moveaxis`` on the level
    itself would do exactly that.

    The cost that is real is the read: a planar file stores each channel in its
    own plane, so one tile of an interleaved image is one read where the same
    tile of a planar image is one read per channel.

    Parameters
    ----------
    data : array-like
        A level's array: a ``TensorStoreArray``, a ``LazyLevel``, a zarr array,
        or a plain ndarray.
    axes : str
        The axes string for ``data`` — ``'CYX'``, ``'YXS'``, ``'YX'``. Pass
        ``ezslide.formats.tiff.infer_axes(data, series.axes)`` if the file may
        not record one.
    channel_axis : int, optional
        Override the axis to move. By default ``C``, else ``S``, else the first
        unknown axis (see :func:`channel_axis_of`).

    Raises
    ------
    ValueError
        If ``data`` has no channel axis at all. A ``YX`` plane has nothing to
        interleave, and what to show instead is the caller's decision.
    """

    def __init__(self, data, axes, channel_axis=None):
        self._data = data
        self._ndim = len(data.shape)
        self._c_ax = channel_axis_of(axes, channel_axis)
        if self._c_ax is None:
            raise ValueError(
                f'a level with axes {axes!r} has no channel axis to interleave'
            )
        self._y, self._x = axes.index('Y'), axes.index('X')
        self.shape = (int(data.shape[self._y]), int(data.shape[self._x]),
                      int(data.shape[self._c_ax]))
        self.dtype = np.dtype(data.dtype)
        self.ndim = 3
        self.size = int(np.prod(self.shape))
        chunks = getattr(data, 'chunks', None)
        self.chunks = ((int(chunks[self._y]), int(chunks[self._x]),
                        self.shape[2]) if chunks else (512, 512, self.shape[2]))

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # Unlike a ``YX`` view, three axes leave an Ellipsis with somewhere to
        # go: ``view[..., 0]`` has to reach the channel axis, not the first one.
        pos = next((i for i, k in enumerate(key) if k is Ellipsis), None)
        if pos is not None:
            fill = max(3 - (len(key) - 1), 0)
            key = key[:pos] + (slice(None),) * fill + key[pos + 1:]
        key = key + (slice(None),) * (3 - len(key))
        full = [slice(None)] * self._ndim
        full[self._y], full[self._x], full[self._c_ax] = key
        out = np.asarray(self._data[tuple(full)])
        # An integer index drops its axis, so the axes that survive the read are
        # the sliced ones, still in source order. Order those, rather than
        # assuming all three are there, and a key like ``view[0, :, :]`` stays
        # correct.
        survivors = [ax for ax in range(self._ndim)
                     if isinstance(full[ax], slice)]
        order = [survivors.index(ax) for ax in (self._y, self._x, self._c_ax)
                 if ax in survivors]
        return np.transpose(out, order)

    def __array__(self, dtype=None, copy=None):
        out = self[:, :, :]
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return np.array(out, copy=True) if copy else out

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f'<InterleavedView shape={self.shape} {self.dtype}>'
