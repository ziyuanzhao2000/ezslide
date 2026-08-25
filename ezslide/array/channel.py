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
"""

from __future__ import annotations

import numpy as np

__all__ = ['ChannelView', 'channel_axis_of', 'n_channels']


def channel_axis_of(axes, prefer=None):
    """The axis index that separates channels, or ``None`` if there is one plane.

    ``C`` if the file records one, else the interleaved sample axis ``S`` — an
    RGB image is one OME channel of three samples, but split apart it is three
    planes. Failing both, the first unknown axis: ``infer_axes`` labels axes it
    had to guess ``?`` and never emits a ``C``, so a multi-channel file that
    records no axes string would otherwise look like a single plane.
    """
    if prefer is not None:
        return prefer
    for label in ('C', 'S', '?'):
        if label in axes:
            return axes.index(label)
    return None


def n_channels(data, axes, channel_axis=None):
    """How many channels ``data`` holds under ``axes``."""
    axis = channel_axis_of(axes, channel_axis)
    return 1 if axis is None else int(data.shape[axis])


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
