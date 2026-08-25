"""Opening a slide without knowing what kind of file it is.

The one thing every caller needs and no single format module can provide:
which class to hand the path to. Both the CLI and the OME-TIFF writer grew
their own private copy of this three-line decision, and anything outside the
package that wants a reader — a viewer, a notebook — grows a third.

``open_slide`` is not a replacement for ``wsidata.open_wsi``. That is the door
into the SpatialData world, and it is the reason :func:`ezslide.pyramid_options`
exists: ``open_wsi`` builds its reader through a registry that forwards no
options, so ``pyramidalize`` has to travel by context variable. When all you
want is the slide itself — with options passed as arguments, every series
reachable by index, and no zarr store written next to the file — open it
directly.
"""

from __future__ import annotations

from pathlib import Path

from .tiff import TiffFile
from .vsi import VsiFile

__all__ = ['open_slide', 'channel_groups', 'VSI_SUFFIXES']

#: Suffixes that mean a cellSens dataset rather than something tifffile reads.
VSI_SUFFIXES = ('.vsi', '.ets')


def slide_class(file):
    """The format class that reads ``file``, chosen by suffix."""
    return VsiFile if Path(str(file)).suffix.lower() in VSI_SUFFIXES else TiffFile


def open_slide(file, **kwargs):
    """Open any slide ezslide reads, dispatching on the file suffix.

    Parameters
    ----------
    file : path
        A ``.vsi`` container or lone ``.ets`` stack, or anything ``tifffile``
        opens — SVS, NDPI, OME-TIFF, plain TIFF.
    **kwargs
        Forwarded to the format class. ``pyramidalize=True`` and
        ``pyramid={...}`` work the same way for either, so a caller that wants
        lazy pyramid levels on a flat input can ask for them without first
        finding out which class it is going to get.

    Returns
    -------
    :class:`~ezslide.formats.tiff.TiffFile` or
    :class:`~ezslide.formats.vsi.VsiFile`
        A context manager; ``slide[i]`` is a series either way.

    Examples
    --------
    .. code-block:: python

        >>> with open_slide('slide.svs') as slide:
        ...     series = slide[0]

        >>> # A flat export gets the pyramid it does not have on disk. This is
        >>> # a no-op on a file that is already pyramidal, so it is safe to ask
        >>> # for unconditionally.
        >>> labels = open_slide('tokens.tif', pyramidalize=True,
        ...                     pyramid={'how': 'mode', 'cache': 'tmp'})
    """
    return slide_class(file)(file, **kwargs)


def channel_groups(slide):
    """Group series that are channels of one image, preserving order.

    A series opts in by returning a non-None ``channel_group_key``; formats
    that do not split channels across series return ``None`` and come back one
    group each. cellSens is the format that needs this — a fluorescence ``.vsi``
    stores every channel as its own ETS stack, so ``slide[0]`` is DAPI and the
    rest of the markers are the following series.

    Parameters
    ----------
    slide : slide or iterable of series
        A :class:`~ezslide.formats.tiff.TiffFile` (its ``.series`` is used) or
        any sequence of series.

    Returns
    -------
    list[list]
        One inner list per image. Single-series images give ``[[series]]``, so
        a caller can treat every format the same way.
    """
    series_list = getattr(slide, 'series', slide)
    groups, index = [], {}
    for series in series_list:
        key = getattr(series, 'channel_group_key', None)
        if key is None:
            groups.append([series])
            continue
        if key in index:
            groups[index[key]].append(series)
        else:
            index[key] = len(groups)
            groups.append([series])
    return groups
