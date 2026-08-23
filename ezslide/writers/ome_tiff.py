"""
writers.ome_tiff — stream an ezslide slide out as a calibrated OME-TIFF.

``tifffile`` supplies more of this than people expect: ``subifds`` builds a real
pyramid, an iterator of tiles plus ``shape``/``dtype``/``tile`` streams pixels
without ever holding a level in memory, and ``MapAnnotation`` is a first-class
place to park arbitrary key/value metadata inside the OME-XML. What it does not
supply is the judgement — which metadata still means something after a
transcode, how channels group into one image, and how to iterate tiles for a
contiguous-sample layout. That is what lives here.

Two things about the output are worth stating plainly:

*Nothing is materialized.* Levels are written tile by tile straight from the
lazy source, so converting a 69094 x 112973 slide costs one tile of memory, not
23 GB.

*Metadata that survives is metadata that is still true.* Everything the reader
recovered — µm/pixel, channel names, objective, magnification, the source path
— is written, the calibration into the OME fields proper and the rest into a
namespaced ``MapAnnotation``. Tags that describe the *source file's byte
layout* (``TileOffsets`` and friends) are deliberately dropped: they would be
false the moment the pixels are re-encoded, and on a slide with 100k tiles they
add about a megabyte of XML per level.

    import ezslide
    ezslide.convert('slide.vsi', 'slide.ome.tif')
    ezslide.convert(src, dst, levels=4, compression='jpeg2000')

Reading one back with ezslide restores the annotation into level metadata, so a
converted slide carries the same information as the original.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import tifffile

from ..formats.tiff import TiffFile, TiffLevel, TiffSeries, infer_axes
from ..formats.vsi import VsiFile

__all__ = ['write_ome_tiff', 'convert', 'OME_NAMESPACE', 'PROVENANCE_NAMESPACE']

#: Namespace of the MapAnnotation holding everything the reader recovered.
OME_NAMESPACE = 'ezslide:metadata'
#: Namespace of the MapAnnotation describing where the file came from.
PROVENANCE_NAMESPACE = 'ezslide:provenance'

#: Tags that describe how the *source* file stored its bytes. Re-encoding makes
#: every one of them a lie, so they are not carried across. The tile geometry
#: tags go too — the output declares its own, which may differ from the source.
_STRUCTURAL_TAGS = frozenset({
    'TileOffsets', 'TileByteCounts', 'StripOffsets', 'StripByteCounts',
    'RowsPerStrip', 'TileWidth', 'TileLength', 'ImageWidth', 'ImageLength',
    'StripOffsetsByteCounts', 'FreeOffsets', 'FreeByteCounts',
    'JPEGTables', 'JPEGInterchangeFormat', 'JPEGInterchangeFormatLength',
})

#: Keys consumed as OME fields rather than repeated in the annotation.
_OME_FIELD_KEYS = frozenset({
    'PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeXUnit', 'PhysicalSizeYUnit',
    'Name', 'ChannelNames',
})

_DEFAULT_TILE = (512, 512)


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------

def _level_tiles(level, tile):
    """Yield ``level``'s tiles in the order tifffile consumes them.

    Row-major within a plane, plane-major across pages, and — unlike
    :func:`ezslide.formats.tiff.iter_tiles` — the sample axis stays *inside*
    each tile for a contiguous layout. Feeding tifffile per-sample tiles for a
    ``YXS`` image is what makes an interleaved source come back out planar.

    Each tile is materialized only as it is handed over, so peak memory is one
    tile regardless of how large the level is.
    """
    axes = infer_axes(level.data, getattr(level, 'axes', None))
    y_ax, x_ax = axes.index('Y'), axes.index('X')
    shape = tuple(level.data.shape)
    th, tw = tile

    # Axes that are neither Y/X nor a contiguous sample axis become separate
    # TIFF pages, and tifffile wants them iterated outermost.
    contig = axes.endswith('YXS') or (axes.endswith('S') and x_ax < axes.index('S'))
    page_axes = [i for i, a in enumerate(axes)
                 if a not in 'YX' and not (contig and a == 'S')]

    for page in np.ndindex(*[shape[i] for i in page_axes]):
        for y in range(0, shape[y_ax], th):
            for x in range(0, shape[x_ax], tw):
                index = [slice(None)] * len(shape)
                for axis, value in zip(page_axes, page):
                    index[axis] = value
                index[y_ax] = slice(y, min(y + th, shape[y_ax]))
                index[x_ax] = slice(x, min(x + tw, shape[x_ax]))
                yield np.ascontiguousarray(np.asarray(level[tuple(index)]))


def _grouped_tiles(levels, tile):
    """Tiles for one pyramid level of a merged multi-channel image.

    ``levels`` is one level taken from each member series, in channel order;
    they are written as consecutive pages of a single ``CYX`` image.
    """
    for level in levels:
        yield from _level_tiles(level, tile)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

def _clean(value):
    """Render a metadata value for the XML without dumping an array into it."""
    if isinstance(value, np.ndarray):
        if value.size > 16:
            return f'<{value.dtype} array, shape {value.shape}>'
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    text = str(value)
    # XML 1.0 forbids most control characters outright; tifffile escapes the
    # markup characters for us but not these.
    return ''.join(c for c in text if c in '\t\n\r' or ' ' <= c <= '￿')


def _annotation(series, source=None, reader=None):
    """``(metadata_map, provenance_map)`` for one output image."""
    md = dict(series.levels[0].metadata)
    keep = {k: _clean(v) for k, v in md.items()
            if str(k) not in _STRUCTURAL_TAGS and str(k) not in _OME_FIELD_KEYS
            and v is not None}
    keep = {str(k): v for k, v in keep.items()}

    provenance = {
        'ConvertedBy': f'ezslide {_version()}',
        'ConvertedAt': datetime.datetime.now(datetime.timezone.utc)
                               .isoformat(timespec='seconds'),
        'SourceLevels': str(len(series.levels)),
    }
    if source is not None:
        provenance['SourcePath'] = str(source)
        provenance['SourceFormat'] = Path(str(source)).suffix.lstrip('.').lower()
    if reader is not None:
        provenance['SourceReader'] = reader
    return keep, provenance


def _version():
    try:
        from importlib.metadata import version
        return version('ezslide')
    except Exception:  # noqa: BLE001 - version is cosmetic, never fatal
        return 'unknown'


def _ome_fields(series, channel_names, source=None, reader=None):
    """The ``metadata=`` dict for the level-0 write of one image."""
    md = series.levels[0].metadata
    meta = {}

    name = md.get('Name') or getattr(series, 'name', None)
    if name:
        meta['Name'] = str(name)

    mpp_x, mpp_y = md.get('PhysicalSizeX'), md.get('PhysicalSizeY')
    if mpp_x is not None:
        meta['PhysicalSizeX'] = float(mpp_x)
        meta['PhysicalSizeXUnit'] = md.get('PhysicalSizeXUnit') or 'µm'
    if mpp_y is not None or mpp_x is not None:
        meta['PhysicalSizeY'] = float(mpp_y if mpp_y is not None else mpp_x)
        meta['PhysicalSizeYUnit'] = md.get('PhysicalSizeYUnit') or 'µm'

    if channel_names:
        meta['Channel'] = {'Name': list(channel_names)}

    keep, provenance = _annotation(series, source=source, reader=reader)
    annotations = []
    if keep:
        annotations.append({'Namespace': OME_NAMESPACE, **keep})
    annotations.append({'Namespace': PROVENANCE_NAMESPACE, **provenance})
    meta['MapAnnotation'] = annotations
    return meta


def _resolution(level, md):
    """``(resolution, unit)`` in pixels/cm, or ``(None, None)`` if uncalibrated.

    OME ``PhysicalSizeX`` alone is enough for ezslide and Bio-Formats to
    recover µm/pixel, but plain TIFF viewers only read the resolution tags, so
    both are written.
    """
    mpp = md.get('PhysicalSizeX')
    if not mpp:
        return None, None
    mpp_y = md.get('PhysicalSizeY') or mpp
    return (1e4 / float(mpp), 1e4 / float(mpp_y)), 'CENTIMETER'


# --------------------------------------------------------------------------
# grouping
# --------------------------------------------------------------------------

def _channel_groups(series_list):
    """Group series that are channels of one image, preserving order.

    A series opts in by returning a non-None ``channel_group_key``; formats
    that do not split channels across series return None and are written one
    image each. The writer therefore never has to know what a ``.vsi`` is.
    """
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


def _group_channel_names(group):
    """Channel names for one merged image, or None to let OME decide."""
    if len(group) > 1:
        return [str(getattr(s, 'channel', None) or s.name) for s in group]
    names = getattr(group[0], 'channel_names', None) or \
        group[0].levels[0].metadata.get('ChannelNames')
    samples = _samples(group[0].levels[0])
    if names and len(names) == 1 and samples == 1:
        return [str(names[0])]
    # An RGB image is one OME channel with three samples, not three channels;
    # naming it per-sample would misdescribe the file.
    return None


def _samples(level):
    axes = infer_axes(level.data, getattr(level, 'axes', None))
    return level.data.shape[axes.index('S')] if 'S' in axes else 1


# --------------------------------------------------------------------------
# the writer
# --------------------------------------------------------------------------

def write_ome_tiff(obj, path, *, levels=None, tile=None, compression='zstd',
                   photometric=None, source=None, reader=None,
                   bigtiff=True, **kwargs):
    """Write a slide, series or level to a pyramidal OME-TIFF.

    Parameters
    ----------
    obj : TiffFile, TiffSeries, TiffLevel, or a wsidata reader
        What to write. A ``TiffFile`` writes every series it holds; series that
        are channels of one image are merged (see ``channel_group_key``).
    path : path
        Output file. ``.ome.tif`` by convention.
    levels : int, optional
        Cap the pyramid depth. ``None`` writes every level the source has;
        ``1`` writes only full resolution.
    tile : (h, w), optional
        Output tile size. Defaults to the source level's chunk shape, so a
        slide is rewritten on the grid it was already stored with.
    compression : str
        Anything ``tifffile`` accepts. The default ``'zstd'`` is lossless —
        re-encoding lossy tiles a second time would defeat the point of
        converting for reproducibility. Pass ``'jpeg2000'`` or ``'jpeg'`` if
        size matters more than fidelity.
    photometric : str, optional
        Overrides the ``rgb``/``minisblack`` choice made from sample count.
    source, reader : optional
        Recorded in the provenance annotation. ``convert`` fills them in.

    Returns
    -------
    Path
        ``path``.
    """
    path = Path(path)
    series_list, source = _as_series(obj, source)
    groups = _channel_groups(series_list)

    with tifffile.TiffWriter(path, bigtiff=bigtiff, ome=True) as writer:
        for group in groups:
            _write_group(writer, group, levels=levels, tile=tile,
                         compression=compression, photometric=photometric,
                         source=source, reader=reader, **kwargs)
    return path


def _as_series(obj, source):
    """Normalize the many things a caller might hand us into a series list."""
    if hasattr(obj, 'reader') and not isinstance(obj, (TiffFile, TiffSeries)):
        obj = obj.reader                       # a wsidata WSIData
    if hasattr(obj, '_reader') and hasattr(obj, 'properties'):
        source = source or getattr(obj, 'file', None)
        obj = obj.reader                       # a wsidata ReaderBase
    if isinstance(obj, TiffFile):
        return list(obj.series), source or obj._file
    if isinstance(obj, TiffSeries):
        return [obj], source
    if isinstance(obj, TiffLevel):
        return [_SingleLevel(obj)], source
    if isinstance(obj, (list, tuple)) and obj:
        return list(obj), source
    # Duck typing last: anything exposing .levels of indexable arrays writes
    # fine, and being strict here would reject perfectly good wrappers.
    if hasattr(obj, 'levels') and len(obj.levels):
        return [obj], source
    if hasattr(obj, 'data') and hasattr(obj, 'metadata'):
        return [_SingleLevel(obj)], source
    msg = (f'cannot write {type(obj).__name__}; expected a TiffFile, '
           'TiffSeries, TiffLevel, a list of series, or a wsidata reader')
    raise TypeError(msg)


class _SingleLevel:
    """Adapter presenting one level as a one-level series."""

    def __init__(self, level):
        self.levels = [level]
        self.name = getattr(level, 'name', None)
        self.channel_group_key = None
        self.channel_names = level.metadata.get('ChannelNames')


def _write_group(writer, group, *, levels, tile, compression, photometric,
                 source, reader, **kwargs):
    """Write one OME image — one series, or several merged as channels."""
    lead = group[0]
    depth = min(len(s.levels) for s in group)
    if levels is not None:
        depth = min(depth, max(1, int(levels)))

    base = lead.levels[0]
    tile = tuple(tile) if tile else _source_tile(base)
    channel_names = _group_channel_names(group)
    samples = _samples(base)
    if photometric is None:
        photometric = 'rgb' if samples >= 3 else 'minisblack'

    md0 = base.metadata
    meta = _ome_fields(lead, channel_names, source=source, reader=reader)

    for i in range(depth):
        members = [s.levels[i] for s in group]
        first = members[0]
        shape, axes = _out_shape(first, len(group))
        resolution, unit = _resolution(first, _scaled(md0, base, first))
        writer.write(
            _grouped_tiles(members, tile) if len(group) > 1
            else _level_tiles(first, tile),
            shape=shape,
            dtype=np.dtype(first.data.dtype),
            tile=tile,
            photometric=photometric,
            compression=compression,
            subifds=depth - 1 if i == 0 else None,
            subfiletype=0 if i == 0 else 1,
            resolution=resolution,
            resolutionunit=unit,
            # OME metadata belongs to the image, which is its first page; a
            # subIFD carrying its own copy is invalid.
            metadata={'axes': axes, **meta} if i == 0 else None,
            **kwargs,
        )


def _out_shape(level, nmembers):
    """``(shape, axes)`` of the written page series for one pyramid level."""
    axes = infer_axes(level.data, getattr(level, 'axes', None))
    shape = tuple(level.data.shape)
    if nmembers > 1:
        # Merged channels: each member contributes one YX plane.
        y, x = shape[axes.index('Y')], shape[axes.index('X')]
        return (nmembers, y, x), 'CYX'
    return shape, axes


def _scaled(md0, base, level):
    """Level-0 metadata rescaled to ``level``'s pixel size."""
    if level is base:
        return md0
    try:
        ratio = base.width / level.width
    except Exception:  # noqa: BLE001
        return md0
    out = dict(md0)
    for key in ('PhysicalSizeX', 'PhysicalSizeY'):
        if out.get(key):
            out[key] = float(out[key]) * ratio
    return out


def _source_tile(level):
    """Reuse the source's own tiling so nothing is re-gridded needlessly."""
    chunks = getattr(level.data, 'chunks', None)
    axes = infer_axes(level.data, getattr(level, 'axes', None))
    if not chunks:
        return _DEFAULT_TILE
    th, tw = chunks[axes.index('Y')], chunks[axes.index('X')]
    # TIFF requires tile dimensions to be multiples of 16.
    if th % 16 or tw % 16 or th < 16 or tw < 16:
        return _DEFAULT_TILE
    return (int(th), int(tw))


# --------------------------------------------------------------------------
# the one-liner
# --------------------------------------------------------------------------

def convert(src, dst, *, series=0, levels=None, tile=None,
            compression='zstd', **kwargs):
    """Convert a slide file to a pyramidal, calibrated OME-TIFF.

    The whole point of the exercise::

        ezslide.convert('slide.vsi', 'slide.ome.tif')

    Parameters
    ----------
    src, dst : path
        Input slide (anything ezslide reads) and output ``.ome.tif``.
    series : int or 'all'
        Which series to write. The default ``0`` is the primary image —
        ``VsiFile`` orders its stacks slide-first, so this is the slide rather
        than the label scan. ``'all'`` writes every series in the file.
    levels, tile, compression
        As for :func:`write_ome_tiff`.
    """
    src = Path(src)
    slide = _open(src)
    try:
        target = slide if series == 'all' else _with_siblings(slide, series)
        return write_ome_tiff(target, dst, levels=levels, tile=tile,
                              compression=compression, source=src,
                              reader=type(slide).__name__, **kwargs)
    finally:
        slide.close()


def _with_siblings(slide, series):
    """The selected series plus any sibling channels of the same image.

    Selecting series 0 of a fluorescence scan and getting only DAPI would be a
    quiet way to lose four fifths of a slide, so a selection is widened to the
    whole channel group.
    """
    chosen = slide[series]
    key = getattr(chosen, 'channel_group_key', None)
    if key is None:
        return chosen
    return [s for s in slide.series
            if getattr(s, 'channel_group_key', None) == key]


def _open(path):
    """Open with the format class that matches the suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in ('.vsi', '.ets'):
        return VsiFile(path)
    return TiffFile(path)
