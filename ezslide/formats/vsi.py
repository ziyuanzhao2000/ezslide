"""
formats.vsi — Olympus / EVIDENT cellSens ``.vsi`` slides on the ezslide stack.

:mod:`ezslide.vendors.vsitiff` already does the hard part: it synthesizes
BigTIFF IFDs over the untouched ``.ets`` pixel container and hands back a real
``tifffile.TiffFile``. Everything ezslide builds on top of tifffile —
``aszarr()``, the global chunk cache, the tensorstore views, ``lazy_pyramid`` —
therefore works on a ``.vsi`` without knowing what it is reading, which is why
this module subclasses :mod:`ezslide.formats.tiff` instead of restating it.

What this module adds is the part the prosthetic cannot carry. The synthetic
IFDs hold only the thirteen tags needed to decode tiles, so a ``.vsi`` opened
through them has no resolution, no name, and no channel identity. Those live
in the ``.vsi`` tag tree, which :mod:`ezslide.vendors.vsimeta` parses and
``vsitiff`` pairs with each ``.ets`` stack; :class:`VsiSeries` folds that
pairing into the same ``PhysicalSizeX`` / ``PhysicalSizeY`` keys a real slide
would have supplied, so ``SlideProperties.mpp`` comes out calibrated rather
than ``None``.

    import ezslide
    slide = ezslide.VsiFile('slide.vsi')
    print(slide)
    print(slide[0].levels[0].metadata['PhysicalSizeX'])

Through wsidata, use the registered reader::

    from wsidata import open_wsi
    wsi = open_wsi('slide.vsi', reader='vsi_zarr')
"""

import itertools
from dataclasses import replace as _dataclass_replace

import numpy as np

from ..vendors import vsitiff
from .tiff import TiffFile, TiffSeries

__all__ = ['VsiFile', 'VsiSeries', 'open_vsi']


#: Tile tables are per-tile arrays that can run to hundreds of thousands of
#: entries on a whole slide. They say nothing a caller wants and they are what
#: ``SlideProperties.raw`` would stringify, so they are dropped from the
#: level metadata of a synthetic IFD.
_BULKY_TAGS = ('TileOffsets', 'TileByteCounts', 'StripOffsets',
               'StripByteCounts')


class VsiSeries(TiffSeries):
    """One ``.ets`` stack, calibrated from the ``.vsi`` tag tree.

    Identical to :class:`~ezslide.formats.tiff.TiffSeries` in every respect
    except metadata: the levels come from synthetic IFDs, so their tags
    describe the tile grid and nothing else, and the physical scale, name and
    channels are injected from the paired ``PyramidMetadata``.
    """

    def __init__(self, ets, channel=None, plane=None, pyramidalize=False,
                 pyramid=None):
        # Set before super().__init__: TiffSeries.__getattr__ delegates through
        # self._series, so an attribute miss before that exists would recurse.
        self._ets = ets
        self._channel = channel
        self._plane = dict(plane or {})
        super().__init__(ets.series, pyramidalize=pyramidalize, pyramid=pyramid)
        for level in self._levels:
            for tag in _BULKY_TAGS:
                level._metadata.pop(tag, None)

    @property
    def ets(self):
        """The underlying :class:`vsitiff.EtsPyramid`."""
        return self._ets

    @property
    def name(self):
        if self._channel is None:
            return self._ets.name
        return f'{self._ets.name} ({self._channel})'

    @property
    def channel(self):
        """Label of the plane this series selects, or None if it is the whole
        stack (an RGB brightfield scan is one plane with three samples)."""
        return self._channel

    @property
    def plane(self):
        """``{ets_axis: value}`` selected out of the ETS coordinate space."""
        return dict(self._plane)

    @property
    def channel_group_key(self):
        """Identify the stack this plane belongs to, for the OME writer.

        ETS keeps every channel of a scan in one ``.ets`` chunk table and
        ezslide splits them into a series each, which is the right shape for
        reading but the wrong shape for OME: there, a fluorescence scan is one
        image with N channels. Series sharing this key are merged back into a
        single ``CYX`` image on write. ``None`` for a brightfield stack, which
        is already one image.
        """
        if self._channel is None:
            return None
        return (str(self._ets.path), self._ets.stack)

    @property
    def is_overview(self):
        """True for the label / macro / overview stacks, not the slide."""
        return self._ets.is_overview

    @property
    def magnification(self):
        return self._ets.magnification

    @property
    def channel_names(self):
        if self._channel is not None:
            return [self._channel]
        return self._ets.channel_names

    def parse_metadata(self, kind):
        """Populate every level with the calibration from the .vsi tag tree.

        Runs the ordinary TIFF pass first, which finds nothing here — a
        synthetic IFD carries no resolution tags — then overwrites with the
        real values. Doing it per level rather than scaling level 0 keeps the
        in-file levels and any lazy levels on the same footing, since each
        one's ratio to level 0 is a fact about its own extent.
        """
        super().parse_metadata(kind)

        ets = self._ets
        meta = ets.meta
        shared = {'Name': self.name, 'Codec': ets.header.codec,
                  'EtsFile': str(ets.path), 'Stack': ets.stack}
        if self.channel_names:
            shared['ChannelNames'] = list(self.channel_names)
        if self._channel is not None:
            shared['Channel'] = self._channel
            shared['PlaneIndex'] = ', '.join(f'axis{a}={v}'
                                             for a, v in sorted(self._plane.items()))
        if meta is not None:
            for key, value in (
                ('Magnification', meta.magnification),
                ('NumericalAperture', meta.numerical_aperture),
                ('Objective', '; '.join(meta.objective_names) or None),
                ('Device', '; '.join(meta.device_names) or None),
                ('StackType', meta.stack_type),
                ('StageOriginX', meta.origin_x),
                ('StageOriginY', meta.origin_y),
                ('BitDepth', meta.bit_depth),
            ):
                if value is not None:
                    shared[key] = value

        base_width = self._levels[0].width
        for level in self._levels:
            parsed = dict(level._parsed_metadata)
            parsed.update(shared)
            if ets.mpp_x is not None and level.width:
                downsample = base_width / level.width
                parsed['PhysicalSizeX'] = ets.mpp_x * downsample
                parsed['PhysicalSizeY'] = (ets.mpp_y or ets.mpp_x) * downsample
                parsed['PhysicalSizeXUnit'] = 'µm'
                parsed['PhysicalSizeYUnit'] = 'µm'
            level._parsed_metadata = parsed


class VsiFile(TiffFile):
    """A ``.vsi`` dataset (or a lone ``.ets``) with the ezslide TiffFile API.

    Each ``.ets`` stack becomes one series, so ``slide[0]`` is a series exactly
    as it is for an SVS. The stacks are ordered slide-first: a cellSens dataset
    stores the label and overview scans as their own stacks, and they are
    written *before* the real slide, so the file's own order would make
    ``slide[0]`` the label.

    Parameters
    ----------
    file : path
        The ``.vsi`` container, or a single ``.ets`` file.
    pyramidalize, pyramid
        As for :class:`~ezslide.formats.tiff.TiffFile`. ETS stacks carry a
        real pyramid, so this normally does nothing.
    base_size : (h, w) or {stack_name: (h, w)}, optional
        Override the true level-0 size. By default it comes from the .vsi tag
        tree, falling back to the padded tile grid.
    include_all_ets : bool
        Also load ``blob_*.ets`` sidecars, which are serialized masks and
        overlays rather than image planes.
    slide_first : bool
        Order the series slide-first (the default). Set False to keep the
        order the stacks are discovered in.
    expand_planes : bool
        Give every channel (and any other extra ETS axis) its own series, so
        a fluorescence scan exposes all of its channels. With False only the
        first plane of each stack is read, which is what ``ets_to_tifffile``
        does on its own.
    """

    def __init__(self, file, kind='vsi', *args, **kwargs):
        super().__init__(file, kind, *args, **kwargs)

    def _open(self, file, *args, pyramidalize=False, pyramid=None,
              base_size=None, include_all_ets=False, slide_first=True,
              expand_planes=True, **kwargs):
        self._vsi = vsitiff.VsiFile(file, base_size=base_size,
                                    include_all_ets=include_all_ets, **kwargs)
        # The .vsi container is a genuine TIFF holding the label/macro
        # thumbnails; let it answer attribute lookups. It is None when the
        # input was a bare .ets, or when the container failed to open.
        self._tifffile = self._vsi.container

        pyramids = list(self._vsi.pyramids)
        if not pyramids:
            failed = ', '.join(f'{p.name} ({exc})' for p, exc in self._vsi.failures)
            raise ValueError(
                f'{file}: no readable .ets pyramid found'
                + (f'; failures: {failed}' if failed else '')
            )
        if slide_first:
            pyramids.sort(key=_slide_first_key)

        # Overlay streams this class opened itself, which the vsitiff VsiFile
        # does not know about and so will not close.
        self._extra_tifs = []
        self._ets_pyramids, series = [], []
        for parent in pyramids:
            planes = (_expand_planes(parent) if expand_planes
                      else [(parent, None, {})])
            for ets, channel, plane in planes:
                if ets is not parent:
                    self._extra_tifs.append(ets.tif)
                self._ets_pyramids.append(ets)
                series.append(VsiSeries(ets, channel=channel, plane=plane,
                                        pyramidalize=pyramidalize,
                                        pyramid=pyramid))
        return series

    @property
    def ets_pyramids(self):
        """The :class:`vsitiff.EtsPyramid` behind each series, same order."""
        return list(self._ets_pyramids)

    @property
    def vsi(self):
        """The underlying :class:`vsitiff.VsiFile`."""
        return self._vsi

    @property
    def metadata(self):
        """The parsed ``.vsi`` tag tree, or None if it could not be read."""
        return self._vsi.metadata

    @property
    def thumbnails(self):
        """Label / macro images stored as ordinary IFDs in the .vsi."""
        return self._vsi.thumbnails

    def __repr__(self):
        lines = [f'VsiFile from {self._file} with {len(self.series)} series: ']
        for series in self._series:
            shape = series.levels[0].shape
            tail = ' (overview)' if series.is_overview else ''
            lines.append(f'  {series.name!r} {shape} '
                         f'{len(series.levels)} levels{tail}')
        return ' \n'.join(lines)

    def close(self):
        # The vsitiff VsiFile owns the container and one overlay stream per
        # stack; the per-plane streams are ours. Tolerant of a half-built
        # object, since __del__ reaches an instance whose _open raised.
        for tif in getattr(self, '_extra_tifs', ()):
            tif.close()
        self._extra_tifs = []
        vsi = getattr(self, '_vsi', None)
        if vsi is not None:
            vsi.close()


def _slide_first_key(pyramid):
    """Sort image stacks before overviews, then by descending level-0 area."""
    height, width = pyramid.level_shapes[0][:2]
    return (pyramid.is_overview, -int(height) * int(width))


def _extra_axes(header):
    """ETS coordinate axes beyond Y/X and the pyramid, that actually vary.

    In a multichannel scan the channel is one of these: the tiles of every
    channel live in the same ``.ets`` chunk table, distinguished only by a
    coordinate. ``ets_to_tifffile`` selects one value per axis, so without
    expansion a five-channel slide reads as its first channel alone.
    """
    level_axis = vsitiff.infer_level_axis(header.coords)
    out = []
    for axis in range(2, header.ndim):
        if axis == level_axis:
            continue
        values = np.unique(header.coords[:, axis])
        if values.size > 1:
            out.append((axis, [int(v) for v in values]))
    return out


def _expand_planes(parent):
    """``[(EtsPyramid, label, {axis: value})]`` — one entry per extra-axis plane.

    Returns the pyramid unchanged when there is nothing to expand, which is
    the brightfield case: an RGB scan is a single plane of three samples.
    """
    axes = _extra_axes(parent.header)
    if not axes:
        return [(parent, None, {})]

    names = parent.meta.channel_names if parent.meta is not None else []
    # The channel axis is the one whose cardinality matches the channel names
    # the tag tree lists; anything else is labelled by its raw coordinate.
    channel_axis = next((a for a, values in axes if len(values) == len(names)
                         and names), None)

    out = []
    base_size = parent.meta.base_size if parent.meta is not None else None
    for combo in itertools.product(*[values for _, values in axes]):
        plane = dict(zip([a for a, _ in axes], combo))
        label = ' '.join(
            names[value] if axis == channel_axis and value < len(names)
            else f'axis{axis}={value}'
            for axis, value in sorted(plane.items()))
        if getattr(parent.tif, '_vsitiff_index', None) == plane:
            out.append((parent, label, plane))       # reuse the open stream
            continue
        tif = vsitiff.ets_to_tifffile(parent.path, header=parent.header,
                                      base_size=base_size, index=plane)
        out.append((_dataclass_replace(parent, tif=tif), label, plane))
    return out


def open_vsi(file, **kwargs) -> VsiFile:
    return VsiFile(file, **kwargs)
