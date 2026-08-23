"""
ezslide command line — the swiss-army knife for turning slides into OME-TIFF.

Everything here is *composition*. The library provides the mechanism —
streaming tiles, rechunking, pyramid generation, metadata preservation — and
this module supplies the policy that only makes sense at a command line: which
files become which channels, whether RGB is split apart, what to call things,
and where to stop.

That separation is possible because the writer is duck-typed (see
:mod:`ezslide.writers.ome_tiff`). Merging separate files into one
multi-channel image and splitting an RGB image into three channels are both
done here with small adapter classes; the writer learns nothing about either.

    ezslide convert slide.vsi slide.ome.tif
    ezslide convert DAPI.tif CY5.tif out.ome.tif --channel-names DAPI CY5
    ezslide convert he.tif out.ome.tif --split-rgb --tile-size 1024
    ezslide convert labels.tif out.ome.tif --mask
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from .array.rechunk import DEFAULT_MAX_MEM
from .formats.tiff import TiffFile, infer_axes
from .formats.vsi import VsiFile
from .writers.ome_tiff import write_ome_tiff

__all__ = ['main', 'convert_command']

_VSI_SUFFIXES = ('.vsi', '.ets')


# --------------------------------------------------------------------------
# adapters — the whole reason the library needs no new features
# --------------------------------------------------------------------------

class _SampleView:
    """One sample of an interleaved level, presented as a ``YX`` level.

    ``--split-rgb``: an RGB image is one OME channel of three samples, which is
    right for brightfield but wrong when you want to merge it with fluorescence
    channels. Slicing the sample axis off turns each into an ordinary
    single-channel plane. The same idea as ``pyramid_assemble.py``'s
    ``SampleSplitter``, expressed as a level rather than a raw array.
    """

    def __init__(self, level, sample):
        self._level = level          # NOTE: a plain attribute, not a tifffile
        self._sample = sample        # series; the writer only duck-types it
        self.axes = 'YX'
        self.data = _SampleArray(level.data, level, sample)
        self.metadata = dict(level.metadata)
        self.metadata.pop('ChannelNames', None)

    @property
    def width(self):
        return self.data.shape[1]

    def __getitem__(self, key):
        return self.data[key]


class _SampleArray:
    """Array view of one sample; carries the shape/chunks the planner reads."""

    def __init__(self, data, level, sample):
        axes = infer_axes(data, getattr(level, 'axes', None))
        self._data = data
        self._sample = sample
        self._s_ax = axes.index('S')
        self._y, self._x = axes.index('Y'), axes.index('X')
        self.shape = (data.shape[self._y], data.shape[self._x])
        chunks = getattr(data, 'chunks', None)
        self.chunks = ((chunks[self._y], chunks[self._x]) if chunks
                       else (512, 512))
        self.dtype = np.dtype(data.dtype)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (2 - len(key))
        full = [slice(None)] * (max(self._y, self._x, self._s_ax) + 1)
        full[self._y], full[self._x] = key[0], key[1]
        full[self._s_ax] = self._sample
        return np.asarray(self._data[tuple(full)])


class _AsChannel:
    """A series relabelled as one channel of a shared multi-channel image.

    Gives every member the same ``channel_group_key``, which is all
    ``_channel_groups`` needs to merge them into a single ``CYX`` OME image.
    """

    def __init__(self, series, key, name, levels=None):
        self._series = series
        self.levels = list(levels if levels is not None else series.levels)
        self.channel = name
        self.channel_group_key = key
        self.channel_names = [name] if name else None
        self.name = getattr(series, 'name', None)


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------

def _open(path):
    return VsiFile(path) if Path(path).suffix.lower() in _VSI_SUFFIXES \
        else TiffFile(path)


def _pick_series(slide, spec, scene):
    """The series a path refers to, honouring a ``,N`` page suffix."""
    if spec is not None:
        return slide[spec]
    return slide[scene or 0]


def _split_spec(text):
    """``path,3`` -> ``(path, 3)``; ``path`` -> ``(path, None)``.

    The suffix convention ``pyramid_assemble.py`` uses to name one page of a
    multi-page input.
    """
    head, sep, tail = str(text).rpartition(',')
    if sep and tail.isdigit():
        return head, int(tail)
    return str(text), None


def _collect(paths, args):
    """Open every input and return the series list to hand the writer."""
    opened, series = [], []
    for path_spec in paths:
        path, page = _split_spec(path_spec)
        if not Path(path).exists():
            _die(f"input not found: {path}")
        slide = _open(path)
        opened.append(slide)
        chosen = _pick_series(slide, page, args.scene)
        series.append((Path(path).stem, chosen))

    planes = []
    for stem, ser in series:
        base = ser.levels[0]
        axes = infer_axes(base.data, getattr(base, 'axes', None))
        if args.split_rgb and 'S' in axes:
            n = base.data.shape[axes.index('S')]
            for i in range(n):
                planes.append((f'{stem}:{i}', ser,
                               [_SampleView(lv, i) for lv in ser.levels]))
        else:
            planes.append((stem, ser, None))

    _check_compatible(planes)

    names = args.channel_names
    if names and len(names) != len(planes):
        _die(f"--channel-names has {len(names)} entries but the output will "
             f"have {len(planes)} channels")

    # One input, no explicit names: leave it exactly as the reader saw it, so
    # an RGB brightfield stays one channel of three samples.
    if len(planes) == 1 and not names and planes[0][2] is None:
        return [planes[0][1]], opened

    key = ('ezslide-cli', id(planes))
    out = []
    for i, (stem, ser, levels) in enumerate(planes):
        name = names[i] if names else stem
        out.append(_AsChannel(ser, key, name, levels))
    return out, opened


def _check_compatible(planes):
    """Every channel of one image must agree on shape and dtype."""
    ref = None
    for stem, ser, levels in planes:
        base = (levels or ser.levels)[0]
        axes = infer_axes(base.data, getattr(base, 'axes', None))
        shape = (base.data.shape[axes.index('Y')],
                 base.data.shape[axes.index('X')])
        dtype = np.dtype(base.data.dtype)
        if ref is None:
            ref = (stem, shape, dtype)
        elif shape != ref[1]:
            _die(f"{stem}: shape {shape} does not match {ref[0]} {ref[1]}")
        elif dtype != ref[2]:
            _die(f"{stem}: dtype {dtype} does not match {ref[0]} {ref[2]}")


def _die(message):
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# the convert command
# --------------------------------------------------------------------------

def convert_command(args):
    out = Path(args.output)
    if out.exists() and not args.overwrite:
        _die(f"{out} exists; pass --overwrite to replace it")

    if args.num_threads:
        import tifffile
        tifffile.TIFF.MAXWORKERS = args.num_threads
        tifffile.TIFF.MAXIOWORKERS = args.num_threads * 5

    series, opened = _collect(args.inputs, args)
    overrides = {}
    if args.pixel_size:
        overrides.update(PhysicalSizeX=args.pixel_size, PhysicalSizeXUnit='µm',
                         PhysicalSizeY=args.pixel_size, PhysicalSizeYUnit='µm')
    if args.name:
        overrides['Name'] = args.name

    tile = (args.tile_size, args.tile_size)
    print(f"Writing {out}")
    print(f"    inputs      : {len(args.inputs)} -> {len(series)} channel(s)")
    print(f"    tile        : {tile[0]}x{tile[1]}  compression={args.compression}")
    print(f"    pyramid     : {'generate' if args.pyramid else 'as-is'}"
          f"  downsample={'mode (mask)' if args.mask else 'mean'}")
    try:
        write_ome_tiff(
            series if len(series) > 1 else series[0], out,
            tile=tile, compression=args.compression, levels=args.levels,
            pyramid=args.pyramid, downsample='mode' if args.mask else 'mean',
            metadata=overrides or None, max_mem=args.max_mem,
            source=args.inputs[0], reader='ezslide-cli',
        )
    finally:
        for slide in opened:
            try:
                slide.close()
            except Exception:  # noqa: BLE001 - never mask a write error
                pass

    size = out.stat().st_size / 1e6
    print(f"    wrote       : {size:,.1f} MB")
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        prog='ezslide',
        description='Read whole-slide images and write calibrated OME-TIFF.')
    subs = parser.add_subparsers(dest='command', required=True)

    c = subs.add_parser(
        'convert', help='convert slides to a pyramidal OME-TIFF',
        description='Convert one or more images into a single pyramidal, '
                    'calibrated OME-TIFF. Several inputs become several '
                    'channels of one image.')
    c.add_argument('inputs', nargs='+', metavar='INPUT',
                   help='input slides. Append ",N" to select one page.')
    c.add_argument('output', metavar='OUTPUT.ome.tif')
    c.add_argument('--tile-size', type=int, default=512, metavar='PIXELS',
                   help='output tile size, a multiple of 16 (default 512)')
    c.add_argument('--compression', default='zstd',
                   help='tile codec (default zstd, lossless)')
    c.add_argument('--levels', type=int, default=None, metavar='N',
                   help='cap the pyramid depth')
    c.add_argument('--no-pyramid', dest='pyramid', action='store_false',
                   help='write only the levels the input already has')
    c.add_argument('--mask', action='store_true',
                   help='label/mask image: downsample by nearest neighbour so '
                        'labels are never averaged into values that do not exist')
    c.add_argument('--split-rgb', action='store_true',
                   help='split interleaved RGB into three separate channels')
    c.add_argument('--channel-names', nargs='+', metavar='NAME',
                   help='names for the output channels, one per channel')
    c.add_argument('--pixel-size', type=float, default=None, metavar='MICRONS',
                   help='override the physical pixel size')
    c.add_argument('--name', default=None, help='override the image name')
    c.add_argument('--scene', type=int, default=0, metavar='N',
                   help='scene/series to read from each input (default 0)')
    c.add_argument('--max-mem', type=int, default=DEFAULT_MAX_MEM,
                   metavar='BYTES', help='ceiling on the rechunk read buffer')
    c.add_argument('--num-threads', type=int, default=0, metavar='N',
                   help='tifffile worker threads (default: leave alone)')
    c.add_argument('--overwrite', action='store_true',
                   help='replace OUTPUT if it exists')
    c.set_defaults(pyramid=True, func=convert_command)
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.tile_size % 16:
        _die(f"--tile-size must be a multiple of 16, got {args.tile_size}")
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
