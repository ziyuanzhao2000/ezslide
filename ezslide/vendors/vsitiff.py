"""
vsitiff — read Olympus / EVIDENT cellSens ``.vsi`` virtual slides using
``tifffile`` as the *only* TIFF-like parsing and decoding engine.

The idea
--------
A ``.vsi`` dataset is two things:

1. ``slide.vsi`` — a real little-endian TIFF file (magic ``II*\\0``). It holds
   the label / macro / overview thumbnails as ordinary IFDs plus a proprietary
   nested metadata tree. ``tifffile.TiffFile`` opens this directly, today.

2. ``_slide_/stackN/frame_t.ets`` — the actual pyramid pixel data. This is
   *not* TIFF. It is an Olympus "SIS/ETS" container: a small header, an
   "additional header" describing pixel type / tile size / codec, and a flat
   chunk table of ``(coordinate..., offset, nbytes)`` records.

The crucial observation is that an ETS file is **structurally isomorphic to a
tiled TIFF**. A tiled TIFF IFD is, in essence, exactly:

    TileWidth, TileLength, Compression, Photometric, TileOffsets[], TileByteCounts[]

...and ETS gives us every one of those. The tile payloads themselves are
plain JPEG / JPEG2000 / PNG / raw streams — the same codecs tifffile already
dispatches to imagecodecs for.

So instead of writing a decoder, we write a **~200 byte prosthetic**: we
synthesize BigTIFF IFDs whose ``TileOffsets`` point *into the untouched ETS
file*, and present tifffile with a virtual stream

    [ 16-byte BigTIFF header ] [ ...original .ets bytes... ] [ synthetic IFDs ]

Every ETS byte offset simply shifts by 16. Nothing is copied, nothing is
re-encoded, no temporary file is written. From that point on tifffile *is*
the reader: lazy per-tile access, `asarray(key=...)`, multi-resolution
`series[0].levels`, `aszarr()` → zarr/dask/napari, `segments()` streaming,
and imagecodecs-backed JPEG/J2K decode.

Caveats worth knowing
---------------------
* The ETS layout constants below are reverse-engineered (they agree with the
  Bio-Formats ``CellSensReader``). This module validates aggressively rather
  than trusting them: the additional-header magic is checked, the codec is
  sniffed from actual tile magic bytes, and the pyramid axis is *inferred*
  from the tile grid geometry instead of assumed.
* ETS stores whole padded tiles, so the tile grid is an upper bound on the
  true image size. Pass ``base_size=(h, w)`` if you know it (or read it from
  the .vsi metadata) and every level is cropped consistently. Declaring a
  smaller ImageWidth/Length than the grid is legal TIFF and tifffile crops
  the edge tiles for you.

Usage
-----
    import vsitiff
    vsi = vsitiff.VsiFile('slide.vsi')
    print(vsi.pyramids)                 # discovered .ets stacks
    p = vsi.largest_pyramid()
    print(p.shape, p.dtype, p.level_shapes)
    tile = p.asarray(level=2)           # numpy
    z = p.aszarr()                      # multiscale zarr store
    d = p.to_dask()                     # list of dask arrays, one per level
    region = p.read_region((4000, 3000), (1024, 1024), level=0)
"""

from __future__ import annotations

import io
import os
import struct
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

try:
    from .vsimeta import PyramidMetadata, parse_vsi_metadata
except ImportError:  # metadata parsing is optional
    PyramidMetadata = None  # type: ignore[assignment,misc]
    parse_vsi_metadata = None  # type: ignore[assignment]

__all__ = ['VsiFile', 'EtsPyramid', 'EtsHeader', 'open_vsi', 'ets_to_tifffile',
           'ets_level0_grid', 'write_ome_tiff']


# --------------------------------------------------------------------------
# 1. ETS container parsing (the only non-tifffile code in this module)
# --------------------------------------------------------------------------

# Olympus pixel-type enum -> numpy dtype.  Matches Bio-Formats CellSensReader.
_PIXEL_TYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    8: np.float32,
    9: np.float64,
}

# Olympus codec enum -> friendly name.  Used only as a *hint*; the real codec
# is sniffed from the tile's magic bytes, which is far more reliable.
_CODEC_HINT = {0: 'raw', 2: 'jpeg', 3: 'jpeg2000', 5: 'jpeg2000', 8: 'png', 9: 'bmp'}

# codec name -> TIFF Compression tag value understood by tifffile
_TIFF_COMPRESSION = {
    'raw': 1,        # NONE
    'jpeg': 7,       # JPEG (new-style)
    'jpeg2000': 34712,  # JP2 / J2K, dispatched to imagecodecs.jpeg2k_decode
    'png': 34933,    # PNG-in-TIFF
    'bmp': None,     # not representable; handled out-of-band
}


def _sniff_codec(payload: bytes) -> str:
    """Identify a tile codec from its leading bytes."""
    if payload[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if payload[:4] == b'\xff\x4f\xff\x51':          # raw J2K codestream (SOC)
        return 'jpeg2000'
    if payload[4:8] == b'jP  ' or payload[:4] == b'\x00\x00\x00\x0c':
        return 'jpeg2000'                            # JP2 box container
    if payload[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    if payload[:2] == b'BM':
        return 'bmp'
    return 'raw'


@dataclass
class EtsHeader:
    """Everything the ETS container tells us."""

    path: Path
    ndim: int
    tile_width: int
    tile_height: int
    tile_depth: int
    samples: int
    dtype: np.dtype
    codec: str
    codec_id: int
    colorspace: int
    quality: int
    coords: np.ndarray            # (nchunks, ndim) int32
    offsets: np.ndarray           # (nchunks,) uint64 — absolute in the .ets file
    bytecounts: np.ndarray        # (nchunks,) uint64
    file_size: int


def parse_ets(path: str | os.PathLike) -> EtsHeader:
    """Parse an Olympus ``.ets`` chunk container.

    Only the table of contents is read — no pixel data is touched.
    """
    path = Path(path)
    size = path.stat().st_size
    with open(path, 'rb') as fh:
        head = fh.read(64)
        if head[:3] != b'SIS':
            msg = f'{path.name}: not an Olympus SIS/ETS file (magic={head[:4]!r})'
            raise ValueError(msg)

        # SIS header (little-endian throughout)
        (_magic, _header_size, _version, ndim, add_offset, _add_size,
         _reserved, chunk_offset, nchunks) = struct.unpack_from(
            '<4siiiQiiQi', head, 0
        )

        if not (0 < ndim <= 16) or not (0 < add_offset < size):
            msg = (f'{path.name}: implausible ETS header '
                   f'(ndim={ndim}, additional_header_offset={add_offset})')
            raise ValueError(msg)

        # --- additional header: pixel format + tile geometry + codec ---
        fh.seek(add_offset)
        add = fh.read(64)
        if add[:3] != b'ETS':
            msg = (f'{path.name}: additional header magic is {add[:4]!r}, '
                   'expected ETS — layout assumption broken')
            raise ValueError(msg)
        (pixel_type, samples, colorspace, codec_id, quality,
         tile_w, tile_h, tile_d) = struct.unpack_from('<8i', add, 8)

        if pixel_type not in _PIXEL_TYPES:
            msg = f'{path.name}: unknown ETS pixel type {pixel_type}'
            raise ValueError(msg)
        dtype = np.dtype(_PIXEL_TYPES[pixel_type])
        if not (0 < samples <= 4) or not (0 < tile_w <= 1 << 16) \
                or not (0 < tile_h <= 1 << 16):
            msg = (f'{path.name}: implausible tile geometry '
                   f'({tile_w}x{tile_h}, {samples} samples) — the additional '
                   f'header at {add_offset} does not look like ETS field order')
            raise ValueError(msg)

        # --- chunk table: coordinate vector + offset + length, per tile ---
        rec = struct.Struct(f'<i{ndim}iQii')
        fh.seek(chunk_offset)
        buf = fh.read(rec.size * nchunks)
        if len(buf) < rec.size * nchunks:
            msg = f'{path.name}: chunk table truncated'
            raise ValueError(msg)

        coords = np.empty((nchunks, ndim), np.int32)
        offsets = np.empty(nchunks, np.uint64)
        counts = np.empty(nchunks, np.uint64)
        for i in range(nchunks):
            vals = rec.unpack_from(buf, i * rec.size)
            coords[i] = vals[1:1 + ndim]
            offsets[i] = vals[1 + ndim]
            counts[i] = vals[2 + ndim]

        if offsets.max() + counts[int(np.argmax(offsets))] > size:
            msg = f'{path.name}: chunk offsets point past end of file'
            raise ValueError(msg)

        # --- sniff the real codec from the first tile ---
        fh.seek(int(offsets[0]))
        codec = _sniff_codec(fh.read(min(16, int(counts[0]))))

    hint = _CODEC_HINT.get(codec_id)
    if hint is not None and hint != codec:
        warnings.warn(
            f'{path.name}: header codec id {codec_id} suggests {hint!r} but '
            f'tile magic says {codec!r}; trusting the tile.',
            stacklevel=2,
        )

    return EtsHeader(
        path=path, ndim=ndim, tile_width=tile_w, tile_height=tile_h,
        tile_depth=tile_d, samples=samples, dtype=dtype, codec=codec,
        codec_id=codec_id, colorspace=colorspace, quality=quality,
        coords=coords, offsets=offsets, bytecounts=counts, file_size=size,
    )


def infer_level_axis(coords: np.ndarray) -> int | None:
    """Find which extra coordinate axis is the resolution pyramid.

    Rather than assuming "the last axis" we test each candidate axis for
    pyramid-like behaviour: as the axis value increases, the tile grid should
    shrink roughly geometrically. This is self-validating and survives files
    with extra Z/C/T axes in unexpected positions.
    """
    ndim = coords.shape[1]
    best, best_score = None, 0.0
    for axis in range(2, ndim):
        values = np.unique(coords[:, axis])
        if values.size < 2:
            continue
        extents = [
            max(coords[coords[:, axis] == v, 0].max(),
                coords[coords[:, axis] == v, 1].max()) + 1
            for v in values
        ]
        ratios = [a / b for a, b in zip(extents, extents[1:]) if b > 0]
        if not ratios:
            continue
        score = sum(1.4 <= r <= 3.0 for r in ratios) / len(ratios)
        if score > best_score:
            best, best_score = axis, score
    return best if best_score >= 0.5 else None


def ets_level0_grid(h: EtsHeader) -> tuple[int, int]:
    """``(grid_height, grid_width)`` in pixels of the full-resolution level.

    This is the *padded* extent — the true image is somewhere inside the last
    row/column of tiles. Used to sanity-check metadata pairing.
    """
    axis = infer_level_axis(h.coords)
    if axis is None:
        sel = np.ones(len(h.coords), bool)
    else:
        sel = h.coords[:, axis] == h.coords[:, axis].min()
    nx = int(h.coords[sel, 0].max()) + 1
    ny = int(h.coords[sel, 1].max()) + 1
    grid_w, grid_h = nx * h.tile_width, ny * h.tile_height
    if grid_w > MAX_PLAUSIBLE_PIXELS or grid_h > MAX_PLAUSIBLE_PIXELS:
        msg = (f'{h.path.name}: level-0 tile grid computes to {grid_w}x{grid_h} px, '
               f'which exceeds the {MAX_PLAUSIBLE_PIXELS} px sanity limit. '
               f'axis 0 max={int(h.coords[sel, 0].max())}, '
               f'axis 1 max={int(h.coords[sel, 1].max())}, '
               f'tile={h.tile_width}x{h.tile_height}, ndim={h.ndim}. '
               'Either the level axis was mis-inferred or an axis holds values '
               'rather than tile indices — run diagnose_vsi.py on this file.')
        raise ValueError(msg)
    return grid_h, grid_w


# --------------------------------------------------------------------------
# 2. The prosthetic: synthesize BigTIFF IFDs over the ETS byte range
# --------------------------------------------------------------------------

# No real whole-slide image is a million pixels on a side; anything larger is
# a parsing failure that must not be allowed to masquerade as an image shape.
MAX_PLAUSIBLE_PIXELS = 1 << 20

_BIGTIFF_HEADER_SIZE = 16
_SHORT, _LONG, _LONG8 = 3, 4, 16
_TYPESIZE = {_SHORT: 2, _LONG: 4, _LONG8: 8}


class _OverlayStream(io.RawIOBase):
    """A seekable read-only stream presenting ``prefix + ets_file + suffix``.

    tifffile accepts any binary stream, so this is all we need to make it
    believe an untouched .ets file is a BigTIFF. No copying, no temp files.
    Thread-safe so ``TiffFile`` can use its threaded tile readers.
    """

    def __init__(self, prefix: bytes, path: Path, ets_size: int, suffix: bytes,
                 name: str = 'ets-overlay.tif') -> None:
        super().__init__()
        self._prefix = prefix
        self._suffix = suffix
        self._ets_size = ets_size
        self._fh = open(path, 'rb')  # noqa: SIM115
        self._lock = threading.Lock()
        self._pos = 0
        self._p = len(prefix)
        self._s = self._p + ets_size
        self._size = self._s + len(suffix)
        self.name = name

    # -- io protocol ------------------------------------------------------
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, b) -> int:  # type: ignore[override]
        view = memoryview(b).cast('B')
        want = len(view)
        got = 0
        with self._lock:
            pos = self._pos
            while got < want and pos < self._size:
                if pos < self._p:                       # header region
                    chunk = self._prefix[pos:self._p][: want - got]
                elif pos < self._s:                     # untouched .ets bytes
                    self._fh.seek(pos - self._p)
                    chunk = self._fh.read(min(want - got, self._s - pos))
                else:                                   # synthetic IFDs
                    off = pos - self._s
                    chunk = self._suffix[off:off + (want - got)]
                if not chunk:
                    break
                view[got:got + len(chunk)] = chunk
                got += len(chunk)
                pos += len(chunk)
            self._pos = pos
        return got

    def close(self) -> None:
        if not self.closed:
            try:
                self._fh.close()
            finally:
                super().close()


@dataclass
class _LevelSpec:
    width: int
    height: int
    tile_width: int
    tile_height: int
    samples: int
    dtype: np.dtype
    compression: int
    photometric: int
    offsets: list[int]
    bytecounts: list[int]
    reduced: bool
    subsampling: tuple[int, int] | None = None


def _sample_format(dtype: np.dtype) -> int:
    if dtype.kind == 'u':
        return 1
    if dtype.kind == 'i':
        return 2
    if dtype.kind == 'f':
        return 3
    return 1


def _build_ifds(levels: list[_LevelSpec], data_base: int) -> bytes:
    """Serialize BigTIFF IFDs. ``data_base`` is where this blob will live."""
    # Pass 1 — how big is each IFD?
    entry_counts = []
    for lv in levels:
        n = 13 + (1 if lv.photometric == 6 else 0)
        entry_counts.append(n)
    ifd_sizes = [8 + 20 * n + 8 for n in entry_counts]
    ifd_offsets, cursor = [], data_base
    for sz in ifd_sizes:
        ifd_offsets.append(cursor)
        cursor += sz
    extra_base = cursor

    ifd_blob = bytearray()
    extra_blob = bytearray()

    def stash(payload: bytes) -> int:
        off = extra_base + len(extra_blob)
        extra_blob.extend(payload)
        if len(extra_blob) % 2:
            extra_blob.append(0)
        return off

    for i, lv in enumerate(levels):
        entries: list[tuple[int, int, int, bytes]] = []

        def add(tag: int, ftype: int, values) -> None:
            values = list(values)
            payload = b''.join(
                struct.pack({_SHORT: '<H', _LONG: '<I', _LONG8: '<Q'}[ftype], v)
                for v in values
            )
            if len(payload) <= 8:
                field_ = payload.ljust(8, b'\0')
            else:
                field_ = struct.pack('<Q', stash(payload))
            entries.append((tag, ftype, len(values), field_))

        sf = _sample_format(lv.dtype)
        bits = lv.dtype.itemsize * 8
        ntiles = len(lv.offsets)

        add(254, _LONG, [1 if lv.reduced else 0])          # NewSubfileType
        add(256, _LONG, [lv.width])                        # ImageWidth
        add(257, _LONG, [lv.height])                       # ImageLength
        add(258, _SHORT, [bits] * lv.samples)              # BitsPerSample
        add(259, _SHORT, [lv.compression])                 # Compression
        add(262, _SHORT, [lv.photometric])                 # Photometric
        add(277, _SHORT, [lv.samples])                     # SamplesPerPixel
        add(284, _SHORT, [1])                              # PlanarConfig=contig
        add(322, _LONG, [lv.tile_width])                   # TileWidth
        add(323, _LONG, [lv.tile_height])                  # TileLength
        add(324, _LONG8, lv.offsets)                       # TileOffsets
        add(325, _LONG8, lv.bytecounts)                    # TileByteCounts
        add(339, _SHORT, [sf] * lv.samples)                # SampleFormat
        if lv.photometric == 6:
            add(530, _SHORT, list(lv.subsampling or (2, 2)))

        entries.sort(key=lambda e: e[0])
        assert len(entries) == entry_counts[i], (len(entries), entry_counts[i])
        assert ntiles == len(lv.bytecounts)

        ifd_blob.extend(struct.pack('<Q', len(entries)))
        for tag, ftype, count, field_ in entries:
            ifd_blob.extend(struct.pack('<HHQ', tag, ftype, count))
            ifd_blob.extend(field_)
        nxt = ifd_offsets[i + 1] if i + 1 < len(levels) else 0
        ifd_blob.extend(struct.pack('<Q', nxt))

    assert len(ifd_blob) == sum(ifd_sizes)
    return bytes(ifd_blob) + bytes(extra_blob), ifd_offsets[0]


def ets_to_tifffile(
    ets_path: str | os.PathLike,
    *,
    header: EtsHeader | None = None,
    base_size: tuple[int, int] | None = None,
    index: dict[int, int] | None = None,
    photometric: int | None = None,
) -> tifffile.TiffFile:
    """Open an ``.ets`` file **as a tifffile.TiffFile**, zero-copy.

    Parameters
    ----------
    base_size:
        True ``(height, width)`` of level 0 if known. Defaults to the padded
        tile-grid extent.
    index:
        For files with extra Z/C/T axes, ``{axis: value}`` selecting the plane.
        Defaults to the first value on each extra axis.
    photometric:
        Override the TIFF PhotometricInterpretation if auto-detection is wrong
        (common toggle: 2 = RGB vs 6 = YCbCr for JPEG tiles).
    """
    h = header or parse_ets(ets_path)
    compression = _TIFF_COMPRESSION.get(h.codec)
    if compression is None:
        msg = f'codec {h.codec!r} has no TIFF equivalent tifffile can decode'
        raise NotImplementedError(msg)

    if photometric is None:
        if h.samples >= 3:
            # JPEG tiles from cellSens are almost always YCbCr-encoded; other
            # codecs deliver RGB directly.
            photometric = 6 if h.codec == 'jpeg' else 2
        else:
            photometric = 1  # MINISBLACK
    if h.codec != 'jpeg' and photometric == 6:
        photometric = 2

    level_axis = infer_level_axis(h.coords)
    other_axes = [
        a for a in range(2, h.ndim) if a != level_axis
        and np.unique(h.coords[:, a]).size > 1
    ]
    index = dict(index or {})
    keep = np.ones(len(h.coords), bool)
    for axis in other_axes:
        value = index.get(axis, int(np.unique(h.coords[:, axis])[0]))
        index[axis] = value
        keep &= h.coords[:, axis] == value

    if level_axis is None:
        level_values = [0]
        level_of = np.zeros(len(h.coords), np.int32)
    else:
        level_values = list(np.unique(h.coords[keep, level_axis]))
        level_of = h.coords[:, level_axis]

    # Level sizes must be derived from ONE base size, not from each level's own
    # padded tile grid: ETS pads the last tile, so per-level grids give ragged
    # ratios (e.g. 768->512 instead of 768->384) and tifffile then refuses to
    # group the pages into a pyramid at all.
    if base_size is None:
        sel0 = keep & (level_of == level_values[0]) if level_axis is not None else keep
        base_size = (
            (int(h.coords[sel0, 1].max()) + 1) * h.tile_height,
            (int(h.coords[sel0, 0].max()) + 1) * h.tile_width,
        )

    specs: list[_LevelSpec] = []
    for li, lv in enumerate(level_values):
        sel = keep & (level_of == lv) if level_axis is not None else keep
        cx, cy = h.coords[sel, 0], h.coords[sel, 1]
        offs, cnts = h.offsets[sel], h.bytecounts[sel]
        nx, ny = int(cx.max()) + 1, int(cy.max()) + 1

        # Tile order in a TIFF IFD is strictly row-major; ETS chunk order is
        # not guaranteed, and sparse grids are legal. Scatter into a dense
        # grid, leaving holes as zero-length tiles (tifffile yields zeros).
        grid_off = np.zeros(ny * nx, np.uint64)
        grid_cnt = np.zeros(ny * nx, np.uint64)
        flat = cy.astype(np.int64) * nx + cx.astype(np.int64)
        grid_off[flat] = offs + _BIGTIFF_HEADER_SIZE      # <-- the whole trick
        grid_cnt[flat] = cnts

        grid_w, grid_h = nx * h.tile_width, ny * h.tile_height
        factor = 2 ** li
        width = -(-base_size[1] // factor)
        height = -(-base_size[0] // factor)
        if width > grid_w or height > grid_h:
            warnings.warn(
                f'{h.path.name}: level {li} grid {grid_w}x{grid_h} is smaller '
                f'than the 2**{li} downsample of the base size; the pyramid may '
                'not be a strict power-of-two chain. Falling back to grid size.',
                stacklevel=2,
            )
            width, height = min(width, grid_w), min(height, grid_h)

        specs.append(_LevelSpec(
            width=width, height=height,
            tile_width=h.tile_width, tile_height=h.tile_height,
            samples=h.samples, dtype=h.dtype, compression=compression,
            photometric=photometric,
            offsets=[int(v) for v in grid_off],
            bytecounts=[int(v) for v in grid_cnt],
            reduced=li > 0,
        ))

    data_base = _BIGTIFF_HEADER_SIZE + h.file_size
    blob, first_ifd = _build_ifds(specs, data_base)
    prefix = struct.pack('<2sHHHQ', b'II', 43, 8, 0, first_ifd)
    assert len(prefix) == _BIGTIFF_HEADER_SIZE

    stream = _OverlayStream(prefix, h.path, h.file_size, blob,
                            name=h.path.stem + '.ets.tif')
    tif = tifffile.TiffFile(stream, name=h.path.stem + '.ets.tif')
    tif.filehandle.lock = True          # allow tifffile's threaded tile reads
    tif._vsitiff_header = h             # keep provenance attached
    tif._vsitiff_index = index
    return tif


# --------------------------------------------------------------------------
# 3. Friendly front end
# --------------------------------------------------------------------------

@dataclass
class EtsPyramid:
    """One ``.ets`` stack, presented as a tifffile pyramid."""

    path: Path
    header: EtsHeader
    tif: tifffile.TiffFile
    stack: str = ''
    meta: 'PyramidMetadata | None' = None

    # -- metadata surfaced from the .vsi tag tree -------------------------
    @property
    def name(self) -> str:
        if self.meta is not None and self.meta.name:
            return self.meta.name
        return self.stack or self.path.stem

    @property
    def mpp(self) -> float | None:
        """Micrometres per pixel at level 0, or None if uncalibrated."""
        return None if self.meta is None else self.meta.mpp

    @property
    def mpp_x(self) -> float | None:
        return None if self.meta is None else self.meta.mpp_x

    @property
    def mpp_y(self) -> float | None:
        return None if self.meta is None else self.meta.mpp_y

    def level_mpp(self, level: int) -> float | None:
        """µm/pixel at a given pyramid level, from the true level ratio."""
        if self.mpp_x is None:
            return None
        base_w = self.level_shapes[0][1]
        return self.mpp_x * base_w / self.level_shapes[level][1]

    @property
    def channel_names(self) -> list[str]:
        names = list(self.meta.channel_names) if self.meta is not None else []
        want = self.header.samples
        if len(names) == want:
            return names
        if not names and want == 3 and self.header.codec != 'raw':
            return ['R', 'G', 'B']
        return names

    @property
    def magnification(self) -> float | None:
        return None if self.meta is None else self.meta.magnification

    @property
    def is_overview(self) -> bool:
        return bool(self.meta is not None and self.meta.is_overview)

    @property
    def series(self) -> tifffile.TiffPageSeries:
        return self.tif.series[0]

    @property
    def levels(self) -> list:
        return list(self.series.levels)

    @property
    def level_shapes(self) -> list[tuple[int, ...]]:
        return [tuple(lv.shape) for lv in self.series.levels]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.series.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.series.dtype

    @property
    def codec(self) -> str:
        return self.header.codec

    def asarray(self, level: int = 0, **kwargs):
        """Fully decode one pyramid level into a numpy array."""
        return self.series.levels[level].asarray(**kwargs)

    def aszarr(self, **kwargs):
        """Multiscale zarr store — feeds dask, napari, ome-zarr writers."""
        return self.series.aszarr(**kwargs)

    @staticmethod
    def _zarr_array(store):
        """tifffile stores open as an Array or a multiscale Group; normalize."""
        import zarr
        obj = zarr.open(store, mode='r')
        if hasattr(obj, 'shape'):
            return obj
        return obj['0']

    def zarr_levels(self) -> list:
        """One lazy zarr array per pyramid level (chunk == ETS tile)."""
        return [self._zarr_array(lv.aszarr()) for lv in self.series.levels]

    def to_dask(self):
        """One dask array per pyramid level, chunked on the native tiles."""
        import dask.array as da
        return [da.from_array(z, chunks=z.chunks, inline_array=True)
                for z in self.zarr_levels()]

    def read_region(self, origin: tuple[int, int], size: tuple[int, int],
                    level: int = 0):
        """Crop ``size=(w, h)`` at ``origin=(x, y)`` *in that level's* coords.

        Only the intersecting tiles are fetched and decoded — this is the
        payoff of letting tifffile own the tile index.
        """
        z = self._zarr_array(self.series.levels[level].aszarr())
        x, y = origin
        w, hgt = size
        return np.asarray(z[y:y + hgt, x:x + w])

    def close(self) -> None:
        self.tif.close()

    def __repr__(self) -> str:
        mpp = f' {self.mpp_x:.4g}um/px' if self.mpp_x else ''
        ch = f' {self.channel_names}' if self.channel_names else ''
        return (f'<EtsPyramid {self.name!r} {self.level_shapes[0]} {self.dtype} '
                f'{len(self.level_shapes)} levels codec={self.codec}{mpp}{ch}>')


class VsiFile:
    """An Olympus ``.vsi`` dataset: thumbnails via tifffile, pyramids via ETS."""

    def __init__(self, path: str | os.PathLike, *,
                 base_size: tuple[int, int] | dict[str, tuple[int, int]] | None = None,
                 include_all_ets: bool = False,
                 lazy: bool = True) -> None:
        """``base_size`` may be a single ``(h, w)`` applied to every stack, or
        a ``{stack_name: (h, w)}`` mapping. Omit it to use the padded tile
        grid of level 0."""
        self.path = Path(path)
        self.base_size = base_size
        self.include_all_ets = include_all_ets
        self.non_image_ets: list[Path] = []
        self._pyramids: list[EtsPyramid] = []
        self._failed: list[tuple[Path, Exception]] = []
        self._metadata = None

        # (a) the .vsi itself is a genuine TIFF — tifffile reads it as-is
        self.container: tifffile.TiffFile | None = None
        if self.path.suffix.lower() == '.vsi':
            try:
                self.container = tifffile.TiffFile(self.path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f'could not open {self.path.name} as TIFF: {exc}',
                              stacklevel=2)

        self.ets_paths = self._discover_ets()
        if not lazy:
            self.pyramids  # noqa: B018

    # -- discovery --------------------------------------------------------
    def _discover_ets(self) -> list[Path]:
        if self.path.suffix.lower() == '.ets':
            return [self.path]
        # cellSens puts the data in a sibling folder named  _<stem>_
        candidates = [
            self.path.parent / f'_{self.path.stem}_',
            self.path.with_suffix(''),
        ]
        found: list[Path] = []
        for folder in candidates:
            if folder.is_dir():
                found.extend(folder.rglob('*.ets'))

        # Only frame_*.ets files hold image planes. Siblings such as
        # blob_21_f_Frame#0.ets are serialized masks / vector overlays: they
        # reuse the SIS/ETS container but their "tiles" are 1-D byte runs
        # (tile_y == 1) and their coordinates are blob IDs, not tile indices.
        # Including one produces a nonsense multi-gigapixel grid.
        image, other = [], []
        for p in sorted(dict.fromkeys(found), key=lambda q: (q.parent.name, q.name)):
            (image if p.name.startswith('frame_') else other).append(p)
        self.non_image_ets = other
        if other and not self.include_all_ets:
            warnings.warn(
                f'{self.path.name}: ignoring {len(other)} non-image .ets file(s) '
                f'({", ".join(q.name for q in other[:3])}); these are blobs/masks, '
                'not image planes. Pass include_all_ets=True to force-load them.',
                stacklevel=3,
            )
        return image + (other if self.include_all_ets else [])

    # -- thumbnails -------------------------------------------------------
    @property
    def thumbnails(self) -> list[np.ndarray]:
        """Label / macro / overview images stored inside the .vsi TIFF."""
        if self.container is None:
            return []
        out = []
        for page in self.container.pages:
            try:
                out.append(page.asarray())
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f'thumbnail page skipped: {exc}', stacklevel=2)
        return out

    @property
    def vsi_metadata(self) -> dict:
        """Whatever tifffile can pull out of the .vsi container."""
        if self.container is None:
            return {}
        meta: dict = {}
        page = self.container.pages[0]
        for tag in page.tags:
            try:
                meta[tag.name] = tag.value
            except Exception:  # noqa: BLE001, S110
                pass
        if getattr(self.container, 'is_sis', False):
            meta['sis'] = self.container.sis_metadata
        return meta

    # -- pyramids ---------------------------------------------------------
    @property
    def metadata(self):
        """Parsed .vsi tag tree (true sizes, channel names, µm/px)."""
        if self._metadata is None and parse_vsi_metadata is not None \
                and self.path.suffix.lower() == '.vsi':
            try:
                self._metadata = parse_vsi_metadata(self.path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f'metadata parse failed: {exc}', stacklevel=2)
                self._metadata = False
        return self._metadata or None

    def _pair_metadata(self, headers: list[EtsHeader]) -> list:
        """Match tag-tree volumes to .ets stacks, with a geometric check.

        Mispairing would silently attach the wrong scale bar and channel names
        to a slide, so a candidate is only accepted when its declared size
        actually fits inside that stack's padded tile grid — i.e. it is at most
        the grid extent and within one tile of it. Volumes that fit nothing
        (typically the label/macro images, which live as TIFF IFDs in the .vsi
        rather than as .ets stacks) are skipped rather than force-matched.
        """
        meta = self.metadata
        if meta is None:
            return [None] * len(headers)
        candidates = meta.image_pyramids()

        def fits(c, h, grid_w: int, grid_h: int) -> bool:
            return (grid_w - h.tile_width < c.width <= grid_w
                    and grid_h - h.tile_height < c.height <= grid_h)

        grids = []
        for h in headers:
            try:
                grids.append(ets_level0_grid(h))
            except ValueError as exc:
                warnings.warn(str(exc), stacklevel=2)
                grids.append(None)

        # Pass 1: in-order greedy, which is how the volumes are laid out.
        out, cursor, used = [], 0, set()
        for h, grid in zip(headers, grids):
            chosen = None
            if grid is not None:
                grid_h, grid_w = grid
                for j in range(cursor, len(candidates)):
                    if fits(candidates[j], h, grid_w, grid_h):
                        chosen, cursor = candidates[j], j + 1
                        used.add(j)
                        break
            out.append(chosen)

        # Pass 2: for anything still unmatched, accept a UNIQUELY fitting
        # unused volume regardless of order. Requiring uniqueness keeps this
        # from guessing when two volumes are plausibly the same size.
        for i, (h, grid) in enumerate(zip(headers, grids)):
            if out[i] is not None or grid is None:
                continue
            grid_h, grid_w = grid
            hits = [j for j, c in enumerate(candidates)
                    if j not in used and fits(c, h, grid_w, grid_h)]
            if len(hits) == 1:
                out[i] = candidates[hits[0]]
                used.add(hits[0])

        for i, (h, grid) in enumerate(zip(headers, grids)):
            if out[i] is not None:
                continue
            grid_w, grid_h = (grid[1], grid[0]) if grid else ('?', '?')
            warnings.warn(
                f'{h.path.parent.name}/{h.path.name}: no .vsi metadata volume '
                f'matches the {grid_w}x{grid_h} tile grid; falling back to grid '
                'size (no µm/px or channel names for this stack)',
                stacklevel=2,
            )
        return out

    @property
    def pyramids(self) -> list[EtsPyramid]:
        if not self._pyramids:
            headers, paths = [], []
            for p in self.ets_paths:
                try:
                    headers.append(parse_ets(p))
                    paths.append(p)
                except Exception as exc:  # noqa: BLE001
                    self._failed.append((p, exc))
                    warnings.warn(f'skipping {p.name}: {exc}', stacklevel=2)

            metas = self._pair_metadata(headers)
            for p, hdr, meta in zip(paths, headers, metas):
                try:
                    bs = self.base_size
                    if isinstance(bs, dict):
                        bs = bs.get(p.parent.name)
                    if bs is None and meta is not None:
                        bs = meta.base_size          # true size, from the .vsi
                    tif = ets_to_tifffile(p, header=hdr, base_size=bs)
                    self._pyramids.append(
                        EtsPyramid(path=p, header=hdr, tif=tif,
                                   stack=p.parent.name, meta=meta)
                    )
                except Exception as exc:  # noqa: BLE001
                    self._failed.append((p, exc))
                    warnings.warn(f'skipping {p.name}: {exc}', stacklevel=2)
        return self._pyramids

    def largest_pyramid(self) -> EtsPyramid:
        if not self.pyramids:
            msg = f'no readable .ets pyramids found for {self.path}'
            raise FileNotFoundError(msg)
        def area(p: EtsPyramid) -> int:
            h, w = p.level_shapes[0][:2]
            # A mis-parsed stack can report an absurd shape and would then
            # always "win" this comparison, silently becoming the slide.
            return 0 if max(h, w) > MAX_PLAUSIBLE_PIXELS else int(h) * int(w)

        best = max(self.pyramids, key=area)
        if area(best) == 0:
            msg = (f'{self.path.name}: every stack has an implausible shape '
                   f'{[p.level_shapes[0] for p in self.pyramids]}; parsing failed')
            raise ValueError(msg)
        return best

    @property
    def failures(self) -> list[tuple[Path, Exception]]:
        return list(self._failed)

    def close(self) -> None:
        for p in self._pyramids:
            p.close()
        if self.container is not None:
            self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<VsiFile {self.path.name} {len(self.ets_paths)} stacks>'


def open_vsi(path, **kwargs) -> VsiFile:
    return VsiFile(path, **kwargs)


def write_ome_tiff(pyramid: EtsPyramid, out_path: str | os.PathLike, *,
                   compression: str = 'jpeg2000', levels: int | None = None,
                   tile: tuple[int, int] | None = None, **kwargs) -> Path:
    """Transcode a pyramid to a calibrated pyramidal OME-TIFF.

    The .vsi metadata is what makes this worth doing: the output carries real
    µm/pixel and channel names, so downstream tools get a correct scale bar
    instead of pixel units.

    Note this *re-encodes* every tile — unlike reading, it is not zero-copy,
    and on a whole slide it is slow. Use ``levels`` to cap the pyramid depth.
    """
    out_path = Path(out_path)
    series = pyramid.series
    nlevels = len(series.levels) if levels is None else min(levels, len(series.levels))
    tile = tile or (pyramid.header.tile_height, pyramid.header.tile_width)

    metadata: dict = {'axes': 'YXS' if pyramid.header.samples > 1 else 'YX'}
    if pyramid.mpp_x:
        metadata.update(PhysicalSizeX=pyramid.mpp_x, PhysicalSizeXUnit='µm',
                        PhysicalSizeY=pyramid.mpp_y or pyramid.mpp_x,
                        PhysicalSizeYUnit='µm')
    names = pyramid.channel_names
    if names:
        # OME semantics: an RGB image is ONE channel with three samples, not
        # three channels. Only emit a per-channel name list when the planes
        # really are separate channels.
        metadata['Channel'] = ({'Name': names} if pyramid.header.samples == 1
                               else {'Name': [pyramid.name or names[0]],
                                     'SamplesPerPixel': [pyramid.header.samples]})
    if pyramid.name:
        metadata['Name'] = pyramid.name

    zlevels = pyramid.zarr_levels()
    with tifffile.TiffWriter(out_path, bigtiff=True, ome=True) as tw:
        for i in range(nlevels):
            tw.write(
                np.asarray(zlevels[i]),
                subifds=nlevels - 1 if i == 0 else None,
                subfiletype=0 if i == 0 else 1,
                tile=tile, compression=compression,
                photometric='rgb' if pyramid.header.samples >= 3 else 'minisblack',
                metadata=metadata if i == 0 else None,
                resolution=(1e4 / pyramid.level_mpp(i), 1e4 / pyramid.level_mpp(i))
                if pyramid.mpp_x else None,
                resolutionunit='CENTIMETER' if pyramid.mpp_x else None,
                **kwargs,
            )
    return out_path