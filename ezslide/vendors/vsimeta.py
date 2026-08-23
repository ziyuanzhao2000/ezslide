"""
vsimeta — parse the Olympus/EVIDENT ``.vsi`` metadata tag tree.

Companion to :mod:`vsitiff`. The ``.ets`` files carry pixels but only a
*padded tile grid*, no channel identity and no calibration. All of that lives
in a nested tag tree inside the ``.vsi`` container, which begins at byte
offset 8 — immediately after the 8-byte TIFF header, interleaved with (but
independent of) the TIFF IFD chain that ``tifffile`` walks for the label and
macro thumbnails.

Structure
---------
A *tag container* is a 24-byte header::

    int16  header_size      (always 24)
    int16  version          (always 21321)
    int32  volume_version
    int64  data_field_offset   -- relative to the container start
    int32  flags               -- low 28 bits are the field count
    int32  (reserved)

followed by a **singly linked list** of fields, each 16 bytes::

    int32  field_type       -- packed flags + a type code in the low 24 bits
    int32  tag              -- meaning is context-dependent (see below)
    uint32 next_field       -- offset of the next field, relative to the
                               CONTAINER start, not this field. 0 ends the list.
    int32  data_size
    [int32 second_tag]      -- only when the "extra tag" flag is set

``field_type`` bit 28 marks a field whose payload is itself one or more nested
containers, which is what makes the whole thing a tree. Bit 30 means the
value is stored *in* ``data_size`` rather than after the header.

The one thing that will bite you
--------------------------------
Tag numbers are **not globally unique** — they are scoped to their enclosing
volume. ``2003`` is ``DIMENSION_SIZE`` in one context and ``DISPLAY_LIMITS``
in another; ``2018`` is both ``EXTERNAL_FILE_PROPERTIES`` and
``RWC_FRAME_ORIGIN``. This parser therefore keys raw metadata by
``(prefix, tag)`` and disambiguates the structural tags using the same
previous-tag context the format itself relies on.

Provenance
----------
The tag numbers and binary layout below are format facts confirmed against
the Bio-Formats ``CellSensReader``. The implementation here is independent;
no Bio-Formats code is reproduced. If you vendor this into a project, note
that Bio-Formats itself is GPL-2.0 — reading its source for constants does
not make your project GPL, but copying its code would.
"""

from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ['VsiMetadata', 'PyramidMetadata', 'parse_vsi_metadata']


# -- field type codes -------------------------------------------------------
_CHAR, _UCHAR, _SHORT, _USHORT, _INT, _UINT = 1, 2, 3, 4, 5, 6
_LONG, _ULONG, _FLOAT, _DOUBLE = 7, 8, 9, 10
_BOOLEAN, _TCHAR, _DWORD, _TIMESTAMP, _DATE = 12, 13, 14, 17, 18
_RGB, _BGR = 269, 270
_UNICODE_TCHAR = 8192

_INT_ARRAY_TYPES = frozenset({
    256, 257, 258, 259, 267, 274, 275, 276, 277,   # INT_2/3/4/RECT/INTERVAL/ARRAY_n
    271, 272, 273,                                  # FIELD_TYPE, MEM_MODEL, COLOR_SPACE
    8195, 8199, 8200, 8470,                         # DIM_INDEX_1/2, VOLUME_INDEX, PIXEL_INFO
})
_DOUBLE_ARRAY_TYPES = frozenset({
    11, 260, 261, 262, 263, 264, 265, 266, 268, 279, 280,
})

# -- structural volume types (field_type low bits, when bit 28 is set) ------
_NEW_VOLUME_HEADER = 0
_PROPERTY_SET_VOLUME = 1
_NEW_MDIM_VOLUME_HEADER = 2

# -- tags we actually act on ------------------------------------------------
_IMAGE_FRAME_VOLUME = 2002
_EXTERNAL_FILE_PROPERTIES = 2018      # collides with RWC_FRAME_ORIGIN
_DOCUMENT_PROPERTIES = 2109
_SLIDE_PROPERTIES = 2062
_DIMENSION_DESCRIPTION_VOLUME = 2007

_IMAGE_BOUNDARY = 2053                # INT_RECT -> (x, y, width, height)
_TILE_ORIGIN = 2410
_RWC_FRAME_SCALE = 2019               # DOUBLE_2 -> (µm/px x, µm/px y)
_RWC_FRAME_ORIGIN = 2018
_RWC_FRAME_UNIT = 2020
_CHANNEL_NAME = 2419
_STACK_NAME = 2030
_STACK_TYPE = 2074
_OBJECTIVE_MAG = 120060
_NUMERICAL_APERTURE = 120061
_OBJECTIVE_NAME = 120063
_DEVICE_NAME = 120116
_DEVICE_MANUFACTURER = 120133
_BIT_DEPTH = 100049
_SLIDE_NAME = 2061
_VALUE = 268435458

_VOLUME_PREFIX = {
    2043: 'Microscope ',
    2417: 'Channel Wavelength ',
    120062: 'Objective Working Distance ',
    2017: 'Timestamp ',
    20051: 'Calibration Function ',
}

_TAG_NAMES = {
    _IMAGE_BOUNDARY: 'Image boundary',
    _TILE_ORIGIN: 'Tile origin',
    _RWC_FRAME_SCALE: 'Frame scale (um/px)',
    _RWC_FRAME_UNIT: 'Frame unit',
    _CHANNEL_NAME: 'Channel name',
    _STACK_NAME: 'Stack name',
    _STACK_TYPE: 'Stack type',
    _OBJECTIVE_MAG: 'Objective magnification',
    _NUMERICAL_APERTURE: 'Numerical aperture',
    _OBJECTIVE_NAME: 'Objective name',
    _DEVICE_NAME: 'Device name',
    _DEVICE_MANUFACTURER: 'Device manufacturer',
    _BIT_DEPTH: 'Bit depth',
    _SLIDE_NAME: 'Slide name',
    2055: 'Specimen', 2057: 'Tissue', 2058: 'Preparation',
    2059: 'Staining', 2060: 'Slide info',
    100002: 'Exposure time (us)', 2015: 'Creation time',
}

_STACK_TYPES = {
    0: 'default', 1: 'overview', 2: 'sample mask', 4: 'focus image',
    8: 'EFI sharpness map', 16: 'EFI height map', 32: 'EFI texture map',
    64: 'EFI stack', 256: 'macro image',
}

_HEADER = struct.Struct('<hhiqii')      # 24 bytes
_FIELD = struct.Struct('<iiIi')         # 16 bytes
_SENTINEL_TAG = -494804095
_MAX_DEPTH = 64


@dataclass
class PyramidMetadata:
    """Metadata for one image volume described in the .vsi tag tree."""

    index: int
    name: str | None = None
    stack_type: str | None = None
    width: int | None = None            # TRUE pixel width of level 0
    height: int | None = None           # TRUE pixel height of level 0
    mpp_x: float | None = None          # micrometres per pixel, X
    mpp_y: float | None = None
    origin_x: float | None = None       # stage position, micrometres
    origin_y: float | None = None
    tile_origin_x: int | None = None
    tile_origin_y: int | None = None
    channel_names: list[str] = field(default_factory=list)
    channel_wavelengths: list[float] = field(default_factory=list)
    magnification: float | None = None
    numerical_aperture: float | None = None
    objective_names: list[str] = field(default_factory=list)
    device_names: list[str] = field(default_factory=list)
    bit_depth: int | None = None
    tags: dict = field(default_factory=dict)

    @property
    def base_size(self) -> tuple[int, int] | None:
        """``(height, width)`` — the shape ``vsitiff`` wants, or None."""
        if self.width is None or self.height is None:
            return None
        return (self.height, self.width)

    @property
    def mpp(self) -> float | None:
        """Scalar µm/pixel; warns if the pixels are not square."""
        if self.mpp_x is None:
            return None
        if self.mpp_y is not None and self.mpp_x > 0:
            if abs(self.mpp_y - self.mpp_x) / self.mpp_x > 0.01:
                warnings.warn(
                    f'anisotropic pixels: {self.mpp_x} x {self.mpp_y} um; '
                    'use mpp_x / mpp_y explicitly',
                    stacklevel=2,
                )
        return self.mpp_x

    @property
    def is_overview(self) -> bool:
        name = (self.name or '').lower()
        return name in {'overview', 'macro image', 'label'} or \
            self.stack_type in {'overview', 'macro image'}

    def __repr__(self) -> str:
        size = f'{self.width}x{self.height}' if self.width else '?'
        mpp = f'{self.mpp_x:.4g}um/px' if self.mpp_x else 'no scale'
        ch = f' {self.channel_names}' if self.channel_names else ''
        return (f'<PyramidMetadata #{self.index} {self.name or "unnamed"!s} '
                f'{size} {mpp}{ch}>')


@dataclass
class VsiMetadata:
    """Everything the .vsi tag tree yielded."""

    path: Path
    pyramids: list[PyramidMetadata] = field(default_factory=list)
    document: dict = field(default_factory=dict)
    truncated: bool = False

    def image_pyramids(self) -> list[PyramidMetadata]:
        """Volumes that describe an actual image plane (have a boundary)."""
        return [p for p in self.pyramids if p.width and p.height]

    def __repr__(self) -> str:
        return (f'<VsiMetadata {self.path.name} '
                f'{len(self.image_pyramids())}/{len(self.pyramids)} volumes>')


# --------------------------------------------------------------------------

class _TagTreeParser:
    def __init__(self, data: bytes, path: Path) -> None:
        self.d = data
        self.size = len(data)
        self.path = path
        self.pyramids: list[PyramidMetadata] = []
        self.document: dict = {}
        self.index = -1
        self.previous_tag = 0
        self.truncated = False
        self._visited: set[int] = set()

    # -- helpers ----------------------------------------------------------
    def _current(self) -> PyramidMetadata | None:
        if self.index < 0:
            return None
        while self.index >= len(self.pyramids):
            self.pyramids.append(PyramidMetadata(index=len(self.pyramids)))
        return self.pyramids[self.index]

    def _decode(self, real_type: int, pos: int, size: int):
        d, end = self.d, pos + size
        if size <= 0 or end > self.size:
            return None
        if real_type in (_CHAR, _UCHAR):
            return d[pos]
        if real_type in (_SHORT, _USHORT):
            return struct.unpack_from('<h', d, pos)[0]
        if real_type in (_INT, _UINT, _DWORD, 271, 272, 273):
            return struct.unpack_from('<i', d, pos)[0]
        if real_type in (_LONG, _ULONG, _TIMESTAMP):
            return struct.unpack_from('<q', d, pos)[0]
        if real_type == _FLOAT:
            return struct.unpack_from('<f', d, pos)[0]
        if real_type in (_DOUBLE, _DATE):
            return struct.unpack_from('<d', d, pos)[0]
        if real_type == _BOOLEAN:
            return bool(d[pos])
        if real_type in (_TCHAR, _UNICODE_TCHAR):
            raw = d[pos:end]
            if real_type == _UNICODE_TCHAR and size % 2 == 0:
                text = raw.decode('utf-16-le', 'replace')
            else:
                text = raw.decode('utf-8', 'replace')
            return text.rstrip('\x00').strip()
        if real_type in _INT_ARRAY_TYPES:
            n = size // 4
            return list(struct.unpack_from(f'<{n}i', d, pos)) if n else []
        if real_type in _DOUBLE_ARRAY_TYPES:
            n = size // 8
            return list(struct.unpack_from(f'<{n}d', d, pos)) if n else []
        if real_type in (_RGB, _BGR) and size >= 3:
            r, g, b = d[pos], d[pos + 1], d[pos + 2]
            return (r, g, b) if real_type == _RGB else (b, g, r)
        return None

    # -- the interesting part ---------------------------------------------
    def _record(self, tag: int, value, prefix: str) -> None:
        pyr = self._current()
        name = _TAG_NAMES.get(tag, f'tag_{tag}')
        key = f'{prefix}{name}'
        target = pyr.tags if pyr is not None else self.document
        if key in target and target[key] != value:
            target.setdefault(f'{key} (list)', []).append(value)
        else:
            target[key] = value

        if pyr is None:
            return

        if tag == _IMAGE_BOUNDARY and isinstance(value, list) and len(value) >= 4:
            # INT_RECT is (x, y, width, height); first one wins, matching the
            # writer's own convention.
            if pyr.width is None and value[2] > 0 and value[3] > 0:
                pyr.width, pyr.height = value[2], value[3]
        elif tag == _TILE_ORIGIN and isinstance(value, list) and len(value) >= 2:
            pyr.tile_origin_x, pyr.tile_origin_y = value[0], value[1]
        elif tag == _RWC_FRAME_SCALE and isinstance(value, list) and len(value) >= 2:
            if pyr.mpp_x is None and value[0] > 0:
                pyr.mpp_x, pyr.mpp_y = abs(value[0]), abs(value[1])
        elif tag == _RWC_FRAME_ORIGIN and isinstance(value, list) and len(value) >= 2:
            if pyr.origin_x is None:
                pyr.origin_x, pyr.origin_y = value[0], value[1]
        elif tag == _CHANNEL_NAME and isinstance(value, str) and value:
            if value not in pyr.channel_names:
                pyr.channel_names.append(value)
        elif tag == _STACK_NAME and isinstance(value, str):
            if pyr.name is None and value not in ('', '0'):
                pyr.name = value
        elif tag == _STACK_TYPE:
            with_int = value if isinstance(value, int) else None
            pyr.stack_type = _STACK_TYPES.get(with_int, pyr.stack_type)
        elif tag == _OBJECTIVE_MAG:
            pyr.magnification = _as_float(value, pyr.magnification)
        elif tag == _NUMERICAL_APERTURE:
            pyr.numerical_aperture = _as_float(value, pyr.numerical_aperture)
        elif tag == _OBJECTIVE_NAME and isinstance(value, str) and value:
            pyr.objective_names.append(value)
        elif tag == _DEVICE_NAME and isinstance(value, str) and value:
            pyr.device_names.append(value)
        elif tag == _BIT_DEPTH and isinstance(value, int):
            pyr.bit_depth = value
        elif tag == _VALUE and prefix == 'Channel Wavelength ':
            wl = _as_float(value, None)
            if wl is not None:
                pyr.channel_wavelengths.append(wl)

    def parse_container(self, fp: int, prefix: str = '', depth: int = 0) -> int:
        """Parse one tag container; return the position where it ended."""
        if depth > _MAX_DEPTH or fp < 0 or fp + 24 >= self.size:
            return fp
        if fp in self._visited:          # cycle guard: malformed files exist
            return fp
        self._visited.add(fp)

        header_size, version, _vol_ver, data_offset, flags, _ = \
            _HEADER.unpack_from(self.d, fp)
        if header_size != 24:
            return fp
        if version != 21321:
            warnings.warn(
                f'{self.path.name}: tag container at {fp} has version {version}, '
                'expected 21321 — layout may differ',
                stacklevel=2,
            )
        count = flags & 0x0FFFFFFF
        if count > self.size:
            return fp

        pos = fp + data_offset
        if pos < 0 or pos >= self.size:
            return fp

        for _ in range(count):
            if pos + 16 >= self.size:
                self.truncated = True
                break

            field_type, tag, next_field, data_size = _FIELD.unpack_from(self.d, pos)
            p = pos + 16
            extra_tag = (field_type >> 27) & 1
            extended = (field_type >> 28) & 1
            inline = (field_type >> 30) & 1
            real_type = field_type & 0xFFFFFF

            second_tag = -1
            if extra_tag:
                if p + 4 > self.size:
                    break
                second_tag = struct.unpack_from('<i', self.d, p)[0]
                p += 4

            if tag < 0:
                if not inline and 0 < data_size and p + data_size <= self.size:
                    p += data_size
                return p

            # Volume-to-pyramid bookkeeping. This is what disambiguates the
            # 2018 tag collision: EXTERNAL_FILE_PROPERTIES only counts when it
            # directly follows an IMAGE_FRAME_VOLUME.
            if (tag == _EXTERNAL_FILE_PROPERTIES
                    and self.previous_tag == _IMAGE_FRAME_VOLUME):
                self.index += 1
            elif tag in (_DOCUMENT_PROPERTIES, _SLIDE_PROPERTIES):
                self.index = -1
            self.previous_tag = tag
            self._current()

            if extended and real_type == _NEW_VOLUME_HEADER:
                end = min(p + data_size, self.size)
                child_prefix = _VOLUME_PREFIX.get(tag, '')
                q = p
                while q < end:
                    start = q
                    q = self.parse_container(q, child_prefix, depth + 1)
                    if q <= start:
                        break
            elif extended and real_type in (_PROPERTY_SET_VOLUME,
                                            _NEW_MDIM_VOLUME_HEADER):
                child_prefix = (_VOLUME_PREFIX.get(tag, '')
                                if real_type == _NEW_MDIM_VOLUME_HEADER
                                else prefix)
                self.parse_container(p, child_prefix, depth + 1)
            else:
                value = data_size if inline else self._decode(real_type, p, data_size)
                if value is not None:
                    self._record(tag, value, prefix)

            # Fields form a linked list; next_field is relative to the
            # CONTAINER start, not to this field.
            if next_field == 0 or tag == _SENTINEL_TAG:
                nxt = fp + data_size + 32
                return nxt if 0 <= fp + data_size and nxt < self.size else p
            nxt = fp + next_field
            if not (0 <= nxt < self.size):
                break
            pos = nxt

        return pos


def _as_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_vsi_metadata(path: str | Path, *, max_bytes: int | None = None
                       ) -> VsiMetadata:
    """Parse the tag tree of a ``.vsi`` file.

    The tree is walked in memory. A ``.vsi`` container is small (the pixels
    live in the ``.ets`` sidecars), typically a few MB, but ``max_bytes``
    caps the read if you hit an unusual file.
    """
    path = Path(path)
    data = path.read_bytes() if max_bytes is None else \
        path.open('rb').read(max_bytes)

    if data[:4] not in (b'II\x2a\x00', b'II\x2b\x00'):
        msg = (f'{path.name}: not a little-endian TIFF container '
               f'(magic {data[:4]!r}); .vsi files are always little-endian')
        raise ValueError(msg)

    parser = _TagTreeParser(data, path)
    parser.parse_container(8)            # tag tree starts right after the header

    meta = VsiMetadata(path=path, pyramids=parser.pyramids,
                       document=parser.document, truncated=parser.truncated)
    if not meta.pyramids:
        warnings.warn(f'{path.name}: no metadata volumes found in tag tree',
                      stacklevel=2)
    return meta