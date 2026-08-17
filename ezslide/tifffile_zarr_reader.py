from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from wsidata import SlideProperties
from wsidata.reader import ReaderBase
from .tifffile_zarr import TiffFile
from wsidata.reader._reader_registry import register


# wsidata's open_wsi() accepts **kwargs but never forwards them: it builds the
# reader through READERS.try_open() -> _open_reader(), which calls
# reader_cls(path) with no options at all. The constructor is therefore
# unreachable from the caller, and the only thing open_wsi lets you vary is
# which class gets instantiated.
#
# This context variable is the side channel around that. Options set here are
# read by any reader constructed inside the block — and open_wsi builds its
# reader synchronously, so a `with` around the call brackets it exactly.
#
# ContextVar rather than a plain class attribute so concurrent opens in
# different threads or asyncio tasks cannot see each other's settings.
_overrides: ContextVar[dict] = ContextVar("ezslide_pyramid_overrides", default={})

#: key used inside the override dict for the on/off flag, kept apart from the
#: options forwarded to TiffSeries.pyramidalize()
_ENABLE_KEY = "pyramidalize"


@contextmanager
def pyramid_options(pyramidalize=None, **options):
    """Scope pyramid settings over readers constructed inside the block.

    The intended use is choosing a reduction for the image at hand, since
    ``how`` is a fact about what the pixels mean rather than a preference:
    intensity data must be averaged, label maps must not be.

        >>> with pyramid_options(how='mode'):
        ...     masks = open_wsi(mask_path, reader='tifffile_zarr_pyramid')

    Averaging a label map is silent corruption — ``block_reduce`` accumulates
    in float, rounds and casts back, so labels 3 and 7 average to label 5,
    which may not exist or may belong to an unrelated object. Nothing about
    the result looks wrong. Note that ``'mode'`` is nearest-neighbour sampling
    of the window corner, not a majority vote, so thin structures can vanish
    at depth; see ``zarr_pyramid.block_reduce``.

    Parameters
    ----------
    pyramidalize : bool, optional
        Force pyramidalization on or off, overriding the reader class's
        default. ``None`` (the default) leaves that decision to the class, so
        ``pyramid_options(how='mode')`` alone only changes *how* a pyramid is
        built, not *whether*. Setting it True means the plain
        ``tifffile_zarr`` reader pyramidalizes too, which is usually simpler
        than registering another reader class.
    **options
        Forwarded to ``TiffSeries.pyramidalize`` — ``how``, ``levels``,
        ``factor``, ``cache``, ``store``, ``materialize_below``,
        ``min_extent``, and anything else ``lazy_pyramid`` accepts.

    Notes
    -----
    Precedence, lowest to highest: the reader class's ``pyramid_defaults``,
    then this block, then explicit constructor arguments. Option dicts are
    merged key by key, so ``how`` can be overridden here without discarding
    the class's ``cache`` setting. Nested blocks merge the same way, innermost
    winning.

    Readers snapshot the resolved settings at construction, so
    ``detach_reader()`` / ``create_reader()`` later — long after the block has
    exited — still rebuilds with the settings the reader was opened under.

    A new thread starts with an empty context, so a reader constructed in a
    worker thread does not inherit a block entered on the caller's thread
    (use ``contextvars.copy_context().run(...)`` if you need that). The
    fallback is the class default, which is the conservative direction.
    """
    merged = {**_overrides.get(), **options}
    if pyramidalize is not None:
        merged[_ENABLE_KEY] = bool(pyramidalize)
    token = _overrides.set(merged)
    try:
        yield merged
    finally:
        _overrides.reset(token)


@register("tifffile_zarr")
class TiffFileZarrReader(ReaderBase):
    name = "tifffile_zarr"
    pkg_namespaces = ["tifffile", "zarr"]          # <- the fix
    pkgs = ["tifffile", "zarr"]                    # pip names, for error messages
    extensions = (".ndpi", ".tif", ".tiff", ".svs", ".scn", ".bif", ".qptiff")
    supports_scenes = False

    #: Subclasses configure pyramidalization declaratively by overriding these;
    #: ``pyramid_options()`` overrides both at call time. Never mutated in
    #: place — ``_resolve_pyramid`` always builds a fresh dict.
    pyramidalize_default = False
    pyramid_defaults = {}

    def __init__(self, file, series=0, pyramidalize=None, pyramid=None, **kwargs):
        self.file = str(file)
        self._series_idx = series
        self._pyramidalize, self._pyramid = self._resolve_pyramid(pyramidalize, pyramid)
        self._kwargs = kwargs
        self.create_reader()
        self.properties = self._build_properties()

    @classmethod
    def _resolve_pyramid(cls, pyramidalize, pyramid):
        """Merge class defaults < ``pyramid_options()`` < explicit arguments.

        ``pyramidalize`` is tri-state: ``None`` means "not specified", which is
        what lets an enclosing block or the class default decide, while an
        explicit ``False`` overrides both.
        """
        override = dict(_overrides.get())
        scoped_enable = override.pop(_ENABLE_KEY, None)

        enabled = cls.pyramidalize_default
        if scoped_enable is not None:
            enabled = scoped_enable
        if pyramidalize is not None:
            enabled = bool(pyramidalize)

        return enabled, {**cls.pyramid_defaults, **override, **(pyramid or {})}

    def create_reader(self):
        # also runs on re-attach, so the pyramid options have to be kept around
        self.set_reader(TiffFile(self.file,
                                 pyramidalize=self._pyramidalize,
                                 pyramid=self._pyramid,
                                 **self._kwargs))

    def detach_reader(self):
        if self._reader is not None:            # NOT self.reader
            self._reader.close()
        self.set_reader(None)

    @property
    def series(self):
        return self.reader[self._series_idx]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.reader._series[key]
        elif isinstance(key, str):
            for series in self.reader._series:
                if series.name == key:
                    return series
        raise KeyError(f"Series {key} not found in {self.file}")
    
    def _build_properties(self):
        s = self.series
        shapes = [(lv.height, lv.width) for lv in s.levels]
        h0, w0 = shapes[0]
        md = s.levels[0].metadata
        mpp = md.get("PhysicalSizeX")
        return SlideProperties(
            shape=(h0, w0),
            n_level=len(shapes),
            level_shape=shapes,
            level_downsample=[h0 / h for h, _ in shapes],
            mpp=float(mpp) if mpp is not None else None,
            bounds=(0, 0, w0, h0),
            raw={str(k): str(v) for k, v in md.items()},
        )

    def get_region(self, x, y, width, height, level=0, **kwargs):
        level = self.translate_level(level)
        lv = self.series.levels[level]
        ds = self.properties.level_downsample[level]
        idx = [slice(None)] * len(lv.shape)
        idx[lv.y_ax] = slice(int(y / ds), int(y / ds) + height)
        idx[lv.x_ax] = slice(int(x / ds), int(x / ds) + width)
        arr = lv[tuple(idx)]                      # WriteableZarrArray.__getitem__
        arr = arr.compute() if hasattr(arr, "compute") else np.asarray(arr)
        if lv.axes[:2] != "YX":                   # e.g. 'SYX' -> 'YXS'
            arr = np.moveaxis(arr, lv.axes.index("S"), -1)
        return arr

    def get_thumbnail(self, size, **kwargs):
        img = self.series.thumbnail
        img.thumbnail((size, size) if isinstance(size, int) else size)
        return np.asarray(img)
    
@register("tifffile_zarr_pyramid")
class PyramidTiffFileZarrReader(TiffFileZarrReader):
    """``tifffile_zarr`` with lazy pyramids on by default.

    Convenience only: ``with pyramid_options(pyramidalize=True)`` around
    ``open_wsi(..., reader='tifffile_zarr')`` does the same thing without a
    second registered reader.
    """

    name = "tifffile_zarr_pyramid"
    pyramidalize_default = True
    pyramid_defaults = {"cache": "tmp"}
