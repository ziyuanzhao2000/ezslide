import numpy as np
import tifffile


tag_registries = [tifffile.TIFF.TAGS,
                  tifffile.TIFF.GPS_TAGS,
                  tifffile.TIFF.IOP_TAGS,
                  tifffile.TIFF.UIC_TAGS,
                  tifffile.TIFF.EXIF_TAGS,
                  tifffile.TIFF.NDPI_TAGS]

def get_tag_name(tag_code):
    for tag_registry in tag_registries:
        if tag_code in tag_registry:
            return tag_registry[tag_code]
    return tag_code


def merge_dicts(dicts, names):
    if len(dicts) != len(names):
        raise ValueError("Number of dictionaries and names must match")
    
    if not dicts:
        return {}
    
    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())
    
    result = {}
    
    for key in all_keys:
        values = [d.get(key) for d in dicts if key in d]
        
        if len(values) == len(dicts):
            are_equal = True
            first_val = values[0]
            
            for v in values[1:]:
                if isinstance(first_val, np.ndarray) or isinstance(v, np.ndarray):
                    if not np.array_equal(first_val, v, equal_nan=True):
                        are_equal = False
                        break
                elif first_val != v:
                    are_equal = False
                    break
            
            if are_equal:
                result[key] = values[0]
                continue
        
        for i, d in enumerate(dicts):
            if key in d:
                result[f"{names[i]}{key}"] = d[key]
    
    return result

def recover_mpp(level):
    """recover mpp using tifffile's TiffPageSeries.mpp (tifffile>=2026.5.2)"""
    parsed = {}
    mpp = level.mpp
    if mpp is not None:
        mpp_x, mpp_y = mpp
        parsed['PhysicalSizeX'] = mpp_x
        parsed['PhysicalSizeY'] = mpp_y
        parsed['PhysicalSizeXUnit'] = 'µm'
        parsed['PhysicalSizeYUnit'] = 'µm'
    return parsed

def TIFFParser(level):
    return recover_mpp(level)


#: Namespaces of the MapAnnotations ``ezslide.writers.ome_tiff`` emits. Kept
#: here as literals rather than imported, so the read path does not drag in the
#: writer.
_EZSLIDE_NAMESPACES = ('ezslide:metadata', 'ezslide:provenance')


def recover_ome_annotations(tifffile_obj, image_index=0):
    """Metadata recovered from an OME-TIFF's XML, for one image in the file.

    Two things come back that ``tifffile`` will not hand over on its own:

    * ``ChannelNames``, read from the OME ``Channel`` elements. True of any
      OME-TIFF, not only ones ezslide wrote.
    * whatever ``ezslide.writers.ome_tiff`` stored in its ``MapAnnotation`` —
      objective, magnification, source path, original codec. ``tifffile``
      writes these happily but only parses ``modulo`` annotations back, so a
      converted slide would otherwise return with its calibration and none of
      its provenance. Closing that loop is the reason the writer records them.

    Results are keyed per image, since one file may hold several.
    """
    if tifffile_obj is None:
        return {}
    # Parsed once per file, not once per level: a cellSens dataset can hold
    # fifty series of nine levels each, and the XML does not change.
    cached = getattr(tifffile_obj, '_ezslide_annotations', None)
    if cached is None:
        cached = _parse_ome_annotations(tifffile_obj)
        try:
            tifffile_obj._ezslide_annotations = cached
        except AttributeError:
            pass
    if not cached:
        return {}
    return cached[image_index] if image_index < len(cached) else {}


def _parse_ome_annotations(tifffile_obj):
    """``[{...}, ...]`` — recovered metadata per OME Image, in document order."""
    xml = getattr(tifffile_obj, 'ome_metadata', None)
    if not xml:
        return []
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return []

    # Annotations live once at the root and are referenced by ID from each
    # image, so resolve them before walking the images.
    by_id = {}
    for element in root.iter():
        if not element.tag.endswith('MapAnnotation'):
            continue
        if element.get('Namespace') not in _EZSLIDE_NAMESPACES:
            continue
        entries = {}
        for value in element:
            for entry in value:
                key = entry.get('K')
                if key:
                    entries[key] = entry.text
        by_id[element.get('ID')] = entries

    images = []
    for image in root.iter():
        if not image.tag.endswith('}Image'):
            continue
        recovered = {}
        for ref in image.iter():
            if ref.tag.endswith('AnnotationRef'):
                recovered.update(by_id.get(ref.get('ID'), {}))
        names = [c.get('Name') for c in image.iter()
                 if c.tag.endswith('}Channel') and c.get('Name')]
        if names:
            recovered['ChannelNames'] = names
        images.append(recovered)
    return images


def NDPIParser(metadata):
    return metadata

def infer_axes(array, axes=None):
    """Fall back to a positional axes string when none is supplied."""
    if axes is not None:
        return axes
    num_axes = len(array.shape)
    if num_axes == 2:
        return 'YX'
    elif num_axes > 2:
        return '?' * (num_axes - 2) + 'YX'
    return axes


def pyramid_factors(axes, factor=2):
    """Per-axis reduction for a TIFF axes string: ``factor`` on Y and X, 1 elsewhere.

    ``lazy_pyramid``'s default ``(1, 1, 2, 2)`` assumes a trailing-YX layout,
    which is wrong for the orders tifffile actually reports for slides
    (``YXS`` for interleaved RGB, ``SYX`` for planar). Derive it instead.
    """
    return tuple(int(factor) if a in 'YX' else 1 for a in axes)


def array_repr_html(array):
    if hasattr(array, '_repr_html_'):
        return array._repr_html_()
    return repr(array)


def iter_tiles(array, axes, tile_size=None):
    """Iterate over an array in tiles, in Y/X raster order.

    Parameters
    ----------
    array : array-like
        The array to iterate over — a zarr array, a ``TensorStoreArray``, or
        a lazy pyramid level. Each tile comes back in whatever the array
        returns from ``__getitem__``, so a lazy array yields lazy views.
    axes : str
        Axes string for the array, must contain 'Y' and 'X'.
    tile_size : (height, width), optional
        Tile shape. Defaults to the array's chunk shape along Y and X.

    """
    axes = infer_axes(array, axes)
    y_ax, x_ax = axes.index('Y'), axes.index('X')
    shape = array.shape
    y_size, x_size = shape[y_ax], shape[x_ax]
    if tile_size is None:
        y_step, x_step = array.chunks[y_ax], array.chunks[x_ax]
    else:
        y_step, x_step = tile_size
    iter_axes = [i for i in range(len(shape)) if i not in (y_ax, x_ax)]
    iter_shape = [shape[i] for i in iter_axes]
    for index in np.ndindex(*iter_shape):
        for y in range(0, y_size, y_step):
            for x in range(0, x_size, x_step):
                full_index = list(index)
                if y_ax > x_ax:
                    full_index.insert(x_ax, slice(x, x + x_step, 1))
                    full_index.insert(y_ax, slice(y, y + y_step, 1))
                else:
                    full_index.insert(y_ax, slice(y, y + y_step, 1))
                    full_index.insert(x_ax, slice(x, x + x_step, 1))
                yield array[tuple(full_index)]


class TiffPage:
    def __init__(self, tiffpage):
        import dask.array as da            # deferred; see the module docstring
        import zarr
        from ..array.cache import make_cache_store
        from ..array.tensorstore_array import as_tensorstore
        self._page = tiffpage

        base_store = self._page.aszarr()
        cached_store = make_cache_store(base_store)
        self._zarr = zarr.open(cached_store)
        # Lazy by default: indexing composes a tensorstore view and decodes
        # nothing. eagerfy() restores zarr's materialize-on-index behaviour.
        self.data = as_tensorstore(self._zarr)
        self.delayed_data = da.from_zarr(cached_store)
        self._store = cached_store

    def __getattr__(self, name):
        return getattr(self._page, name)


    def _repr_html_(self):
        return array_repr_html(self.delayed_data)
        
    def __repr__(self):
        return self.delayed_data.__repr__()
    
    def __str__(self):
        return self.delayed_data.__str__()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def eagerfy(self):
        """Return numpy from ``page[sel]`` from now on. Returns ``self``."""
        self.data.eagerfy()
        return self

    def lazify(self):
        """Return lazy tensorstore views from ``page[sel]`` again."""
        self.data.lazify()
        return self

    def tiles(self, tile_size=None):
        return iter_tiles(self.data, self.axes, tile_size)

class TiffLevel():
    def __init__(self, tifflevel, level_id):
        import dask.array as da            # deferred; see the module docstring
        import zarr
        from ..array.cache import make_cache_store
        from ..array.tensorstore_array import as_tensorstore
        self._level = tifflevel
        self.level_id = level_id
        self._pages = [TiffPage(page) for page in tifflevel.pages]
        
        base_store = self._level.aszarr()
        cached_store = make_cache_store(base_store)
        initial_array = zarr.open(cached_store)
        if isinstance(initial_array, zarr.Group): # this occurs at level 0
            initial_array = initial_array['0']
            self.delayed_data = da.from_zarr(cached_store, component='0')
        else:
            self.delayed_data = da.from_zarr(cached_store)
        self._zarr = initial_array
        self.data = as_tensorstore(initial_array)
        self._store = cached_store
        self._metadata = dict([
            (get_tag_name(tag.code), tag.value) for tag in self._level.pages[0].tags
        ])
        self._parsed_metadata = {}

    @property
    def metadata(self):
        metadata = {}
        metadata.update(self._parsed_metadata)
        metadata.update(self._metadata)
        return metadata

    @property
    def pages(self):
        return self._pages

    @property
    def x_ax(self):
        return self.axes.index('X') 

    @property
    def y_ax(self):
        return self.axes.index('Y')

    @property
    def width(self):
        return self.shape[self.x_ax]
    
    @property
    def height(self):
        return self.shape[self.y_ax]

    def __getattr__(self, name):
        if name in ['name']:
            attr = self.metadata.get(name)
            if attr is not None:
                return attr
        return getattr(self._level, name)

    @property
    def is_multiscale(self):
        return (self.is_pyramidal or self.level_id > 0)
    
    def _repr_html_(self):
        if self.is_multiscale:
            return f"<h3>Pyramid level {self.level_id},\n</h3>" + array_repr_html(self.delayed_data)
        else:
            return array_repr_html(self.delayed_data)

    def __repr__(self):
        return f"Pyramid level {self.level_id},\n" + self.delayed_data.__repr__()
        
    def __str__(self):
        return self.__repr__()
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def eagerfy(self):
        """Materialize on index, for this level and each of its pages.

        A ``LazyTiffLevel`` has neither a tensorstore nor pages — its data is
        a computed view that already returns numpy — so this is a no-op there
        and the cascade from a series stays uniform.
        """
        for arr in (self.data, *self._pages):
            if hasattr(arr, 'eagerfy'):
                arr.eagerfy()
        return self

    def lazify(self):
        """Undo ``eagerfy``, for this level and each of its pages."""
        for arr in (self.data, *self._pages):
            if hasattr(arr, 'lazify'):
                arr.lazify()
        return self

    def tiles(self, tile_size=None):
        return iter_tiles(self.data, self.axes, tile_size)

    def parse_metadata(self, kind):
        metadata = {}
        # Annotations first: a slide ezslide converted carries its objective,
        # magnification and provenance here, and anything the TIFF tags say
        # about this file should win over what they say about its ancestor.
        metadata.update(recover_ome_annotations(self._parent_tifffile(),
                                                self._image_index()))
        metadata.update(TIFFParser(self._level))
        if kind == 'ndpi':
            parser = NDPIParser
        else:
            parser = lambda x: x
        self._parsed_metadata = parser(metadata)

    def _parent_tifffile(self):
        """The ``tifffile.TiffFile`` this level came out of, if reachable."""
        try:
            return self._level.pages[0].parent
        except (AttributeError, IndexError):
            return None

    def _image_index(self):
        """Position of this level's series in the file, for per-image metadata."""
        tif = self._parent_tifffile()
        if tif is None:
            return 0
        for i, series in enumerate(getattr(tif, 'series', ())):
            if self._level is series or any(self._level is lv
                                            for lv in series.levels):
                return i
        return 0

class LazyTiffLevel(TiffLevel):
    """A pyramid level that is not in the file — computed from the level above.

    Same surface as ``TiffLevel`` (the readers, the writer and ``iter_tiles``
    cannot tell the difference), but ``data`` is a ``LazyLevel``/``CachedLevel``
    view rather than a zarr store over TIFF tiles, and there are no ``pages``
    because there is no IFD behind it.

    Attributes that describe the *data* (``shape``, ``dtype``, ``ndim``) are set
    here; everything else (``axes``, ``kind``, ``name``, ...) falls through to
    the base level, since a downsample changes extent but not layout.
    """

    def __init__(self, parent, array, level_id):
        self._parent = parent
        self.level_id = level_id
        self.data = array
        self.shape = tuple(array.shape)
        self.dtype = np.dtype(array.dtype)
        self.ndim = len(self.shape)
        self._pages = []
        self._store = None
        self._delayed = None

        self.downsample = parent.shape[parent.y_ax] / self.shape[parent.y_ax]
        self._metadata = dict(parent._metadata)
        self._metadata['ImageWidth'] = self.shape[parent.x_ax]
        self._metadata['ImageLength'] = self.shape[parent.y_ax]
        self._parsed_metadata = {}

    def __getattr__(self, name):
        parent = self.__dict__.get('_parent')
        if parent is None:                    # during __init__ / unpickling
            raise AttributeError(name)
        return getattr(parent, name)

    # Anything derived from extent must be defined here rather than left to
    # __getattr__, which would silently answer with level 0's value.
    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def nbytes(self):
        return self.size * self.dtype.itemsize

    @property
    def delayed_data(self):
        """Dask view, built on first use — only ``__repr__`` needs it."""
        import dask.array as da
        if self._delayed is None:
            self._delayed = da.from_array(
                self.data, chunks=self.data.chunks,
                name=f"lazy-level-{self.level_id}-{id(self):x}",
                meta=np.empty((0,) * self.ndim, self.dtype))
        return self._delayed

    def parse_metadata(self, kind):
        """Inherit the base level's parsed metadata, with spacing rescaled.

        The tags come from level 0's IFD, so the physical pixel size recorded
        there describes level 0. This level's pixels are ``downsample`` times
        larger.
        """
        parsed = dict(self._parent._parsed_metadata)
        for key in ('PhysicalSizeX', 'PhysicalSizeY'):
            if parsed.get(key) is not None:
                parsed[key] = parsed[key] * self.downsample
        self._parsed_metadata = parsed


def _thumbnail_grayscale(img):
    """Normalize a single-channel array to 0-255 uint8 for a PIL preview.

    PIL's mode "I;16" (what ``Image.fromarray`` produces for a raw uint16
    array) can't be resized at every reduction factor: ``Image.thumbnail()``
    raises ``ValueError: image has wrong mode`` from its internal ``reduce()``
    fast path once the ratio is large enough — reliably so for the kind of
    16x+ downscale a real slide's thumbnail needs. uint8 grayscale ("L" mode)
    resizes at any ratio, and a coarse preview has no use for 16-bit
    precision anyway.
    """
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) * (255.0 / (hi - lo))
    else:
        img = np.zeros_like(img)
    return img.astype(np.uint8)


class TiffSeries():
    def __init__(self, tiffseries, pyramidalize=False, pyramid=None):
        self._series = tiffseries
        self._levels = [TiffLevel(level, level_id) \
                        for level_id, level in enumerate(tiffseries.levels)]
        self._pyramidalized = False
        if pyramidalize and not tiffseries.is_pyramidal:
            self.pyramidalize(**(pyramid or {}))

    def pyramidalize(self, levels=8, factor=2, how='mean', cache='tmp',
                     store=None, materialize_below=None, min_extent=256,
                     **kwargs):
        """Append lazy downsampled levels on top of level 0.

        Nothing is read here: the returned levels know their shapes and compute
        pixels only when indexed. See ``lazy_pyramid`` for the cost model.

        Parameters
        ----------
        levels, factor, how, min_extent
            Pyramid depth, per-step XY reduction, reduction kernel (use
            ``'mode'`` for label images), and the extent at which to stop.
        cache, store, materialize_below
            Forwarded to ``lazy_pyramid``. ``cache='tmp'`` keeps computed
            blocks on disk rather than in RAM — level 1 alone is ~25% of the
            slide, which is why ``'memory'`` is a poor default here.
            ``materialize_below`` is ``None`` so that opening a file stays
            instant; set it to 3 or so if you will pan around the deep levels,
            accepting that construction then reads the whole slide once.
        """
        from ..array.pyramid import lazy_pyramid
        from ..array.tensorstore_array import TensorStoreArray
        base = self._levels[0]
        axes = infer_axes(base.data, getattr(base, 'axes', None))
        # ``block_reduce`` reshapes and pads real numpy blocks, so the pyramid
        # gets an eager handle on level 0 — a separate wrapper, so whichever
        # mode a caller has put ``base.data`` in is left alone.
        source = base.data.eager_view() if isinstance(base.data, TensorStoreArray) \
            else base.data
        stack = lazy_pyramid(source, levels=levels,
                             factors=pyramid_factors(axes, factor), how=how,
                             cache=cache, store=store,
                             materialize_below=materialize_below,
                             min_extent=min_extent, **kwargs)
        self._levels.extend(LazyTiffLevel(base, arr, level_id)
                            for level_id, arr in enumerate(stack[1:], 1))
        self._pyramidalized = len(stack) > 1
        return self._levels

    def __getattr__(self, name):
        return getattr(self._series, name)

    @property
    def is_pyramidal(self):
        """True once lazy levels exist, even though the file has only one."""
        return self._series.is_pyramidal or self._pyramidalized

    @property
    def metadata(self):
        if len(self._levels) == 1:
            return self._levels[0].metadata
        else:
            return merge_dicts(dicts=[level.metadata for level in self._levels],
                               names=[f'level_{i}.' for i in range(len(self._levels))])

    @property
    def levels(self):
        return self._levels

    @property
    def channel_group_key(self):
        """Key shared by series that are channels of one multi-channel image.

        ``None`` — the default and the answer for any format that puts a whole
        image in one series — means "write me as my own image". A format whose
        reader splits channels across series (see ``VsiSeries``) returns a key
        identifying the parent image, and the OME writer merges the group into
        a single ``CYX`` image with per-channel names. Keeping the key here
        rather than in the writer is what lets the writer stay ignorant of any
        particular file format.
        """
        return None

    @property
    def name(self):
        if hasattr(self._series, 'name'):
            return self._series.name
        else:
            return self._levels[0].name
        
    # alias
    @property
    def is_multiscale(self):
        return self.is_pyramidal
    
    @property
    def thumbnail(self):
        """A small PIL preview of the coarsest pyramid level.

        Grayscale for a plane with no channel axis, RGB for an exactly-3-channel
        uint8 plane (PIL can only composite 8-bit samples) — interleaved or
        planar — and the first channel alone, in grayscale, for anything else:
        a true multiplex/IF stack (any channel count, typically uint16), where
        there's no single obviously "right" color composite for a low-res
        preview, and reading only one channel avoids pulling all of them (49,
        for a CyCIF stack) into memory for a 250px image.
        """
        from PIL import Image
        from ..array.channel import channel_axis_of

        level = self._levels[-1]
        axes = self.axes
        c_ax = channel_axis_of(axes)

        if c_ax is None:
            img = _thumbnail_grayscale(np.asarray(level.data[:]))
            return Image.fromarray(img)                               # MINISBLACK

        if level.shape[c_ax] == 3 and np.dtype(level.dtype) == np.uint8:
            img = np.asarray(level.data[:])
            if c_ax != len(axes) - 1:
                img = np.moveaxis(img, c_ax, -1)
            return Image.fromarray(img)                              # RGB

        idx = [slice(None)] * len(axes)
        idx[c_ax] = 0
        img = _thumbnail_grayscale(np.asarray(level.data[tuple(idx)]))
        return Image.fromarray(img)                                   # first channel

    @property
    def data(self):
        if len(self._levels) == 1:
            return self._levels[0].data
        else:
            return [level.data for level in self._levels]

    def multiscale(self, eager=True):
        """Every level's array as a list, one entry per level.

        The difference from :attr:`data` is that this is *always* a list, even
        for a single-level series, so a consumer that walks a pyramid does not
        have to branch on how many levels the file turned out to have.

        ``eager=True`` also flips the series to materialize on index, which is
        the contract a viewer's chunk loader expects: it asks for a region and
        wants an ndarray back. It costs nothing up front — the levels still
        read only the region indexed.

        .. code-block:: python

            >>> levels = series.multiscale()
            >>> viewer.add_image(levels, multiscale=len(levels) > 1)
        """
        if eager:
            self.eagerfy()
        return [level.data for level in self._levels]

    @property
    def channel_names(self):
        """Names the file records for its channels, or ``None``.

        Recovered from the OME ``Channel`` elements of any OME-TIFF, including
        the ones ezslide's own writer produces, so a converted slide keeps the
        marker names it was written with. ``VsiSeries`` overrides this with the
        names from the cellSens tag tree.
        """
        names = self._levels[0].metadata.get('ChannelNames')
        return [str(name) for name in names] if names else None

    @property
    def pixel_size(self):
        """``(y, x)`` micrometres per pixel at level 0, or ``None``.

        Broader than ``tifffile``'s ``.mpp``, which reads the TIFF resolution
        tags only: this is whatever the format's parser recovered, so OME XML
        and the cellSens tag tree answer here too.
        """
        md = self._levels[0].metadata
        y, x = md.get('PhysicalSizeY'), md.get('PhysicalSizeX')
        if y is None or x is None:
            return None
        try:
            return float(y), float(x)
        except (TypeError, ValueError):
            return None

    @property
    def level_shapes(self):
        """``[(height, width), ...]`` per level, whatever axis order is used."""
        return [(level.height, level.width) for level in self._levels]

    @property
    def downsamples(self):
        """Each level's linear shrink factor relative to level 0."""
        base = self._levels[0].height
        return [base / level.height for level in self._levels]

    def __repr__(self):
        lines = [
                f'Image {self.name!r}' if self.name else 'Image' + f'of type {self.kind}',
                f'Data type: {str(self.dtype)}',
                f"Axes order: {self.axes}",
                f'Pyramidal with {len(self.levels)} levels:' if self.is_multiscale else '',
            ]
        if self.is_multiscale:
            for level_id, level in enumerate(self.levels):
                lines.append(f'  Level {level_id}, data shape: {level.shape}, chunk shape: {level.data.chunks}')
        else:
            lines.append(f'Data shape: {self.levels[0].shape}')

        return ' \n'.join(s for s in lines if s)
    
    def __getitem__(self, key):
        if isinstance(key, int) or len(key) == 0:
            return self.levels[key]
        else:
            return self.levels[key[0]][key[1:]]

    def __setitem__(self, key, value):
        if isinstance(key, int) or len(key) == 0:
            self.levels[key] = value
        else:
            self.levels[key[0]][key[1:]] = value

    def eagerfy(self):
        """Make every level of this series materialize on index."""
        for level in self._levels:
            level.eagerfy()
        return self

    def lazify(self):
        """Make every level of this series return lazy views again."""
        for level in self._levels:
            level.lazify()
        return self

    def parse_metadata(self, kind):
        for level in self._levels:
            level.parse_metadata(kind)


class TiffFile():
    def __init__(self, file, kind=None, *args, pyramidalize=False,
                 pyramid=None, **kwargs):
        """
        Parameters
        ----------
        pyramidalize : bool
            Give every series that is not already pyramidal a lazy pyramid, so
            that flat TIFFs expose the same multi-level interface as an SVS.
            Series that are pyramidal in the file are left alone. Costs nothing
            at open time — levels compute on demand.
        pyramid : dict, optional
            Options for ``TiffSeries.pyramidalize``, e.g.
            ``{'how': 'mode', 'materialize_below': 3}``.
        """
        self._file = file
        self._tifffile = None
        self._series = self._open(file, *args, pyramidalize=pyramidalize,
                                  pyramid=pyramid, **kwargs)
        self._kind = kind if kind else self.series[0].kind

        try:
            for series in self._series:
                series.parse_metadata(self._kind)
        except Exception as e:
            print(f"Warning: could not parse metadata due to {e}")

    def _open(self, file, *args, pyramidalize=False, pyramid=None, **kwargs):
        """Open the container and return its series as ezslide ``TiffSeries``.

        The single place that assumes the file is a TIFF. A subclass that
        reads a container ``tifffile`` cannot open by itself overrides this —
        see ``formats.vsi.VsiFile`` — and is responsible for setting
        ``self._tifffile`` to whatever should answer the attribute lookups
        that fall through ``__getattr__`` (``None`` if nothing should).
        """
        self._tifffile = tifffile.TiffFile(file, *args, **kwargs)
        self._zarr_store = tifffile.imread(file, *args, **kwargs, aszarr=True)
        return [TiffSeries(series, pyramidalize=pyramidalize, pyramid=pyramid)
                for series in self._tifffile.series]

    @property
    def series(self):
        return self._series

    @property
    def kind(self):
        return self._kind 

    @property
    def data(self):
        if len(self._series) == 1:
            return self._series[0].data
        else:
            return [series.data for series in self._series]
        
    def __getattr__(self, name):
        # __dict__ rather than self._tifffile: an override of _open that fails
        # part-way, or a container with no TIFF behind it, would otherwise send
        # every missing attribute into infinite recursion.
        backing = self.__dict__.get('_tifffile')
        if backing is None:
            raise AttributeError(name)
        return getattr(backing, name)

    def __repr__(self):
        lines = [f'TiffFile ({self.kind}) from {self._file} with {len(self.series)} image series: ']
        for series in self._series:
            lines.append(f'  Series {series.name!r} with {len(series.levels)} levels')
        return ' \n'.join(s for s in lines if s)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._series[key]
        elif isinstance(key, str):
            for series in self.series:
                if series.name == key:
                    return series

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._series[key] = value
        elif isinstance(key, str):
            for id, series in enumerate(self.series):
                if series.name == key:
                    self._series[id] = value
                    break
    
    def eagerfy(self):
        """Make every series in the file materialize on index.

        The escape hatch for code that wants zarr's old contract back —
        ``PIL``, ``np.pad``, anything that types its input as ``ndarray``:

            >>> slide = TiffFile(path).eagerfy()
            >>> slide[0].levels[0][0:512, 0:512]   # np.ndarray, read now
        """
        for series in self._series:
            series.eagerfy()
        return self

    def lazify(self):
        """Make every series in the file return lazy views again."""
        for series in self._series:
            series.lazify()
        return self

    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
        
    def to_ome_tiff(self, path, **kwargs):
        """Write this slide out as a pyramidal, calibrated OME-TIFF.

        Thin sugar over :func:`ezslide.write_ome_tiff`; imported lazily so the
        read path never pulls in the writer.
        """
        from ..writers.ome_tiff import write_ome_tiff
        return write_ome_tiff(self, path, source=self._file, **kwargs)

    def close(self):
        if self._tifffile is not None:
            self._tifffile.close()
