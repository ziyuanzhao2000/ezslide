import zarr
import dask.array as da
import numpy as np
import tifffile
import re
from math import log
from PIL import Image
from .global_chunk_cache import make_cache_store
from .lazy_pyramid import lazy_pyramid


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
    array : zarr array
        The array to iterate over.
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
        self._page = tiffpage

        base_store = self._page.aszarr()
        cached_store = make_cache_store(base_store)
        self.data = zarr.open(cached_store)
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

    def tiles(self, tile_size=None):
        return iter_tiles(self.data, self.axes, tile_size)

class TiffLevel():
    def __init__(self, tifflevel, level_id):
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
        self.data = initial_array
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

    def tiles(self, tile_size=None):
        return iter_tiles(self.data, self.axes, tile_size)

    def parse_metadata(self, kind):
        metadata = {}
        metadata.update(TIFFParser(self._level))
        if kind == 'ndpi':
            parser = NDPIParser
        else:
            parser = lambda x: x
        self._parsed_metadata = parser(metadata)

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

    @property
    def delayed_data(self):
        """Dask view, built on first use — only ``__repr__`` needs it."""
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
        base = self._levels[0]
        axes = infer_axes(base.data, getattr(base, 'axes', None))
        stack = lazy_pyramid(base.data, levels=levels,
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
        img = self._levels[-1].data[:]
        if len(self.axes) == 2:
            return Image.fromarray(img) # MINISBLACK
        elif len(self.axes) == 3:
            if self.axes[:2] == 'YX' and img.shape[2] == 3:
                return Image.fromarray(img) # RGB
            elif self.axes[1:] == 'YX' and img.shape[0] == 3:
                return Image.fromarray(img.transpose(1,2,0)) # RGB

    @property
    def data(self):
        if len(self._levels) == 1:
            return self._levels[0].data
        else:
            return [level.data for level in self._levels]
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
        self._tifffile = tifffile.TiffFile(file, *args, **kwargs)
        self._zarr_store = tifffile.imread(file, *args, **kwargs, aszarr=True)
        self._series = [TiffSeries(series, pyramidalize=pyramidalize,
                                   pyramid=pyramid)
                        for series in self._tifffile.series]
        self._kind = kind if kind else self.series[0].kind 

        try:
            for series in self._series:
                series.parse_metadata(self._kind)
        except Exception as e:
            print(f"Warning: could not parse metadata due to {e}")
            
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
        return getattr(self._tifffile, name)

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
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
        
    def close(self):
        self._tifffile.close()

def remove_invalid_xml_chars(text):
    xml_compliant_text = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', text)
    return xml_compliant_text

class TiffWriter(tifffile.TiffWriter):
    def __init__(self, file, *args, **kwargs):
        self.file = file
        super().__init__(file, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tifffile, name)
    
    def write(self, 
            tiff_obj, 
            metadata={},
            write_tiles=True, 
            *args, **kwargs):
        
        if isinstance(tiff_obj, TiffLevel):
            level = tiff_obj
            serialized = dict([(remove_invalid_xml_chars(str(key)), 
                                remove_invalid_xml_chars(str(val))) for key, val in level.metadata.items()])
            metadata.update(serialized)
            metadata.update({'MapAnnotation': serialized,})
            tile_size = kwargs.get("tile", None)
            if tile_size is None:
                min_size = min(level.width, level.height)
                if min_size > 1024:
                    tile_size = (1024, 1024)
                else:
                    s = 2**(int(log(min_size, 2)))
                    tile_size = (s, s)
            print(tile_size)
            kwargs['tile']=tile_size
            shape = level.data.shape
            if write_tiles:
                data = level.tiles(tile_size)
                if level.axes == 'YXS':
                    shape = tuple([shape[i] for i  in [2, 0, 1]])
            else:
                data = level.data[:]
            if level.level_id == 0: # base level
                metadata.update({'Name': level.name})
            if not 'photometric' in kwargs:
                if isinstance(level.metadata['BitsPerSample'], int):
                    photometric = 'MINISBLACK'
                elif isinstance(level.metadata['BitsPerSample'], tuple):
                    if len(level.metadata['BitsPerSample']) == 3:
                        photometric = 'RGB'
                    else:
                        photometric = 'MINISBLACK'
                kwargs['photometric'] = photometric
            super().write(data, 
                        metadata=metadata, 
                        shape=shape,
                        dtype=level.data.dtype,
                        *args, **kwargs)
        elif isinstance(tiff_obj, TiffSeries):
            series = tiff_obj
            subresolutions = len(series.levels) - 1
            if subresolutions == 0:
                self.write(series.levels[0], 
                        metadata=metadata, 
                        write_tiles=write_tiles,
                        *args,
                        **kwargs)
            else:
                self.write(series.levels[0], 
                        metadata=metadata, 
                        write_tiles=write_tiles,
                        subifds=subresolutions,
                        *args,
                        **kwargs)
                for level in series.levels[1:]:
                    self.write(level, 
                        metadata=metadata, 
                        write_tiles=write_tiles,
                        subfiletype=1,
                        *args,
                        **kwargs)
        elif isinstance(tiff_obj, TiffFile):
            file = tiff_obj
            for series in file.series:
                self.write(series, 
                           metadata=metadata,
                           write_tiles=write_tiles,
                           *args,
                           **kwargs)