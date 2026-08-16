import numpy as np
from wsidata import SlideProperties
from wsidata.reader import ReaderBase
from .tifffile_zarr import TiffFile
from wsidata.reader._reader_registry import register


class TiffFileZarrReader(ReaderBase):
    name = "tifffile_zarr"
    pkg_namespaces = ["tifffile", "zarr"]          # <- the fix
    pkgs = ["tifffile", "zarr"]                    # pip names, for error messages
    extensions = (".ndpi", ".tif", ".tiff", ".svs", ".scn", ".bif", ".qptiff")
    supports_scenes = False

    def __init__(self, file, series=0, **kwargs):
        self.file = str(file)
        self._series_idx = series
        self._kwargs = kwargs
        self.create_reader()
        self.properties = self._build_properties()

    def create_reader(self):
        self.set_reader(TiffFile(self.file, **self._kwargs))

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
    
register('tifffile_zarr')(TiffFileZarrReader)