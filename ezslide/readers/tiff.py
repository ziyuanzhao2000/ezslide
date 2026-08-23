"""wsidata readers for anything ``tifffile`` opens directly."""

from wsidata.reader._reader_registry import register

from ..formats.tiff import TiffFile
from .base import ZarrSlideReader


@register("tifffile_zarr")
class TiffFileZarrReader(ZarrSlideReader):
    """SVS, NDPI, OME-TIFF and friends, as lazy tensorstore-backed levels."""

    name = "tifffile_zarr"
    extensions = (".ndpi", ".tif", ".tiff", ".svs", ".scn", ".bif", ".qptiff")
    file_cls = TiffFile


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
