"""wsidata reader adapters, one module per format.

Importing this package registers all three readers with wsidata, which is what
makes ``open_wsi('slide.vsi')`` find them by extension:

``tifffile_zarr``
    SVS / NDPI / OME-TIFF / TIFF — :class:`~ezslide.readers.tiff.TiffFileZarrReader`
``tifffile_zarr_pyramid``
    the same, with lazy pyramids on by default
``vsi_zarr``
    Olympus cellSens — :class:`~ezslide.readers.vsi.VsiZarrReader`

The registry names are independent of these module names and are what callers
pass as ``open_wsi(..., reader=...)``.

To add a format, subclass :class:`~ezslide.readers.base.ZarrSlideReader` in a
new module here and import it below, so the ``@register`` runs.
"""

from .base import ZarrSlideReader, pyramid_options
from .datatree import patch_to_datatree, to_datatree
from .tiff import PyramidTiffFileZarrReader, TiffFileZarrReader
from .vsi import VsiZarrReader

__all__ = ["ZarrSlideReader", "pyramid_options", "TiffFileZarrReader",
           "PyramidTiffFileZarrReader", "VsiZarrReader",
           "patch_to_datatree", "to_datatree"]
