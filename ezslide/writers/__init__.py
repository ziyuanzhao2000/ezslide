"""Writing slides back out.

The mirror image of :mod:`ezslide.readers`, with one difference worth knowing:
readers are wsidata adapters registered in a global registry, while writers are
plain functions you call. wsidata has no writer of its own — it persists
analysis results to a SpatialData store and never writes pixels — so this is
the only route from an ezslide slide back to a file.

``ome_tiff``
    :func:`~ezslide.writers.ome_tiff.convert` and
    :func:`~ezslide.writers.ome_tiff.write_ome_tiff`: streaming, pyramidal,
    calibrated OME-TIFF that carries the reader's metadata across.
"""

from .ome_tiff import (OME_NAMESPACE, PROVENANCE_NAMESPACE, convert,
                       write_ome_tiff)

__all__ = ["convert", "write_ome_tiff", "OME_NAMESPACE", "PROVENANCE_NAMESPACE"]
