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

Loaded lazily: the names below are declared in ``__init__.pyi`` and resolve on
first use, so the read path never pays for the writer.
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
