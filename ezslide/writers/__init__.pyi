# Type stub and lazy-import manifest; see ezslide/__init__.pyi for how it works.
# Within-package (level 1) imports only -- lazy_loader rejects anything else.

from .ome_tiff import (
    OME_NAMESPACE as OME_NAMESPACE,
    PROVENANCE_NAMESPACE as PROVENANCE_NAMESPACE,
    convert as convert,
    write_ome_tiff as write_ome_tiff,
)

__all__ = ["convert", "write_ome_tiff", "OME_NAMESPACE", "PROVENANCE_NAMESPACE"]
