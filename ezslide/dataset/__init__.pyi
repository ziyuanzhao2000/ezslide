# Type stub and lazy-import manifest; see ezslide/__init__.pyi for how it works.
# Within-package (level 1) imports only -- lazy_loader rejects anything else.

from .patch import (
    PatchBlockDataset as PatchBlockDataset,
    PatchDataset as PatchDataset,
)
from .tile import tile_images as tile_images

__all__ = ["PatchDataset", "PatchBlockDataset", "tile_images"]
