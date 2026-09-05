"""Torch Datasets for streaming WSI patches: PatchDataset (naive per-tile),
PatchBlockDataset (chunk-aligned block dedup), and the tile_images() entry
point that selects between them or delegates to wsidata's TileImagesDataset.

Loaded lazily via __init__.pyi so importing ezslide does not require torch;
only importing ezslide.dataset (or its re-exported names) does.
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
