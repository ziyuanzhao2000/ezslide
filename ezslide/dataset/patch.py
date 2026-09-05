"""Per-tile and block-batched torch Datasets for streaming WSI patches.

Both classes match the constructor signature and per-item dict output of
``wsidata.dataset.image.TileImagesDataset``, so they are drop-in
alternatives selectable through :func:`ezslide.dataset.tile_images`. They
differ only in how pixels are fetched:

``PatchDataset``
    One read per tile, like ``TileImagesDataset``, but goes through
    tensorstore's async ``Future`` API when the reader is tensorstore-backed
    instead of blocking on ``reader.get_region()``.
``PatchBlockDataset``
    Groups overlapping tiles into disk-aligned blocks and decodes each block
    once, cutting individual patches out of it with plain numpy slicing.
    Falls back to ``PatchDataset``'s per-tile behavior when the tile grid is
    not uniform (block dedup needs a fixed patch size and level).

Backend detection happens once, in ``__init__`` -- every reader
``ezslide.register_readers()`` registers subclasses ``ZarrSlideReader`` and
is tensorstore-backed by construction, so an ``isinstance`` check is enough;
no per-read probing of the underlying array is needed. When the reader is
*not* tensorstore-backed (e.g. wsidata's own ``OpenSlideReader``), every read
falls back to the plain, serial ``reader.get_region()`` call, matching
``TileImagesDataset``.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import cached_property

import cv2
import numpy as np
from torch.utils.data import Dataset
from wsidata import shapes2tiles

from ..array.tensorstore_array import tensorstore_context
from ..readers.base import ZarrSlideReader


def _color_norm_fn(color_norm):
    if color_norm is None:
        return lambda x: x
    from wsidata._normalizer import ColorNormalizer

    cn = ColorNormalizer(method=color_norm)
    return lambda x: cn(x)


def _level_chunks(reader, level):
    """Chunk shape of a pyramid level, or None if unavailable."""
    try:
        lv = reader.series.levels[level]
        return getattr(lv, "data", lv).chunks
    except Exception:
        return None


def _aligned_block_size(chunks, patch_hw, target_px):
    """Pick a (height, width) read block, snapped to the chunk grid when known.

    Reading on the chunk grid means each compressed chunk is decoded exactly
    once for the whole block instead of once per patch that overlaps it.
    Ported from mesoslide.tools._embed_patch._aligned_block_size.
    """
    patch_h, patch_w = patch_hw
    chunk_h = chunk_w = None
    if chunks is not None:
        chunk_h, chunk_w = int(chunks[0]), int(chunks[1])

    def _snap(chunk, minimum):
        if not chunk or chunk > 2 * target_px:
            return max(target_px, minimum)
        return max(max(1, int(round(target_px / chunk))) * chunk, minimum)

    return _snap(chunk_h, patch_h), _snap(chunk_w, patch_w)


def _group_patches_into_blocks(xmin, ymin, patch_w, patch_h, block_h, block_w):
    """Bucket uniform-size patches into spatial blocks, one read per block.

    Ported from mesoslide.tools._embed_patch._group_patches_into_blocks,
    specialized to a single uniform patch size (callers verify uniformity
    first). Returns a list of (row_indices, y0, y1, x0, x1).
    """
    xmax = xmin + patch_w
    ymax = ymin + patch_h
    by, bx = ymin // block_h, xmin // block_w
    key = by * (int(bx.max()) + 1) + bx
    order = np.argsort(key, kind="stable")
    cuts = np.flatnonzero(np.diff(key[order])) + 1
    blocks = []
    for rows in np.split(order, cuts):
        blocks.append((
            rows,
            int(ymin[rows].min()), int(ymax[rows].max()),
            int(xmin[rows].min()), int(xmax[rows].max()),
        ))
    return blocks


class PatchDataset(Dataset):
    """Per-tile patch reader, API-compatible with wsidata's TileImagesDataset."""

    def __init__(self, wsi, key="tiles", target_key=None, transform=None,
                 color_norm=None, target_transform=None, image_size=None,
                 async_batch=None, num_workers=0):
        tiles_gdf = wsi[key]
        self.color_norm = color_norm
        self._tile_requests = shapes2tiles(wsi, key, image_size=image_size)

        bounds = tiles_gdf.bounds
        self._minx = bounds["minx"].to_numpy()
        self._miny = bounds["miny"].to_numpy()
        self._maxx = bounds["maxx"].to_numpy()
        self._maxy = bounds["maxy"].to_numpy()

        self.tissue_ids = (tiles_gdf["tissue_id"].to_numpy()
                            if "tissue_id" in tiles_gdf.columns else None)
        self.targets = (tiles_gdf[target_key].to_numpy()
                         if target_key is not None else None)

        self.transform = transform
        self.target_transform = target_transform

        # Expected DataLoader worker count in this process, used to size
        # tensorstore's data_copy_concurrency limit so N worker processes
        # don't each claim the whole machine's threads.
        self._num_workers = max(1, num_workers)

        # Fixed for the life of the dataset -- checked once, not per read.
        self.reader = wsi.reader
        self._tensorstore_backed = isinstance(self.reader, ZarrSlideReader)
        self.async_batch = (self._tensorstore_backed if async_batch is None
                             else async_batch)

        # Safe to pickle to a spawned DataLoader worker; reopens lazily on
        # first real access in whichever process ends up using it.
        self.reader.detach_reader()

    @cached_property
    def _cn_func(self):
        return _color_norm_fn(self.color_norm)

    def __len__(self):
        return len(self._tile_requests)

    # -- pixel reads ---------------------------------------------------------

    def _level_view(self, tile_req):
        """Lazy tensorstore view for one tile. Tensorstore-backed reader only."""
        tensorstore_context(num_workers=self._num_workers)
        level = self.reader.translate_level(tile_req.level)
        ds = self.reader.properties.level_downsample[level]
        y0 = int(tile_req.y / ds)
        x0 = int(tile_req.x / ds)
        arr = self.reader.series.levels[level]
        return arr[y0:y0 + tile_req.height, x0:x0 + tile_req.width]

    def _read_one(self, tile_req):
        if self._tensorstore_backed:
            return self._level_view(tile_req)._ts.read().result()
        return self.reader.get_region(tile_req.x, tile_req.y, tile_req.width,
                                       tile_req.height, level=tile_req.level)

    def _read_many(self, tile_reqs):
        """Batched read for __getitems__: issue every future before waiting
        on any of them when tensorstore-backed; serial reads otherwise."""
        if self._tensorstore_backed and self.async_batch:
            futures = [self._level_view(tr)._ts.read() for tr in tile_reqs]
            return [f.result() for f in futures]
        return [self._read_one(tr) for tr in tile_reqs]

    # -- postprocessing (matches TileImagesDataset.__getitem__) --------------

    def _postprocess(self, idx, tile):
        tile_req = self._tile_requests[idx]
        if tile_req.dsize is not None:
            tile = cv2.resize(tile, tile_req.dsize)
        tile = self._cn_func(tile)

        out_w = tile.shape[1]
        tile_w_base = self._maxx[idx] - self._minx[idx]
        downsample = tile_w_base / out_w if out_w > 0 else 1.0

        if self.transform:
            tile = self.transform(tile)

        result = {
            "image": tile,
            "x": int(self._minx[idx]),
            "y": int(self._miny[idx]),
            "tissue_id": int(self.tissue_ids[idx]) if self.tissue_ids is not None else -1,
            "downsample": downsample,
        }
        if self.targets is not None:
            target = self.targets[idx]
            if self.target_transform:
                target = self.target_transform(target)
            result["target"] = target
        return result

    def __getitem__(self, idx):
        tile = self._read_one(self._tile_requests[idx])
        return self._postprocess(idx, tile)

    def __getitems__(self, indices):
        tiles = self._read_many([self._tile_requests[i] for i in indices])
        return [self._postprocess(i, t) for i, t in zip(indices, tiles)]


class PatchBlockDataset(PatchDataset):
    """PatchDataset that decodes disk-aligned blocks once and reuses them for
    every tile that overlaps them, instead of redecoding shared pixels per
    tile. Falls back to PatchDataset's per-tile reads when the tile grid is
    ragged (block dedup needs one fixed patch size and level).

    Port of mesoslide.tools._embed_patch.PatchBlockDataset to ezslide's
    channel-last (H, W, C) layout and to per-tile (not per-block) output, so
    the public API matches wsidata's TileImagesDataset exactly.
    """

    def __init__(self, wsi, key="tiles", target_key=None, transform=None,
                 color_norm=None, target_transform=None, image_size=None,
                 async_batch=None, num_workers=0, cache_size=4, block_px=4096):
        super().__init__(wsi, key=key, target_key=target_key, transform=transform,
                          color_norm=color_norm, target_transform=target_transform,
                          image_size=image_size, async_batch=async_batch,
                          num_workers=num_workers)
        self.cache_size = cache_size
        self._block_cache = OrderedDict()
        # Probing chunk/level metadata below reopens the reader (via
        # .series), which PatchDataset.__init__ already detached for
        # pickling -- detach again once done.
        self._block_mode = self._setup_blocks(block_px)
        self.reader.detach_reader()

    def _setup_blocks(self, block_px):
        levels = {tr.level for tr in self._tile_requests}
        heights = {tr.height for tr in self._tile_requests}
        widths = {tr.width for tr in self._tile_requests}
        if len(levels) != 1 or len(heights) != 1 or len(widths) != 1:
            return False

        self._patch_level = next(iter(levels))
        self._patch_h = next(iter(heights))
        self._patch_w = next(iter(widths))

        level = self.reader.translate_level(self._patch_level)
        ds = self.reader.properties.level_downsample[level]
        self._ymin_lvl = np.array([int(tr.y / ds) for tr in self._tile_requests])
        self._xmin_lvl = np.array([int(tr.x / ds) for tr in self._tile_requests])

        chunks = _level_chunks(self.reader, level) if self._tensorstore_backed else None
        block_h, block_w = _aligned_block_size(
            chunks, (self._patch_h, self._patch_w), block_px)

        self._blocks = _group_patches_into_blocks(
            self._xmin_lvl, self._ymin_lvl, self._patch_w, self._patch_h,
            block_h, block_w)
        self._patch_to_block = np.empty(len(self._tile_requests), dtype=np.int64)
        for block_id, (rows, *_bounds) in enumerate(self._blocks):
            self._patch_to_block[rows] = block_id
        return True

    def _cache_put(self, block_id, block):
        self._block_cache[block_id] = block
        if len(self._block_cache) > self.cache_size:
            self._block_cache.popitem(last=False)

    def _read_block_uncached(self, block_id):
        _, y0, y1, x0, x1 = self._blocks[block_id]
        level = self.reader.translate_level(self._patch_level)
        if self._tensorstore_backed:
            tensorstore_context(num_workers=self._num_workers)
            view = self.reader.series.levels[level][y0:y1, x0:x1]
            return view._ts.read().result()
        ds = self.reader.properties.level_downsample[level]
        return self.reader.get_region(int(x0 * ds), int(y0 * ds),
                                       x1 - x0, y1 - y0, level=self._patch_level)

    def _read_block(self, block_id):
        if block_id in self._block_cache:
            self._block_cache.move_to_end(block_id)
            return self._block_cache[block_id]
        block = self._read_block_uncached(block_id)
        self._cache_put(block_id, block)
        return block

    def _read_blocks_async(self, block_ids):
        """Return {block_id: array} for every block a batch needs.

        Collects results in a local dict before touching the shared LRU
        cache, so a batch spanning more distinct blocks than cache_size
        cannot evict a block this same call still needs -- inserting into
        the cache eagerly per-block let an eviction remove a block before
        it had been sliced into a patch yet.
        """
        unique_ids = list(dict.fromkeys(block_ids))
        blocks = {b: self._block_cache[b] for b in unique_ids if b in self._block_cache}
        pending = [b for b in unique_ids if b not in blocks]
        if pending:
            if self._tensorstore_backed and self.async_batch:
                tensorstore_context(num_workers=self._num_workers)
                level = self.reader.translate_level(self._patch_level)
                futures = {}
                for block_id in pending:
                    _, y0, y1, x0, x1 = self._blocks[block_id]
                    futures[block_id] = self.reader.series.levels[level][y0:y1, x0:x1]._ts.read()
                for block_id, future in futures.items():
                    blocks[block_id] = future.result()
            else:
                for block_id in pending:
                    blocks[block_id] = self._read_block_uncached(block_id)
            for block_id in pending:
                self._cache_put(block_id, blocks[block_id])
        return blocks

    def _patch_from_block(self, idx, block):
        _, y0, _y1, x0, _x1 = self._blocks[self._patch_to_block[idx]]
        y = self._ymin_lvl[idx] - y0
        x = self._xmin_lvl[idx] - x0
        return block[y:y + self._patch_h, x:x + self._patch_w]

    def __getitem__(self, idx):
        if not self._block_mode:
            return super().__getitem__(idx)
        block = self._read_block(self._patch_to_block[idx])
        return self._postprocess(idx, self._patch_from_block(idx, block))

    def __getitems__(self, indices):
        if not self._block_mode:
            return super().__getitems__(indices)
        block_ids = [self._patch_to_block[i] for i in indices]
        blocks = self._read_blocks_async(block_ids)
        tiles = [self._patch_from_block(i, blocks[b])
                 for i, b in zip(indices, block_ids)]
        return [self._postprocess(i, t) for i, t in zip(indices, tiles)]
