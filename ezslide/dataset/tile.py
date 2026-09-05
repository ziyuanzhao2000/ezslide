"""Entry point for building a per-tile patch dataset, matching the
signature of wsidata.WSIData.ds.tile_images() with two extra parameters
selecting the implementation.
"""

from __future__ import annotations

from .patch import PatchBlockDataset, PatchDataset


def tile_images(wsi, tile_key="tiles", target_key=None, transform=None,
                 target_transform=None, color_norm=None, image_size=None,
                 backend="ezslide", block=True, num_workers=0, cache_size=4,
                 async_batch=None):
    """Build a per-tile patch dataset for a WSIData object.

    Parameters
    ----------
    wsi : wsidata.WSIData
    tile_key, target_key, transform, target_transform, color_norm, image_size
        Same meaning as ``wsidata.WSIData.ds.tile_images()``.
    backend : {"ezslide", "wsidata"}, default "ezslide"
        "ezslide" uses this package's PatchDataset/PatchBlockDataset.
        "wsidata" delegates to ``wsi.ds.tile_images(...)``.
    block : bool, default True
        When backend="ezslide", use the block-deduping PatchBlockDataset
        (default) instead of the naive per-tile PatchDataset.
    num_workers : int, default 0
        Expected DataLoader worker count. Forwarded to the dataset so it can
        size tensorstore's data_copy_concurrency limit to
        max(1, os.cpu_count() // num_workers) instead of leaving every
        worker process free to claim the whole machine's threads. Ignored
        when backend="wsidata".
    cache_size : int, default 4
        backend="ezslide", block=True only: number of decoded blocks kept in
        memory at once.
    async_batch : bool, optional
        backend="ezslide" only: whether __getitems__ issues tensorstore
        futures for a whole DataLoader batch before waiting on any of them.
        Defaults to True when the reader is tensorstore-backed, False
        otherwise (see PatchDataset). Set explicitly to force serial
        per-tile reads even on a tensorstore-backed reader, e.g. to compare
        against the async path.

    Returns
    -------
    torch.utils.data.Dataset
    """
    if backend == "wsidata":
        return wsi.ds.tile_images(
            tile_key=tile_key, target_key=target_key, transform=transform,
            target_transform=target_transform, color_norm=color_norm,
            image_size=image_size)
    if backend != "ezslide":
        raise ValueError(f"Unknown backend {backend!r}; expected 'ezslide' or 'wsidata'")

    kwargs = dict(key=tile_key, target_key=target_key, transform=transform,
                  target_transform=target_transform, color_norm=color_norm,
                  image_size=image_size, num_workers=num_workers,
                  async_batch=async_batch)
    if block:
        return PatchBlockDataset(wsi, cache_size=cache_size, **kwargs)
    return PatchDataset(wsi, **kwargs)
