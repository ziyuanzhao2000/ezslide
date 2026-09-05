"""The way in: open a whole slide image and remember where it came from.

``wsidata.open_wsi()`` returns a ``WSIData`` backed by a Zarr store, but
nothing in that store records which slide file produced it or which reader
opened it. :func:`open_wsi` here is a thin wrapper that fills that gap by
writing a ``wsi_source`` block into ``attrs`` (the same mechanism wsidata
already uses for ``slide_properties``, which is known to round-trip through
``write()``/``read_zarr()``), and :func:`read_wsi` reads it back to
reconstruct a full ``WSIData`` from a store path alone.

Calling ``wsidata.open_wsi()`` directly still works exactly as before; this
module is purely additive.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["open_wsi", "read_wsi", "WSI_SOURCE_KEY"]

WSI_SOURCE_KEY = "wsi_source"


def open_wsi(wsi, store="auto", reader=None, scene=None, **kwargs):
    """Open a whole slide image, recording its source path and reader in attrs.

    Same signature and behavior as :func:`wsidata.open_wsi`. The returned
    ``WSIData``'s ``attrs["wsi_source"]`` records the resolved slide path and
    reader name, so it can later be reconstructed with :func:`read_wsi`
    without re-supplying the path.

    Parameters
    ----------
    wsi, store, reader, scene, **kwargs
        Forwarded to :func:`wsidata.open_wsi`.

    Returns
    -------
    :class:`wsidata.WSIData`
    """
    from wsidata import open_wsi as _open_wsi
    from wsidata.reader.spatialdata_image2d import SpatialDataImage2DReader

    slide_data = _open_wsi(wsi, store=store, reader=reader, scene=scene, **kwargs)

    if WSI_SOURCE_KEY not in slide_data.attrs:
        r = slide_data.reader
        if r is not None and not isinstance(r, SpatialDataImage2DReader):
            slide_data.attrs[WSI_SOURCE_KEY] = {
                "path": str(Path(r.file).resolve()),
                "reader": r.name,
            }
    return slide_data


def read_wsi(store, **kwargs):
    """Reconstruct a WSIData from a Zarr store alone, using its recorded ``wsi_source``.

    Equivalent to ``open_wsi(<recorded slide path>, store=store,
    reader=<recorded reader>, **kwargs)``, so any :func:`open_wsi` keyword
    (``attach_images``, ``attach_thumbnail``, ``scene``, ...) still applies.

    Parameters
    ----------
    store : str or Path
        Path to a Zarr store previously written by a ``WSIData`` opened
        through this module's :func:`open_wsi`.
    **kwargs
        Forwarded to :func:`open_wsi`.

    Returns
    -------
    :class:`wsidata.WSIData`

    Raises
    ------
    ValueError
        If the store doesn't exist, has no recorded ``wsi_source`` (e.g. it
        was written via plain ``wsidata.open_wsi()``/``SpatialData.write()``,
        or before this feature existed), or the recorded slide path can't be
        resolved.
    """
    from spatialdata import read_zarr

    store = Path(store)
    if not store.exists():
        raise ValueError(f"Store '{store}' does not exist.")

    sdata = read_zarr(store)
    source = sdata.attrs.get(WSI_SOURCE_KEY)
    if source is None:
        raise ValueError(
            f"Store '{store}' has no recorded WSI source (attrs['{WSI_SOURCE_KEY}']). "
            "It may have been written by plain wsidata.open_wsi()/SpatialData.write(), "
            f"or before this feature was added. Use ezslide.open_wsi(<slide path>, "
            f"store='{store}') instead."
        )

    path = Path(source["path"])
    if not path.exists():
        candidate = store.parent / path.name
        if candidate.exists():
            path = candidate
        else:
            raise ValueError(
                f"Recorded WSI source '{source['path']}' for store '{store}' no longer "
                f"exists (also checked '{candidate}'). Re-open explicitly with "
                f"ezslide.open_wsi(<slide path>, store='{store}')."
            )

    kwargs.setdefault("reader", source.get("reader"))
    kwargs.setdefault("store", store)
    return open_wsi(path, **kwargs)
