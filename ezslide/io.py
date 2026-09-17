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
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import pandas as pd
    from wsidata import WSIData

__all__ = [
    "open_wsi", "read_wsi", "WSI_SOURCE_KEY",
    "SLIDE_ID", "resolve_manifest", "iter_slides", "open_slides",
]

WSI_SOURCE_KEY = "wsi_source"
SLIDE_ID = "slide_id"


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
    from wsidata.reader._reader_registry import READERS
    from wsidata.reader.spatialdata_image2d import SpatialDataImage2DReader

    if reader is None or reader not in READERS:
        # reader=None means auto-detect by extension, which needs ezslide's
        # readers (tifffile_zarr, vsi_zarr, ...) in the registry to find them;
        # an explicit reader name not yet in the registry needs the same.
        # register_readers() is idempotent, so this is a no-op once it's run.
        from ._registry import register_readers

        register_readers()

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


def resolve_manifest(
    slides_table: "pd.DataFrame", store_col: str, slide_id_col: str
) -> list[tuple[str, str]]:
    """Normalise a cohort manifest to a list of ``(slide_id, store)`` pairs."""
    if store_col not in slides_table.columns:
        raise ValueError(
            f"slides_table has no '{store_col}' column; pass store_col= to name "
            f"the column holding the .zarr store paths. Got: {list(slides_table.columns)}"
        )
    stores = slides_table[store_col].tolist()
    if slide_id_col in slides_table.columns:
        ids = [str(v) for v in slides_table[slide_id_col]]
    else:
        ids = [Path(str(s)).stem for s in stores]
    return list(zip(ids, stores))


def iter_slides(
    slides_table: "pd.DataFrame",
    *,
    store_col: str = "store",
    slide_id_col: str = SLIDE_ID,
    attach_images: bool = False,
    close: bool = True,
) -> Iterator[tuple[str, "WSIData"]]:
    """Yield ``(slide_id, wsi)`` one slide at a time, closing each before advancing.

    Peak memory is one slide, not the whole cohort, regardless of how many
    stores ``slides_table`` lists.

    Parameters
    ----------
    slides_table
        DataFrame with one row per slide. Must have a column of Zarr store
        paths (``store_col``). A ``slide_id`` column is used if present,
        otherwise ids are derived from the store filenames.
    store_col
        Column holding the ``.zarr`` store paths written by ``wsi.write()``.
    slide_id_col
        Column holding slide ids.
    attach_images
        Reattach the WSI pixels. Needed for anything that reads image data
        (patch extraction, plotting); unnecessary for table-only work.
    close
        Close each slide's reader after yielding. Set False only if the
        caller keeps references to the yielded slides beyond the loop body.

    Yields
    ------
    (slide_id, wsi) : (str, WSIData)

    Examples
    --------
    >>> import pandas as pd, ezslide
    >>> manifest = pd.DataFrame({"store": sorted(glob("cohort/*.zarr"))})
    >>> for slide_id, wsi in ezslide.iter_slides(manifest):
    ...     table = wsi.tables["tiles_table"]
    """
    for slide_id, store in resolve_manifest(slides_table, store_col, slide_id_col):
        wsi = read_wsi(store, attach_images=attach_images)
        try:
            yield slide_id, wsi
        finally:
            if close:
                try:
                    wsi.close()
                except Exception:
                    # A reader that never attached (attach_images=False on some
                    # backends) has nothing to detach; not worth failing the loop.
                    pass


def open_slides(
    slides_table: "pd.DataFrame",
    *,
    store_col: str = "store",
    slide_id_col: str = SLIDE_ID,
    attach_images: bool = True,
) -> dict[str, "WSIData"]:
    """Open a whole cohort at once, returning ``{slide_id: WSIData}``.

    The eager counterpart to :func:`iter_slides`, for callers that need
    random access to pixels across slides -- patch extraction and galleries,
    where the rows being read come from many slides interleaved.

    This is cheaper than it sounds: a ``WSIData`` holds a lazy reader plus
    the slide's tables, so the cost is the tables, not the pixels. It is the
    tile *reads* that are expensive, and those stay bounded by the selected
    subset.
    """
    from tqdm import tqdm
    return {
        slide_id: wsi
        for slide_id, wsi in tqdm(iter_slides(
            slides_table,
            store_col=store_col,
            slide_id_col=slide_id_col,
            attach_images=attach_images,
            close=False,
        ))
    }
