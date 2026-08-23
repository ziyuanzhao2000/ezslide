"""ezslide — lazy whole-slide image access on tifffile, zarr and tensorstore.

The package is layered, bottom up:

:mod:`ezslide.array`
    Format-agnostic lazy-array machinery: the shared chunk cache, tensorstore
    views, block reduction, lazy pyramids, and the rechunker that retiles an
    array from one chunk grid onto another. Knows nothing about slides.
:mod:`ezslide.vendors`
    Upstream code kept pristine for re-syncing. ``vsitiff`` turns an Olympus
    ``.ets`` container into a synthetic BigTIFF that tifffile can read, using
    the ``.vsi`` tag tree that ``vsimeta`` parses.
:mod:`ezslide.formats`
    The slide model — ``TiffFile`` / ``TiffSeries`` / ``TiffLevel`` over a
    container. ``formats.tiff`` for anything tifffile opens directly,
    ``formats.vsi`` for cellSens datasets.
:mod:`ezslide.readers`
    The wsidata adapters, registered as ``tifffile_zarr``,
    ``tifffile_zarr_pyramid`` and ``vsi_zarr``.
:mod:`ezslide.writers`
    The way back out. ``convert(src, dst)`` streams any slide ezslide reads
    into a pyramidal OME-TIFF that keeps the calibration and metadata the
    reader recovered, generating pyramid levels for a flat input.
:mod:`ezslide.cli`
    The ``ezslide`` command. Composition only — merging files into channels,
    splitting RGB, mask downsampling — built on the writer's duck-typed
    protocol rather than on anything the library had to grow.

Each tier depends only on the ones above it in that list. Adding a format means
a module in ``formats`` that subclasses ``TiffFile`` and overrides ``_open()``,
plus a module in ``readers`` that subclasses ``ZarrSlideReader`` and names it.
"""

from . import readers, writers           # importing readers registers them
from .readers import ZarrSlideReader, pyramid_options
from .array.tensorstore_array import (TensorStoreArray, as_tensorstore,
                                      tensorstore_context)
from .formats.tiff import TiffFile
from .formats.vsi import VsiFile, VsiSeries, open_vsi
from .array.rechunk import RechunkPlan, iter_rechunked, plan_rechunk
from .writers import convert, write_ome_tiff

__all__ = ["readers", "writers", "ZarrSlideReader", "pyramid_options",
           "TensorStoreArray", "as_tensorstore", "tensorstore_context",
           "TiffFile", "VsiFile", "VsiSeries", "open_vsi",
           "convert", "write_ome_tiff",
           "plan_rechunk", "iter_rechunked", "RechunkPlan"]
