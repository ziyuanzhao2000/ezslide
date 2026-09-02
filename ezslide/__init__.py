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

"""