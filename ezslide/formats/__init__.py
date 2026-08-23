"""The slide model: ``TiffFile`` / ``TiffSeries`` / ``TiffLevel`` over a container.

``tiff``
    Anything ``tifffile`` opens directly — SVS, NDPI, OME-TIFF, plain TIFF.
    Defines the model the other formats subclass.
``vsi``
    Olympus / EVIDENT cellSens datasets, via the vendored ETS prosthetic in
    :mod:`ezslide.vendors.vsitiff`. Subclasses the model rather than
    reimplementing it: an ``.ets`` stack is presented as a synthetic BigTIFF,
    so only the metadata handling differs.

Adding a format means subclassing ``TiffFile`` and overriding ``_open()``.
"""
