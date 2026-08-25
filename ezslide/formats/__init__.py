"""The slide model: ``TiffFile`` / ``TiffSeries`` / ``TiffLevel`` over a container.

``tiff``
    Anything ``tifffile`` opens directly — SVS, NDPI, OME-TIFF, plain TIFF.
    Defines the model the other formats subclass.
``vsi``
    Olympus / EVIDENT cellSens datasets, via the vendored ETS prosthetic in
    :mod:`ezslide.vendors.vsitiff`. Subclasses the model rather than
    reimplementing it: an ``.ets`` stack is presented as a synthetic BigTIFF,
    so only the metadata handling differs.
``open``
    ``open_slide`` — the suffix-to-class decision every caller needs and no
    single format module can make — and ``channel_groups``, which reassembles
    the series a format split into channels.

Adding a format means subclassing ``TiffFile``, overriding ``_open()``, and
naming its suffixes in ``open``.
"""

from .open import channel_groups, open_slide, slide_class

__all__ = ["open_slide", "channel_groups", "slide_class"]
