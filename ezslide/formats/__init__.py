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

Loaded lazily: the names below are declared in ``__init__.pyi`` and resolve on
first use, so nothing here imports tifffile until a slide class is asked for.
"""

import lazy_loader as _lazy

__getattr__, __dir__, __all__ = _lazy.attach_stub(__name__, __file__)
