"""Format-agnostic lazy-array machinery.

Nothing here knows what a slide is. These are the pieces the format modules in
:mod:`ezslide.formats` compose over whatever container they have opened:

``cache``
    One process-wide LRU chunk pool shared by every zarr store ezslide opens.
``tensorstore_array``
    ``TensorStoreArray`` — indexing composes a view and decodes nothing until
    the result is materialized.
``reduce``
    ``block_reduce`` and friends: the downsampling kernels, including the
    ``'mode'`` kernel that label maps need.
``pyramid``
    ``lazy_pyramid`` — pyramid levels that know their shape up front and
    compute pixels only when indexed.
``rechunk``
    ``plan_rechunk`` / ``iter_rechunked`` — read an array on one chunk grid and
    emit it on another, each source chunk decoded once, within a stated memory
    budget. What the OME-TIFF writer uses to retile a slide.
"""

from .rechunk import DEFAULT_MAX_MEM, RechunkPlan, iter_rechunked, plan_rechunk

__all__ = ["plan_rechunk", "iter_rechunked", "RechunkPlan", "DEFAULT_MAX_MEM"]
