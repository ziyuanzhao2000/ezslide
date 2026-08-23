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
"""
