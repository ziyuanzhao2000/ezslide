__all__ = ['register_readers']


def register_readers():
    """Import ezslide's wsidata adapters, registering them with ``open_wsi``.

    Call this before letting :func:`wsidata.open_wsi` pick a reader by file
    extension. Importing ezslide no longer does it for you, because that would
    pull wsidata — and spatialdata, anndata, cv2 and geopandas behind it — into
    every process that only wanted to read a TIFF.

    Also patches :func:`wsidata.open_wsi`'s ``attach_images=True`` path
    (:func:`ezslide.readers.datatree.patch_to_datatree`) so it resolves
    channel count, dtype and channel names from the slide instead of assuming
    3-channel ``uint8`` RGB — needed for multiplexed IF (``CYX``) and mono
    (``YX``) images to attach correctly, not just standard H&E/brightfield
    slides.

    Returns
    -------
    module
        :mod:`ezslide.readers`, so ``readers = ezslide.register_readers()``
        works if you want the module as well as the side effect.

    Examples
    --------
    .. code-block:: python

        >>> import ezslide
        >>> from wsidata import open_wsi
        >>> ezslide.register_readers()
        >>> wsi = open_wsi('slide.vsi')          # finds 'vsi_zarr' by suffix
    """
    from . import readers
    from .readers.datatree import patch_to_datatree

    patch_to_datatree()
    return readers
