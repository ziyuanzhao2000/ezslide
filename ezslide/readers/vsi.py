"""wsidata reader for Olympus / EVIDENT cellSens ``.vsi`` slides."""

from wsidata.reader._reader_registry import register

from ..formats.vsi import VsiFile
from .base import ZarrSlideReader


@register("vsi_zarr")
class VsiZarrReader(ZarrSlideReader):
    """Read a ``.vsi`` dataset (or a bare ``.ets``) through ezslide.

    Everything the base reader does — lazy tensorstore levels, the shared
    chunk cache, ``pyramid_options()`` — applies unchanged; only the container
    it opens differs. Two things are specific to cellSens:

    * a dataset holds several ``.ets`` stacks (the slide plus the label and
      overview scans, and one series per channel of a fluorescence scan),
      which are exposed as scenes. ``VsiFile`` orders them slide-first, so
      scene 0 is the image you want;
    * the calibration comes from the ``.vsi`` tag tree rather than from TIFF
      tags, which ``VsiSeries`` has already folded into level metadata by the
      time properties are built.

        >>> from wsidata import open_wsi
        >>> wsi = open_wsi('slide.vsi', reader='vsi_zarr')
        >>> wsi = open_wsi('slide.vsi', reader='vsi_zarr', scene=1)  # overview
    """

    name = "vsi_zarr"
    extensions = (".vsi", ".ets")
    supports_scenes = True
    file_cls = VsiFile

    def __init__(self, file, scene=None, **kwargs):
        super().__init__(file, series=scene, **kwargs)

    def _resolve_series(self, scene):
        # Only now are the stacks known, so only now can the index be checked.
        scene = self.validate_scene(scene, len(self.reader.series))
        return 0 if scene is None else scene

    def _build_properties(self):
        props = super()._build_properties()
        all_series = self.reader.series
        props.scene = self._series_idx
        props.n_scenes = len(all_series)
        props.scene_names = [s.name for s in all_series]
        props.magnification = self.series.magnification
        return props
