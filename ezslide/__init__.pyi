from . import array as array
from . import formats as formats
from . import readers as readers
from . import vendors as vendors
from . import writers as writers

from ._registry import register_readers as register_readers

from .io import (
    open_wsi as open_wsi,
    read_wsi as read_wsi,
)

from .array.channel import (
    ChannelView as ChannelView,
    InterleavedView as InterleavedView,
)
from .array.rechunk import (
    RechunkPlan as RechunkPlan,
    iter_rechunked as iter_rechunked,
    plan_rechunk as plan_rechunk,
)
from .array.tensorstore_array import (
    TensorStoreArray as TensorStoreArray,
    as_tensorstore as as_tensorstore,
    tensorstore_context as tensorstore_context,
)
from .formats.open import (
    channel_groups as channel_groups,
    open_slide as open_slide,
)
from .formats.tiff import TiffFile as TiffFile
from .formats.vsi import (
    VsiFile as VsiFile,
    VsiSeries as VsiSeries,
    open_vsi as open_vsi,
)
from .readers import (
    ZarrSlideReader as ZarrSlideReader,
    pyramid_options as pyramid_options,
    patch_to_datatree as patch_to_datatree,
)
from .writers import (
    convert as convert,
    write_ome_tiff as write_ome_tiff,
)

__all__ = [
    "array", "formats", "readers", "vendors", "writers",
    "register_readers",
    "ZarrSlideReader", "pyramid_options", "patch_to_datatree",
    "TensorStoreArray", "as_tensorstore", "tensorstore_context",
    "ChannelView", "InterleavedView",
    "TiffFile", "VsiFile", "VsiSeries", "open_vsi",
    "open_slide", "channel_groups",
    "convert", "write_ome_tiff",
    "plan_rechunk", "iter_rechunked", "RechunkPlan",
    "open_wsi", "read_wsi",
]
