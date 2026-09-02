# Type stub and lazy-import manifest for ezslide.
#
# This file has two readers. Type checkers and IDEs read it statically, which
# is how they still see the public API now that __init__.py binds no names at
# import time. lazy_loader.attach_stub parses it with ast at runtime to build
# the name -> module map behind the package's __getattr__. One list, both
# consumers, no drift: a symbol added here is importable and type-visible, and
# a symbol added only to a submodule is neither.
#
# lazy_loader's _StubVisitor accepts only within-package imports (ast level 1).
# An absolute import here -- `from typing import ...` included -- raises
# ValueError when the package is imported. The __all__ below is ignored by the
# visitor (attach derives its own, sorted) and exists for type checkers.

from . import array as array
from . import formats as formats
from . import readers as readers
from . import vendors as vendors
from . import writers as writers

from ._registry import register_readers as register_readers

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
)
from .writers import (
    convert as convert,
    write_ome_tiff as write_ome_tiff,
)

__all__ = [
    "array", "formats", "readers", "vendors", "writers",
    "register_readers",
    "ZarrSlideReader", "pyramid_options",
    "TensorStoreArray", "as_tensorstore", "tensorstore_context",
    "ChannelView", "InterleavedView",
    "TiffFile", "VsiFile", "VsiSeries", "open_vsi",
    "open_slide", "channel_groups",
    "convert", "write_ome_tiff",
    "plan_rechunk", "iter_rechunked", "RechunkPlan",
]
