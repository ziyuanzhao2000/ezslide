# Type stub and lazy-import manifest; see ezslide/__init__.pyi for how it works.
# Within-package (level 1) imports only -- lazy_loader rejects anything else.

from .channel import (
    ChannelView as ChannelView,
    InterleavedView as InterleavedView,
    channel_axis_of as channel_axis_of,
    n_channels as n_channels,
)
from .rechunk import (
    DEFAULT_MAX_MEM as DEFAULT_MAX_MEM,
    RechunkPlan as RechunkPlan,
    iter_rechunked as iter_rechunked,
    plan_rechunk as plan_rechunk,
)

__all__ = [
    "plan_rechunk", "iter_rechunked", "RechunkPlan", "DEFAULT_MAX_MEM",
    "ChannelView", "InterleavedView", "channel_axis_of", "n_channels",
]
