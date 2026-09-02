# Type stub and lazy-import manifest; see ezslide/__init__.pyi for how it works.
# Within-package (level 1) imports only -- lazy_loader rejects anything else.

from .open import (
    channel_groups as channel_groups,
    open_slide as open_slide,
    slide_class as slide_class,
)

__all__ = ["open_slide", "channel_groups", "slide_class"]
