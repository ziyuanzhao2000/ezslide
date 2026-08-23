"""Upstream code, kept pristine so it can be re-synced.

Do not refactor these files. They are maintained outside ezslide and are
vendored verbatim; the only local divergence is the relative import at the top
of ``vsitiff.py`` (``from .vsimeta import ...``, which upstream writes as
``from vsimeta import ...``). Anything ezslide wants to change about their
behavior belongs in :mod:`ezslide.formats.vsi` instead.

``vsimeta``
    Parses the nested tag tree inside a ``.vsi`` container — the true pixel
    sizes, channel names and µm/px that the ``.ets`` sidecars do not carry.
``vsitiff``
    Synthesizes BigTIFF IFDs over an untouched ``.ets`` file so ``tifffile``
    can read Olympus tiles with no copying and no decoding of its own.

Note that ``vsitiff.VsiFile`` is *not* the same class as
``ezslide.formats.vsi.VsiFile``; the latter wraps the former.
"""
