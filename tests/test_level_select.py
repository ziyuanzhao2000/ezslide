"""Tests for ``ezslide.select_level_for_downsample`` / ``ezslide.resolve_display_level``.

Runnable either way::

    pytest tests/
    python tests/test_level_select.py    # no pytest needed
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ezslide import select_level_for_downsample, resolve_display_level
from wsidata import SlideProperties

LEVEL_DOWNSAMPLE = [1.0, 4.0, 16.0, 64.0]


def test_picks_level_0_when_target_is_1_or_less():
    level, ds = select_level_for_downsample(LEVEL_DOWNSAMPLE, 1.0, len(LEVEL_DOWNSAMPLE))
    assert (level, ds) == (0, 1.0)

    level, ds = select_level_for_downsample(LEVEL_DOWNSAMPLE, 0.5, len(LEVEL_DOWNSAMPLE))
    assert (level, ds) == (0, 1.0)


def test_picks_the_coarsest_qualifying_level():
    # target=20 -> level 2 (downsample 16) is the coarsest level whose
    # downsample doesn't exceed the target; level 3 (64) is forbidden.
    level, ds = select_level_for_downsample(LEVEL_DOWNSAMPLE, 20.0, len(LEVEL_DOWNSAMPLE))
    assert (level, ds) == (2, 16.0)


def test_picks_the_exact_level_when_target_matches():
    level, ds = select_level_for_downsample(LEVEL_DOWNSAMPLE, 4.0, len(LEVEL_DOWNSAMPLE))
    assert (level, ds) == (1, 4.0)


def test_clamps_to_the_lowest_resolution_level_when_target_exceeds_all():
    level, ds = select_level_for_downsample(LEVEL_DOWNSAMPLE, 1000.0, len(LEVEL_DOWNSAMPLE))
    assert (level, ds) == (3, 64.0)


def _props(shape=(4096, 8192)):
    n = len(LEVEL_DOWNSAMPLE)
    return SlideProperties(
        shape=shape,
        n_level=n,
        level_shape=[(shape[0] // int(d), shape[1] // int(d)) for d in LEVEL_DOWNSAMPLE],
        level_downsample=LEVEL_DOWNSAMPLE,
    )


def test_resolve_display_level_matches_axes_size():
    props = _props(shape=(4096, 8192))  # h0, w0
    # Rendering into a ~400x400px axes with oversample=1.5 wants roughly
    # max(8192/(400*1.5), 4096/(400*1.5)) ~= 13.65 -> coarsest level <= that
    # is level 1 (downsample 4.0); level 2 (16.0) would under-resolve.
    level, ds = resolve_display_level(props, 400, 400, oversample=1.5)
    assert (level, ds) == (1, 4.0)


def test_resolve_display_level_falls_back_to_full_resolution_without_axes_size():
    props = _props()
    level, ds = resolve_display_level(props, None, None)
    assert (level, ds) == (0, 1.0)


def test_resolve_display_level_uses_level_0_for_a_large_axes():
    props = _props(shape=(512, 512))
    level, ds = resolve_display_level(props, 2000, 2000, oversample=1.5)
    assert (level, ds) == (0, 1.0)


# --------------------------------------------------------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception as exc:                             # noqa: BLE001
            failed.append((name, exc))
            print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - len(failed)}/{len(tests)} passed')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(_main())
