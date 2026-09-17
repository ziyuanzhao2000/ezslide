"""Tests for ``ezslide.iter_slides`` / ``ezslide.open_slides`` / ``ezslide.resolve_manifest``.

Runnable either way::

    pytest tests/
    python tests/test_cohort.py    # no pytest needed

Fixtures are synthesized plain RGB TIFFs, same pattern as ``test_io.py``, so
nothing here needs tile tables or a real slide.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezslide

ezslide.register_readers()


def _assert_raises(exc_type, substr, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        assert substr in str(exc), f'{substr!r} not in {exc!r}'
    else:
        raise AssertionError(f'expected {exc_type.__name__} containing {substr!r}')


def _cohort(n=3):
    """A manifest DataFrame of n freshly written .zarr stores in a temp dir."""
    d = tempfile.mkdtemp(prefix='ezslide-cohort-test-')
    stores = []
    for i in range(n):
        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        path = os.path.join(d, f'slide{i}.tif')
        tifffile.imwrite(path, rgb, tile=(128, 128), photometric='rgb')
        store = os.path.join(d, f'slide{i}.zarr')
        wsi = ezslide.open_slide(path, store=store, reader='tifffile_zarr')
        wsi.write(store)
        wsi.close()
        stores.append(store)
    return pd.DataFrame({'store': stores})


def test_resolve_manifest_derives_ids_from_filenames():
    manifest = _cohort(2)
    pairs = ezslide.resolve_manifest(manifest, 'store', ezslide.SLIDE_ID)
    assert [sid for sid, _ in pairs] == ['slide0', 'slide1']


def test_resolve_manifest_uses_slide_id_column_when_present():
    manifest = _cohort(2)
    manifest[ezslide.SLIDE_ID] = ['a', 'b']
    pairs = ezslide.resolve_manifest(manifest, 'store', ezslide.SLIDE_ID)
    assert [sid for sid, _ in pairs] == ['a', 'b']


def test_resolve_manifest_rejects_a_manifest_without_the_store_column():
    _assert_raises(
        ValueError, 'store_col',
        ezslide.resolve_manifest, pd.DataFrame({'path': ['x']}), 'store', ezslide.SLIDE_ID,
    )


def test_iter_slides_yields_every_slide():
    manifest = _cohort(3)
    seen = [(sid, wsi.properties.shape) for sid, wsi in ezslide.iter_slides(manifest)]
    assert len(seen) == 3
    assert len({sid for sid, _ in seen}) == 3


def test_iter_slides_is_reiterable():
    manifest = _cohort(2)
    first = [sid for sid, _ in ezslide.iter_slides(manifest)]
    second = [sid for sid, _ in ezslide.iter_slides(manifest)]
    assert first == second


def test_iter_slides_attach_images_controls_pixels():
    manifest = _cohort(1)
    for _, wsi in ezslide.iter_slides(manifest, attach_images=False):
        assert 'wsi' not in wsi.images
    for _, wsi in ezslide.iter_slides(manifest, attach_images=True):
        assert 'wsi' in wsi.images


def test_open_slides_returns_a_mapping_with_pixels():
    manifest = _cohort(2)
    slides = ezslide.open_slides(manifest)
    try:
        assert len(slides) == 2
        assert all('wsi' in w.images for w in slides.values())
    finally:
        for w in slides.values():
            w.close()


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
