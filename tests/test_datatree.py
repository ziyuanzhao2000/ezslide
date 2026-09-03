"""Tests for the multiplex-IF / mono patch to wsidata's ``to_datatree``.

Runnable either way::

    pytest tests/
    python tests/test_datatree.py    # no pytest needed

Fixtures are synthesized (see ``ezslide-test-api``'s ``_fixtures``), so
nothing here needs a real slide.
"""

import os
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezslide

ezslide.register_readers()

from wsidata.reader._reader_registry import READERS

from ezslide.readers.datatree import _channel_coords, patch_to_datatree, to_datatree

CHANNELS = ['DAPI', 'CD3', 'CD8', 'PANCK']


def _fixtures():
    """(rgb_path, label_path, cycif_path, sources) in a fresh temp dir.

    Mirrors ``tests/test_api.py``'s ``_fixtures`` exactly, kept standalone
    here so this file has no import-order dependency on that one.
    """
    d = tempfile.mkdtemp(prefix='ezslide-test-')
    rgb = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    label = np.random.randint(0, 20, (2048, 2048), dtype=np.uint8)
    cycif = np.random.randint(0, 4095, (4, 1024, 1024), dtype=np.uint16)

    paths = [os.path.join(d, n) for n in ('rgb.tif', 'label.tif', 'cycif.ome.tif')]
    tifffile.imwrite(paths[0], rgb, tile=(256, 256), photometric='rgb')
    tifffile.imwrite(paths[1], label)                      # flat, untiled
    tifffile.imwrite(paths[2], cycif, tile=(256, 256), ome=True,
                     metadata={'axes': 'CYX', 'Channel': {'Name': CHANNELS}})
    return (*paths, {'rgb': rgb, 'label': label, 'cycif': cycif})


# --------------------------------------------------------------------------
# get_region
# --------------------------------------------------------------------------

def test_get_region_handles_cyx_without_s_axis():
    _, _, cycif, src = _fixtures()
    reader = READERS.try_open(cycif, reader='tifffile_zarr')
    try:
        region = reader.get_region(0, 0, 64, 64, level=0)
        assert region.shape == (64, 64, 4)
        assert np.array_equal(region, np.moveaxis(src['cycif'][:, :64, :64], 0, -1))
    finally:
        reader.detach_reader()


def test_get_region_mono_stays_2d():
    _, label, _, src = _fixtures()
    reader = READERS.try_open(label, reader='tifffile_zarr')
    try:
        region = reader.get_region(0, 0, 64, 64, level=0)
        assert region.ndim == 2
        assert np.array_equal(region, src['label'][:64, :64])
    finally:
        reader.detach_reader()


def test_get_region_rgb_unchanged():
    rgb, _, _, src = _fixtures()
    reader = READERS.try_open(rgb, reader='tifffile_zarr')
    try:
        region = reader.get_region(0, 0, 64, 64, level=0)
        assert region.shape == (64, 64, 3)
        assert np.array_equal(region, src['rgb'][:64, :64])
    finally:
        reader.detach_reader()


# --------------------------------------------------------------------------
# _channel_coords
# --------------------------------------------------------------------------

def test_channel_coords_uses_matching_names():
    assert _channel_coords(4, CHANNELS, np.uint16) == CHANNELS


def test_channel_coords_falls_back_on_mismatched_names():
    assert _channel_coords(4, ['only', 'three', 'names'], np.uint16) == \
        ['c0', 'c1', 'c2', 'c3']


def test_channel_coords_three_channels_no_names_is_rgb():
    assert _channel_coords(3, None, np.uint8) == ['r', 'g', 'b']


def test_channel_coords_mono():
    assert _channel_coords(1, None, np.uint8) == ['c0']
    assert _channel_coords(1, ['DAPI'], np.uint8) == ['DAPI']


# --------------------------------------------------------------------------
# patch_to_datatree
# --------------------------------------------------------------------------

def test_patch_to_datatree_patches_both_bindings():
    import wsidata.io._wsi as wsi_module
    import wsidata.reader as reader_module

    patch_to_datatree()
    assert wsi_module.to_datatree is to_datatree
    assert reader_module.to_datatree is to_datatree


# --------------------------------------------------------------------------
# end-to-end through open_wsi
# --------------------------------------------------------------------------

def test_open_wsi_attaches_multiplex_image_with_correct_channels():
    _, _, cycif, src = _fixtures()
    patch_to_datatree()
    from wsidata import open_wsi

    wsi = open_wsi(cycif, reader='tifffile_zarr', attach_images=True, store=None)
    scale0 = wsi.images['wsi']['scale0']['image']
    assert scale0.shape == (4, 1024, 1024)
    assert list(scale0.coords['c'].values) == CHANNELS
    assert scale0.dtype == np.uint16
    np.testing.assert_array_equal(
        scale0.transpose('y', 'x', 'c').values, np.moveaxis(src['cycif'], 0, -1))


def test_open_wsi_attaches_mono_image():
    _, label, _, src = _fixtures()
    patch_to_datatree()
    from wsidata import open_wsi

    wsi = open_wsi(label, reader='tifffile_zarr', attach_images=True, store=None)
    scale0 = wsi.images['wsi']['scale0']['image']
    assert scale0.shape == (1, 2048, 2048)
    assert list(scale0.coords['c'].values) == ['c0']
    assert scale0.dtype == np.uint8
    np.testing.assert_array_equal(scale0.values[0], src['label'])


def test_open_wsi_attaches_rgb_image_unchanged():
    rgb, _, _, src = _fixtures()
    patch_to_datatree()
    from wsidata import open_wsi

    wsi = open_wsi(rgb, reader='tifffile_zarr', attach_images=True, store=None)
    scale0 = wsi.images['wsi']['scale0']['image']
    assert scale0.shape == (3, 2048, 2048)
    assert list(scale0.coords['c'].values) == ['r', 'g', 'b']
    assert scale0.dtype == np.uint8
    np.testing.assert_array_equal(scale0.transpose('y', 'x', 'c').values, src['rgb'])


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
