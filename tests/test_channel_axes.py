"""Tests for channel-first and unlabeled (``Q``) channel axes.

Covers region slicing, the reader, PatchDataset/PatchBlockDataset, the
OME-TIFF writer and ``ezslide convert --channel-names``.

Runnable either way::

    pytest tests/
    python tests/test_channel_axes.py    # no pytest needed

Fixtures are synthesized, so nothing here needs a real slide.
"""

import contextlib
import io
import os
import re
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezslide
from ezslide.array.channel import (channel_axis_of, level_region_index,
                                   n_planar_channels, slice_level_region,
                                   to_channel_last)
from ezslide.cli import main as cli_main
from ezslide.formats.open import open_wsi

CHANNELS = ['DAPI', 'CD3', 'CD8', 'PANCK']


def _fixtures():
    """(qyx_path, cyx_path, iyx_path, data) in a fresh temp dir.

    ``qyx``: tifffile "shaped" metadata, read back as ``QYX``.
    ``cyx``: OME-TIFF with an explicit ``CYX`` axes string.
    ``iyx``: plain multi-page TIFF with no metadata, read back as ``IYX``.
    """
    d = tempfile.mkdtemp(prefix='ezslide-test-')
    data = np.random.randint(0, 4095, (4, 1024, 1024), dtype=np.uint16)
    paths = [os.path.join(d, n) for n in ('qyx.tif', 'cyx.ome.tif', 'iyx.tif')]
    tifffile.imwrite(paths[0], data, tile=(256, 256), photometric='minisblack')
    tifffile.imwrite(paths[1], data, tile=(256, 256), ome=True,
                     metadata={'axes': 'CYX', 'Channel': {'Name': CHANNELS}})
    tifffile.imwrite(paths[2], data, tile=(256, 256), photometric='minisblack',
                     metadata=None)
    return (*paths, data)


def _ome_summary(path):
    """(axes, SizeC, SizeT, channel names, pixels) of an OME-TIFF."""
    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata
        return (tif.series[0].axes,
                int(re.search(r'SizeC="(\d+)"', xml).group(1)),
                int(re.search(r'SizeT="(\d+)"', xml).group(1)),
                re.findall(r'<Channel [^>]*Name="([^"]+)"', xml),
                tif.series[0].asarray())


def _convert(*argv):
    """Run ``ezslide convert``; return (exit code, stdout + stderr)."""
    out = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            cli_main(['convert', *argv, '--overwrite'])
        except SystemExit as exc:
            code = exc.code or 0
    return code, out.getvalue()


# --------------------------------------------------------------------------
# axis resolution
# --------------------------------------------------------------------------

def test_fixture_axes_match_tifffile_labels():
    qyx, cyx, iyx, _ = _fixtures()
    for path, axes in [(qyx, 'QYX'), (cyx, 'CYX'), (iyx, 'IYX')]:
        with tifffile.TiffFile(path) as tif:
            assert tif.series[0].axes == axes, (path, tif.series[0].axes)


def test_channel_axis_of_q_and_i():
    assert channel_axis_of('QYX') == 0
    assert channel_axis_of('YXQ') == 2
    assert channel_axis_of('QYXS') == 3        # S wins over Q
    assert channel_axis_of('IYX') is None


def test_n_planar_channels():
    data = np.zeros((4, 8, 9))
    assert n_planar_channels(data, 'CYX') == 4
    assert n_planar_channels(data, 'QYX') == 4
    assert n_planar_channels(data, '?YX') == 4
    assert n_planar_channels(np.zeros((8, 9, 3)), 'YXS') == 1
    assert n_planar_channels(np.zeros((3, 8, 9)), 'SYX') == 1
    assert n_planar_channels(np.zeros((8, 9)), 'YX') == 1


# --------------------------------------------------------------------------
# region slicing
# --------------------------------------------------------------------------

def test_level_region_index_and_to_channel_last():
    _, cyx, _, data = _fixtures()
    lv = open_wsi(cyx)[0].levels[0]
    idx = level_region_index(lv, 200, 32, 100, 64)
    assert idx == (slice(None), slice(200, 232), slice(100, 164))
    region = to_channel_last(lv, data[idx])
    assert region.shape == (32, 64, 4)


def test_slice_level_region_channel_first():
    qyx, cyx, _, data = _fixtures()
    want = np.moveaxis(data[:, 200:232, 100:164], 0, -1)
    for path in (qyx, cyx):
        lv = open_wsi(path)[0].levels[0]
        np.testing.assert_array_equal(slice_level_region(lv, 200, 32, 100, 64), want)


def test_reader_get_region_on_qyx():
    ezslide.register_readers()
    from wsidata.reader._reader_registry import READERS

    qyx, _, _, data = _fixtures()
    reader = READERS.try_open(qyx, reader='tifffile_zarr')
    try:
        region = reader.get_region(100, 200, 64, 32, level=0)
        np.testing.assert_array_equal(
            region, np.moveaxis(data[:, 200:232, 100:164], 0, -1))
    finally:
        reader.detach_reader()


# --------------------------------------------------------------------------
# PatchDataset / PatchBlockDataset
# --------------------------------------------------------------------------

def test_patch_datasets_on_cyx_are_exact():
    try:
        import lazyslide as zs
        from ezslide.dataset.patch import PatchBlockDataset, PatchDataset
    except ImportError:
        print('  SKIP  torch or lazyslide unavailable')
        return
    ezslide.register_readers()
    from wsidata import open_wsi as wsidata_open_wsi

    _, cyx, _, data = _fixtures()
    wsi = wsidata_open_wsi(cyx, reader='tifffile_zarr', store=None)
    zs.pp.tile_tissues(wsi, 256, background_filter=False, tissue_key=None)

    for cls in (PatchDataset, PatchBlockDataset):
        for async_batch in (False, True):
            ds = cls(wsi, key='tiles', async_batch=async_batch)
            assert len(ds) == 16
            items = ds.__getitems__(list(range(len(ds)))) + [ds[5]]
            for item in items:
                x, y = item['x'], item['y']
                np.testing.assert_array_equal(
                    np.asarray(item['image']),
                    np.moveaxis(data[:, y:y + 256, x:x + 256], 0, -1),
                    err_msg=f'{cls.__name__} async_batch={async_batch}')


# --------------------------------------------------------------------------
# OME-TIFF writer and CLI
# --------------------------------------------------------------------------

def test_convert_qyx_writes_channels_not_timepoints():
    qyx, _, _, data = _fixtures()
    out = os.path.join(os.path.dirname(qyx), 'qyx_plain.ome.tif')
    code, log = _convert(qyx, out)
    assert code == 0, log
    assert '1 -> 4 channel(s)' in log
    axes, size_c, size_t, names, pixels = _ome_summary(out)
    assert (axes, size_c, size_t, names) == ('CYX', 4, 1, [])
    np.testing.assert_array_equal(pixels, data)


def test_convert_names_channels_of_single_multichannel_input():
    qyx, _, _, data = _fixtures()
    out = os.path.join(os.path.dirname(qyx), 'qyx_named.ome.tif')
    code, log = _convert(qyx, out, '--channel-names', *CHANNELS)
    assert code == 0, log
    assert '1 -> 4 channel(s)' in log
    axes, size_c, size_t, names, pixels = _ome_summary(out)
    assert (axes, size_c, size_t, names) == ('CYX', 4, 1, CHANNELS)
    np.testing.assert_array_equal(pixels, data)


def test_convert_keeps_names_of_cyx_input():
    _, cyx, _, data = _fixtures()
    out = os.path.join(os.path.dirname(cyx), 'cyx_copy.ome.tif')
    code, log = _convert(cyx, out)
    assert code == 0, log
    axes, size_c, _, names, pixels = _ome_summary(out)
    assert (axes, size_c, names) == ('CYX', 4, CHANNELS)
    np.testing.assert_array_equal(pixels, data)


def test_convert_rejects_wrong_channel_name_count():
    qyx, _, _, _ = _fixtures()
    out = os.path.join(os.path.dirname(qyx), 'bad.ome.tif')
    code, log = _convert(qyx, out, '--channel-names', 'a', 'b')
    assert code == 1
    assert '--channel-names has 2 entries but qyx has 4 channel(s)' in log


def test_convert_merges_separate_files_with_names():
    qyx, _, _, data = _fixtures()
    d = os.path.dirname(qyx)
    planes = [os.path.join(d, f'ch{i}.tif') for i in range(2)]
    for path, plane in zip(planes, data):
        tifffile.imwrite(path, plane)
    out = os.path.join(d, 'merged.ome.tif')
    code, log = _convert(*planes, out, '--channel-names', 'A', 'B')
    assert code == 0, log
    assert '2 -> 2 channel(s)' in log
    axes, size_c, _, names, pixels = _ome_summary(out)
    assert (axes, size_c, names) == ('CYX', 2, ['A', 'B'])
    np.testing.assert_array_equal(pixels, data[:2])


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
