"""Tests for the format-neutral API: open_slide, multiscale, channel views, calibration.

Runnable either way::

    pytest tests/
    python tests/test_api.py        # no pytest needed

Fixtures are synthesized, so nothing here needs a real slide. The VSI paths
(channel groups spread across series) still need a cellSens dataset; see the
note on ``test_channel_groups_single_series``.
"""

import os
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezslide
from ezslide import (ChannelView, InterleavedView, channel_groups,
                     open_slide)
from ezslide.array.channel import channel_axis_of, n_channels
from ezslide.formats.open import slide_class

CHANNELS = ['DAPI', 'CD3', 'CD8', 'PANCK']


def _fixtures():
    """(rgb_path, label_path, cycif_path, sources) in a fresh temp dir."""
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


def _pyramidal(path, how='mean'):
    return open_slide(path, pyramidalize=True,
                      pyramid={'how': how, 'levels': 8, 'cache': 'tmp'})


# --------------------------------------------------------------------------
# open_slide
# --------------------------------------------------------------------------

def test_dispatch_by_suffix():
    assert slide_class('a.svs') is ezslide.TiffFile
    assert slide_class('a.ome.tif') is ezslide.TiffFile
    assert slide_class('a.vsi') is ezslide.VsiFile
    assert slide_class('a.VSI') is ezslide.VsiFile      # case-insensitive
    assert slide_class('a.ets') is ezslide.VsiFile


def test_open_slide_is_a_context_manager():
    rgb, _, _, _ = _fixtures()
    with open_slide(rgb) as slide:
        assert slide[0].level_shapes[0] == (2048, 2048)


# --------------------------------------------------------------------------
# multiscale
# --------------------------------------------------------------------------

def test_multiscale_is_always_a_list():
    rgb, label, _, _ = _fixtures()
    # A flat file with no pyramid asked for still comes back as a list of one,
    # which is the whole point: callers never branch on level count.
    assert isinstance(open_slide(label)[0].multiscale(), list)
    assert len(open_slide(label)[0].multiscale()) == 1
    assert len(_pyramidal(rgb)[0].multiscale()) > 1


def test_multiscale_levels_halve():
    rgb, _, _, _ = _fixtures()
    shapes = [lv.shape[:2] for lv in _pyramidal(rgb)[0].multiscale()]
    assert shapes[0] == (2048, 2048)
    for big, small in zip(shapes, shapes[1:]):
        assert small == (big[0] // 2, big[1] // 2)


def test_multiscale_eagerfies():
    """A viewer indexes a level and must get an ndarray, not a lazy view."""
    rgb, _, _, _ = _fixtures()
    for level in _pyramidal(rgb)[0].multiscale():
        assert isinstance(level[0:32, 0:32], np.ndarray)


def test_multiscale_eager_false_stays_lazy():
    rgb, _, _, _ = _fixtures()
    level = open_slide(rgb)[0].multiscale(eager=False)[0]
    assert not isinstance(level[0:32, 0:32], np.ndarray)


# --------------------------------------------------------------------------
# calibration and geometry
# --------------------------------------------------------------------------

def test_level_shapes_and_downsamples():
    rgb, _, _, _ = _fixtures()
    series = _pyramidal(rgb)[0]
    assert series.level_shapes[0] == (2048, 2048)
    assert series.downsamples[:3] == [1.0, 2.0, 4.0]
    assert len(series.level_shapes) == len(series.downsamples) == len(series.levels)


def test_pixel_size_absent_is_none():
    _, label, _, _ = _fixtures()
    assert open_slide(label)[0].pixel_size is None


def test_pixel_size_read_from_ome():
    d = tempfile.mkdtemp(prefix='ezslide-test-')
    path = os.path.join(d, 'cal.ome.tif')
    tifffile.imwrite(path, np.zeros((512, 512), np.uint8), ome=True,
                     resolution=(1 / 0.325, 1 / 0.325),
                     resolutionunit='MICROMETER')
    size = open_slide(path)[0].pixel_size
    assert size is not None
    assert abs(size[0] - 0.325) < 1e-6 and abs(size[1] - 0.325) < 1e-6


# --------------------------------------------------------------------------
# channels
# --------------------------------------------------------------------------

def test_channel_names_from_ome():
    _, _, cycif, _ = _fixtures()
    assert open_slide(cycif)[0].channel_names == CHANNELS


def test_channel_names_absent_is_none():
    _, label, _, _ = _fixtures()
    assert open_slide(label)[0].channel_names is None


def test_channel_axis_resolution():
    assert channel_axis_of('CYX') == 0
    assert channel_axis_of('YXS') == 2
    assert channel_axis_of('YX') is None
    # infer_axes never emits a C, so a guessed axis has to count as one.
    assert channel_axis_of('?YX') == 0
    assert channel_axis_of('CYXS') == 0        # C wins over S
    assert channel_axis_of('YXS', prefer=0) == 0


def test_n_channels():
    _, _, cycif, _ = _fixtures()
    level = open_slide(cycif)[0].multiscale()[0]
    assert n_channels(level, 'CYX') == 4
    assert n_channels(np.zeros((8, 9)), 'YX') == 1


def test_channel_view_is_pixel_exact_on_every_level_kind():
    _, _, cycif, src = _fixtures()
    levels = _pyramidal(cycif)[0].multiscale()
    assert len(levels) > 1, 'need a synthesized level to compare against'
    for level in levels:
        view = ChannelView(level, 'CYX', 2)
        assert view.ndim == 2 and view.shape == level.shape[1:]
        tile = view[10:74, 20:84]
        assert isinstance(tile, np.ndarray)
        assert np.array_equal(tile, np.asarray(level[2, 10:74, 20:84]))
    # ...and the file level really does match the original pixels.
    assert np.array_equal(ChannelView(levels[0], 'CYX', 2)[0:64, 0:64],
                          src['cycif'][2, 0:64, 0:64])


def test_channel_view_indexing_matches_numpy():
    _, _, cycif, src = _fixtures()
    level = open_slide(cycif)[0].multiscale()[0]
    view, want = ChannelView(level, 'CYX', 1), src['cycif'][1]
    for key in [(slice(0, 32), slice(0, 32)), Ellipsis, 5,
                (slice(None), 7), (3, slice(10, 20))]:
        assert np.array_equal(view[key], want[key]), key


def test_channel_view_passthrough_without_a_channel_axis():
    plane = np.arange(72, dtype=np.uint8).reshape(8, 9)
    view = ChannelView(plane, 'YX', 0)
    assert view.shape == (8, 9)
    assert np.array_equal(np.asarray(view), plane)


def test_channel_view_rejects_a_channel_that_is_not_there():
    _, _, cycif, _ = _fixtures()
    level = open_slide(cycif)[0].multiscale()[0]
    for data, axes, ch in [(level, 'CYX', 9), (np.zeros((8, 9)), 'YX', 1)]:
        try:
            ChannelView(data, axes, ch)
        except IndexError:
            continue
        raise AssertionError(f'expected IndexError for channel {ch}')


def test_channel_view_array_protocol():
    _, _, cycif, src = _fixtures()
    level = open_slide(cycif)[0].multiscale()[0]
    view = ChannelView(level, 'CYX', 0)
    assert np.array_equal(np.asarray(view), src['cycif'][0])
    assert np.array(view, copy=False).shape == (1024, 1024)     # NumPy 2
    assert np.array(view, copy=True).shape == (1024, 1024)
    assert np.asarray(view, dtype=np.float32).dtype == np.float32


def test_interleaved_view_is_pixel_exact_on_every_level_kind():
    _, _, cycif, src = _fixtures()
    levels = _pyramidal(cycif)[0].multiscale()
    assert len(levels) > 1, 'need a synthesized level to compare against'
    for level in levels:
        view = InterleavedView(level, 'CYX')
        assert view.ndim == 3
        assert view.shape == (*level.shape[1:], level.shape[0])
        tile = view[10:74, 20:84]
        assert isinstance(tile, np.ndarray)
        assert np.array_equal(
            tile, np.moveaxis(np.asarray(level[:, 10:74, 20:84]), 0, -1))
    # ...and the file level really does match the original pixels.
    assert np.array_equal(InterleavedView(levels[0], 'CYX')[0:64, 0:64],
                          np.moveaxis(src['cycif'][:, 0:64, 0:64], 0, -1))


def test_interleaved_view_indexing_matches_numpy():
    _, _, cycif, src = _fixtures()
    level = open_slide(cycif)[0].multiscale()[0]
    view, want = InterleavedView(level, 'CYX'), np.moveaxis(src['cycif'], 0, -1)
    for key in [(slice(0, 32), slice(0, 32)), Ellipsis, 5,
                (slice(None), 7), (3, slice(10, 20)),
                (slice(0, 8), slice(0, 8), 2), (Ellipsis, 0)]:
        assert np.array_equal(view[key], want[key]), key


def test_interleaved_view_leaves_an_interleaved_level_alone():
    """``YXS`` data is already in the layout, so the view is a passthrough."""
    rgb = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    view = InterleavedView(rgb, 'YXS')
    assert view.shape == (8, 9, 3)
    assert np.array_equal(np.asarray(view), rgb)
    assert np.array_equal(view[2:5, 1:4], rgb[2:5, 1:4])


def test_interleaved_view_rejects_a_level_without_channels():
    try:
        InterleavedView(np.zeros((8, 9)), 'YX')
    except ValueError:
        return
    raise AssertionError('expected ValueError for a level with no channel axis')


def test_interleaved_view_array_protocol():
    _, _, cycif, src = _fixtures()
    level = open_slide(cycif)[0].multiscale()[0]
    view = InterleavedView(level, 'CYX')
    assert np.array_equal(np.asarray(view), np.moveaxis(src['cycif'], 0, -1))
    assert np.array(view, copy=False).shape == (1024, 1024, 4)   # NumPy 2
    assert np.array(view, copy=True).shape == (1024, 1024, 4)
    assert np.asarray(view, dtype=np.float32).dtype == np.float32


def test_channel_groups_single_series():
    """A format that keeps channels in one series gives one group per series.

    The interesting case — cellSens, one series per channel — needs a real
    ``.vsi``; what is asserted here is the invariant every caller relies on,
    that a group is always a list even when there is nothing to group.
    """
    _, _, cycif, _ = _fixtures()
    slide = open_slide(cycif)
    groups = channel_groups(slide)
    assert groups == [[slide[0]]]
    assert channel_groups(list(slide.series)) == groups     # accepts a sequence


# --------------------------------------------------------------------------
# lazy levels: NumPy 2 and strided indexing
# --------------------------------------------------------------------------

def test_lazy_level_array_protocol_numpy2():
    rgb, _, _, _ = _fixtures()
    level = _pyramidal(rgb)[0].multiscale()[1]
    assert type(level).__name__ in ('LazyLevel', 'CachedLevel')
    assert np.asarray(level).shape == level.shape
    assert np.array(level, copy=False).shape == level.shape   # used to raise
    assert np.array(level, copy=True).shape == level.shape
    assert np.asarray(level, dtype=np.float32).dtype == np.float32


def test_lazy_level_strided_slices_match_numpy():
    _, label, _, _ = _fixtures()
    level = _pyramidal(label)[0].multiscale()[1]
    full = np.asarray(level)
    for key in [(slice(None, None, 4), slice(None, None, 4)),
                (slice(10, 200, 3), slice(5, 180, 7)),
                (slice(None, None, 2), 17),
                (Ellipsis, slice(None, None, 8)),
                (slice(None), slice(None))]:
        got = level[key]
        assert got.shape == full[key].shape, key
        assert np.array_equal(got, full[key]), key


def test_lazy_level_rejects_negative_steps():
    _, label, _, _ = _fixtures()
    level = _pyramidal(label)[0].multiscale()[1]
    try:
        level[::-1]
    except NotImplementedError:
        return
    raise AssertionError('expected NotImplementedError for a negative step')


def test_mode_downsampling_keeps_real_labels():
    """Averaging a label map invents classes; 'mode' must not."""
    _, label, _, src = _fixtures()
    real = set(np.unique(src['label']).tolist())
    mode = np.asarray(_pyramidal(label, how='mode')[0].multiscale()[1][:])
    mean = np.asarray(_pyramidal(label, how='mean')[0].multiscale()[1][:])
    assert set(np.unique(mode).tolist()) <= real
    assert not np.array_equal(mode, mean)


# --------------------------------------------------------------------------
# the tmp pyramid cache
# --------------------------------------------------------------------------

def test_tmp_cache_directory_is_removed():
    import gc
    import glob

    pattern = os.path.join(tempfile.gettempdir(), 'zpyr-*')
    _, label, _, _ = _fixtures()
    before = len(glob.glob(pattern))

    slide = _pyramidal(label)
    _ = np.asarray(slide[0].multiscale()[1][0:32, 0:32])     # force a cache fill
    assert len(glob.glob(pattern)) == before + 1

    slide.close()
    del slide, _
    gc.collect()
    assert len(glob.glob(pattern)) == before


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
