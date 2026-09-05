"""Tests for ``ezslide.open_wsi`` / ``ezslide.read_wsi`` (wsi_source round-trip).

Runnable either way::

    pytest tests/
    python tests/test_io.py    # no pytest needed

Fixtures are synthesized (see ``ezslide-test-api``'s ``_fixtures``), so
nothing here needs a real slide.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ezslide

ezslide.register_readers()


def _assert_raises(substr, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as exc:
        assert substr in str(exc), f'{substr!r} not in {exc!r}'
    else:
        raise AssertionError(f'expected ValueError containing {substr!r}')


def _fixture():
    """(rgb_path, store_path) in a fresh temp dir."""
    d = tempfile.mkdtemp(prefix='ezslide-test-')
    rgb = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    path = os.path.join(d, 'rgb.tif')
    tifffile.imwrite(path, rgb, tile=(256, 256), photometric='rgb')
    store = os.path.join(d, 'rgb.zarr')
    return path, store


# --------------------------------------------------------------------------
# open_wsi records wsi_source
# --------------------------------------------------------------------------

def test_open_wsi_records_wsi_source():
    path, store = _fixture()
    wsi = ezslide.open_wsi(path, store=store, reader='tifffile_zarr')
    assert wsi.attrs['wsi_source'] == {
        'path': os.path.realpath(path),
        'reader': 'tifffile_zarr',
    }


def test_open_wsi_does_not_overwrite_existing_wsi_source():
    path, store = _fixture()
    wsi = ezslide.open_wsi(path, store=store, reader='tifffile_zarr')
    wsi.write(store)
    wsi.close()

    reopened = ezslide.open_wsi(path, store=store, reader='tifffile_zarr')
    assert reopened.attrs['wsi_source']['path'] == os.path.realpath(path)


def test_open_wsi_registers_readers_on_demand_with_explicit_reader():
    # In a fresh interpreter that never called register_readers(), passing
    # reader='tifffile_zarr' explicitly must still work.
    path, _ = _fixture()
    code = (
        'import ezslide\n'
        'from wsidata.reader._reader_registry import READERS\n'
        'print("tifffile_zarr" in READERS)\n'
        f'wsi = ezslide.open_wsi({path!r}, reader="tifffile_zarr", store=None)\n'
        'print("tifffile_zarr" in READERS)\n'
        'print(wsi.reader.name)\n'
    )
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ['False', 'True', 'tifffile_zarr']


def test_open_wsi_registers_readers_on_demand_with_auto_detect():
    # Same, but with reader=None (extension-based auto-detect) — ezslide's
    # readers must be in the registry *before* wsidata picks one by
    # extension, or vsi/tifffile-family slides would never be found.
    path, _ = _fixture()
    code = (
        'import ezslide\n'
        'from wsidata.reader._reader_registry import READERS\n'
        'print("tifffile_zarr" in READERS)\n'
        f'ezslide.open_wsi({path!r}, store=None)\n'
        'print("tifffile_zarr" in READERS)\n'
    )
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ['False', 'True']


# --------------------------------------------------------------------------
# read_wsi reconstructs from the store alone
# --------------------------------------------------------------------------

def test_read_wsi_round_trips():
    path, store = _fixture()
    wsi = ezslide.open_wsi(path, store=store, reader='tifffile_zarr')
    wsi.write(store)
    wsi.close()

    reloaded = ezslide.read_wsi(store)
    assert reloaded.properties.shape == wsi.properties.shape
    assert reloaded.reader.name == 'tifffile_zarr'


def test_read_wsi_raises_without_wsi_source():
    from wsidata import open_wsi as wsidata_open_wsi

    path, store = _fixture()
    wsi = wsidata_open_wsi(path, store=store, reader='tifffile_zarr')
    wsi.write(store)
    wsi.close()

    _assert_raises('wsi_source', ezslide.read_wsi, store)


def test_read_wsi_registers_readers_on_demand():
    # In a fresh interpreter that never called register_readers(), a store
    # written with an ezslide reader (tifffile_zarr) must still round-trip:
    # read_wsi has to register it before handing the reader name to open_wsi.
    path, store = _fixture()
    wsi = ezslide.open_wsi(path, store=store, reader='tifffile_zarr')
    wsi.write(store)
    wsi.close()

    code = (
        'import ezslide\n'
        'from wsidata.reader._reader_registry import READERS\n'
        'print("tifffile_zarr" in READERS)\n'
        f'wsi = ezslide.read_wsi({store!r})\n'
        'print("tifffile_zarr" in READERS)\n'
        'print(wsi.reader.name)\n'
    )
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ['False', 'True', 'tifffile_zarr']


def test_read_wsi_raises_for_missing_store():
    _assert_raises('does not exist', ezslide.read_wsi, '/no/such/store.zarr')


def test_read_wsi_falls_back_to_path_next_to_store():
    path, store = _fixture()
    wsi = ezslide.open_wsi(path, store=store, reader='tifffile_zarr')
    wsi.write(store)
    wsi.close()

    moved = os.path.join(os.path.dirname(store), os.path.basename(path))
    if moved != path:
        shutil.move(path, moved)

    reloaded = ezslide.read_wsi(store)
    assert reloaded.properties.shape == wsi.properties.shape


def test_read_wsi_raises_when_slide_unresolvable():
    path, store = _fixture()
    wsi = ezslide.open_wsi(path, store=store, reader='tifffile_zarr')
    wsi.write(store)
    wsi.close()

    os.remove(path)

    _assert_raises('no longer', ezslide.read_wsi, store)


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
