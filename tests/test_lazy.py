"""Import cost is a feature, so it gets a test.

``import ezslide`` used to cost about 2.8 seconds, nearly all of it wsidata
pulling spatialdata, anndata, geopandas, shapely, cv2 and xarray in behind it,
paid by every process including the CLI and every plain TIFF read. The package
is lazy now, and laziness is easy to lose: one convenience import at the top of
one ``__init__.py`` puts it all back, and nothing else in the suite would
notice.

Every check here runs in a subprocess. Within one pytest session other tests
have already imported half of these modules, so ``sys.modules`` in-process says
nothing about what ``import ezslide`` is responsible for.

Runnable either way, like ``test_api.py``::

    pytest tests/
    python tests/test_lazy.py       # no pytest needed
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Third-party modules no import of ezslide should be responsible for.
HEAVY = ('wsidata', 'spatialdata', 'anndata', 'zarr', 'dask.array',
         'tensorstore', 'geopandas', 'shapely', 'cv2', 'xarray', 'ome_zarr')


def _run(code):
    """Run ``code`` in a clean interpreter, returning its stdout."""
    done = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout


def _loaded(statement):
    """Which of ``HEAVY`` are in ``sys.modules`` after ``statement``."""
    return _run(f'import sys; {statement}; '
                f'print(" ".join(m for m in {HEAVY!r} if m in sys.modules))').split()


def test_bare_import_loads_nothing_heavy():
    assert _loaded('import ezslide') == []


def test_cli_import_loads_nothing_heavy():
    # The console script pays this on every invocation, including --help.
    assert _loaded('import ezslide.cli') == []


def test_open_slide_does_not_need_wsidata():
    # The read path and the wsidata path are separate tiers; reaching for one
    # must not drag in the other.
    assert 'wsidata' not in _loaded('from ezslide import open_slide')


def test_writer_does_not_need_wsidata_or_zarr():
    loaded = _loaded('from ezslide import write_ome_tiff')
    assert 'wsidata' not in loaded
    assert 'zarr' not in loaded


#: Every documented way of asking for the wsidata path.
TRIGGERS = (
    'import ezslide; ezslide.register_readers()',
    'import ezslide; ezslide.readers',
    'import ezslide; ezslide.pyramid_options',
    'import ezslide; ezslide.ZarrSlideReader',
    'import ezslide.readers',
)


def test_readers_register_on_demand():
    """Each documented way in registers all three readers, and only then."""
    for trigger in TRIGGERS:
        out = _run(
            'from wsidata.reader._reader_registry import READERS\n'
            'names = ("tifffile_zarr", "tifffile_zarr_pyramid", "vsi_zarr")\n'
            'import ezslide\n'
            'print(any(n in READERS for n in names))\n'
            f'{trigger}\n'
            'print(all(n in READERS for n in names))\n').split()
        assert out == ['False', 'True'], f'{trigger!r} -> {out}'


def test_public_api_resolves():
    """Every name __all__ advertises is actually reachable through __getattr__."""
    _run('import ezslide\n'
         'missing = [n for n in ezslide.__all__ if not hasattr(ezslide, n)]\n'
         'assert not missing, missing\n')


def test_unknown_attribute_still_raises():
    _run('import ezslide\n'
         'try:\n'
         '    ezslide.no_such_name\n'
         'except AttributeError:\n'
         '    pass\n'
         'else:\n'
         '    raise SystemExit("expected AttributeError")\n')


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
