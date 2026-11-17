import pytest

import fsspec
import numpy as np
import ujson
import zarr

import kerchunk.combine
import kerchunk.zarr
import kerchunk.df
from kerchunk.utils import fs_as_store, refs_as_store


@pytest.mark.parametrize(
    "arrays,chunks,axis",
    [
        (
            [np.arange(10), np.arange(10, 20)],
            (2,),
            0,
        ),
        (
            [np.arange(12), np.arange(12, 36), np.arange(36, 42)],
            (6,),
            0,
        ),
        (
            # Terminal chunk does not need to be filled
            [np.arange(5), np.arange(5, 10), np.arange(10, 17)],
            (5,),
            0,
        ),
        (
            [
                np.broadcast_to(np.arange(6), (10, 6)),
                np.broadcast_to(np.arange(7, 10), (10, 3)),
            ],
            (10, 3),
            1,
        ),
        (
            [
                np.broadcast_to(np.arange(6), (10, 6)).T,
                np.broadcast_to(np.arange(7, 10), (10, 3)).T,
            ],
            (3, 10),
            0,
        ),
    ],
)
def test_success(tmpdir, arrays, chunks, axis, m):
    fns = []
    refs = []
    for i, x in enumerate(arrays):
        fn = f"{tmpdir}/out{i}.zarr"
        g = zarr.open(fn, zarr_format=2)
        arr = g.create_array("x", shape=x.shape, dtype=x.dtype, chunks=chunks)
        arr[:] = x
        fns.append(fn)
        ref = kerchunk.zarr.single_zarr(fn, inline=0)
        refs.append(ref)

    out = kerchunk.combine.concatenate_arrays(
        refs, axis=axis, path="x", check_arrays=True
    )

    store = refs_as_store(out)
    g = zarr.open(store, zarr_format=2)
    assert (g["x"][:] == np.concatenate(arrays, axis=axis)).all()

    try:
        import fastparquet
    except ImportError:
        return
    kerchunk.df.refs_to_dataframe(out, "memory://out.parq")
    storage_options = dict(
        fo="memory://out.parq",
        remote_protocol="file",
        skip_instance_cache=True,
    )
    g = zarr.open("reference://", zarr_format=2, storage_options=storage_options)
    assert (g["x"][:] == np.concatenate(arrays, axis=axis)).all()

    kerchunk.df.refs_to_dataframe(out, "memory://out.parq", record_size=1)
    storage_options = dict(
        fo="memory://out.parq",
        remote_protocol="file",
        skip_instance_cache=True,
    )
    g = zarr.open("reference://", zarr_format=2, storage_options=storage_options)
    assert (g["x"][:] == np.concatenate(arrays, axis=axis)).all()


def test_fail_chunks(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10)
    x2 = np.arange(10, 20)
    g = zarr.open(fn1, zarr_format=2)
    arr = g.create_array("x", shape=x1.shape, dtype=x1.dtype, chunks=(2,))
    arr[:] = x1
    g = zarr.open(fn2, zarr_format=2)
    arr = g.create_array("x", shape=x2.shape, dtype=x2.dtype, chunks=(3,))
    arr[:] = x2

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Incompatible array chunks at index 1.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


def test_fail_shape(tmpdir):
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(12).reshape(6, 2)
    x2 = np.arange(12, 24)
    g = zarr.open(fn1, zarr_format=2)
    arr = g.create_array("x", shape=x1.shape, dtype=x1.dtype)
    arr[:] = x1
    g = zarr.open(fn2, zarr_format=2)
    arr = g.create_array("x", shape=x2.shape, dtype=x2.dtype)
    arr[:] = x2

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    with pytest.raises(ValueError, match=r"Incompatible array shape at index 1.*"):
        kerchunk.combine.concatenate_arrays([ref1, ref2], path="x", check_arrays=True)


def test_irregular_chunk_boundaries(tmpdir):
    """Inputs with non whole-chunk lengths concatenate via seam rewrites"""
    fn1 = f"{tmpdir}/out1.zarr"
    fn2 = f"{tmpdir}/out2.zarr"
    x1 = np.arange(10)
    x2 = np.arange(10, 24)
    g = zarr.open(fn1, zarr_format=2)
    arr = g.create_array("x", shape=x1.shape, dtype=x1.dtype, chunks=(4,))
    arr[:] = x1
    g = zarr.open(fn2, zarr_format=2)
    arr = g.create_array("x", shape=x2.shape, dtype=x2.dtype, chunks=(4,))
    arr[:] = x2

    ref1 = kerchunk.zarr.single_zarr(fn1, inline=0)
    ref2 = kerchunk.zarr.single_zarr(fn2, inline=0)

    out = kerchunk.combine.concatenate_arrays(
        [ref1, ref2], path="x", check_arrays=True
    )

    truth = np.concatenate([x1, x2])
    g = zarr.open(refs_as_store(out), zarr_format=2)
    assert g["x"].shape == truth.shape
    assert (g["x"][:] == truth).all()
    # the seam sits inside output chunks 2 and 3, which must be rewritten;
    # the fully aligned chunks 0 and 1 keep pointing at the source file
    assert set(out["refs"]["x/0"]) == set(ref1["refs"]["x/0"])
    assert set(out["refs"]["x/1"]) == set(ref1["refs"]["x/1"])
    for key in ["x/2", "x/3", "x/4", "x/5"]:
        # seam chunks carry decoded data rather than source references
        assert not isinstance(out["refs"][key], list)


def test_several_irregular_boundaries(tmpdir):
    """Continuous misaligned seams must not accumulate offset errors"""
    arrays = [np.arange(10), np.arange(10, 23), np.arange(23, 30)]
    refs = []
    for i, x in enumerate(arrays):
        fn = f"{tmpdir}/multi{i}.zarr"
        g = zarr.open(fn, zarr_format=2)
        arr = g.create_array("x", shape=x.shape, dtype=x.dtype, chunks=(4,))
        arr[:] = x
        refs.append(kerchunk.zarr.single_zarr(fn, inline=0))

    out = kerchunk.combine.concatenate_arrays(refs, path="x", check_arrays=True)
    truth = np.concatenate(arrays)
    g = zarr.open(refs_as_store(out), zarr_format=2)
    assert g["x"].shape == truth.shape
    assert (g["x"][:] == truth).all()
    # slices crossing every seam (at 10 and 23) and every chunk boundary
    for seam in (10, 23):
        assert (g["x"][seam - 3: seam + 3] == truth[seam - 3: seam + 3]).all()


def test_misaligned_seam_2d(tmpdir):
    """Seam handling on a non-leading axis, including the fill-only tail"""
    x1 = np.broadcast_to(np.arange(6), (10, 6)).copy()
    x2 = np.broadcast_to(np.arange(100, 105), (10, 5)).copy()
    refs = []
    for i, x in enumerate([x1, x2]):
        fn = f"{tmpdir}/two_d{i}.zarr"
        g = zarr.open(fn, zarr_format=2)
        arr = g.create_array("x", shape=x.shape, dtype=x.dtype, chunks=(10, 4))
        arr[:] = x
        refs.append(kerchunk.zarr.single_zarr(fn, inline=0))

    out = kerchunk.combine.concatenate_arrays(
        refs, axis=1, path="x", check_arrays=True
    )
    truth = np.concatenate([x1, x2], axis=1)
    g = zarr.open(refs_as_store(out), zarr_format=2)
    assert g["x"].shape == truth.shape
    assert (g["x"][:] == truth).all()
    # the uniform chunk grid is preserved
    assert ujson.loads(out["refs"]["x/.zarray"])["chunks"] == [10, 4]


def test_misaligned_seam_preserves_codecs_and_fill(tmpdir):
    """Rewritten seam chunks use the original compressor/filters and fill value"""
    import numcodecs

    x1 = np.arange(10, dtype="float64")
    x2 = np.arange(10, 20, dtype="float64")
    refs = []
    for i, x in enumerate([x1, x2]):
        fn = f"{tmpdir}/codec{i}.zarr"
        g = zarr.open(fn, zarr_format=2)
        arr = g.create_array(
            "x",
            shape=x.shape,
            dtype=x.dtype,
            chunks=(4,),
            compressor=numcodecs.Blosc("zstd"),
            filters=[numcodecs.Delta(dtype="float64")],
            fill_value=-99.0,
        )
        arr[:] = x
        refs.append(kerchunk.zarr.single_zarr(fn, inline=0))

    out = kerchunk.combine.concatenate_arrays(refs, path="x")
    meta = ujson.loads(out["refs"]["x/.zarray"])
    assert meta["compressor"]["id"] == "blosc"
    assert meta["filters"][0]["id"] == "delta"
    assert meta["fill_value"] == -99.0

    g = zarr.open(refs_as_store(out), zarr_format=2)
    truth = np.concatenate([x1, x2])
    assert (g["x"][:] == truth).all()


def test_misaligned_seam_slash_separator(tmpdir):
    x1 = np.arange(10)
    x2 = np.arange(10, 20)
    refs = []
    for i, x in enumerate([x1, x2]):
        fn = f"{tmpdir}/slash{i}.zarr"
        g = zarr.open(fn, zarr_format=2)
        arr = g.create_array(
            "x",
            shape=x.shape,
            dtype=x.dtype,
            chunks=(4,),
            chunk_key_encoding={"name": "v2", "separator": "/"},
        )
        arr[:] = x
        refs.append(kerchunk.zarr.single_zarr(fn, inline=0))

    out = kerchunk.combine.concatenate_arrays(refs, path="x")
    truth = np.concatenate([x1, x2])
    g = zarr.open(refs_as_store(out), zarr_format=2)
    assert (g["x"][:] == truth).all()
    assert "x/2/0" not in out["refs"]  # would indicate dot/slash confusion
    assert "x/2" in out["refs"]


def test_misaligned_seam_multiple_chunks_perpendicular(tmpdir):
    """Seam chunks fan out over several chunk indices on the other axes"""
    x1 = np.arange(32).reshape(8, 4)
    x2 = np.arange(100, 108).reshape(2, 4)
    refs = []
    for i, x in enumerate([x1, x2]):
        fn = f"{tmpdir}/fan{i}.zarr"
        g = zarr.open(fn, zarr_format=2)
        arr = g.create_array("x", shape=x.shape, dtype=x.dtype, chunks=(3, 1))
        arr[:] = x
        refs.append(kerchunk.zarr.single_zarr(fn, inline=0))

    out = kerchunk.combine.concatenate_arrays(refs, axis=0, path="x")
    truth = np.concatenate([x1, x2], axis=0)
    g = zarr.open(refs_as_store(out), zarr_format=2)
    assert g["x"].shape == truth.shape
    assert (g["x"][:] == truth).all()
    # every perpendicular column of the seam rows keeps its own values
    assert not (g["x"][6:9, :] == g["x"][6:9, 0:1]).all()


def test_all_fill_seam_chunk_is_omitted(tmpdir):
    """A seam chunk containing only fill values needs no reference at all"""
    fn1 = f"{tmpdir}/fill1.zarr"
    fn2 = f"{tmpdir}/fill2.zarr"
    g = zarr.open(fn1, zarr_format=2)
    arr = g.create_array("x", shape=(6,), chunks=(4,), dtype="int64", fill_value=-5)
    arr[:4] = [1, 2, 3, 4]  # chunk 1 (4:6) is entirely fill
    g = zarr.open(fn2, zarr_format=2)
    arr = g.create_array("x", shape=(6,), chunks=(4,), dtype="int64", fill_value=-5)
    arr[2:6] = [90, 91, 92, 93]  # head (0:2) is also fill

    refs = [
        kerchunk.zarr.single_zarr(fn1, inline=0),
        kerchunk.zarr.single_zarr(fn2, inline=0),
    ]
    out = kerchunk.combine.concatenate_arrays(refs, path="x")
    # output chunk 1 (elements 4:8) is fill from both sides -> key absent
    assert "x/1" not in out["refs"]

    truth = np.array([1, 2, 3, 4, -5, -5, -5, -5, 90, 91, 92, 93])
    g = zarr.open(refs_as_store(out), zarr_format=2)
    assert (g["x"][:] == truth).all()
