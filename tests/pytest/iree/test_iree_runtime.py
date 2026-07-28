import pytest

from iree_utils import requires_iree_runtime, matmul_impl

# IREERuntime bridge: the ctypes wrapper driving xtc_iree_shim.


def test_iree_dtype_mapping():
    # dtype -> IREE element-type spelling; unsupported dtypes raise. Pure Python.
    import numpy as np
    from xtc.runtimes.iree.IREERuntime import _iree_dtype

    assert _iree_dtype(np.dtype("float32")) == b"f32"
    assert _iree_dtype(np.dtype("int32")) == b"i32"
    with pytest.raises(NotImplementedError):
        _iree_dtype(np.dtype("complex64"))


@requires_iree_runtime
def test_runtime_setup_invoke_matches_reference(tmp_path):
    # Drive a compiled matmul directly through the runtime, below the evaluator:
    # setup -> readback invoke populates the output -> it matches numpy, and
    # evaluate_perf returns one timing per repeat.
    import numpy as np
    from xtc.runtimes.iree.IREERuntime import IREERuntime

    impl = matmul_impl(64, 64, 64, "float32", "matmul")
    sch = impl.get_scheduler()
    sch.tile("i", {"i1": 16})
    module = impl.get_compiler(dump_file=str(tmp_path / "m")).compile(sch.schedule())

    a = np.random.rand(64, 64).astype("float32")
    b = np.random.rand(64, 64).astype("float32")
    out = np.empty((64, 64), dtype="float32")
    # IREE compilation/execution leaves sticky FP exception flags in the process
    # that numpy would otherwise report on this (correct) reference matmul.
    with np.errstate(all="ignore"):
        expected = a @ b

    rt = IREERuntime()
    ctx = rt.setup(
        vmfb_path=module.file_name,
        entry_function=module.payload_name,
        single_thread=True,
        inputs=[a, b],
        outputs=[out],
    )
    try:
        rt.invoke(ctx)
        assert np.allclose(out, expected, rtol=1e-4, atol=1e-4)
        times = rt.evaluate_perf(
            ctx=ctx, pmu_events=[], repeat=2, number=1, min_repeat_ms=0
        )
        assert len(times) == 2
        assert all(t >= 0 for t in times)
    finally:
        rt.teardown(ctx)
