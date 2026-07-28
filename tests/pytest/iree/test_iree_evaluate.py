import pytest

from iree_utils import requires_iree, requires_iree_runtime, matmul_impl

I, J, K, DTYPE = 64, 64, 64, "float32"
MATMUL_ARGS = (I, J, K, DTYPE)


def _schedule(impl, sched_func):
    sch = impl.get_scheduler()
    sched_func(sch)
    return sch.schedule()


def _tiled_vectorized(sch):
    sch.tile("i", {"i1": 16})
    sch.tile("j", {"j1": 16})
    sch.tile("k", {"k1": 16})
    sch.vectorize(["j1"])


@requires_iree_runtime
def test_evaluate_validates_and_times():
    # evaluate(validate=True) returns the best per-call time only when the
    # output matches the numpy reference; a mismatch would return a string.
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    result = impl.evaluate(_schedule(impl, _tiled_vectorized))
    assert isinstance(result, float)
    assert result > 0


@requires_iree_runtime
def test_close_unloads_and_allows_reload():
    # close() releases the dlopen handle and drops the singleton; the next
    # IREERuntime() reloads the shim on demand.
    from xtc.runtimes.iree.IREERuntime import IREERuntime

    runtime = IREERuntime()
    assert runtime._library() is not None
    runtime.close()
    assert runtime._lib is None
    assert IREERuntime._instance is None
    # A fresh runtime reloads cleanly and works.
    reloaded = IREERuntime()
    assert reloaded._library() is not None


@requires_iree_runtime
def test_evaluate_nop_schedule():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    result = impl.evaluate(_schedule(impl, lambda sch: None))
    assert isinstance(result, float) and result > 0


@requires_iree_runtime
def test_evaluate_parallelized():
    impl = matmul_impl(*MATMUL_ARGS, "matmul")

    def sched(sch):
        _tiled_vectorized(sch)
        sch.parallelize(["i", "j"])

    schedule = _schedule(impl, sched)
    assert schedule.parallelized is True
    result = impl.evaluate(schedule)
    assert isinstance(result, float) and result > 0


@requires_iree_runtime
def test_executor_execute_succeeds(tmp_path):
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    schedule = _schedule(impl, _tiled_vectorized)
    dump = tmp_path / "matmul_iree"
    module = impl.get_compiler(dump_file=str(dump)).compile(schedule)
    executor = module.get_executor(validate=True)
    assert executor.execute() == 0


def test_pmu_counters_require_single_thread():
    # PMU restriction is a pure policy check in __init__ (per-task perf_event
    # counters miss the local-task pool), so it needs neither compiler nor shim.
    from types import SimpleNamespace
    from xtc.targets.iree.IREEEvaluator import IREEEvaluator

    stub = SimpleNamespace(
        _np_inputs_spec=None, _np_outputs_spec=None, _reference_impl=None
    )
    with pytest.raises(NotImplementedError):
        IREEEvaluator(stub, pmu_counters=["INSTRUCTIONS"], single_thread=False)
    # Single-threaded (local-sync) is allowed.
    IREEEvaluator(stub, pmu_counters=["INSTRUCTIONS"], single_thread=True)


@requires_iree
def test_evaluator_default_thread_policy(tmp_path):
    # A non-parallelized schedule defaults to single-threaded (local-sync),
    # a parallelized one to multi-threaded (local-task). This only inspects the
    # evaluator's thread policy, so it needs the compiler but not the shim.
    impl = matmul_impl(*MATMUL_ARGS, "matmul")
    dump = tmp_path / "matmul_iree"

    seq = impl.get_compiler(dump_file=str(dump)).compile(
        _schedule(impl, _tiled_vectorized)
    )
    assert seq.get_evaluator()._single_thread is True

    impl2 = matmul_impl(*MATMUL_ARGS, "matmul")

    def par(sch):
        _tiled_vectorized(sch)
        sch.parallelize(["i", "j"])

    dump2 = tmp_path / "matmul_iree_par"
    mod = impl2.get_compiler(dump_file=str(dump2)).compile(_schedule(impl2, par))
    assert mod.get_evaluator()._single_thread is False
