"""BGEMM configurations reserved for validation after score candidate comparison."""

import test_benchmark_bgemm as bgemm

from simpler_setup import SceneTestCase, scene_test


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestScoreHoldout(SceneTestCase):
    RTOL = bgemm.TestBenchmarkBgemm.RTOL
    ATOL = bgemm.TestBenchmarkBgemm.ATOL
    CALLABLE = bgemm.TestBenchmarkBgemm.CALLABLE
    generate_args = bgemm.TestBenchmarkBgemm.generate_args
    compute_golden = bgemm.TestBenchmarkBgemm.compute_golden
    CASES = [
        {
            "name": family + mode,
            "platforms": ["a2a3"],
            "manual": True,
            # This study compares the same two-cluster six-thread topology.
            "config": {"aicpu_thread_num": 6},
            "params": dict(params, placement_auto=int(mode == "Adaptive")),
        }
        for family, params in (
            ("Heavy", {"matmul_add_task_num": 256, "incore_data_size": 128, "incore_loop": 8, "grid_k": 2}),
            ("Deep", {"matmul_add_task_num": 512, "incore_data_size": 64, "incore_loop": 2, "grid_k": 8}),
        )
        for mode in ("Shared", "Adaptive")
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
