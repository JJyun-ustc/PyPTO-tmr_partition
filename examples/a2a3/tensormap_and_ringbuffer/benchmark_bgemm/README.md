# benchmark_bgemm — a shape sweep, not a kernel demo

Tiled batched matmul `C = sum over k of A[k] @ B[k]`, with every dimension that
matters driven from `params`. The arithmetic is ordinary; the directory exists
so you can move one axis at a time and watch what it costs.

| Task | Core | Work |
| ---- | ---- | ---- |
| `GEMM` | AIC | one `tile_size × tile_size` matmul |
| `ADD` | AIV | accumulate that tile into `C` |

## The four axes

| Param | Moves | Effect |
| ----- | ----- | ------ |
| `matmul_add_task_num` | task count | more, smaller units of scheduling — the dispatch-overhead axis |
| `incore_data_size` | tile edge | arithmetic per task, quadratically |
| `incore_loop` | tiles per task | arithmetic per task, linearly, with no extra dispatch |
| `grid_k` | accumulation depth | how many partial products land in the same `C` tile — the dependency-chain axis |

The cases are laid out as single-axis moves from `Case1`, which is what makes
them comparable:

| Case | Default Sim? | Default Onboard? | `task_num` | `data_size` | `loop` | `grid_k` | Reads as |
| ---- | ------------ | ---------------- | ---------- | ----------- | ------ | -------- | -------- |
| `Case1` | no | no | 64 | 128 | 4 | 2 | the reference point |
| `Case2` | no | no | 256 | 128 | 4 | 2 | 4× the tasks |
| `Case0` | **yes** | **yes** | 500 | 128 | 4 | 2 | ~8× the tasks |
| `Case3` | no | no | 64 | 128 | 16 | 2 | 4× the work per task, same task count |
| `Case4` | no | no | 64 | 128 | 4 | 4 | twice the accumulation depth |
| `Bgemm64` | no | **yes** | 32 | 64 | 1 | 4 | small tiles, minimal per-task work |

`Case2` against `Case3` is the interesting pair: both quadruple the total
matmul work relative to `Case1`, one by adding tasks and one by making each
task bigger.

## The default run differs by platform

`Case1`–`Case4` are manual on both platforms. `Bgemm64` is manual only on
`a2a3sim`, so the Sim Per-PR run keeps the larger `Case0`, while the Onboard
Per-PR run continues to execute both `Case0` and `Bgemm64`:

```bash
# Sim default — Case0 only, 1 of 6 cases
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3sim

# Onboard default — Case0 + Bgemm64; placement cases are manual
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3 --device 0

# include manual cases: six baseline cases on Sim, plus four placement cases on Onboard
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3sim --manual include

# Sim Daily selection — Case1–Case4 + Bgemm64, 5 of 6 cases
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3sim --manual only
```

If a `--case` selector matches nothing, the error names this as the likely
cause — the selected case is manual and the default mode is `exclude`.

All six baseline cases run on `a2a3sim` as well as `a2a3`, which makes this one of the few
places you can move these axes without holding a die.

## `C` is INOUT, not OUT

Worth knowing before you copy the `CALLABLE` block. `C` is a zero-initialised
accumulator: the AIV kernel reads it from GM, adds the matmul result, and
stores it back, once per `grid_k` iteration. The host-provided zeros therefore
have to be staged host-to-device, which makes `C` **read-before-write** — an
`INOUT`, not a pure `OUT`. Declaring it `OUT` would skip the H2D stage and
accumulate onto whatever the buffer held.

## Run

```bash
pytest examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm --platform a2a3 --device 0 --manual include
```

The manual `ClusterCase0Shared`/`ClusterCase0Adaptive` and
`ClusterScaleShared`/`ClusterScaleAdaptive` pairs request six active AICPU
threads. Each pair runs the same BGEMM graph and deterministic input tensors;
only the C++ task placement request changes. `ClusterCase0` reuses Case0's
500 matmul/add pairs (1000 tasks). `ClusterScale` uses 2048 pairs (4096 tasks) with 64×64 tiles and `grid_k=4`
to expose scheduling pressure across many independent groups. For example:

```bash
python test_benchmark_bgemm.py -p a2a3 -d 0 --case ClusterScaleShared --manual include --rounds 30 --skip-golden
python test_benchmark_bgemm.py -p a2a3 -d 0 --case ClusterScaleAdaptive --manual include --rounds 30 --skip-golden
```

Run each case once with its golden check before using `--skip-golden` for
repeated timing. See [AICPU cluster placement](../../../../docs/user/how-to/use-aicpu-cluster-placement.md)
for the runtime behavior and measurement boundaries.
The current controller retains a fixed 256-node calibration prefix and uses
bounded dependency groups, ready-time final binding, 64-node summary epochs
and restricted single-block cross-cluster pickup. See the
[2026-11-16 investigation](../../../../docs/investigations/2026-11-aicpu-cluster-bgemm.md)
for staged v6/v7 controls. The final Case0/Scale automatic medians are
3.60%/18.79% below same-binary Shared controls in this unlocked-device suite;
the results do not establish an independent benefit from cross-cluster pickup.

Tolerances are `RTOL = ATOL = 1e-3`, with inputs scaled by `0.01` to keep the
`grid_k`-deep float32 accumulation inside them.

For turning a run into numbers rather than pass/fail, see
[`docs/dfx/README.md`](../../../../docs/dfx/README.md) — in particular the
scheduler-overhead model, which is the thing the task-count axis here is built
to probe.

### v7 控制成本分阶段实验

当前自动策略使用有界小组偏好、ready 时绑定、低频单写者摘要以及受限跨侧领取。
完整机制与压力记账见 [放置说明](../../../../docs/user/how-to/use-aicpu-cluster-placement.md)，
结果见 [实验记录](../../../../docs/investigations/2026-11-aicpu-cluster-bgemm.md)。
`reproduce_cluster_study.py` 执行四个独立黄金校验和两批反序的各 15 轮计时；
`run_control_stage.py <new-output-dir> --jobs <N>` 从当前 worktree 构建并追加独立诊断；
`analyze_control_study.py <study-root>` 汇总本次保存的各阶段。
