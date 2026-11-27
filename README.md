# PyPTO TMR: AICPU cluster placement study

## Latest measurements: streaming TMR infrastructure and visibility

Shared median/p95 **36.650/36.731 ms**; B infrastructure **38.207/38.350 ms (+4.25% median)**.
All 720 ordinary samples and 144 warm-ups across three sequence lengths are retained; 12 complete output/40-layer KV golden checks pass.
**Active C/D dispatch is not implemented, overrides are zero, and new gain models are not trained. This is not a completed active A–D comparison.**

- [English results](docs/en/experiments/tmr-cluster/streaming-action-results.md) · [中文报告](docs/zh/experiments/tmr-cluster/streaming-action-results.md)
- [Actual mechanism and limits](docs/en/experiments/tmr-cluster/streaming-action-mechanism.md) · [Reproduce](docs/en/experiments/tmr-cluster/streaming-action-reproduce.md)
- [Samples and complete raw archive](official-results/streaming-action-20261127/)

The `runtime` submodule pins source snapshot `153ba8ae4` on `runtime-streaming-action-20261127`; measured base is `8ebbff773` plus the retained patch.

## Next experiment plan: streaming TMR action scheduling

**Target specification; partially measured, not fully implemented.** Compare default and alternative claims at a real idle slot using only the visible event prefix. Validate A–D and diamond submission timing before residual training.

- [English plan](docs/en/experiments/tmr-cluster/streaming-action-plan.md)
- [中文方案](docs/zh/experiments/tmr-cluster/streaming-action-plan.md)

## Latest completed experiment: QKV round two — MIX, feedback and offline training

Same-round Shared median: **36.831 ms**. Original QKV: **40.005 ms (+8.62%)**;
improved QKV: **37.668 ms (+2.27%)**; learned QKV: **37.616 ms (+2.13%)**.
Thirty formal samples and six retained warm-ups per policy. All four pass output
and complete forty-layer KV golden; all 37 loaded compute-kernel fingerprints match.

Evidence points to centralized pending/S0 waiting as the main original cost.
Ordinary MIX actually assigns 20/23 tasks in the sync-coexistence test, while short
chains expose missing/stale-summary fallbacks. Learned-QKV holdout composite MAE
improves by 14.31% and AICPU shadow inference passes its budget, but its 0.14%
median gain over improved QKV does not establish a stable net scheduling benefit.
Qwen synchronous attention retains its protocol. Regressions and failed attempts remain public.

- [One-page conclusion and formal results](docs/en/experiments/tmr-cluster/qkv-round2.md)
- [Mechanism and complete experiment workflow](docs/en/experiments/tmr-cluster/algorithm-qkv-round2.md)
- [Chinese report](docs/zh/experiments/tmr-cluster/qkv-round2.md)
- [Samples, weights, calibration, replay, diagnostics and commands](official-results/qkv-round2-20261125/)

The round-two runtime snapshot was `8ebbff773`. Data collected 2026-11-25; published 2026-11-26.

## Historical experiment: first QKV prototype

QKV versus Shared on A2A3 Qwen3-14B decode: **36.697 → 39.307 ms median
(+7.11%)**, p95 **36.829 → 39.463 ms (+7.15%)**, 30 formal samples per mode.
Both output/all-layer KV goldens pass; 37 loaded kernel fingerprints match.
The prototype assigns 96.55% of eligible tasks and processes 40,325 natural
multi-task batches. All samples, warmups and failed collection attempts remain available.

- [Mechanism, ownership and complete reproduction](docs/en/experiments/tmr-cluster/algorithm-qkv-ready-batch.md)
- [Results, protocol and limitations](docs/en/experiments/tmr-cluster/qkv-ready-batch.md)
- [Chinese mechanism and workflow](docs/zh/experiments/tmr-cluster/algorithm-qkv-ready-batch.md)
- [Raw evidence and frozen fingerprints](official-results/qkv-ready-batch-20261125/)

The first-round runtime branch was `runtime-qkv-ready-batch`. The policy is frozen;
no additional tuning or ablation follows this result. Compiled binaries,
virtual environments, caches and generic driver logs are excluded from publication.

## Previous experiment: v14

**v14: O scoring and immutable per-side block quotas.** First-ready side correction,
WAIT consumer voting, and cross-side pickup are removed. S executes its own range and
periodically publishes actual queue pressure to O.

Qwen improves over v8 but remains slower than Shared; Bgemm and Paged Attention regress versus v8.
All results are retained, with no tuning during measurement.

- [v14 algorithm](docs/en/experiments/tmr-cluster/algorithm-v14.md)
- [v14 results and diagnostics](docs/en/experiments/tmr-cluster/o-quota.md)
- [Samples and statistics](official-results/v14-o-quota-20261120/summary.json)
- [CSV](official-results/v14-o-quota-20261120/results.csv)
- [Chinese report](docs/zh/experiments/tmr-cluster/o-quota.md)

## Results

Device-wall median milliseconds, 30 samples per case/version. Positive percentages mean slower.

| Case | Shared | v8 (ms) | v14 | v14 vs v8 | v14 vs Shared |
| ---- | -----: | ------: | --: | --------: | ------------: |
| Bgemm | 1.04637 | 1.00804 | 1.15184 | +14.27% | +10.08% |
| Scale | 4.85766 | 3.86222 | 5.67701 | +46.99% | +16.87% |
| Uniform | 2.45813 | 2.20300 | 2.93216 | +33.10% | +19.28% |
| Skew | 3.28223 | 2.90045 | 3.87212 | +33.50% | +17.97% |
| Qwen | 36.10933 | 46.00712 | 40.56889 | -11.82% | +12.35% |

Ascend910B2 physical card 6, logical 0. Six version permutations, five rounds per process,
all first rounds retained: 450 ordinary samples, 15 model golden runs, 15 separate diagnostics.
184 CPU tests and 15 board regression cases pass. No exclusive device lock was available.
After a session interruption, 255 validated samples were retained; batch medians and restart metadata are included.

## Source

```bash
git clone --recurse-submodules https://github.com/JJyun-ustc/PyPTO-tmr_partition.git
cd PyPTO-tmr_partition
```

The previous v14 snapshot is preserved on `runtime-o-quota`; prior study branches remain intact.
Release tag: `v14-o-quota-20261120`. Binaries, environments, caches, and credentials are excluded.

---

## PyPTO

**English** | [Chinese](README.zh-CN.md)

## Overview

PyPTO (pronounced: pai p-t-o) is a high-performance programming framework for AI accelerators, designed to simplify the development of complex fused operators and entire model networks while maintaining high-performance computing capabilities. The framework adopts an innovative **PTO (Parallel Tensor/Tile Operation) programming paradigm**, with a **Tile-based programming model** as its core design philosophy. Through a multi-level intermediate representation (IR) system, it compiles AI model applications built via APIs from high-level Tensor graphs step by step into hardware instructions, ultimately generating executable code that runs efficiently on target platforms.

### Core Features

- **Tile-based Programming Model**: All computations are based on Tiles (hardware-aware data blocks), fully leveraging hardware parallel computing capabilities and memory hierarchy
- **Multi-level Computation Graph Transformation**: Transforms Tensor Graphs into Tile Graphs, Block Graphs, and Execution Graphs through compilation passes, with each step including a series of pass optimization workflows
- **Automated Code Generation**: Compilation results generate low-level PTO virtual instruction code through CodeGen, which is then compiled into executable code for target platforms
- **MPMD Execution Scheduling**: Executable code is loaded to the device side and scheduled to processor cores on the device using MPMD (Multiple Program Multiple Data) approach
- **Complete Toolchain Support**: Full compilation artifacts and runtime performance data can be visualized through IDE-integrated toolchains to identify performance bottlenecks; developers can also control compilation and scheduling behavior through the toolchain
- **Python-friendly API**: Provides intuitive Tensor-level abstractions that align with algorithm developers' thinking patterns, supporting dynamic shapes and symbolic programming
- **Layered Abstraction Design**: Exposes different abstraction levels to different developers - algorithm developers use the Tensor level, performance experts use the Tile level, and system developers use the Block level

### Target Users

- **Algorithm Developers**: Primarily use Tensor-level programming for rapid algorithm implementation and validation, focusing on algorithm logic
- **Performance Optimization Experts**: Can use Tile or Block levels for deep performance tuning to achieve optimal performance
- **System Developers**: Can integrate with third-party frameworks or develop toolchains at Tensor/Tile/Block and PTO virtual instruction set levels

## Getting Started

### Prerequisites

- **Python**: Version 3.10 or higher
- **CMake**: Version 3.15 or higher
- **C++ Compiler**: Supporting C++17 standard (GCC, Clang, or MSVC)
- **nanobind**: Version 2.0.0 or higher (automatically installed during build)
- **scikit-build-core**: Version 0.10.0 or higher (automatically installed during build)

These apply to a from-source install. The development container below carries all of
them, and needs none of them on the host.

### Installation

#### Option A: Development Container (Ascend A2/A3)

The fastest path on a machine with an Ascend card. CANN, PyPTO, `simpler`, and ptoas are
already built into the image, so there is no toolchain to assemble.

The host has to supply three things first: an Ascend NPU driver (26.0.rc1), Docker Engine,
and Ascend Docker Runtime — the plugin that makes `--runtime=ascend` below work at all.
[Development Container](docs/en/user/04-dev-container.md) carries the install and
verification steps for each, and is also where card selection, keeping your source across
containers, and device troubleshooting are covered.

With those in place, pull the image and start a container on card 0. The registry allows
anonymous pulls, so no login is needed:

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
docker pull "$PYPTO_IMAGE"

docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=0 \
  --shm-size=16g \
  "$PYPTO_IMAGE"
```

Inside the container, `pypto-doctor` checks the environment and
`python $PYPTO_SRC/examples/beginner/01_hello_world.py` runs a kernel on the card.

#### Option B: Install from Source

1. **Clone the repository**:

   ```bash
   git clone https://github.com/hw-native-sys/pypto.git
   cd pypto
   ```

2. **Install in development mode** (recommended for development):

   ```bash
   pip install -e .
   ```

   Or with development dependencies:

   ```bash
   # Install CPU-only torch first (avoids pulling ~2GB CUDA dependencies)
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[dev]"
   ```

3. **Install in production mode**:

   ```bash
   pip install .
   ```

The build system uses scikit-build-core to automatically handle CMake configuration and C++ extension compilation.

#### Build Options

- **Build type**: The default build type is `RelWithDebInfo` (optimized with debug symbols). You can override this:

  ```bash
  CMAKE_BUILD_TYPE=Release pip install .
  ```

- **Enable ccache** (optional, for faster rebuilds):

  ```bash
  # ccache will be automatically detected and used if available
  brew install ccache  # macOS
  sudo apt-get install ccache  # Ubuntu/Debian
  ```

### AI Assistant Plugins

Shared PyPTO skills are published from
[`hw-native-sys/pypto-skills`](https://github.com/hw-native-sys/pypto-skills)
as two plugins: `pypto-developer` for repository development workflows and
`pypto-user` for user-facing inspection and profiling workflows. Install either
or both plugins as needed.

Codex:

```bash
codex plugin marketplace add hw-native-sys/pypto-skills
codex plugin add pypto-developer@pypto-skills
codex plugin add pypto-user@pypto-skills
```

Claude Code:

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-developer@pypto-skills
claude plugin install pypto-user@pypto-skills
```

### Running Examples

PyPTO includes examples organized by complexity:

#### 1. Hello World (simplest program)

```bash
python examples/beginner/01_hello_world.py
```

#### 2. Beginner & Intermediate Examples (element-wise, matmul, softmax, ...)

```bash
python examples/intermediate/02_softmax.py
```

#### 3. Model Examples (FFN, paged attention, LLaMA, ...)

```bash
python examples/models/01_ffn.py
```

### Running Tests

To run unit tests:

```bash
# Install with development dependencies (CPU-only torch first)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Run unit tests in parallel
python -m pytest tests/ut -n auto --maxprocesses 8 -v

# Run a specific unit test file
python -m pytest tests/ut/core/test_error.py -n auto --maxprocesses 8 -v
```

For system tests, see `tests/st/README.md`.

## Documentation

The documentation site is published at **<https://www.pypto.ai/pypto/>**.
Its source is the markdown under [`docs/`](docs/), readable directly on GitHub:

| Section | Contents |
| ------- | -------- |
| [User Manual](docs/en/user/index.md) | Getting started, the language guide, the operation reference, debugging |
| [Reference](docs/en/reference/index.md) | PTO ISA — cluster architecture, TPUSH/TPOP, buffer management |
| [Developer](docs/en/dev/index.md) | IR, passes, code generation, backend dispatch |
| [Runtime](https://www.pypto.ai/simpler/) | The `simpler` runtime that executes compiled programs |

Chinese translations live in [`docs/zh/`](docs/zh/) and are selectable from the
site's language menu.

## License

This project is licensed under the **CANN Open Software License Agreement Version 2.0**.

This license grants you a limited, worldwide, royalty-free license to download, use, modify, integrate, and distribute the Software or its derivative works for the purpose of developing software **solely for use in systems with Huawei AI Processors** (including Ascend, Kirin, Yueying, and other Huawei-branded AI chipsets).

See the [LICENSE](LICENSE) file for the full license text.
