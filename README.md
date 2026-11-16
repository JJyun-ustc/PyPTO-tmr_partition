# PyPTO TMR：AICPU cluster 二分放置实验

本仓库保存基于完整 PyPTO 系统的 Ascend 910B2 AICPU cluster 放置研究。
最新版本采用固定 256 节点校准、冻结参数、禁止跨侧领取和 8 节点快照批次。
这里的 cluster 是 **AICPU cluster**，不是 AI Core 组。

- [中文实验说明](docs/zh/experiments/tmr-cluster/experiment.md)
- [环境与复现步骤](docs/zh/experiments/tmr-cluster/environment.md)
- [Python 环境版本记录](docs/zh/experiments/tmr-cluster/environment-observed.json)
- [2026-11-16 原始结果与逐轮数据](official-results/v6-simplified-20261116/summary.json)
- [源码来源与版本](docs/zh/experiments/tmr-cluster/source-provenance.json)

## 获取完整代码

```bash
git clone --recurse-submodules https://github.com/JJyun-ustc/PyPTO-tmr_partition.git
cd PyPTO-tmr_partition
```

`main` 保存 PyPTO；同一 GitHub 仓库的 `runtime-snapshot` 分支保存完整修改后的
runtime。`main` 的 runtime 子模块固定引用该分支的具体提交。此布局保留 PyPTO
原有 runtime 版本一致性检查，核心改动不会仅存在于无法取得的本地子模块里。
第三方 libbacktrace、msgpack-c 保留各自上游及固定版本，许可证随源码保留。

## 当前结论

当前卡上重新配对测量，各策略 30 轮、两批反序，单位为 device_wall 中位数：

| 规模 | 原自动策略 | 简化后的自动策略 | 新版 Shared | 新自动相对 Shared |
| --- | ---: | ---: | ---: | ---: |
| Case0，1000 节点 | 1.419 ms | 1.197 ms | 1.047 ms | +14.32% |
| Scale，4096 节点 | 6.629 ms | 5.809 ms | 4.913 ms | +18.23% |

新版比原自动策略降低 15.66% / 12.37%，但尚未超过 Shared。
这是合并修改的效果，没有逐项消融；本次实现的是快照批处理，**不是并行评分**。
实验未持有独占设备锁，不同物理卡的历史结果不合并；O 始终固定在 CPU 7。

环境文件记录依赖、工具链和复现步骤，不是容器镜像或系统磁盘备份。
虚拟环境、缓存、编译产物和凭据不随源码上传；驱动/CANN 需要在目标机器另行安装。
研究代码沿用上游 CANN Open Software License Agreement Version 2.0，见 [LICENSE](LICENSE)。

---

以下保留上游 PyPTO 项目说明。

# PyPTO

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
