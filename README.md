# PyPTO TMR：AICPU cluster 放置研究

当前发布 **v11 ready亲和研究版本**：在独立运行库分支重新实现，删除旧价格学习、
O负载评分、未来预留与完成负载计数。O只形成依赖亲和，第一次ready选择队列侧，
S按block认领，并允许低频、受限的自动单block跨侧领取。
这里的cluster是AICPU cluster，不是AI Core组。

**实验不支持“普遍优于v8”**：多block与人工突发负载改善，Scale和PA回退。
旧v8在 `runtime-snapshot` 原样保留，新版在 `runtime-ready-affinity`。

- [完整算法、流程与状态记账](docs/zh/experiments/tmr-cluster/algorithm.md)
- [v11中文实验报告：全部中位数/p95、O/S诊断与限制](docs/zh/experiments/tmr-cluster/ready-affinity.md)
- [全部逐轮样本与验证](official-results/v11-ready-affinity-20261119/summary.json)
- [比较CSV](official-results/v11-ready-affinity-20261119/results.csv)
- [A1–A4累积实验](docs/zh/experiments/tmr-cluster/block-pressure.md)
- [历史v8算法](docs/zh/experiments/tmr-cluster/algorithm-v8.md)
- [源码来源与精确提交](docs/zh/experiments/tmr-cluster/source-provenance.json)

## 获取完整代码

```bash
git clone --recurse-submodules https://github.com/JJyun-ustc/PyPTO-tmr_partition.git
cd PyPTO-tmr_partition
```

main保存完整PyPTO，runtime子模块固定到本仓库的v11运行库具体提交。
旧版本可从 `runtime-snapshot` 或先前main提交获取。第三方来源、许可证随源码保留。
原研究worktree中的v8与原始PyPTO未被覆盖。

## 本轮结果

同一张910B2物理卡5重测：独立冻结官方Shared、v8、v11关闭领取、v11开启领取，
O固定CPU7，S使用CPU2–6，两个AICPU cluster。
每组30轮、15正序+15反序；以下为普通device_wall中位数，单位ms。

|负载|官方Shared复核|v8|v11开启领取|v11相对v8|
|---|---:|---:|---:|---:|
|Bgemm|1.0334|0.9904|0.9840|-0.65%|
|Scale|4.8404|3.8404|5.1357|+33.73%|
|Uniform|2.4578|2.1965|2.6628|+21.23%|
|Skew|3.2542|2.8784|3.5130|+22.05%|
|Multi16|1.2200|1.2827|1.1681|-8.93%|
|MultiSkew|4.9481|5.0247|4.6181|-8.09%|
|Fanout|4.8895|4.4633|4.5475|+1.89%|
|Burst|14.0967|14.7075|13.7389|-6.59%|

共1200个普通样本、40次golden及40次独立诊断；184项C++测试、15个板上回归通过。
官方源码唯一runtime src差异是线程上限4→6，官方调度、队列和依赖处理冻结；
这与过去修改运行库中的Shared分支不同。MultiSkew官方前后漂移约2.08%，未持有独占锁，
小幅差异不作确定收益；仿真场景因缺少g++-15未执行。未做O位置对照。

环境记录是依赖/工具链与复现说明，不是磁盘镜像。Git不包含虚拟环境、编译产物、
缓存或凭据；驱动/CANN需另行安装。许可证见 [LICENSE](LICENSE)。

---

以下保留上游PyPTO说明。

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
