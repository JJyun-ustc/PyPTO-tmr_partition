# PyPTO TMR：AICPU cluster 放置研究

当前发布 **v13 O预测预留研究版本**：完整压力评分回到O，S只做阈值式ready亲和修正；
按writer归属低频汇总累计R/E/C，O维护有界预测预留，移除逐FIN压力更新，保留受限跨侧领取。
不实现范围级亲和、kernel时长预测或价格学习。这里是AICPU cluster，不是AI Core组。

**本轮未保住v8性能**：四项中位数均更慢；领取开关消融也没有证明稳定净收益。
新版按要求原样保留全部结果，不根据中途成绩调参或改发最好候选。
v8、v11、v12分支保留，v13独立位于 `runtime-orchestrator-pressure`。

- [完整算法与累计账本](docs/zh/experiments/tmr-cluster/algorithm.md)
- [v13中文实验报告：交错计时、O/S诊断、领取消融](docs/zh/experiments/tmr-cluster/orchestrator-pressure.md)
- [全部样本与验证](official-results/v13-orchestrator-pressure-20261119/summary.json)
- [中位数、p95与各交错组CSV](official-results/v13-orchestrator-pressure-20261119/results.csv)
- [v12算法](docs/zh/experiments/tmr-cluster/algorithm-v12.md) / [v12报告](docs/zh/experiments/tmr-cluster/ready-pressure.md)
- [v8算法](docs/zh/experiments/tmr-cluster/algorithm-v8.md)
- [源码来源](docs/zh/experiments/tmr-cluster/source-provenance.json)

## 获取完整代码

```bash
git clone --recurse-submodules https://github.com/JJyun-ustc/PyPTO-tmr_partition.git
cd PyPTO-tmr_partition
```

main保存完整PyPTO，runtime子模块固定到v13具体提交。旧分支 `runtime-snapshot`、
`runtime-ready-affinity`、`runtime-ready-pressure` 分别保留v8、v11、v12。
原始PyPTO与旧研究worktree均未覆盖。

## 本轮结果

同一张910B2物理卡0/逻辑0，PCIe 0000:C1:00.0；O=CPU7，S=CPU2–6，两个AICPU cluster。
四组平衡Williams版本顺序，每组每版本8轮，每项32轮；所有进程首轮保留。
下表为device_wall中位数，单位ms；正数表示更慢。

|负载|v8|v12|v13关闭领取|v13开启领取|v13相对v8|
|---|---:|---:|---:|---:|---:|
|Bgemm|1.01090|1.00960|1.12555|1.11790|+10.58%|
|Scale|3.80049|5.00003|5.48891|5.48853|+44.42%|
|Uniform|2.20268|2.63624|2.66862|2.70676|+22.88%|
|Skew|2.90953|3.68813|3.54387|3.56969|+22.69%|

512个普通样本、16次golden、16次独立诊断；194个CPU用例、15个板上正确性用例和DFX关闭构建通过。
Scale诊断中S ready略降，但背景反馈与O提交成本上升；WAIT票辅助独立测量，不能声称完成路径零新增开销。
普通性能诊断已触发单block远端领取；多block板上回归未触发远端分支，范围唯一性由并发CPU用例覆盖。
未持独占设备锁，小幅变化不作稳定收益；仿真因缺g++-15未运行；未做O位置对照。
详细p95、交错组结果、全部诊断与限制见报告。

Git保存完整项目、候选源码、指纹、原始研究结果及中文说明；不包含二进制、venv、
缓存或凭据。环境记录是依赖与工具链说明，不是磁盘镜像。许可证见 [LICENSE](LICENSE)。

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
