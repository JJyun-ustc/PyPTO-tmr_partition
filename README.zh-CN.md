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

[English](README.md) | **中文**

## 概述

PyPTO（发音：pai p-t-o）是一个面向 AI 加速器的高性能编程框架，旨在简化复杂融合算子和整网模型的开发，同时保持高性能计算能力。该框架采用创新的 **PTO（Parallel Tensor/Tile Operation）编程范式**，以 **Tile 编程模型**为核心设计理念。通过多级中间表示 (IR) 系统，将通过 API 构建的 AI 模型应用从高层 Tensor 图逐步编译为硬件指令，最终生成在目标平台上高效运行的可执行代码。

### 核心特性

- **Tile 编程模型**：所有计算基于 Tile（硬件感知的数据块），充分利用硬件并行计算能力和内存层级结构
- **多级计算图变换**：通过编译 Pass 将 Tensor 图变换为 Tile 图、Block 图和执行图，每一步都包含一系列 Pass 优化流程
- **自动代码生成 (CodeGen)**：编译结果通过 CodeGen 生成底层 PTO 虚拟指令代码，随后编译为目标平台的可执行代码
- **MPMD 执行调度**：可执行代码被加载到设备端，使用 MPMD（多程序多数据）方式调度到设备上的处理器核心
- **完整工具链支持**：完整的编译产物和运行时性能数据可通过 IDE 集成工具链可视化，以识别性能瓶颈；开发者也可通过工具链控制编译和调度行为
- **Python 友好 API**：提供符合算法开发者思维模式的直观 Tensor 级抽象，支持动态形状和符号编程
- **分层抽象设计**：向不同开发者暴露不同抽象层级——算法开发者使用 Tensor 级，性能专家使用 Tile 级，系统开发者使用 Block 级

### 目标用户

- **算法开发者**：主要使用 Tensor 级编程进行快速算法实现和验证，专注于算法逻辑
- **性能优化专家**：可使用 Tile 或 Block 级进行深度性能调优，以实现最佳性能
- **系统开发者**：可在 Tensor/Tile/Block 和 PTO 虚拟指令集层面与第三方框架集成或开发工具链

## 快速开始

### 前置条件

- **Python**：3.10 或更高版本
- **CMake**：3.15 或更高版本
- **C++ 编译器**：支持 C++17 标准（GCC、Clang 或 MSVC）
- **nanobind**：2.0.0 或更高版本（构建时自动安装）
- **scikit-build-core**：0.10.0 或更高版本（构建时自动安装）

以上是从源码安装的要求。下面的开发容器已经带齐这些，宿主机一个都不需要。

### 安装

#### 方式一：开发容器（昇腾 A2/A3）

有昇腾卡的机器上最快的路径。CANN、PyPTO、`simpler` 和 ptoas 都已构建进镜像，不需要自己拼工具链。

宿主机需要先备齐三样：昇腾 NPU 驱动（26.0.rc1）、Docker Engine，以及 Ascend Docker
Runtime —— 下面的 `--runtime=ascend` 正是靠这个插件才生效。这三样的安装与核验步骤见
[开发容器](docs/zh/user/04-dev-container.md)，选卡、跨容器保留源码和设备侧故障排查也都在同一篇。

备齐之后，拉取镜像并用 0 号卡起一个容器。镜像仓库允许匿名拉取，不需要登录：

```bash
export PYPTO_IMAGE=swr.cn-east-3.myhuaweicloud.com/cloud-pypto/pypto-dev:a3-dist
docker pull "$PYPTO_IMAGE"

docker run --rm -it \
  --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES=0 \
  --shm-size=16g \
  "$PYPTO_IMAGE"
```

进入容器后，用 `pypto-doctor` 自检环境，用 `python $PYPTO_SRC/examples/beginner/01_hello_world.py`
在卡上跑一个内核。

#### 方式二：从源码安装

1. **克隆仓库**：

   ```bash
   git clone https://github.com/hw-native-sys/pypto.git
   cd pypto
   ```

2. **以开发模式安装**（推荐用于开发）：

   ```bash
   pip install -e .
   ```

   或安装开发依赖：

   ```bash
   # 先安装 CPU 版本的 torch（避免下载约 2GB 的 CUDA 依赖）
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -e ".[dev]"
   ```

3. **以生产模式安装**：

   ```bash
   pip install .
   ```

构建系统使用 scikit-build-core 自动处理 CMake 配置和 C++ 扩展编译。

#### 构建选项

- **构建类型**：默认构建类型为 `RelWithDebInfo`（带调试符号的优化版本）。可以覆盖此设置：

  ```bash
  CMAKE_BUILD_TYPE=Release pip install .
  ```

- **启用 ccache**（可选，加快重复构建）：

  ```bash
  # ccache 可用时会自动检测并使用
  brew install ccache  # macOS
  sudo apt-get install ccache  # Ubuntu/Debian
  ```

### AI 助手插件

PyPTO 公共 skills 通过
[`hw-native-sys/pypto-skills`](https://github.com/hw-native-sys/pypto-skills)
发布为两个插件：`pypto-developer` 提供仓库开发工作流，`pypto-user` 提供面向用户的检查和性能分析工作流。可按需安装其中一个或两个插件。

Codex：

```bash
codex plugin marketplace add hw-native-sys/pypto-skills
codex plugin add pypto-developer@pypto-skills
codex plugin add pypto-user@pypto-skills
```

Claude Code：

```bash
claude plugin marketplace add hw-native-sys/pypto-skills
claude plugin install pypto-developer@pypto-skills
claude plugin install pypto-user@pypto-skills
```

### 运行示例

PyPTO 包含按复杂度组织的示例：

#### 1. Hello World（最简单的程序）

```bash
python examples/beginner/01_hello_world.py
```

#### 2. 入门与进阶示例（逐元素、矩阵乘、softmax 等）

```bash
python examples/intermediate/02_softmax.py
```

#### 3. 模型示例（FFN、paged attention、LLaMA 等）

```bash
python examples/models/01_ffn.py
```

### 运行测试

运行单元测试：

```bash
# 安装开发依赖（先安装 CPU 版 torch）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# 并行运行单元测试
python -m pytest tests/ut -n auto --maxprocesses 8 -v

# 运行指定单元测试文件
python -m pytest tests/ut/core/test_error.py -n auto --maxprocesses 8 -v
```

系统测试请参见 `tests/st/README.md`。

## 文档

文档站点发布在 **<https://www.pypto.ai/pypto/>**。
其源文件是 [`docs/`](docs/) 下的 markdown，在 GitHub 上可直接阅读：

| 章节 | 内容 |
| ---- | ---- |
| [用户手册](docs/zh/user/index.md) | 入门、语言指南、操作参考、调试 |
| [参考](docs/zh/reference/index.md) | PTO ISA —— 集群架构、TPUSH/TPOP、缓冲区管理 |
| [开发者](docs/zh/dev/index.md) | IR、passes、代码生成、后端分派 |
| [运行时](https://www.pypto.ai/simpler/) | 执行已编译程序的 `simpler` 运行时 |

英文文档位于 [`docs/en/`](docs/en/)（以英文为准），可从站点的语言菜单切换。

## 许可证

本项目基于 **CANN Open Software License Agreement Version 2.0** 许可。

该许可授予您有限的、全球范围的、免版税的许可，可下载、使用、修改、集成和分发本软件及其衍生作品，用于开发**仅在华为 AI 处理器系统上使用**的软件（包括 Ascend、Kirin、Yueying 及其他华为品牌 AI 芯片组）。

详细许可条款请参阅 [LICENSE](LICENSE) 文件。
