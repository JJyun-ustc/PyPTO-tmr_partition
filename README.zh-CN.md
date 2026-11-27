# PyPTO TMR：AICPU cluster 放置研究

## 下一轮方案：流式 TMR 动作调度

**仅为设计提案，尚无新实现或测量。** S 有真实空槽时，仅用可见事件前缀比较默认领取与替代领取；先完成 A–D 和菱形提交时序验证，再决定是否训练残差模型。

- [中文方案](docs/zh/experiments/tmr-cluster/streaming-action-plan.md)
- [English plan](docs/en/experiments/tmr-cluster/streaming-action-plan.md)

## 最新已完成实验：QKV 第二轮，MIX、反馈与离线训练

同轮重测 Shared，中位数 **36.831 ms**；原 QKV **40.005 ms（+8.62%）**，
改进 QKV **37.668 ms（+2.27%）**，训练型 QKV **37.616 ms（+2.13%）**。
每策略 30 个正式样本及 6 个 warm-up，四策略输出与全部 40 层 KV golden 通过，37 个 kernel 指纹一致。

主要开销指向集中 pending / S0 等待。普通 MIX 在同步共存用例真实选侧 20/23 个任务，
但短链存在大量未初始化/过旧摘要回退。学习型 QKV 的留出综合 MAE 改善 14.31%，
AICPU 影子推理通过预算；相对改进版仅快约 0.14%，未证明稳定净收益。
Qwen 的同步 attention 保留原协议，性能回退与失败尝试全部保留。

- [一页结论与正式结果](docs/zh/experiments/tmr-cluster/qkv-round2.md)
- [最新机制与完整实验流程](docs/zh/experiments/tmr-cluster/algorithm-qkv-round2.md)
- [English report](docs/en/experiments/tmr-cluster/qkv-round2.md)
- [全部样本、训练权重、校准表、回放、诊断与复现命令](official-results/qkv-round2-20261125/)

`runtime` 子模块固定到 `8ebbff773`；数据于 2026-11-25 采集，2026-11-26 发布。

## 历史实验：QKV 初版

A2A3 Qwen3-14B decode 上，QKV 相对 Shared 的中位数从 **36.697 增至 39.307 ms（回退 7.11%）**，
p95 从 **36.829 增至 39.463 ms（回退 7.15%）**，每种模式各 30 个正式样本。
两种模式的输出和全部层 KV golden 通过，37 个实际加载 kernel 指纹一致。
原型覆盖 96.55% 的符合条件任务，处理 40,325 个自然多任务批。全部样本、warm-up 与失败采集尝试保留。

- [最新机制、所有权证明与完整复现流程](docs/zh/experiments/tmr-cluster/algorithm-qkv-ready-batch.md)
- [结果报告、采样协议与限制](docs/zh/experiments/tmr-cluster/qkv-ready-batch.md)
- [English mechanism and workflow](docs/en/experiments/tmr-cluster/algorithm-qkv-ready-batch.md)
- [原始证据与冻结指纹](official-results/qkv-ready-batch-20261125/)

`runtime` 子模块固定到 `runtime-qkv-ready-batch`。策略冻结，本轮不补调参或消融。
发布不包含二进制、venv、缓存及通用驱动日志。

## 历史实验：v14

历史版本 **v14：O 统一评分、双侧固定 block 配额**。删除 S 首次 ready 改侧、WAIT 票数累计和跨侧领取，
由两侧 S 领取各自不重叠的范围，通过周期摘要向 O 反馈实际队列压力。

Qwen decode 相比 v8 改善，但仍慢于 Shared；Bgemm 和 Paged Attention 相比 v8 回退。
全部结果保留，测量中未调参；v14 是机制验证版本。

- [v14 算法](docs/zh/experiments/tmr-cluster/algorithm-v14.md)
- [v14 完整报告与诊断](docs/zh/experiments/tmr-cluster/o-quota.md)
- [全部样本与统计](official-results/v14-o-quota-20261120/summary.json)
- [CSV](official-results/v14-o-quota-20261120/results.csv)
- [v13 报告](docs/zh/experiments/tmr-cluster/orchestrator-pressure.md)
- [v8 算法](docs/zh/experiments/tmr-cluster/algorithm-v8.md)
- [源码来源](docs/zh/experiments/tmr-cluster/source-provenance.json)

## 本轮结果

device_wall 中位数，单位 ms，每项每版本 30 样本，正百分比表示更慢。

| Case | Shared | v8 (ms) | v14 | v14 vs v8 | v14 vs Shared |
| ---- | -----: | ------: | --: | --------: | ------------: |
| Bgemm | 1.04637 | 1.00804 | 1.15184 | +14.27% | +10.08% |
| Scale | 4.85766 | 3.86222 | 5.67701 | +46.99% | +16.87% |
| Uniform | 2.45813 | 2.20300 | 2.93216 | +33.10% | +19.28% |
| Skew | 3.28223 | 2.90045 | 3.87212 | +33.50% | +17.97% |
| Qwen | 36.10933 | 46.00712 | 40.56889 | -11.82% | +12.35% |

物理卡 6 / 逻辑 0，Ascend910B2，六种版本顺序、每进程五轮，全部首轮保留。
450 个普通样本，15 次模型 golden，15 次独立诊断；184 个 CPU 用例和 15 个板上回归通过。
未持独占设备锁；中断后保留 255 个样本续跑，全部批次和恢复记录公开。

## 获取代码

```bash
git clone --recurse-submodules https://github.com/JJyun-ustc/PyPTO-tmr_partition.git
cd PyPTO-tmr_partition
```

历史 v14 提交保留在 `runtime-o-quota` 分支。
v8、v11、v12、v13 分别保留在 `runtime-snapshot`、`runtime-ready-affinity`、
`runtime-ready-pressure`、`runtime-orchestrator-pressure`。
发布标签 `v14-o-quota-20261120`；仓库不包含二进制、venv、缓存或凭据。

---

## PyPTO

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
