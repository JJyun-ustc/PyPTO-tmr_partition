# 环境记录与复现步骤

## 环境范围

本仓库提供完整 PyPTO、修改后的 runtime、固定第三方来源、实验用例、原始证据和
依赖版本记录。它不是运行中容器或系统盘的逐字节备份；不包含虚拟环境、Git 凭据、
编译缓存、设备内存、CANN 安装包或驱动二进制。驱动/CANN 需按机器和授权另行准备。
上游 docker/pypto_dev.Dockerfile 与 docker/scripts/ 保留，可参考其完整部署流程。

| 项目 | 被测环境 |
| --- | --- |
| 架构／系统 | aarch64／openEuler 24.03 LTS-SP2 |
| Python | 3.10.22 |
| CANN／驱动 | 9.0.0／26.1.1 |
| 设备 | Ascend 910B2 |
| GCC | 12.3.1 |
| Torch／NumPy | 2.10.0+cpu／2.2.6 |
| CMake／Ninja | 4.4.2／1.13.0 |
| nanobind／scikit-build-core | 2.15.0／1.0.3 |
| pytest／ruff | 9.1.1／0.14.8 |
| PTOAS | 上游工具链固定 v0.67；直接 BGEMM runtime 例子不使用 PTOAS |
| PTO-ISA | b1d460ee8c001585ede36b68ba3678dc5a226bbf |

完整可见 Python 包版本见 environment-observed.json 和 requirements-observed.txt。
这是观测清单，不是保证能跨系统直接安装的 lockfile；尤其 Torch 的 +cpu 版本需要
对应下载源，CANN Python 包还依赖本机安装。不要直接用清单覆盖系统 Python。

## 获取源码

```bash
git clone --recurse-submodules https://github.com/JJyun-ustc/PyPTO-tmr_partition.git
cd PyPTO-tmr_partition
git submodule status
```

runtime 指向本仓库 runtime-snapshot 分支的固定提交，保留版本校验。
libbacktrace/msgpack-c 指向公开上游的固定提交。PTO-ISA 由 runtime/pto_isa.pin 管理。

## 准备 Python 与工具链

在已安装匹配 Ascend 驱动和 CANN 的机器上，以实际路径加载 CANN：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install cmake==4.4.2 ninja==1.13.0 nanobind==2.15.0 scikit-build-core==1.0.3
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 pytest==9.1.1 pytest-xdist==3.8.0 ruff==0.14.8
source .claude/skills/testing/load-env.sh
```

先编译并安装 runtime 的 Python 绑定，再只构建 a2a3：

```bash
python -m pip install --no-build-isolation -e ./runtime -C build.targets=_task_interface
cd runtime
python simpler_setup/build_runtimes.py --platforms a2a3
cd ..
python -m pip install --no-build-isolation -e . -C build.tool-args="-j$PYPTO_BUILD_JOBS"
```

当前上游 runtime 构建驱动内部按 CPU 数选择并发，上限 32，且会并发构建目标。
它不会被标准 CMAKE_BUILD_PARALLEL_LEVEL 完整覆盖。资源受限时应审阅驱动，
或在首次配置后逐个使用 cmake --build <缓存目录> --parallel 指定并发；不要认为
导出 PYPTO_BUILD_JOBS 就限制了驱动的全部并发。上述首次安装步骤在发布副本中未重跑。

只跑官方 BGEMM 用例不需要编译上层 Python 图；需要完整图代码生成时，还应按
上游 docs/en/user/04-dev-container.md 配置独立 CPython 3.10 PTOAS 环境，使用
仓库 docker/scripts/pypto-install-ptoas.sh 安装固定版本并核对 SHA256。

## 正确性与计时

先确认物理卡与容器逻辑设备的映射、健康状态和当前进程。不要假定物理卡总是 4，
也不要把物理卡号直接当作 Python 的逻辑设备号。本仓库的架构检查默认查询物理卡 0，
若容器没有暴露它，应直接核对实际卡板卡信息和 CANN SoC 配置，保留该事实。

```bash
npu-smi info
npu-smi info -t board -i 4 -c 0
cd runtime
python examples/a2a3/tensormap_and_ringbuffer/benchmark_bgemm/reproduce_cluster_study.py \
  --device 0 --physical-card 4 --output ../official-results/reproduced-run
```

该脚本先检查架构，再对四个配对用例各运行一次 golden；之后按两批反序，
每批 15 轮执行，关闭诊断，保留全部 30 轮并汇总中位数和 p95。
它记录运行前后二进制 SHA256，拒绝覆盖已有输出。存在 task-submit 时，
每个硬件调用通过它执行；不存在则明确记录未加独占锁。它不自动修改 O 的位置。
若当前卡有其他任务，应另选空闲卡或使用队列服务，不应把竞争结果用于对比。

原始记录中二进制哈希只用于核对同一套计时期间是否发生变化。重新编译会改变
Git 构建戳和可能的二进制哈希，不能要求新构建与历史哈希逐字节相等。

## 测试与额外回归

```bash
cd runtime
cmake -S tests/ut/cpp -B tests/ut/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build tests/ut/cpp/build --target test_a2a3_tmr_cluster_partitioner \
  test_a2a3_tmr_scheduler_state test_a2a3_tmr_wiring --parallel 2
ctest --test-dir tests/ut/cpp/build \
  -R 'test_a2a3_tmr_(cluster_partitioner|scheduler_state|wiring)$' \
  --parallel 2 --output-on-failure
```

被测工作区此前通过 17+32+45=94 项测试。额外回归源代码已从原工作区外围目录
整理到 runtime/tests/st/a2a3/tensormap_and_ringbuffer/cluster_partition_study/，
包括自动多 block、sync-start 和 DUMMY；这些脚本通过标准 SceneTest 入口运行。
原来的机器绝对路径没有作为复现依赖继续使用。

上游 GitHub CI 文件原样保留；个人仓库没有上游的 self-hosted runner 和 secrets，
不代表全部上游 CI 可以直接运行。本次发布没有重新构建完整 PyPTO 或重跑 NPU，
保留的是已测源码快照和原始证据，额外发布脚本会单独进行静态检查。
