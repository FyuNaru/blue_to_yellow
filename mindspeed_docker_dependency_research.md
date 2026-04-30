# MindSpeed Core Docker 镜像依赖调研与改进建议

## 依赖识别与合理性分析

### 业务依赖

| 依赖 | 当前版本/来源 | 引入原因 | 合理性 | 是否建议消除 |
| --- | --- | --- | --- | --- |
| CANN 基础镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11`，可切换 `a3/910b`、`openeuler24.03/ubuntu22.04` | 提供 Ascend CANN、Python 环境和 NPU 运行基础 | 必要依赖，MindSpeed 在 NPU 上运行必须依赖 CANN/Ascend 运行环境 | 不建议消除；建议只收敛支持矩阵并固化已验证版本 |
| Ascend 驱动/工具挂载 | `/usr/local/Ascend/driver`、`/usr/local/dcmi`、`npu-smi`、`/etc/ascend_install.info` | 容器访问宿主机 NPU 驱动、设备管理能力 | 必要依赖，镜像内不应内置宿主机驱动 | 不建议消除；建议文档明确宿主机驱动版本要求 |
| MindSpeed | 默认 `master`，可通过 `--mindspeed-branch` 指定 | 核心业务代码 | 必要依赖 | 不消除；建议分支/版本与镜像 tag 对齐 |
| Megatron-LM | 默认 `core_v0.12.1` | MindSpeed Core 训练依赖 Megatron-LM | 必要依赖 | 不消除；建议固定到已适配分支 |
| PyTorch | `torch==2.7.1` | 深度学习框架 | 必要依赖 | 不消除；需要与 `torch_npu`、CANN 版本保持兼容 |
| torch_npu | `torch_npu==2.7.1` | Ascend NPU 上运行 PyTorch | 必要依赖 | 不消除；需与 PyTorch/CANN 版本匹配 |
| transformers | `transformers==4.57.1` | 模型与训练脚本依赖，当前实测发现版本相关问题 | 合理，已固定版本避免漂移 | 不建议消除；建议继续固定版本 |
| MindSpeed `requirements.txt` | `numpy<=1.26.0`、`protobuf`、`sentencepiece`、`einops`、`scipy`、`pandas`、`scikit-learn`、`SQLAlchemy` 等 | MindSpeed 运行、数据处理、模型配置、测试等 Python 依赖 | 合理，但其中部分可能偏开发/测试用途 | 不整体消除；建议后续拆分 runtime/dev/test requirements |

### 构建工具依赖

| 依赖 | 引入原因 | 合理性 | 消除建议 |
| --- | --- | --- | --- |
| Docker | 构建和运行容器镜像 | 必要 | 不消除；需补充最低 Docker 版本，当前代码未约束，需确认 |
| Bash | `build.sh`、repo 配置脚本执行 | 必要 | 不消除 |
| git | 构建期 clone MindSpeed 和 Megatron-LM | 当前必要 | 可改进为构建上下文 COPY 或子模块/源码包，减少构建期网络依赖 |
| gcc/g++/make/build-essential/gcc-c++ | 编译 Python 扩展、源码安装依赖 | 合理 | 若产物运行期不需要，可通过多阶段构建消除运行镜像中的编译工具 |
| cmake | 部分依赖编译需要 | 合理 | 同上，可移到 builder stage |
| ninja/pybind11/wheel/setuptools/pip/packaging | Python 包构建和安装 | 合理 | runtime 可保留 pip，构建工具可后续拆分 |
| curl/wget/ca-certificates | 配置源、下载依赖、HTTPS 访问 | 合理 | 可保留；如严格瘦身可减少重复工具 |
| jq | 通用 JSON 处理工具，当前 Dockerfile 安装但未直接使用 | 合理性偏弱 | 可考虑移除，除非后续脚本需要 |
| apt/yum 源配置脚本 | 切换到华为云源，提升构建稳定性 | 合理 | 不消除；建议说明依赖外网/内网镜像源 |

### 其它依赖

| 依赖 | 引入原因 | 合理性 | 消除建议 |
| --- | --- | --- | --- |
| vim | 容器内调试 | 便利性依赖，非运行必要 | 生产镜像可移除，开发镜像保留 |
| pytest/pytest-mock | 测试依赖 | 开发/验证合理，运行期非必要 | 建议后续拆分 dev/test 镜像或 requirements |
| `--privileged`、`--network host`、`--ipc=host` | NPU 访问、网络通信、训练共享内存 | 训练开发场景合理 | 可研究最小权限启动参数，但短期不建议贸然收敛 |
| 华为云 apt/yum repo | 安装系统包 | 合理 | 需确认生产环境是否允许访问；可提供内网源替换能力 |

## 镜像依赖调研和改进建议

### 1. 当前基础操作系统和容器软件的版本约束

当前镜像基于 Ascend CANN 基础镜像构建，默认基础镜像为：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11
```

当前默认约束如下：

```text
基础操作系统：openEuler 24.03
可选操作系统：Ubuntu 22.04、openEuler 24.03
CANN 基础镜像版本：9.0.0-beta.2
NPU 类型：默认 910b，可选 a3/910b
Python：3.11
CPU 架构：支持 x86_64、aarch64
PyTorch：2.7.1
torch_npu：2.7.1
Megatron-LM：core_v0.12.1
transformers：4.57.1
镜像名称：mindspeed-core
默认 tag：master-910b-openeuler24.03-py3.11-aarch64
```

需确认：

```text
Docker Engine / containerd / 宿主机驱动版本当前没有在仓库中显式约束，需要补充实际验证环境版本。
```

### 2. 系统当前提供的镜像现状

当前系统提供源码构建形式的 Docker 镜像方案，包含：

```text
docker/Dockerfile
docker/build.sh
docker/configure_apt_repo.sh
docker/configure_yum_repo.sh
docker/OVERVIEW.md
docker/OVERVIEW.zh.md
```

镜像通过 `docker/build.sh` 构建，支持按如下维度选择基础镜像：

```text
NPU 类型：a3 / 910b
操作系统：openeuler24.03 / ubuntu22.04
CANN 基础镜像版本：默认 9.0.0-beta.2
Python 版本：默认 3.11
MindSpeed 分支：默认 master
Megatron-LM 分支：默认 core_v0.12.1
```

需确认：

```text
当前是否已经对外发布预构建镜像到镜像仓库；从代码看，目前主要是提供 Dockerfile 和 build.sh，本地构建得到 mindspeed-core 镜像。
```

### 3. 系统当前验证的操作系统及版本现状

建议谨慎填写：

```text
代码层面当前支持 openEuler 24.03 和 Ubuntu 22.04 两类 CANN 基础镜像。
默认验证路径为 openEuler 24.03 + CANN 9.0.0-beta.2 + 910b + Python 3.11 + aarch64。
Ubuntu 22.04 和 a3 路径已提供构建参数入口，但实际完整训练验证情况需进一步确认。
```

需确认：

```text
本地实测成功的镜像是否为 openEuler 24.03 + 910b + aarch64。
是否已经在 Ubuntu 22.04、a3、x86_64 上完成构建或训练验证。
```

### 4. 系统对操作系统的依赖分析

```text
MindSpeed 镜像对操作系统的直接依赖主要体现在系统包管理器、系统编译工具链、NUMA 库、网络源配置和 CANN 基础镜像标签格式上。

业务运行本身主要依赖 Ascend CANN、torch_npu、PyTorch、MindSpeed 和 Megatron-LM。操作系统更多承担基础运行环境、包安装、编译和调试能力。当前 Dockerfile 已通过 OS_FAMILY 抽象 Ubuntu/openEuler 差异：Ubuntu 使用 apt，openEuler 使用 yum；两者安装的核心能力基本一致，包括 gcc/g++、make/build-essential、cmake、git、curl/wget、numa 开发库等。

因此，系统不是强绑定某一个 Linux 发行版，但强依赖 CANN 官方基础镜像中已经验证过的 OS 组合。随意切换到未验证 OS 可能导致 CANN、torch_npu、系统包源、编译工具链或运行库兼容性问题。
```

### 5. 系统对操作系统验证/镜像的改进策略

```text
后续建议将操作系统支持从“可配置”进一步收敛为明确的支持矩阵，并对每个组合给出验证状态。建议优先维护默认组合：openEuler 24.03 + CANN 9.0.0-beta.2 + 910b + Python 3.11 + aarch64。

改进策略包括：

1. 建立镜像支持矩阵，明确 OS、CANN、NPU 类型、CPU 架构、Python、PyTorch、torch_npu、Megatron-LM 的组合关系。
2. 对默认组合进行完整构建验证、import 验证、npu-smi 验证和最小训练任务验证。
3. 对 Ubuntu 22.04、a3、x86_64 等非默认组合标注为“支持构建”或“已完成训练验证”，避免文档歧义。
4. 将构建依赖和运行依赖拆分，后续可引入多阶段构建，减少 gcc、cmake、vim、pytest 等非运行必要依赖进入最终镜像。
5. 固定关键 Python 依赖版本，尤其是 PyTorch、torch_npu、transformers、Megatron-LM，降低构建结果漂移风险。
6. 减少构建期网络不确定性，例如将 MindSpeed/Megatron-LM ref 固定到 tag/commit，必要时支持源码包或内部镜像源。
7. 补充容器运行时要求，包括 Docker 版本、宿主机 Ascend 驱动版本、CANN/driver 兼容关系、NPU 设备挂载和最小权限运行参数。
```

## 待确认信息

```text
1. Docker Engine / containerd 的最低版本或实际验证版本
2. 宿主机 Ascend 驱动版本
3. 是否有正式发布的预构建镜像仓库地址
4. Ubuntu 22.04 是否完成真实构建/训练验证
5. a3 是否完成真实构建/训练验证
6. x86_64 是否完成真实构建/训练验证
7. 默认镜像最终对应 MindSpeed master 还是 26.0.0_core_r0.12.1 分支
```
