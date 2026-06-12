# TorchTitanTurbo + FSDPTurbo Qwen3.5 验证手册

## 1. 验证目标

按以下顺序验证，避免同时引入环境、模型规模和 FSDPTurbo 三类问题：

```text
环境检查
  -> 单卡 micro debugmodel
  -> 四卡 PyTorch FSDP baseline
  -> 四卡 FSDPTurbo custom FSDP
  -> 真实 Qwen3.5-35B-A3B
```

前三个阶段用于验证训练链路，不需要完整 C4 数据集，也不需要真实
Qwen3.5-35B-A3B 权重。

## 2. 数据和模型文件

### Debug 验证

TorchTitanTurbo 的 Qwen3.5 debug 配置使用：

```text
hf_assets_path = ./tests/assets/tokenizer
dataset = c4_test
```

需要：

- TorchTitan 源码仓中的 `tests/assets/tokenizer`。
- TorchTitan 提供的 `c4_test` 测试数据配置。

由于路径是相对于当前工作目录解析的，建议从 TorchTitan 源码根目录运行。

### 真实 35B 验证

真实配置使用：

```text
hf_assets_path = ./assets/hf/Qwen3.5-35B-A3B-Base
dataset = c4
```

需要：

- Qwen3.5-35B-A3B tokenizer 和模型 assets。
- C4 或其他与 TorchTitan dataloader 兼容的文本训练数据。
- 足够的 NPU 卡数和显存。

下载官方模型文件不等于已经准备好训练数据。模型权重和 C4 文本数据是两类不同资源。

## 3. 路径设置

按服务器实际位置修改：

```bash
export TORCHTITAN_ROOT=/path/to/torchtitan
export TORCHTITAN_TURBO_ROOT=/path/to/TorchTitanTurbo
export FSDP_TURBO_ROOT=/path/to/FSDPTurbo
```

切换到接入分支并安装：

```bash
git -C "${TORCHTITAN_TURBO_ROOT}" fetch origin
git -C "${TORCHTITAN_TURBO_ROOT}" switch feature/fsdpturbo-fsdp-integration

python -m pip install -e "${TORCHTITAN_ROOT}"
python -m pip install -e "${FSDP_TURBO_ROOT}"
python -m pip install -e "${TORCHTITAN_TURBO_ROOT}"
```

## 4. 环境检查

```bash
python - <<'PY'
import torch
import torch_npu
import torchtitan
import torchtitanturbo
import fsdp_turbo

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("torchtitan:", torchtitan.__file__)
print("torchtitanturbo:", torchtitanturbo.__file__)
print("fsdp_turbo:", fsdp_turbo.__file__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())

from torch.nn.attention.varlen import varlen_attn
from fsdp_turbo.distributed.fine_grained_fully_shard import get_fsdp_strategy

print("varlen_attn:", varlen_attn)
print("custom strategy:", type(get_fsdp_strategy("custom")).__name__)
PY
```

成功标准：

```text
所有模块导入成功
torch.npu.is_available() 为 True
NPU 数量符合预期
能够导入 varlen_attn
能够创建 CustomFSDPStrategy
```

## 5. 检查测试 tokenizer

```bash
test -d "${TORCHTITAN_ROOT}/tests/assets/tokenizer" \
  && echo "tokenizer assets found" \
  || echo "tokenizer assets missing"
```

如果目录不存在，应使用与当前 TorchTitanTurbo 匹配的完整 TorchTitan 源码 checkout，
不要继续在不完整的 wheel 环境中运行 debug 配置。

创建日志目录：

```bash
mkdir -p "${TORCHTITAN_ROOT}/logs"
cd "${TORCHTITAN_ROOT}"
```

## 6. 单卡 Micro Debug Baseline

```bash
bash "${TORCHTITAN_TURBO_ROOT}/examples/train_qwen3_5_npu.sh" \
  qwen3_5_35b_a3b_micro_debugmodel \
  2 \
  2>&1 | tee logs/qwen35-micro-baseline.log
```

该配置是随机初始化的微型 MoE 模型，主要验证：

```text
数据加载
模型构建
forward
backward
optimizer step
```

成功标准：

```text
完成 2 step
loss 不是 NaN/Inf
日志最后出现 Training completed
```

## 7. 四卡 PyTorch FSDP Baseline

```bash
NGPU=4 bash "${TORCHTITAN_TURBO_ROOT}/examples/train_qwen3_5_npu.sh" \
  qwen3_5_35b_a3b_debugmodel \
  2 \
  --training.seq_len 64 \
  --parallelism.data_parallel_shard_degree 4 \
  2>&1 | tee logs/qwen35-native-fsdp.log
```

成功标准：

```text
NPUs: 4
FSDP backend: torch
日志出现 Applied Qwen3.5 FSDP
完成 2 step
loss 不是 NaN/Inf
```

## 8. 四卡 FSDPTurbo Custom FSDP

保持模型、卡数、序列长度和 FSDP degree 不变，只切换 backend：

```bash
TORCHTITAN_TURBO_FSDP_BACKEND=fsdp_turbo_custom \
NGPU=4 bash "${TORCHTITAN_TURBO_ROOT}/examples/train_qwen3_5_npu.sh" \
  qwen3_5_35b_a3b_debugmodel \
  2 \
  --training.seq_len 64 \
  --parallelism.data_parallel_shard_degree 4 \
  2>&1 | tee logs/qwen35-fsdpturbo.log
```

成功标准：

```text
FSDP backend: fsdp_turbo_custom
日志出现 Applying FSDP backend=fsdp_turbo_custom
完成 forward、backward 和 optimizer step
完成 2 step
loss 不是 NaN/Inf
```

到此可以说明：

```text
TorchTitanTurbo + FSDPTurbo 已在 Qwen3.5 MoE debugmodel 上初步打通。
```

这还不能证明真实 35B 权重加载、checkpoint 或性能优化已经验证完成。

## 9. 失败时保存的信息

每次运行至少记录：

```text
完整命令
torch、torch_npu、TorchTitan 版本
代码分支和 commit
卡数和并行配置
第一处 traceback
最后一个成功执行的阶段
```

只处理第一处真实错误，不要同时修改多个配置或功能。

## 10. 真实 Qwen3.5-35B-A3B 前置检查

debugmodel 打通后，再确认：

```text
模型 assets 路径正确
官方权重 shape 与 TorchTitanTurbo 的 _35b_a3b() 配置一致
state-dict adapter 能映射文本 backbone 权重
视觉和 MTP 权重如何处理
初始 checkpoint 的加载方式
训练数据下载和缓存位置
所需卡数与显存预算
```

真实模型配置默认使用：

```text
local_batch_size = 1
seq_len = 4096
activation checkpoint = full
```

不要在 debug baseline 之前直接运行真实 35B 模型。
