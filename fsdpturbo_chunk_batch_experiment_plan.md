# FSDPTurbo Chunk Batch Size 长跑实验计划

## 1. 实验背景

当前需要验证 `chunk batch size` 特性在 FSDPTurbo + Qwen3 30B 场景下的功能稳定性、性能影响和显存收益。

已知当前基础环境：

```text
模型：Qwen3 30B
机器：单机 8 卡 NPU
代码分支：chunk-batch-size
测试脚本：tests/system_tests/qwen3/test_qwen3.py
启动脚本：tests/system_tests/qwen3/run_test_qwen3.sh
dispatcher：eager
```

当前已经跑通过的基础配置：

```text
TENSOR_PARALLEL_SIZE=2
EXPERT_PARALLEL_SIZE=8
EXPERT_FULLY_SHARD_PARALLEL_SIZE=1
FULLY_SHARD_PARALLEL_SIZE=4
MAX_LENGTH=512
BATCH_SIZE=2
CHUNK_MBS=1 或 2
```

## 2. 实验目标

本次实验主要回答以下问题：

1. `ENABLE_CHUNK_BATCH=0/1` 两种模式是否都可以长时间稳定运行。
2. 开启 chunk batch 后，loss 是否保持正常，无 NaN/Inf，无明显异常波动。
3. 在相同 batch size 和 sequence length 下，step time 是否变化。
4. 在更大 batch size 或更长 sequence length 下，chunk batch 是否可以提升可跑上限。
5. profiler 是否可以正常采集，并能否用于解释性能和显存差异。

## 3. 指标口径

每组实验至少记录以下指标：

```text
配置
是否成功跑完
平均 step time
tokens/s
rank max peak memory
loss 起始值和结束值
是否出现 NaN/Inf
是否 OOM
异常日志
```

建议性能统计口径：

```text
NUM_STEPS=100
前 10 step 不计入平均耗时
统计 step 10-99 的平均 step time
```

tokens/s 计算方式：

```text
tokens/s = BATCH_SIZE * MAX_LENGTH / avg_step_time
```

如果后续确认脚本中的 `BATCH_SIZE` 是全局 batch，则该值可作为全局 tokens/s。若为单卡 batch，则需要再乘以数据并行维度。

显存口径：

```text
脚本打印的 max memory 来自 torch.npu.max_memory_allocated()
建议统计所有 rank 的 max memory，并取最大值
```

注意：`max_memory_allocated()` 只统计 PyTorch/NPU allocator 管理的 Tensor 显存，不一定包含 HCCL buffer、NPU runtime workspace、算子临时 workspace 等全部设备占用。必要时需要结合 profiler 或 npu-smi。

## 4. 阶段 0：脚本功能冒烟

目的：确认 main 已跑通后，特性分支的基本脚本也可以运行。

建议配置：

```text
S0: BS=2, SEQ=512, chunk off, NUM_STEPS=3
S1: BS=2, SEQ=512, chunk=2, NUM_STEPS=3
S2: BS=2, SEQ=512, chunk=1, NUM_STEPS=3
```

判断标准：

```text
训练可以完成
loss 可以正常打印
chunk on 时日志出现 Applying chunkmbs to module: model.layers.0
无 NaN/Inf
无 OOM
```

## 5. 阶段 0.5：Profiler 冒烟与可观测性验证

Profiler 不应混入正式性能长跑，因为 profiler 会改变 step time。它应该先作为独立验证项，确认能够采集到有效 trace。

建议配置：

```text
P0: BS=2, SEQ=512, chunk off, NUM_STEPS=8, ENABLE_PROFILER=1, PROFILE_MEMORY=1
P1: BS=2, SEQ=512, chunk=1, NUM_STEPS=8, ENABLE_PROFILER=1, PROFILE_MEMORY=1
```

建议输出目录：

```text
PROFILE_DIR=/home2/w00857719/FSDPTurbo/profile_smoke_off
PROFILE_DIR=/home2/w00857719/FSDPTurbo/profile_smoke_chunk1
```

判断标准：

```text
PROFILE_DIR 下生成 profiler trace 文件
tensorboard/profiler 可以打开
可以看到 NPU op 和 CPU op
PROFILE_MEMORY=1 时可以看到 memory 相关信息
chunk on 时可以观察到更多分段执行行为
```

当前代码中需要注意：

```python
schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=5000)
```

如果只跑几十或几百步，`skip_first=5000` 会导致基本采不到 trace。正式使用 profiler 前建议将其改成环境变量，或临时改成：

```python
schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=2, repeat=1, skip_first=2)
```

建议后续支持以下环境变量：

```text
PROFILE_SKIP_FIRST
PROFILE_WAIT
PROFILE_WARMUP
PROFILE_ACTIVE
PROFILE_REPEAT
```

## 6. 阶段 1：固定配置长跑，确认稳定性

目的：在默认可跑配置上验证 chunk on/off 的稳定性。

配置：

```text
A0: BS=2, SEQ=512, chunk off
A1: BS=2, SEQ=512, CHUNK_MBS=2
A2: BS=2, SEQ=512, CHUNK_MBS=1
```

统一设置：

```text
NUM_STEPS=100
ENABLE_PROFILER=0
PROFILE_MEMORY=0
```

判断标准：

```text
三组都可以稳定跑完
loss 无 NaN/Inf
step time 在跳过前 10 step 后相对稳定
显存峰值无异常增长
```

## 7. 阶段 2：固定 Sequence Length，扫描 Batch Size

目的：验证 chunk batch 是否可以提升 batch size 可跑上限。

固定参数：

```text
MAX_LENGTH=512
NUM_STEPS=100
ENABLE_PROFILER=0
```

实验矩阵：

```text
BS=2:
  chunk off
  CHUNK_MBS=2
  CHUNK_MBS=1

BS=4:
  chunk off
  CHUNK_MBS=2
  CHUNK_MBS=1

BS=6:
  chunk off
  CHUNK_MBS=2
  CHUNK_MBS=1

BS=8:
  chunk off
  CHUNK_MBS=4
  CHUNK_MBS=2
  CHUNK_MBS=1
```

重点观察：

```text
baseline 是否 OOM
chunk 是否能跑通 baseline OOM 的配置
相同 BS 下 chunk 对 step time 的影响
相同 BS 下 chunk 对 peak memory 的影响
```

如果 baseline OOM 但 chunk 能跑通，该结果是最直接的收益证明。

## 8. 阶段 3：固定 Batch Size，扫描 Sequence Length

目的：验证长序列下 chunk batch 对激活显存压力的影响。

优先配置：

```text
BATCH_SIZE=2
NUM_STEPS=100
ENABLE_PROFILER=0
```

实验矩阵：

```text
SEQ=512:
  chunk off
  CHUNK_MBS=1

SEQ=1024:
  chunk off
  CHUNK_MBS=1

SEQ=1536:
  chunk off
  CHUNK_MBS=1

SEQ=2048:
  chunk off
  CHUNK_MBS=1
```

如果 `BATCH_SIZE=2` 下差异不明显，可以继续尝试：

```text
BATCH_SIZE=4
SEQ=512 / 1024 / 1536
chunk off / CHUNK_MBS=2 / CHUNK_MBS=1
```

## 9. 阶段 4：关键配置 Profiler 深入采集

目的：对阶段 2/3 中有代表性的配置进行 profiler 解释。

不要对所有配置都开 profiler。建议只选：

```text
K0: 最大的 baseline 可跑配置，chunk off
K1: 与 K0 相同 BS/SEQ，chunk on
K2: baseline OOM 但 chunk 能跑的配置，chunk on
K3: step time 差异明显的配置，chunk off/on 各一组
```

建议设置：

```text
NUM_STEPS=8 或 10
ENABLE_PROFILER=1
PROFILE_MEMORY=1
```

profiler schedule 建议：

```text
skip_first=2
wait=1
warmup=1
active=2
repeat=1
```

观察重点：

```text
NPU 算子时间分布
MoE dispatcher 相关算子耗时
FSDP all-gather / reduce-scatter 相关耗时
chunk on/off 的 forward 分段行为
memory timeline 是否能看到峰值变化
```

## 10. 推荐优先执行的最小实验集

如果时间有限，优先跑以下 8 组：

```text
1. BS=2, SEQ=512, chunk off
2. BS=2, SEQ=512, CHUNK_MBS=2
3. BS=2, SEQ=512, CHUNK_MBS=1

4. BS=4, SEQ=512, chunk off
5. BS=4, SEQ=512, CHUNK_MBS=2
6. BS=4, SEQ=512, CHUNK_MBS=1

7. BS=2, SEQ=1024, chunk off
8. BS=2, SEQ=1024, CHUNK_MBS=1
```

执行建议：

```text
先不开 profiler
每组 NUM_STEPS=100
每组记录 step time、tokens/s、rank max memory、loss
若某组 OOM，记录 OOM 位置和日志
```

## 11. 启动命令模板

chunk off：

```bash
MODEL_PATH=/path/to/Qwen3-30B \
DATASET_DISK_PATH=/path/to/dataset \
BATCH_SIZE=2 \
MAX_LENGTH=512 \
NUM_STEPS=100 \
ENABLE_CHUNK_BATCH=0 \
ENABLE_PROFILER=0 \
bash tests/system_tests/qwen3/run_test_qwen3.sh
```

chunk on：

```bash
MODEL_PATH=/path/to/Qwen3-30B \
DATASET_DISK_PATH=/path/to/dataset \
BATCH_SIZE=2 \
MAX_LENGTH=512 \
NUM_STEPS=100 \
ENABLE_CHUNK_BATCH=1 \
CHUNK_MBS=1 \
ENABLE_PROFILER=0 \
bash tests/system_tests/qwen3/run_test_qwen3.sh
```

profiler 冒烟：

```bash
MODEL_PATH=/path/to/Qwen3-30B \
DATASET_DISK_PATH=/path/to/dataset \
BATCH_SIZE=2 \
MAX_LENGTH=512 \
NUM_STEPS=8 \
ENABLE_CHUNK_BATCH=1 \
CHUNK_MBS=1 \
ENABLE_PROFILER=1 \
PROFILE_MEMORY=1 \
PROFILE_DIR=/home2/w00857719/FSDPTurbo/profile_smoke_chunk1 \
bash tests/system_tests/qwen3/run_test_qwen3.sh
```

## 12. 实验记录模板

```text
实验编号：
分支/commit：
模型路径：
数据集路径：
TP / EP / EXPERT_FSDP / FSDP：
dispatcher：
BATCH_SIZE：
MAX_LENGTH：
ENABLE_CHUNK_BATCH：
CHUNK_MBS：
NUM_STEPS：
是否开启 profiler：

是否跑通：
失败原因：
平均 step time：
tokens/s：
rank max peak memory：
loss start：
loss end：
是否 NaN/Inf：
备注：
```

## 13. 结论判断标准

优先级从高到低：

1. baseline OOM，但 chunk 能跑通更大的 batch size 或 sequence length。
2. 相同配置下，chunk 的 rank max peak memory 更低。
3. 相同配置下，chunk 的 step time 不明显劣化，或有可解释的性能收益。
4. loss 曲线和 baseline 对齐，无 NaN/Inf。
5. profiler 可以解释 chunk on/off 的执行差异。

如果只看到 step time 差异，但看不到 `max_memory_allocated()` 差异，不能直接判定特性无效。需要结合更大 batch/sequence 的可跑上限实验，以及 profiler 或 npu-smi 的整卡显存观察。
