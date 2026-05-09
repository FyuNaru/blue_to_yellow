# MindSpeed Lite Chunk Batch Size 需求分析文档

## 1. 背景

当前需求来自 FSDP2 大模型训练场景。FSDP2 可以把参数 shard 到多张卡上，降低常驻参数显存，但每个 transformer block 执行时仍需要 all-gather 当前 block 的完整参数。对于参数量大、hidden size 宽、MoE block 较重的模型，单个 block 执行时的显存峰值可能由以下几部分叠加：

```text
当前 block all-gather 参数
+ 当前 batch 的激活
+ 通信 buffer / copy 临时空间
+ 框架缓存
```

当这些开销叠加后，留给激活的空间不足，训练就很难继续提高 batch size 或 sequence length。`chunk batch size` 要解决的就是这个问题：**不要让指定模块一次处理完整 batch，而是在模块内部按 batch 维切成多个 chunk 依次执行，再把输出拼回去。**

## 2. 需求目标

该需求要实现的是模块级 forward chunking，而不是 dataloader batch 切分，也不是梯度累积。

例如外层 batch size 是 8，`chunk_mbs=2`：

```text
原始执行：
block(input[0:8]) -> output[0:8]

chunk 执行：
block(input[0:2]) -> output[0:2]
block(input[2:4]) -> output[2:4]
block(input[4:6]) -> output[4:6]
block(input[6:8]) -> output[6:8]
cat(outputs)      -> output[0:8]
```

对训练主循环来说，batch size、loss、backward、optimizer step 都不应该变化；变化只发生在被选中的模块内部。

## 3. 开发时需要先想清楚的问题

### 3.1 包在哪一层

这个功能不适合默认包整个模型，也不适合一开始就包到很细的 attention/MLP 内部。比较合理的第一版粒度是 transformer block，例如：

```text
model.layers.{*}
```

原因是 block 通常也是 FSDP、recompute 的常见包装粒度，输入输出结构相对稳定，收益也更容易观察。

### 3.2 哪些输入能切

不能递归切所有 tensor。模型 forward 里有些 tensor 不一定是 batch 语义，例如 `position_ids`、`cache_position`、`past_key_values`、某些 attention mask 或 MoE metadata。切错会直接导致 shape 错误，或者更麻烦的精度不一致。

因此需求里应该让用户显式指定切哪些参数：

```python
chunk_arg_indexs=[0]
chunk_kwarg_names=[]
batch_dim=0
```

第一版不要做过度自动推断，宁可配置麻烦一点，也要避免静默切错。

### 3.3 输出怎么拼

最基本要支持：

- `Tensor`：按 `batch_dim` 做 `torch.cat`。
- `tuple/list`：逐项拼接。

复杂输出对象不建议第一版承诺全支持。遇到不能拼的结构应该直接报错，而不是返回一个不确定的结果。

### 3.4 和 recompute / FSDP 的顺序

原始需求里明确说 wrapper 要在“重计算外面、fully_shard 里面”。落到 MindSpeed Lite 里，顺序应理解为：

```text
TP / EP -> recompute -> chunk_batch -> FSDP
```

也就是逻辑嵌套为：

```text
FSDP( ChunkBatch( Recompute( original_forward ) ) )
```

这个顺序的含义是：每个 chunk 进入的是 recompute 包装后的 forward，同时最终模块仍由 FSDP 管理。

## 4. 功能边界

本需求应该包含：

- 通过配置开关启用/关闭。
- 配置 `chunk_mbs`。
- 配置目标模块匹配规则。
- 配置 `batch_dim`。
- 配置需要切分的 `args` 下标和 `kwargs` 名称。
- batch size 不能整除 `chunk_mbs` 时，最后一个 chunk 正常处理剩余样本。
- 匹配不到模块、找不到输入、输出不能拼接时明确报错。

本需求不应该包含：

- 不实现每个 chunk forward 后立即 backward。
- 不替代梯度累积。
- 不自动选择最优 `chunk_mbs`。
- 不承诺所有模型都能降低峰值显存。
- 不承诺支持任意复杂输出对象。

## 5. 为什么它可能降低显存

它降低的是目标模块单次 forward 的激活工作集。原来 block 一次处理完整 batch，现在每次只处理一个 chunk，模块内部临时 tensor 和激活峰值可能下降。

但它不是万能的。如果峰值主要来自以下部分，收益可能不明显：

- FSDP all-gather 后的参数。
- optimizer state。
- 通信 buffer。
- 框架缓存。
- 其它不随 batch 线性变化的临时空间。

所以验证时不能只看“开了功能是否训练成功”，还要看峰值显存来源是否真的和目标模块激活有关。

## 6. 验收标准

功能正确性：

- 开启后能看到目标模块被应用 chunk wrapper。
- `batch_size > chunk_mbs` 时确实拆成多个 chunk。
- `batch_size <= chunk_mbs` 时走原始 forward。
- 输出 shape 与不开启功能一致。
- loss 可以正常计算，backward 和 optimizer step 可以正常执行。

精度对齐：

- 小模型或短序列下，对比 `chunk_batch=False` 和 `chunk_batch=True`。
- 固定随机种子，尽量关闭 dropout 等随机行为。
- 对比 loss、输出 shape、梯度是否在可接受误差内。

显存对比：

- 每步前重置峰值统计，例如 `torch.npu.reset_peak_memory_stats()`。
- 对比 `chunk_batch=False`、`chunk_mbs=batch_size`、`chunk_mbs=batch_size/2`、`chunk_mbs=1`。
- 如果峰值没降，需要进一步确认峰值是否主要来自参数或通信，而不是激活。

## 7. 实现拆解

建议拆成四块：

```text
配置层：
MindSpeedLiteConfig 增加 chunk_batch 和 chunk_batch_plan。

模块选择：
根据 apply_modules 匹配目标模块，例如 model.layers.{*}。

forward wrapper：
推断 batch size，按 chunk_mbs 切分输入，逐 chunk 调原 forward，最后拼接输出。

主流程接入：
放在 recompute 之后、FSDP 之前。
```

核心逻辑可以概括为：

```python
for start in range(0, full_batch_size, chunk_mbs):
    end = min(start + chunk_mbs, full_batch_size)
    micro_args = slice_selected_args(args, start, end)
    micro_kwargs = slice_selected_kwargs(kwargs, start, end)
    outputs.append(forward_func(*micro_args, **micro_kwargs))

return concat_outputs(outputs)
```

## 8. 结论

这个需求的核心判断很简单：**在不改变外层训练语义的前提下，把指定模块的一次大 batch forward 拆成多次小 batch forward，用更多计算调度开销换更低的模块内激活峰值。**

开发时最需要把握三个点：

- 只切有 batch 语义的输入。
- 输出必须能可靠拼回原结构。
- 包装顺序必须在 recompute 外、FSDP 内。

如果这三点成立，这个功能就可以作为 MindSpeed Lite 中一个相对独立的显存优化能力接入。
