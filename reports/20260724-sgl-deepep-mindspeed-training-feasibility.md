# SGL DeepEP 用于 MindSpeed MoE 训练的阶段性调研报告

> - 任务提出时间：2026-07-20
> - 报告时间：2026-07-24
> - 当前阶段：源码调研与方案对齐，尚未完成 NPU 实机验证

## 1. 摘要

本次任务源于以下需求：

> SGL 中已经有面向 Prefill 的 DeepEP 融合实现和算子代码，需要评估训练是否能够直接使用，或者需要进行改造。

经过对 `sgl-kernel-npu`、Megatron-LM、MindSpeed 和 vLLM-Ascend 相关代码的初步调研，当前判断如下：

1. 训练侧第一阶段应重点评估 DeepEP **normal dispatch/combine**，而不是直接接入完整的 `fused_deep_moe`。
2. DeepEP normal 可以覆盖 MindSpeed 当前 MoE 链路中的 token 重排、Expert Parallel 通信和结果归并，具备替换 `permute + AllToAllV + unpermute` 的潜力。
3. 当前 `deep_ep` 包主要提供正向接口，没有注册训练所需的 autograd/backward。直接 `import deep_ep` 可以完成安装和正向冒烟验证，但不能据此认定训练可以直接使用。
4. 从数学关系看，反向可以复用 dispatch/combine 两类能力：正向 combine 的反向需要 dispatch，正向 dispatch 的反向需要 combine；但 Router 权重梯度、通信句柄、反向加权和多流调度仍需训练侧适配。
5. 当前代码中的 normal 算子主要使用通信及 Vector 侧资源，不包含 Expert GMM，具备与 GMM/FA 并行的硬件基础；但当前 Python/C++ 适配层没有提供可直接用于训练 overlap 的完整 event/autograd 机制。
6. 建议采用“先依赖 wheel 做 BF16 正向 POC，再补最小反向，最后接入 MindSpeed overlap”的渐进路线。是否最终将算子代码迁入训练仓，需要根据 POC 结果和维护边界再决定。

本次组会希望优先与领导对齐：目标硬件和模型规格、第一阶段交付边界、是否允许依赖独立 `deep_ep.whl`、是否要求首版即支持反向 overlap。

### 1.1 组会口头汇报建议

可以用下面这段话开场：

> 这四天我先把 SGL DeepEP 的 normal、low-latency 和 fused 三条路径区分开了，并顺着源码追到了 Python、C++ 和昇腾算子。当前结论是，训练最值得验证的是 normal dispatch/combine：它可以独立打成 `deep_ep.whl`，正向有希望替换 MindSpeed 现在的 token 重排和 AllToAllV；但现有包没有训练 autograd，normal 的 event 接口也不足以直接完成 overlap。因此我建议先做 BF16 非 overlap 的 forward POC，再按 combine 反向复用 dispatch、dispatch 反向复用 combine 的关系补最小 backward，最后才进入 MindSpeed overlap。今天主要想对齐首测硬件和模型规格，以及第一版是只交可行性验证，还是需要交付可训练代码。

如果只讨论一个结论：

> **不是“import 后直接训练”，而是“正向算子可直接验证，训练语义和 overlap 需要分阶段适配”。**

## 2. 背景介绍

### 2.1 MoE 为什么需要通信

普通 Transformer 层中的 FFN 只有一套权重；MoE 将 FFN 替换为多套 Expert FFN，并由 Router 为每个 token 选择 Top-K Expert。

当不同 Expert 分布在不同 NPU 上时，一个 token 可能需要被发送到另一张 NPU 执行 Expert 计算。因此 MoE 层会增加如下过程：

```text
输入 token
  → Router 选择 Expert
  → Dispatch：按 Expert 重排并跨卡发送 token
  → 本地 Expert FFN
  → Combine：将 Expert 输出发回原卡并按 Router 权重归并
  → 输出 token
```

将 Expert 分布在不同设备上的并行方式称为 Expert Parallel，简称 EP。

### 2.2 当前 MindSpeed 的通用实现

MindSpeed 当前基于 Megatron 的 AllToAll token dispatcher，主要流程为：

```text
本地 permute
  → AllToAllV（发送 token 和 Router 概率）
  → 按本地 Expert 再排序
  → Expert FFN
  → 撤销本地 Expert 排序
  → AllToAllV（将 Expert 输出发回）
  → unpermute + Router 权重归并
```

AllToAllV 是通用的变长集合通信。它负责跨卡传输，但 token 如何重排、如何按 Expert 组织、如何恢复原始顺序，主要由训练框架完成。

### 2.3 DeepEP 的定位

DeepEP 是面向 MoE Expert Parallel 的专用通信方案。`sgl-kernel-npu` 中的 DeepEP-Ascend 是其昇腾实现，将 MoE token 布局、重排和通信放入专用算子中。

`sgl-kernel-npu` 虽然属于 SGLang 生态，但 DeepEP 会单独生成 `deep_ep.whl`，可以独立安装，不需要将整个 SGLang 推理框架引入 MindSpeed。

### 2.4 Normal、Low-Latency 和 Fused

| 模式 | 主要场景 | 处理范围 |
| --- | --- | --- |
| Normal | Prefill、大 token batch、训练候选 | 独立 dispatch/combine，Expert 计算保留在框架侧 |
| Low-Latency | Decode、小 token batch | 低延迟 dispatch/combine |
| Fused Deep MoE | 推理融合 | dispatch + GMM1 + SwiGLU + GMM2 + combine |

对话中的 “MegaMoE / fuseddeepep” 对应完整融合方向。该路径将 Expert FFN 也放进算子，更偏向推理；训练需要权重梯度、输入梯度和 Router 梯度，因此第一阶段更适合评估 normal。

### 2.5 “通信计算掩盖”是什么

通信计算掩盖是让通信和计算同时执行。例如通信耗时 5 ms、GMM 耗时 8 ms：

```text
串行：5 ms + 8 ms = 13 ms
并行：总耗时理想情况下接近 8 ms
```

Normal dispatch/combine 本身不执行 Expert GMM，因此具备与 Cube 侧 GMM/FA 并行的资源基础。但真正实现掩盖，还需要训练框架管理通信流、计算流、event 依赖和 tensor 生命周期。

## 3. 调研范围与代码基线

### 3.1 已检查代码仓

| 代码仓 | 本地基线 | 调研目的 |
| --- | --- | --- |
| `sgl-kernel-npu` | `main@d6617476`，2026-07-23 | DeepEP Python API、C++ 适配层、normal/fused 算子及构建方式 |
| Megatron-LM | `core_r0.16.0@ddc0d677` | MindSpeed 所依赖的 MoE AllToAll dispatcher 和普通 autograd |
| MindSpeed | `feat/core-r016-verl-qwen3-system-tests@ef86bb23` | 昇腾 MoE overlap、自定义 backward 和 feature patch |
| vLLM-Ascend | `releases/v0.18.0@bfec76c8` | 对比推理侧 EP 调用路径，避免将相似功能误认为同一算子 |

当前 MindSpeed 工作区存在与本任务无关的本地修改，本次仅做只读调研，后续实现应使用独立 worktree 和确认后的目标分支。

### 3.2 版本变化提示

任务于 2026-07-20 提出，当前 `sgl-kernel-npu` 主分支在任务提出后仍有更新：

- 2026-07-22：补充 A5 normal per-token FP8。
- 2026-07-22：补充 A5 fused Deep MoE。
- 2026-07-23：重写 DeepEP 和 fused MoE 文档。

因此，对话中“normal 尚未支持 A5”属于当时状态；当前代码已经提供 A5/CANN 9.0 构建路径。后续验证必须固定 commit、CANN 和硬件版本，避免调研期间基线漂移。

## 4. `sgl-kernel-npu` 中 DeepEP 的实现和调用流程

### 4.1 代码结构

```text
python/deep_ep/
  deep_ep/buffer.py                  # 用户主入口 Buffer
  deep_ep/strategies/                # normal / low-latency 策略
  setup.py                           # 生成 deep_ep wheel

csrc/deepep/
  pybind_extension.cpp               # Python 与 C++ 的绑定
  deep_ep.cpp                         # C++ 适配层、输出分配和算子调用
  ops/                                # A3/A5 等平台算子
  ops2/                               # A2 平台相关算子

tests/python/deepep/                  # normal、low-latency、fused 测试
```

构建脚本会分别生成：

```text
deep_ep*.whl
sgl_kernel_npu*.whl
```

因此 DeepEP 与 SGLang 运行时是可拆分的。

### 4.2 Normal dispatch 调用链

当前默认 normal 路径可以概括为：

```text
deep_ep.Buffer.get_dispatch_layout()
  → DefaultNormalCommStrategy.get_dispatch_layout()
  → deep_ep_cpp.Buffer.get_dispatch_layout()
  → aclnnDispatchLayout

deep_ep.Buffer.dispatch()
  → DefaultNormalCommStrategy.dispatch()
  → deep_ep_cpp.Buffer.intranode_dispatch()
  → aclnnNotifyDispatch
  → aclnnCamMoeDispatchNormal
```

各阶段职责：

1. `get_dispatch_layout` 根据 Top-K Expert 结果计算每个 rank、每个 Expert 的 token 数及布局信息。
2. `aclnnNotifyDispatch` 交换通信所需的 token 计数和 offset 元数据。
3. `aclnnCamMoeDispatchNormal` 完成 token 重排、跨 rank 搬运以及可选量化。
4. Python 接口返回按本地 Expert 排列的 token、每个本地 Expert 的 token 数，以及后续 combine 所需的 handle。

### 4.3 Normal combine 调用链

```text
deep_ep.Buffer.combine()
  → DefaultNormalCommStrategy.combine()
  → deep_ep_cpp.Buffer.intranode_combine()
  → aclnnCamMoeCombineNormal
```

Combine 使用 dispatch 返回的索引和通信信息，将 Expert 输出发回 token 原始 rank，并根据 Router 权重完成归并。

### 4.4 Fused Deep MoE 调用链

```text
deep_ep.Buffer.fused_deep_moe()
  → deep_ep_cpp.Buffer.fused_deep_moe()
  → aclnnFusedDeepMoe

或：

deep_ep.Buffer.fused_deep_moe(fuse_mode=DISPATCH_FFN_COMBINE)
  → deep_ep_cpp.Buffer.dispatch_ffn_combine()
  → aclnnDispatchFFNCombine
```

完整融合路径包含：

```text
Routing/Dispatch
  → GMM1
  → Dequant/SwiGLU/Quant
  → GMM2
  → Dequant/Combine
```

当前 fused 测试中的 Expert 权重设置为 `requires_grad=False`，代码中未提供训练 backward。因此它可以作为后续更长期的训练融合方向，但不适合作为第一阶段“能否直接训练”的切入点。

### 4.5 当前源码确认的关键差距

#### 1. 没有 autograd/backward

`deep_ep.Buffer` 和 `deep_ep_cpp.Buffer` 仅暴露 forward 风格的 dispatch/combine/fused 接口，没有 `torch.autograd.Function` 或 backward 注册。

这意味着直接执行：

```python
output = deep_ep_buffer.combine(...)
loss = output.sum()
loss.backward()
```

不能自动得到完整的输入、Expert 权重和 Router 梯度。

#### 2. 当前 normal event 接口不能直接承担训练 overlap

Python API 保留了 `previous_event`、`async_finish` 和 `EventOverlap` 等兼容参数，但在当前 commit 的 default normal 路径中：

- `EventOverlap.current_stream_wait()` 为空操作。
- C++ `EventHandle.current_stream_wait()` 为空操作。
- normal 路径返回的 event 为 `nullopt`。

因此不能仅通过设置 `async_finish=True` 就认定通信已经能够与 MindSpeed GMM/FA 正确掩盖。需要进一步确认是否通过训练侧外部 stream/event 调度即可满足，还是必须修改 DeepEP adapter。

#### 3. Forward handle 与反向加权需要专门设计

Forward combine 会使用 dispatch 保存的 Router 权重。在反向中：

- combine 的反向需要将输出梯度 dispatch 到各个 Expert，并乘对应 Router 权重。
- dispatch 的反向需要将各 Expert 的输入梯度 combine 回原 token，但此时不能重复乘 Router 权重。

因此不能机械地把 forward handle 原样复用到全部反向阶段，需要设计反向 handle、权重和索引的保存方式。

#### 4. 规格需要逐项对齐

需要确认：

- A3/A5 及 CANN 版本。
- 单机 HCCS、跨机 HCCS/RDMA 拓扑。
- EP size、Expert 数、每卡本地 Expert 数。
- hidden size、Top-K、每卡 token 数。
- BF16/INT8/FP8 通信模式。
- shared expert、EPLB、token padding/drop。
- MindSpeed GMM 所需的 `tokens_per_expert` 是计数还是前缀和。

## 5. MindSpeed 当前实现与反向机制

### 5.1 普通 AllToAll dispatcher

Megatron `MoEAlltoAllTokenDispatcher` 的主流程是：

```text
dispatch_preprocess:
  routing_map/probs
  → 计算 input_splits/output_splits
  → permute

token_dispatch:
  → AllToAllV(tokens)
  → AllToAllV(probs)

dispatch_postprocess:
  → 按本地 Expert 再排序

experts:
  → GMM1/SwiGLU/GMM2

combine_preprocess:
  → 撤销本地 Expert 排序

token_combine:
  → AllToAllV(expert outputs)

combine_postprocess:
  → unpermute
  → Router 权重归并
```

当前 `token_combine` 的 docstring 中出现了 “DeepEP kernels” 描述，但实际代码仍调用 `all_to_all(...)`，当前 checkout 尚未真正接入 DeepEP。

### 5.2 普通 AllToAll 的 backward

Megatron 将 AllToAll 封装为 `_AllToAll(torch.autograd.Function)`：

```text
forward:
  AllToAll(input, output_splits, input_splits)

backward:
  AllToAll(grad_output, input_splits, output_splits)
```

反向仍然使用同一种 AllToAll 通信，只是交换发送和接收 split，使梯度沿原通信路径返回。

普通 MoE 路径中的 permute、Expert FFN、unpermute 和 Router 权重计算由 PyTorch autograd 串联，不需要 MoE 层手工实现完整 backward。

### 5.3 MindSpeed overlap 路径

开启 `--moe-alltoall-overlap-comm` 后，MindSpeed 会使用 `MoELayerOverlapAllToAll(torch.autograd.Function)`。

该路径手工管理：

- 正向子图的保存与 detach。
- 反向重计算。
- 异步 AllToAll 发起和等待时机。
- GMM backward 与通信的重叠。
- shared expert 的并行执行。
- 中间 tensor 的释放和恢复。

因此其 backward 代码复杂的主要原因是性能调度和显存管理，而不是 MoE 数学关系本身。DeepEP 第一版 POC 不应直接进入这条复杂路径，应先在非 overlap 路径验证正确性。

## 6. 训练是否可以直接使用 DeepEP

### 6.1 直接 `import deep_ep` 可以验证什么

可以验证：

- wheel 能否在目标 CANN/NPU 环境安装。
- `Buffer` 和 EP 通信组能否初始化。
- layout/dispatch/combine 是否满足目标模型规格。
- DeepEP forward 是否与当前 MindSpeed AllToAll forward 对齐。
- 单独的 dispatch/combine 性能。

不能直接验证：

- 完整训练 backward。
- Router 梯度。
- Expert 输入和权重梯度与基线一致。
- 与 GMM/FA 的实际 overlap。

### 6.2 正向和反向的对应关系

可以将正向简化为：

```text
x ──Dispatch──> expert_input
  ──Expert FFN──> expert_output
  ──Weighted Combine──> y
```

反向为：

```text
dy ──Dispatch + Router 权重──> expert_output_grad
   ──Expert backward──> expert_input_grad
   ──Unweighted Combine──> dx
```

因此：

| 正向操作 | 反向需要 |
| --- | --- |
| Dispatch | Combine，将 Expert 输入梯度归并回原 token |
| Weighted Combine | Dispatch，将输出梯度按路由发送给 Expert |
| Expert FFN | Expert backward，计算输入梯度和权重梯度 |

DeepEP 的两个前向通信能力在反向中具备复用基础，但需要训练侧自定义 autograd 来组织，并补齐 Router 概率梯度。

### 6.3 当前阶段结论

```text
DeepEP normal：
  正向具备直接 POC 条件；
  完整训练不能直接使用，需要适配 backward 和 overlap。

Fused Deep MoE：
  当前更偏推理，首版不建议作为训练接入目标。
```

## 7. 候选实现方案

### 方案 A：直接依赖 `deep_ep.whl` 做最小 Forward POC

做法：

1. 在目标 NPU 环境构建并安装 `deep_ep.whl`。
2. 构造与 MindSpeed 相同的 EP group、Top-K 和 Expert 布局。
3. 使用同一输入、Router 结果和 Expert 权重，对比：

```text
MindSpeed AllToAll forward
vs.
DeepEP dispatch + MindSpeed Expert + DeepEP combine
```

优点：

- 改动最小。
- 最快确认硬件、形状和 forward 精度是否满足。
- 不立即承担算子代码维护成本。

缺点：

- 只能解决第一阶段可行性。
- 尚无训练 backward 和 overlap。

建议：作为第一阶段推荐方案。

### 方案 B：保留 wheel 依赖，在 MindSpeed 增加 DeepEP dispatcher 和 autograd

做法：

1. 增加独立的 `MoEDeepEPTokenDispatcher` 或 feature flag。
2. forward 调用 DeepEP dispatch/combine。
3. 通过自定义 `torch.autograd.Function` 实现：
   - combine backward → dispatch gradient。
   - Expert backward → PyTorch/MindSpeed 现有 Expert。
   - dispatch backward → unweighted combine gradient。
   - Router probability gradient。
4. 首版使用同步/BF16 路径，与 AllToAll 基线对齐。

优点：

- DeepEP 仍作为独立依赖升级。
- MindSpeed 只维护训练语义、autograd 和调度。
- 责任边界相对清晰。

缺点：

- 当前 DeepEP adapter 的 event/handle 可能需要扩展。
- wheel、CANN、torch_npu 和自定义 OPP 存在版本耦合。

建议：若 Forward POC 通过，优先评估该生产化方向。

### 方案 C：抽取 normal 算子和 adapter 到训练仓

做法：

- 将 `CamMoeDispatchNormal`、`CamMoeCombineNormal` 及必要的 C++/OPP 构建代码迁入 MindSpeed 或训练算子仓。
- 根据训练 backward、stream/event 和显存管理需求修改接口。

优点：

- 可以针对训练定制 handle、backward 和 overlap。
- 发布节奏和版本由训练侧控制。

缺点：

- 需要维护算子源码、CANN 适配、构建和测试。
- 后续难以自动跟随 SGL DeepEP 优化。
- 需要保留 MIT 许可证和原版权声明。

建议：只有在 wheel adapter 无法满足训练 overlap，或组织上要求训练侧独立发布时采用。

### 方案 D：直接扩展完整 Fused Deep MoE 支持训练

需要为 fused 算子补齐：

- GMM1/GMM2 权重梯度。
- 输入和激活梯度。
- SwiGLU backward。
- Router 概率梯度。
- 训练精度与混合精度策略。
- 激活保存/重计算。

该方案范围最大、风险最高，当前不建议作为第一阶段目标。

### 方案比较

| 方案 | 正向验证速度 | 训练改造量 | Overlap 自主性 | 维护成本 | 建议 |
| --- | --- | --- | --- | --- | --- |
| A. wheel Forward POC | 最快 | 低 | 低 | 低 | 第一阶段推荐 |
| B. wheel + MindSpeed autograd | 中 | 中 | 中 | 中 | Forward 通过后优先 |
| C. 算子迁入训练仓 | 中 | 高 | 高 | 高 | adapter 受限时考虑 |
| D. 完整 fused 训练化 | 慢 | 很高 | 高 | 很高 | 暂不建议 |

## 8. 组会待对齐问题

### 8.1 目标边界

1. 第一阶段目标是形成可行性结论，还是要求提交可训练代码？
2. 第一版只验证 normal dispatch/combine，是否可以暂不考虑 fused MegaMoE？
3. 第一版是否允许只支持同步 BF16，不做量化和 overlap？
4. 最终希望以 `deep_ep.whl` 依赖交付，还是要求算子代码进入 MindSpeed/训练算子仓？

### 8.2 目标环境

5. 首个验证平台是 A3 还是 A5？对应 CANN、PyTorch、torch_npu 版本是什么？
6. 首版验证单机 HCCS，还是必须包含跨机？
7. 目标 MindSpeed/Megatron 分支是否确定为 `core_r0.16.0` 组合？

### 8.3 目标模型

8. 优先模型和规格是什么：hidden size、Top-K、Expert 数、EP size、每卡 token 数？
9. 是否包含 shared expert、EPLB、token drop/padding、TP 扩展 EP？
10. Router 权重放在 Expert 内部还是 combine 阶段，需要以哪条现有训练链路为精度基线？

### 8.4 性能目标

11. 首要收益目标是替换 AllToAllV 的单算子性能，还是完整 MoE 层耗时？
12. 首阶段要求与 GMM overlap，还是还需要覆盖 shared expert/FA overlap？
13. 成功标准是 step time、吞吐、MFU，还是 dispatch/combine latency 和 overlap 比例？

## 9. 建议实施规划

以下工期从目标硬件、模型规格和代码分支确认后开始计算。

| 阶段 | 主要工作 | 可验收产物 | 预估 |
| --- | --- | --- | --- |
| 0. 规格对齐 | 固定硬件/CANN/模型/EP/分支和精度基线 | 规格矩阵、测试输入、成功标准 | 0.5 天 |
| 1. 安装与 Forward POC | 构建 wheel，跑 layout/dispatch/combine，对比 MindSpeed forward | 冒烟脚本、输出精度报告、接口差距清单 | 1～2 天 |
| 2. 最小 backward | 自定义 autograd，验证输入/Expert/Router 梯度 | 梯度对齐测试、反向设计说明 | 2～3 天 |
| 3. MindSpeed 普通路径接入 | 新增 dispatcher/feature flag，跑最小 MoE 训练 | 可训练分支、单步/短训精度结果 | 2～3 天 |
| 4. Overlap 与性能 | 接入异步流，分析 GMM/FA 掩盖和显存 | Profiling 时间线、性能对比、风险结论 | 3～5 天 |
| 5. 生产化决策 | 决定 wheel 依赖或迁移算子，补充支持矩阵和测试 | 最终方案、维护边界、CI/ST 计划 | 1～2 天 |

建议阶段门：

```text
Forward 不对齐：停止进入 backward，先解决规格/语义问题。
Backward 不对齐：不进入完整训练和性能优化。
无可观性能收益：重新评估是否值得迁移算子。
```

## 10. 验证指标

### 10.1 正确性

- Forward 输出与 AllToAll 基线对齐。
- `hidden_states.grad` 对齐。
- Expert GMM 权重梯度对齐。
- Router probs/logits 梯度对齐。
- 多 step loss 曲线和参数更新结果一致。
- Top-K 重复路由、空 Expert、负载不均衡、0 token 等边界场景正确。

### 10.2 性能

- layout、dispatch、combine 单阶段耗时。
- 完整 MoE layer forward/backward 耗时。
- 通信与 GMM/FA 的 overlap 时间和掩盖比例。
- HBM 峰值和临时 buffer。
- 是否存在 D2H 同步、stream wait 或动态 shape 引入的空洞。
- 不同 token 数、EP size、节点数下的扩展性。

## 11. 当前已完成工作与未完成项

### 11.1 已完成

- 建立 MoE、EP、dispatch/combine、normal/fused 的概念边界。
- 确认 DeepEP 可单独打包为 `deep_ep.whl`，不需要引入完整 SGLang。
- 追踪 normal dispatch 的 Python → strategy → C++ → AscendC 调用链。
- 追踪 normal combine 和 fused Deep MoE 调用链。
- 确认当前代码没有训练 autograd/backward。
- 确认当前 default normal event 接口不足以直接认定训练 overlap 可用。
- 追踪 Megatron AllToAll dispatcher 的 forward 和 `_AllToAll.backward`。
- 追踪 MindSpeed `MoELayerOverlapAllToAll` 的手写 backward 和异步通信路径。
- 对比 vLLM-Ascend 当前 EP 调用，确认相似功能不等于同一份 normal 算子。
- 形成四类接入方案、关键风险和阶段性验证计划。

### 11.2 尚未完成

- 未在 A3/A5 上构建和安装 `deep_ep.whl`。
- 未完成 DeepEP 与 MindSpeed 的 forward 数值对齐。
- 未实现最小 DeepEP autograd/backward。
- 未完成 Router/Expert/input 梯度对齐。
- 未进行单算子和完整 MoE layer profiling。
- 未形成最终的 wheel 依赖或算子迁移结论。

## 12. 本次组会建议结论

建议本次组会先确定以下三点：

1. **第一目标**：以 DeepEP normal 为对象，先完成 BF16、单机多卡、非 overlap 的 forward/gradient POC。
2. **默认路线**：先使用独立 `deep_ep.whl`，由 MindSpeed 补训练 autograd；只有 adapter 无法满足训练 overlap 时，再讨论迁移算子代码。
3. **验收顺序**：规格满足 → forward 对齐 → backward 对齐 → 最小训练 → overlap 性能，避免在正确性未闭环前直接进入复杂 overlap 改造。

如果以上边界能够对齐，下一阶段可以立即输出最小 Forward POC 的测试设计和接口映射表。

## 附录：关键源码位置

### `sgl-kernel-npu`

- `README.md`
- `python/deep_ep/README.md`
- `python/deep_ep/deep_ep/buffer.py`
- `python/deep_ep/deep_ep/strategies/normal_strategy.py`
- `python/deep_ep/deep_ep/utils.py`
- `python/deep_ep/setup.py`
- `csrc/deepep/pybind_extension.cpp`
- `csrc/deepep/deep_ep.cpp`
- `csrc/deepep/event.hpp`
- `csrc/deepep/ops/op_kernel/cam_moe_dispatch_normal*`
- `csrc/deepep/ops/op_kernel/cam_moe_combine_normal*`
- `tests/python/deepep/test_intranode.py`
- `tests/python/deepep/test_fused_deep_moe.py`

### Megatron-LM / MindSpeed

- `megatron/core/transformer/moe/moe_layer.py`
- `megatron/core/transformer/moe/token_dispatcher.py`
- `megatron/core/tensor_parallel/mappings.py`
- `mindspeed/core/transformer/moe/moe_feature/overlap/moe_layer_overlap_all2all.py`
- `mindspeed/core/transformer/moe/moe_feature/overlap/comm_utils.py`
- `mindspeed/features_manager/moe/moe_alltoall_overlap.py`
