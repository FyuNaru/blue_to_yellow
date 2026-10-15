# TorchTitan v0.3.0 NPU 适配修改说明与社区 main 对比

核对日期：2026-10-15。用于代码评审与后续上游提交筛选。

## 一、评审结论（省流版）

**v0.3.0 基线保留全部 8 个提交；向社区 main 提交时不能整套照搬。**

| 提交与修改点 | 为什么修改 v0.3.0 | 当前 main 的情况 | 上游处理建议 |
| --- | --- | --- | --- |
| `712a72fbf` 包入口先导入 Torch | 避免 SPMD 与 NPU 扩展循环初始化，阻塞模块导入 | 包入口仍未提前导入 Torch | 保留候选；需用 main 配套依赖复现确认 |
| `7d9046995` RoPE 复数缓存通过实数视图索引 | 当前 NPU 不支持原路径对复数缓存的 gather | main 改为 token-first 的一维位置索引；此前未修改 main 方法的 NPU 检查已通过 | 不提交旧 workaround；没有证据证明当前 main 需要它 |
| `98186b504` DistMuon 不再限定 CUDA | 原校验直接拒绝 NPU 参数 | main 已取消 CUDA 限制，只要求每进程一个设备 | 不重复提交；两者实现并非完全相同 |
| `6b610e26b` 损失单测先导入 Torch | 测试入口直接先导入 SPMD，仍可触发循环初始化 | main 损失测试仍先导入 SPMD | 单独评估测试入口，不认定包入口修复自动覆盖它 |
| `104c4c55f` 损失参考输入统一二维布局 | 不同计算布局的浮点舍入差异导致逐位梯度比较失败 | main 已直接构造二维 logits 和一维 labels | 不重复提交旧补丁；必须说明测试比较范围变化 |
| `dcae8fef8` DistMuon 单测使用当前加速器 | CUDA 卡数检查使 NPU 测试被跳过，且设备名固定为 cuda | main 对应测试仍检查 CUDA 卡数、返回 cuda | 可作为独立候选；按 main 测试结构重新适配与验证 |
| `77861dd12` AC 测试设备适配并补 FullAC 正确性对照 | 显存测试仅支持 CUDA，原正确性测试没有覆盖 NPU FullAC | main 已重写测试，已有 Full/default 输出、梯度对照，但不是我们的设备适配方式 | 正确性覆盖不重复加；重新评估 NPU 测试设备及内存覆盖 |
| `9f67de7d7` SelectiveAC 识别二维 aten.matmul | NPU 暴露的二维 matmul 未被原 mm/linear 策略识别，保存策略没有生效 | 旧 activation_checkpoint.py 已被 activation_remat.py / torch_remat 体系替代 | 作为 v0.3.0 专用适配保留；新体系不能直接移植旧策略 |

“main 没有相同修改”仅代表静态源码结论，不等于确认存在相同运行时错误。

## 二、比较范围与验证边界

- 基线：官方 `v0.3.0`，提交 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`。
- 适配分支：[FyuNaru/torchtitan:feat/npu-v030-review](https://github.com/FyuNaru/torchtitan/tree/feat/npu-v030-review)，HEAD `9f67de7d7345707134594acd47489b32c9cb4a15`，已推送。
- 本次 main 比较快照：`d51e4b3c6bf409e938dba55c84d9eb58d680371b`，`[MoE] Organize common modules and unify feed-forward fields (#5345)`。
- 共 8 个逻辑提交，涉及 9 个文件，新增 186 行、删除 37 行；其中 4 项包含生产代码修改，另外 4 项为测试修改。生产修复附带的原生回归测试也计入文件差异。
- 本次重新检查了 main 源码，没有重新执行训练或 main 全套 NPU 测试。下文运行结果来自此前保存的验证日志，不是本次重新运行结果。
- 既有验证主要使用 A5（Ascend 950DT）、Torch 2.14、TorchNPU git9c3f664、TritonAscend 3.2.2，以及独立环境中的 TorchNPU 兼容补丁；走 Inductor 的 default 路径，不是 TE 路径。
- TorchTitan 补丁不包含 TorchNPU 环境修复，不能声称仅打该 patch 就能在任意环境跑通全部六类用例。

## 三、逐项修改说明

### 1. 包入口提前导入 Torch：712a72fbf

**文件：** `torchtitan/__init__.py`。

**原问题：** 导入 TorchTitan 子模块时，可能先进入 `spmd_types`，后者又触发 Torch 加载 NPU 扩展；NPU 扩展的导入链间接加载 Torch 的 FSDP，FSDP 访问尚未初始化完成的 SPMD 符号，形成循环导入。错误发生在模块加载阶段，不是模型计算公式出错。不是 NPU 扩展本身直接调用该 SPMD 函数。

**修改：** 在包入口显式 `import torch`，让设备扩展初始化先于后续子模块的 SPMD 导入。不显式依赖 `torch_npu`，不吞异常，不增加 NPU 专属分支。

#### 1.1 SPMD 是什么，为什么这里用到它？

SPMD 是 Single Program, Multiple Data，即“单程序、多数据”：多张卡执行同一套程序，各自处理不同数据。本处实际导入的是 `spmd_types` 包，它为分布式张量提供标注及检查，不是负责卡间传输的通信库。

例如，一个 8 行张量可以在两张卡上各放 4 行，也可以每张卡都放完整 8 行。两种存放方式的计算/通信要求不同。`spmd_types` 帮助描述和检查这类区别，避免把切分张量误当成完整或复制张量处理。

#### 1.2 为什么先 import Torch 就能避开循环？

Python 的 `import` 会执行初始化代码。模块开始加载时就会登记到模块缓存中，因此另一个模块再次导入它时，可能得到“已经创建、但还没执行完”的对象，而不是重新从头加载。

失败路径（根据保存的 traceback 简化）：

```text
开始 import spmd_types
  → spmd_types._backward_hooks 执行 import torch，暂停等待
    → Torch 初始化到设备扩展加载阶段，加载 TorchNPU
      → TorchNPU 分布式模块间接加载 Torch FSDP
        → FSDP import spmd_types，拿到正在等待的 SPMD 半成品
          → 调用 register_local_autograd_function，但该名字还没导出
```

实际报错为 `partially initialized module 'spmd_types' has no attribute 'register_local_autograd_function'`。不是这个函数一定不存在，而是访问得太早。

先导入 Torch 时，顺序变成：

```text
开始 import torch
  → Torch 已准备好核心功能，随后加载设备扩展
    → 间接进入 FSDP，首次开始 import spmd_types
      → SPMD 内部 import torch，取得已初始化到较后阶段的 Torch
      → 当前 SPMD 初始化所需的 Torch 功能已经存在，SPMD 能继续完成
    → FSDP 再调用 SPMD 函数，此时函数已导出
```

“用户代码还没手动 import SPMD”不代表其他模块不能先替用户导入它。关键也不是所有模块都必须完全初始化，而是**此刻要用的符号是否已经准备好**。此修复改变了依赖环的进入顺序，不代表根本消除了依赖环，更不是所有 Python 程序都必须先导入 Torch。

#### 1.3 新增代码逐行含义

```python
# Load device extensions before modules that import spmd_types.
import torch  # noqa: F401
```

- 注释说明这个导入是为了初始化顺序，不是为了在此文件调用某个 Torch API。
- `import torch` 提前触发 Torch 及自动加载设备扩展的初始化。
- `noqa: F401` 告诉检查器：这是有意保留的初始化导入，不要因为本文件没有引用 `torch` 变量就按“未使用导入”处理。
- 该修复只覆盖经过 `torchtitan` 包入口的导入；其他脚本自己先导入 SPMD，仍可能绕过它，见第 4 项。

**已有证据：** v0.3.0 后续用例在该修改基础上执行。此前对 main `0a3480fe2` 的诊断中，原生 `import torchtitan.components.loss` 在现有验证栈失败，加此修改后成功；另一个导入入口前后均成功，不能声称所有导入都会失败。

**main 对比：** 本次快照包入口仍没有 `import torch`。但 main 固定的 SPMD 依赖比现有验证环境更新，必须在匹配依赖下再确认。社区分支 `fix/npu-main-compat` 目前只包含此候选，已推送；自动创建 PR 曾失败，不视为已创建。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/torchtitan/__init__.py)

### 2. RoPE 复数缓存实数视图索引：7d9046995

**文件：** `torchtitan/models/common/rope.py`；新增原生测试 `tests/unit_tests/test_rope_accelerator.py`。

**原问题：** RoPE 为位置编码缓存复数值。v0.3.0 按每个 batch 的 positions 选择缓存，当前 NPU 对原路径的复数 gather 不支持，训练进入该路径会失败。

**修改：** 仅当缓存位于 NPU 时，先用 `view_as_real` 展开复数的实部、虚部，在实数视图上按原位置规则取值，再用 `view_as_complex` 恢复。保存两部分数值及 packed positions 语义，不把复数强转成实数而丢失虚部。其他设备保持原路径。

**已有证据：** 新原生测试对照 CPU 的 RoPE 输出及 query/key 梯度，覆盖 FP32/BF16 和 prefix/singleton/packed 三类位置输入；原始四卡 default 训练在包含该修改的基线上完成 10 步。

**main 对比：** main 改为 token-first 输入，使用 `rope_cache[positions]`，不是 v0.3.0 的每 batch gather。此前在未修改 main `0a3480fe2` 方法上执行的 13 项直接 NPU 检查已通过，覆盖三种 dtype、位置索引及输出/梯度；这些不是完整 main 训练。当前快照仍是该一维索引结构。因此不提交旧 workaround，但 v0.3.0 补丁继续保留。

#### 2.1 逐行解释实数视图适配

RoPE 是旋转位置编码。这里缓存每个位置对应的复数旋转系数；一个复数 `a+bj` 包含实部 a 和虚部 b。修复的目标是换一种方式取缓存，不是更换位置编码公式。

```python
if self.cache.device.type == "npu":
    real_cache = torch.view_as_real(self.cache).flatten(-2)
    reshaped = _reshape_for_broadcast(real_cache, query.shape, positions)
    return torch.view_as_complex(reshaped.unflatten(-1, (-1, 2)))
```

1. `if self.cache.device.type == "npu"`：仅对 NPU 缓存启用新增路径，CPU/CUDA 等仍执行原代码。
2. `torch.view_as_real(self.cache)`：把每个复数视作 `[实部, 虚部]`。例如 `[1+2j, 3+4j]` 变成 `[[1,2],[3,4]]`；这是视图表达，不是丢掉虚部的类型转换。
3. `.flatten(-2)`：从倒数第二维开始合并最后两维，得到 `[1,2,3,4]`。原缓存形状 `[最大位置数,D/2]`（复数），先变为 `[最大位置数,D/2,2]`，再变为 `[最大位置数,D]`（实数）。D 表示一个注意力头的维度。
4. `_reshape_for_broadcast(real_cache, query.shape, positions)`：复用原有位置选择与形状调整逻辑。此时缓存为实数，可以避开原复数 gather 路径；位置编号不变。使用 `query.shape` 是因为展开后的缓存宽度已经是 D，与 query 最后一维一致。
5. `reshaped.unflatten(-1, (-1, 2))`：把最后一维重新分成若干组，每组 2 个数。例如 `[1,2,3,4]` 恢复为 `[[1,2],[3,4]]`。外面的 `-1` 指最后一维，括号内的 `-1` 表示自动推算组数。
6. `torch.view_as_complex(...)`：把相邻实数对重新解释为复数，恢复 `[1+2j,3+4j]`，供原有 RoPE 乘法继续使用。

完整过程是“复数 → 实数对 → 按位置选择 → 恢复复数”，不是降低精度或修改位置值。对每 batch 的 positions，辅助函数输出可具有 `[B,L,1,D]` 实数形状，恢复后为 `[B,L,1,D/2]` 复数形状；中间的 1 用于广播到不同注意力头。

**注释范围提醒：** 分支中的注释写了“NPU gather/index 不支持复数缓存”，范围偏大。准确结论应限定为“当前验证栈下 v0.3.0 的复数缓存 gather 路径不受支持”，不能据此断言所有复数索引都不支持。本次只在报告澄清，未修改已推送源码。

#### 2.2 为什么 main 不再需要相同补丁？具体例子

设位置缓存第 0、1、2、3 行分别为 C0、C1、C2、C3，每个 C 是一行复数系数。两条样本各有 3 个 token：

```text
positions = [[0,1,0],
             [2,0,1]]
期望缓存 = [[C0,C1,C0],
            [C2,C0,C1]]
```

v0.3.0 query 是 `[B,L,H,D]`，保留样本 B 和长度 L 两维。通用 positions 路径先用 `expand` 将缓存视图扩展到每条样本，再用 `torch.gather(..., dim=1, ...)` 分别取值。`expand` 不意味着必须物理复制缓存。该路径对复数缓存的 gather 正是原阻塞点。

main query 改为 `[T,H,D]`，第一维直接排列 token；在这个简化例子中 T=6，对应位置为 `[0,1,0,2,0,1]`。因此直接执行：

```python
selected_cache = rope_cache[positions]
# [C0,C1,C0,C2,C0,C1]
return selected_cache.view(num_tokens, 1, cache_width)
```

两者为每个 token 取到的系数相同，但底层操作从原每 batch 的复数 gather 变成一维位置索引。**不支持某条复数 gather，不等于所有复数索引都不支持。** main 并没有取消复数缓存或 RoPE，只是不再走原失败路径；此前实际 NPU 方法检查也通过。

注意：旧版并非每次都 gather。不传 positions 时走前缀切片；能确定为单行的 positions 另有直接索引分支。main 的整个模型布局迁移也不只是 flatten 两行，样本边界仍由其他代码维护。此例只解释缓存选择的等价关系。

#### 2.3 为什么新增 test_rope_accelerator.py，原测试不够吗？

是新建的测试文件，不是仓库原有用例，也没有替换或删除原 `test_rope.py`。原文件已有形状、dtype、位置边界等测试，但未覆盖本次完整的 CPU/加速器输出及 query/key 梯度对照组合。

新增测试用同样输入，对比原 CPU 路径与加速器路径，检查取位置、实部/虚部还原及梯度。组合为 FP32/BF16 × prefix/singleton/packed × CPU/可用加速器，共 12 个参数化实例；无设备时加速器分支跳过，跳过不算 NPU 通过。prefix 是不显式传位置，singleton 是一行位置，packed 例子包含多行和位置重置。

新增测试是修复后的回归保护，原训练失败才是实际问题证据。单独建文件不是解决报错的必要条件；后续可以将这些覆盖并入已有 `test_rope.py`，但不应把新增测试说成“原测试未经修改就通过”。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/torchtitan/models/common/rope.py#L384)

### 3. DistMuon 参数存储设备校验：98186b504

**文件：** `torchtitan/distributed/flex_shard/dist_muon.py`；新增原生测试 `tests/unit_tests/flex_shard/test_dist_muon_storage.py`。

**原问题：** `_validate_parameter_storage` 要求参数必须在 CUDA，NPU 参数直接报 `DistMuon requires one CUDA device per process`。这是校验限制，不是已证明 Muon 数学计算不支持 NPU。

**修改：** 使用当前加速器类型校验，保留“每个进程只有一个参数存储设备”及 DTensor 参数约束；增加 7 项 CPU/mock 正反例测试。

**已有证据：** 7 项新增回归及原优化器/调度器 28 项单测通过。原始四卡 `kimi_k2_5_muon_fsdp+ep` 完成配方规定的 1 步训练；不是长期收敛或 GPU/NPU 逐位对齐证明。

**main 对比：** 已只要求 `len(local_devices) == 1`，取消设备类型限制。main 比我们的校验更宽松；只能说同一 CUDA 限制已解除，不能说实现完全一致。不向 main 加回我们的额外设备类型限制。

#### 3.1 原来确实报错，还是只是新增测试？

原始四卡 Muon 训练确实因为 CUDA 硬编码报错，不是为了增加测试而假设的问题。生产代码修复解除该入口阻塞；新测试用于防止以后再次把设备限定死，或误把原本不合法的参数放过去。

例如，每个进程负责本机一张 NPU，参数都在该设备上，应该允许。参数同时分布在同一进程的两张设备上，仍不符合原单设备约束，不能为支持 NPU 而一并放开。

#### 3.2 为什么新建 test_dist_muon_storage.py？

原 `test_dist_muon.py` 主要运行 2/4 进程分布式优化器场景；新增文件只检查 `_validate_parameter_storage` 这个小函数的规则，避免每个输入边界测试都启动多卡通信。

7 个 CPU/mock 实例包括：当前 CUDA、当前 PrivateUse1 加速器设备可接受的 2 个正例；空参数、CPU 参数、同进程多设备、没有当前加速器、参数设备类型与当前加速器不一致的 5 个反例。PrivateUse1 是 PyTorch 预留给外部设备扩展的后端槽位，在当前 NPU 环境中被注册为 npu。

这里 mock 是临时模拟设备对象和 DTensor 类型，不需要真的创建 CUDA/NPU 张量。因此它证明的是校验逻辑，不是实际设备计算能力；真实 NPU 行为另由训练和原分布式测试验证。

文件可以合并，不是技术上必须新增。若并入原文件，应放在不会被整个多卡类的 skip 条件覆盖的位置，保留其轻量、无设备也可检查的作用。两份新增测试都不是仓外临时运行器，而是附在修复提交中的正式测试代码；本次没有移动或删除它们。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/torchtitan/distributed/flex_shard/dist_muon.py#L257)

### 4. 损失单测导入顺序：6b610e26b

**文件：** `tests/unit_tests/test_loss.py`。

**原问题：** 测试文件先导入 SPMD，再导入 Torch；测试收集或 spawn 子进程能在进入 TorchTitan 包入口之前触发循环初始化。因此第 1 项包入口修复不自动覆盖这个测试入口。

**修改：** 把 Torch 移到 SPMD 前导入；增加 `usort: skip` 保留有意安排的顺序。没有修改损失实现或断言。

#### 4.1 为什么已经修了包入口，还要修测试文件？

测试文件自己先执行 `import spmd_types`，可能在随后导入任何 `torchtitan` 模块之前就失败。第 1 项只有进入 TorchTitan 包入口后才生效，不能回到过去纠正这个顺序。多进程测试通过 spawn 启动新 Python 进程时，也会重新执行测试文件导入，不能依赖父进程已经初始化。

```python
import torch  # usort: skip

import spmd_types as spmd
```

`import torch` 提前准备依赖；`import spmd_types as spmd` 中的 `spmd` 只是包的本地简称，不是另一种通信后端。`usort: skip` 是为了防止自动排序又把两者恢复成字母顺序，并不关闭整个文件的质量检查。循环初始化的完整原因见第 1.2 节。

**已有证据：** 原始收集失败被消除，后续能够执行损失单测；其后发现的梯度逐位差异属于第 5 项另一个问题，不能混为一次修复。

**main 对比：** 测试已迁至 `tests/unit_tests/cpu/components/test_loss.py`，仍先导入 SPMD。需使用 main 固定依赖验证该入口后，决定是否单独提交测试导入修复；不按缺少同样补丁直接认定 main 失败。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/tests/unit_tests/cpu/components/test_loss.py#L12)

### 5. 损失参考输入统一二维布局：104c4c55f

**文件：** `tests/unit_tests/test_loss.py`。

**原问题：** 原参考实现用 `[B,V,L]`，生产实现用 `[B*L,V]`。B 为 batch，L 为序列长度，V 为词表大小。两种布局表达相同数学损失，但浮点归约顺序可能不同，导致 NPU 反向梯度的 `torch.equal` 失败；诊断最大绝对差约 `8.73e-10`。单独布局探针在 CPU 也观察到舍入差异，不能据此直接认定 TorchNPU 算子错误。

**修改：** 仅把参考 logits/labels 展平为 `[B*L,V]` / `[B*L]`，与生产路径保持一致；保留原逐位断言、测试场景和生产损失公式。

**已有证据：** 修订后第四类 32 项单测通过，包含 NPU 分布式损失测试；CPU 对照通过。两个原始四卡 `2d_eager` 配方各完成 10 步。

**评审必须说明：** 原跨布局逐位梯度测试确实失败，不能写成原测试未经修改就通过。本修改校正的是参考输入布局，验证目标不再包含跨布局逐位不变性；不是数值容差放宽。

#### 5.1 用一个小形状理解布局变化

假设 B=2、L=3、V=5，即两条样本、每条三个 token、每个 token 对五个候选词打分。原 logits 是 `[2,3,5]`。旧参考代码 transpose 成 `[2,5,3]`，把词表维放在中间；生产路径展平成 `[6,5]`，把所有 token 排成 6 行，每行 5 个分数。

两者数学上都计算 6 个 token 的交叉熵，但计算布局变化可能改变浮点归约顺序。浮点数不是无限精确的小数，同一个数学公式在不同执行顺序下，不保证最后一位相同。`torch.equal` 要求逐位一致，所以很小差异也会失败。

修改是把参考 logits 改为 `flatten(0, 1).float()`、标签改为 `flatten(0, 1)`。不是修改标签对应关系，也不是将断言改成允许误差。评审应明确接受“同布局对照”的测试目的，不能把它包装成修复了所有跨布局数值差异。

**main 对比：** main 测试直接创建 `[T,V]` logits、`[T]` labels，参考调用不再使用三维 transpose；逐位比较仍保留。相同方向已覆盖，不重复提交 v0.3.0 的 flatten 补丁。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/tests/unit_tests/cpu/components/test_loss.py#L470)

### 6. DistMuon 单测使用当前加速器：dcae8fef8

**文件：** `tests/unit_tests/flex_shard/test_dist_muon.py`。

**原问题：** 测试以 CUDA 卡数决定是否运行，并把设备类型固定为 cuda；NPU 机器会跳过，跳过不能记作 NPU 通过。

**修改：** 用 `torch.accelerator.device_count()` 检查数量，设备类型使用 PyTorch 分布式测试框架的 `DEVICE_TYPE`。不改 Muon 计算逻辑。

#### 6.1 为什么不能把原来的 skipped 记成通过？

原测试问的是“有没有至少两张/四张 CUDA 卡”，NPU 机器即使有 8 张 NPU，也不满足这个条件，因此测试主体根本不执行。设备名固定为 cuda 也会让后续张量创建与通信选择不匹配。此次只把设备选择接到当前测试设备机制，不减少卡数要求、通信步骤或优化器对照断言。

本项适配的是原分布式测试；第 3 项新增的 storage 文件是小范围规则检查，两者不能相互替代。测试激活后发现生产设备校验失败仍应修生产代码，而不是再把测试跳过。

**已有证据：** 两项原生分布式测试实际运行于 NPU/HCCL，分别覆盖 2/4 进程、flat checkpoint 恢复及参数存储布局约束。依赖第 3 项生产设备限制修复。

**main 对比：** 当前对应文件仍用 `torch.cuda.device_count()` 并返回 cuda。作为独立测试可移植性候选保留，但 main 的测试参数化、目录及多 GPU 标记已经演进，需重新适配并确认 CI 收集/执行方式，不直接 cherry-pick。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/tests/unit_tests/gpu/distributed/flex_shard/test_dist_muon.py#L46)

### 7. AC 单测设备适配与 FullAC 正确性：77861dd12

**文件：** `tests/unit_tests/test_activation_checkpoint.py`。

**原问题：** 显存测试只使用 CUDA；正确性测试主要在 CPU 上运行，且缺少完整 FullAC 输出/梯度对照。AC 指激活检查点：通过少存中间结果、反向时重新计算来节省内存。

**修改：** 使用当前加速器及其内存统计；把正确性测试放到可用设备，添加 FullAC 与无检查点模型的输出、输入梯度、参数梯度对照。保留原显存预期和数值比较标准。

**已有证据：** 完成第 8 项策略修复后，AC 文件 4 项通过，其中包括真实 NPU 数值与显存检查；CPU 对照因无加速器跳过显存项。该提交本身不是 SAC 显存问题的全部修复。

#### 7.1 FullAC、SelectiveAC 与正确性检查分别是什么意思？

- FullAC：更广泛地对被包裹区域做重新计算，以少保存中间结果换取额外计算。
- SelectiveAC/SAC：选择性保存部分操作结果，其余优先重算，在内存和计算量之间取舍。
- 正确性对照：从相同权重、相同输入出发，与不使用检查点的模型比较输出、输入梯度和参数梯度，检查节省内存的手段有没有改变计算结果。

本项将 `.cuda()` / `torch.cuda.memory_stats()` 等改为当前设备及其内存接口，并补上 FullAC 对照。没有通过把预期 3 MiB 改成 0 MiB 来消除失败。事实上，测试开始在 NPU 上真实执行后暴露了第 8 项策略问题，这正是区分“测试能执行”和“功能正确”的必要性。

**main 对比：** 原生测试已重写，使用 torch_remat，已有 full/default 策略的输出及梯度零容差对照。不能重复添加相同测试目的；是否需要 NPU 设备和内存覆盖，应基于新测试设计重新评估。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/tests/unit_tests/gpu/distributed/test_activation_checkpoint.py#L161)

### 8. SelectiveAC 识别二维 aten.matmul：9f67de7d7

**文件：** `torchtitan/distributed/activation_checkpoint.py`。

**原问题：** 原选择性检查点策略识别 mm/linear，但 NPU 可能将二维矩阵乘法保留为 `aten.matmul`。同类计算因算子名字不同，未进入原“隔次保存”等策略，测试观察到 SAC 保存内存为 0 MiB，而原策略预期为 3 MiB。

**修改：** 仅对两个输入均为二维的 `aten.matmul`，复用已有 mm 计数、指定形状重算及保存决策。只有原 save_ops 包含 mm 时才给二维 matmul 启用保存别名，不把高维 matmul 一并加入，也不覆盖子类自定义保存集。

**已有证据：** AC 与 DistMuon 合计 6 项原生测试通过；最终基线的原始四卡 `1d_compile_sac_op` 完成 10 步，仍使用 Inductor。

**main 对比：** 旧文件不存在，已经切换为 `activation_remat.py`，依赖 torch_remat 按模型区域配置保存/重算。旧的逐算子计数策略没有可直接粘贴的对应点，需针对新体系重新验证，当前不作为 main 提交候选。

#### 8.1 原来是什么报错？不是 matmul 算子不支持

原来确实有测试失败：NPU 上 SAC 显存测试的保存量为 0 MiB，原断言预期为 3 MiB。不是抛出“matmul 不支持”，也不是已经发现模型输出算错。原策略没有识别这个二维乘法，落到默认的 `PREFER_RECOMPUTE`，没有按预定隔次保存规则工作。

因此“保存得更少”在这里不是自动变好：它偏离了用户配置/原策略的内存与重算取舍，可能增加重算开销。本轮没有测出性能差值，不对变慢幅度作结论。补丁后原显存断言及输出/梯度检查通过；不是只看到训练能结束就认为策略生效。

#### 8.2 mm、matmul、linear 的区别

| 操作 | 含义 | 例子 |
| --- | --- | --- |
| `aten.mm` | 两个二维矩阵相乘 | `[8,16] × [16,32] → [8,32]` |
| `aten.matmul` | 更通用的乘法接口，支持二维及批量等情况 | 两个二维输入时，可执行与 mm 相同的矩阵乘法 |
| `aten.linear` | 线性层：输入乘权重转置，可加偏置 | `Linear(16,32)` 把每个 token 的 16 个特征映射成 32 个 |

不是“NPU 的乘法叫 matmul，GPU 的乘法叫 mm”。两种设备都可能出现这些操作；模型实现、算子分解与编译路径决定策略最终看到哪种名字。本轮 NPU 路径保留了二维 matmul，而原策略只识别 mm/linear。linear 本身不是 mm 的简单别名，只是内部包含同类矩阵乘法。

#### 8.3 为什么限定两个输入都为二维？是谁要求的？

这是本次最小补丁的范围，不是 NPU/PyTorch 规定检查点只能用于二维。两个二维输入的 matmul 与原 mm 规则直接对应；三维例如 `[4,8,16] × [4,16,32]` 是 4 组乘法，还可能涉及广播，不能未经验证就全部纳入原二维矩阵计数和权重形状规则。

原策略读取第二个输入形状匹配线性层权重，且按矩阵操作出现次数决定隔次保存。把高维 matmul 一并纳入，可能改变注意力等其他计算的保存/重算策略，也会改变计数。当前失败与验证针对二维路径，因此只补二维；多维若需要适配，须另行设计与验证。

#### 8.4 修改及相关逻辑逐行解释

下面保留相关原有行以便理解；不是每一行都是本次新增，主要修改是识别二维 matmul、统一 `is_mm` 判断以及保存集别名条件。

```python
mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.linear.default)
```

原有集合：策略本来就识别的两种矩阵计算。

```python
mode = "recompute" if ctx.is_recompute else "forward"
mm_count_key = f"{mode}_mm_count"
```

原有逻辑：前向和重算阶段分别计数，避免重算阶段直接接着前向次数往下数。

```python
is_2d_matmul = (
    func == torch.ops.aten.matmul.default
    and args[0].ndim == 2
    and args[1].ndim == 2
)
```

- `func` 是当前被策略检查的操作，`args` 是它的输入。
- 第一行条件确认操作为 matmul，而不是任意操作。
- 后两行分别确认两个输入为二维。Python 的 `and` 从左到右短路求值，非 matmul 不会执行后续输入检查。

```python
is_mm = func in mm_ops or is_2d_matmul
if is_mm:
    weight_shape = args[1].shape
```

将“原 mm/linear 或二维 matmul”统一成一个判断；满足时取第二个输入形状，进入原有形状匹配及计数规则。

```python
    if func == torch.ops.aten.linear.default:
        weight_shape = torch.Size((weight_shape[1], weight_shape[0]))
```

原有逻辑：linear 权重按 `[输出维度,输入维度]` 存储，匹配集合按 mm 右操作数 `[输入维度,输出维度]` 记录，因此仅对 linear 交换形状顺序。此处调整的是比较用的形状，不是实际转置权重张量。

```python
    if weight_shape in mm_recompute_shapes:
        return CheckpointPolicy.PREFER_RECOMPUTE
    meta[mm_count_key] += 1
```

原有逻辑现在也覆盖二维 matmul：若匹配指定的强制重算形状，返回优先重算；否则当前阶段的矩阵计数加一。被该形状规则提前返回的操作不参与后面的计数。

```python
if func in save_ops or (
    is_2d_matmul and torch.ops.aten.mm.default in save_ops
):
```

原操作本就在保存集合中，或“它是二维 matmul 且原集合包含 mm”，就进入保存决策。第二部分是新增别名关系：既然配置允许保存 mm，就让对应二维 matmul 遵循相同规则。不能无条件保存，因为子类可能刻意移除了 mm；否则会破坏自定义策略。

```python
    if is_mm and meta[mm_count_key] % 2 == 0:
        return CheckpointPolicy.PREFER_RECOMPUTE
    return CheckpointPolicy.MUST_SAVE
return CheckpointPolicy.PREFER_RECOMPUTE
```

- 对进入保存决策的矩阵操作，计数第 2、4、6……次时优先重算；其余要求保存。
- 非上述保存分支的操作默认优先重算。
- `PREFER_RECOMPUTE` 是检查点框架的策略偏好，不能泛化成任何上下文中绝对保证重算；`MUST_SAVE` 是要求保存。
- 在没有指定形状提前返回、且 mm 在保存集合中的简单序列里，效果为“第 1 次保存、第 2 次优先重算、第 3 次保存、第 4 次优先重算”。修改前漏识别的二维 matmul 不会进入这套 mm 规则，修改后才会进入。

本项不改变矩阵乘法数值公式，也不把 matmul 替换成 mm；只改变检查点策略如何分类当前操作。与第 7 项测试设备适配是两个独立修改，不能混成一个“为了让断言通过”的修改。

[main 源码](https://github.com/pytorch/torchtitan/blob/d51e4b3c6bf409e938dba55c84d9eb58d680371b/torchtitan/distributed/activation_remat.py)

## 四、两条分支的处理策略

1. **v0.3.0 基线分支：** 8 个提交全部保留，继续用于导出聚合 patch、复现已验证用例。不要仅因 main 已有另一种实现，就改动这条已验证基线。
2. **社区 main 分支：** 不直接提交整套 patch。优先确认包入口导入问题；测试入口导入、DistMuon 测试设备适配、新 AC 测试设备覆盖分别评估。已覆盖项不重复提交，实现已替换项重新分析。
3. **环境修复独立：** strict_sum / strict_reduction、组合调度器等属于 TorchNPU，不加入 TorchTitan PR。default 与 TE 验证结果分开记录，不能互相替代。
4. **未通过项保留：** 变长注意力配方仍受当前 NPU FlashAttention 算子缺失阻塞；在线验证数据获取也曾阻塞验证。不能把本报告的修改说明解读为六类用例全部打通。

## 五、已有详细验证记录

以下为原 TorchTitan 工作目录中的验证材料路径，不属于本仓库；本次仅发布报告，没有同步运行日志或测试脚本。日志所称“本目录”指原 `verification/submission_20261015/` 目录。

- RoPE 适配记录：`verification/torchtitan_v030_20261012/REVIEW_COMPLEX_20261015.md`
- 优化器与设备限制验证：`verification/group3_20261015/README.md`
- 损失导入、布局与数值诊断：`verification/group4_20261015/README.md`
- AC、DistMuon 与编译验证：`verification/group6_20261015/README.md`
- main 导入与未修改 RoPE 方法的历史检查日志：本目录 `main_loss_import_before.log`、`main_loss_import_after.log`、`rope_main_unpatched.log`；检查对象为此前 `0a3480fe2`，不是对本次最新快照重新实测。

本报告已补充评审问答中的初始化调用链、测试覆盖边界、形状示例及逐行代码解读。main 对比仍以第二节固定快照为准，补充说明时未再次刷新 main 或重跑训练。仅修改仓外报告，未修改源码、分支内容或服务器环境，未移动测试、未新建 PR。
