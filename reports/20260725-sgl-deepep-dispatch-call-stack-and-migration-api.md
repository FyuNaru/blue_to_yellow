# SGL DeepEP Normal Dispatch 调用栈与训练迁移接口

> - 调研对象：`sgl-kernel-npu`
> - 源码基线：`main@d66174769bf36b6cd1b8247ced156ce94287bcb8`
> - 调研范围：`DEEP_USE_MODE=default` 下的 normal dispatch，重点关注 A3/A5 intranode
> - 目标：明确原仓调用栈、接口参数、需要迁移的代码，以及面向训练的目标接口

## 1. 结论

`sgl-kernel-npu` 中的 normal dispatch 不是一个可以单独复制的 NPU 算子，而是一条由三个
自定义算子组成的流水线：

```text
get_dispatch_layout
  └─ DispatchLayout
       分析 Top-K 路由，生成 rank/Expert 布局和发送索引

dispatch
  ├─ NotifyDispatch
  │    各 EP rank 交换计数和 offset 元数据
  └─ CamMoeDispatchNormal
       按 Expert 重排 token，并完成 EP 通信
```

因此，要提供和 DeepEP 相同的调用方式，最少需要迁移两个公开方法：

```python
buffer.get_dispatch_layout(...)
buffer.dispatch(...)
```

不能只复制 `CamMoeDispatchNormal`。

面向训练的第一版建议：

```text
保留 DeepEP 兼容 API
只支持 normal + BF16 + 同步 + 单机 HCCS
在兼容 API 外增加 autograd 包装
先验证 forward，再补 backward
```

## 2. dispatch 在 MoE 中做什么

假设：

```text
EP size = 2
全局 Expert 数 = 4

rank 0：Expert 0、Expert 1
rank 1：Expert 2、Expert 3
```

rank 0 当前有三个 token，每个 token 选择两个 Expert：

```text
token    Top-K Expert    Router 权重
x0       [0, 2]          [0.60, 0.40]
x1       [3, 2]          [0.70, 0.30]
x2       [1, 0]          [0.55, 0.45]
```

dispatch 后应得到：

```text
rank 0:
  Expert 0: x0, x2
  Expert 1: x2

rank 1:
  Expert 2: x0, x1
  Expert 3: x1
```

因此 dispatch 同时完成两类工作：

1. 按 Expert 重新排列 token。
2. Expert 不在本 rank 时，通过 EP 通信将 token 发到目标 rank。

输出 `recv_x` 已经按本地 Expert 连续排列，可以交给 GMM：

```text
recv_x
  ├─ 本地 Expert 0 的所有 token
  ├─ 本地 Expert 1 的所有 token
  └─ ...
```

`num_recv_tokens_per_expert_list` 告诉 GMM 每段分别有多少行。

## 3. 用户侧完整调用方式

仓库测试中的基本用法为：

```python
import deep_ep

buffer = deep_ep.Buffer(ep_group)

(
    num_tokens_per_rank,
    num_tokens_per_rdma_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    event,
) = buffer.get_dispatch_layout(
    topk_idx,
    num_experts,
)

(
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
    num_recv_tokens_per_expert_list,
    handle,
    event,
) = buffer.dispatch(
    x=x,
    num_tokens_per_rank=num_tokens_per_rank,
    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
    num_tokens_per_expert=num_tokens_per_expert,
    is_token_in_rank=is_token_in_rank,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
)
```

参考测试：

```text
tests/python/deepep/test_common.py
tests/python/deepep/test_intranode.py
```

当前实现中必须先执行 `get_dispatch_layout()`。它除了返回 Python 可见结果，还会把
`send_token_idx_small`、`notify_send_data` 保存到 C++ `Buffer` 内部；后续 dispatch 会读取
这些隐藏状态。

## 4. Buffer 初始化调用栈

### 4.1 Python import

入口：

```text
python/deep_ep/deep_ep/__init__.py
```

主要工作：

1. 设置 `ASCEND_CUSTOM_OPP_PATH`。
2. 将自定义算子的 `op_api/lib` 加入动态库路径。
3. 导出 `Buffer`、`Config` 和 `EventOverlap`。

因此独立 wheel 不只是 Python 代码，还需要包含：

```text
deep_ep_cpp.so
自定义算子 OPP 目录
op_api 动态库
```

### 4.2 Python Buffer

入口：

```text
python/deep_ep/deep_ep/buffer.py::Buffer.__init__
```

调用过程：

```text
deep_ep.Buffer(ep_group)
  ├─ 从 ProcessGroup 获取 rank 和 world size
  ├─ 从 torch_npu HCCL backend 获取通信域名称
  ├─ 创建 deep_ep_cpp.Buffer
  └─ 根据 DEEP_USE_MODE 选择 normal/low-latency strategy
```

默认：

```bash
DEEP_USE_MODE=default
```

对应：

```text
DefaultNormalCommStrategy
```

策略注册与选择：

```text
python/deep_ep/deep_ep/ep_strategy.py
python/deep_ep/deep_ep/strategies/normal_strategy.py
```

### 4.3 C++ Buffer

Python 创建：

```python
self.runtime = deep_ep_cpp.Buffer(
    self.rank,
    self.group_size,
    num_nvl_bytes,
    num_rdma_bytes,
    low_latency_mode,
    moe_all_to_all_group_name,
)
```

调用：

```text
csrc/deepep/pybind_extension.cpp
  └─ csrc/deepep/deep_ep.cpp::Buffer::Buffer
```

C++ `Buffer` 保存：

```text
当前 EP rank
EP world size
HCCL 通信域
长序列 round 配置
notify_send_data
send_token_idx_small
```

最后两个字段是 layout 与 dispatch 之间的内部状态。

## 5. get_dispatch_layout 完整调用栈

```text
用户
└─ deep_ep.Buffer.get_dispatch_layout()
   # python/deep_ep/deep_ep/buffer.py
   └─ DefaultNormalCommStrategy.get_dispatch_layout()
      # python/deep_ep/deep_ep/strategies/normal_strategy.py
      └─ deep_ep_cpp.Buffer.get_dispatch_layout()
         # csrc/deepep/pybind_extension.cpp
         └─ Buffer::get_dispatch_layout()
            # csrc/deepep/deep_ep.cpp
            └─ EXEC_NPU_CMD(aclnnDispatchLayout)
               ├─ ops/op_host/op_api/aclnn_dispatch_layout.{h,cpp}
               ├─ ops/op_host/dispatch_layout.cpp
               ├─ ops/op_host/dispatch_layout_tiling.cc
               ├─ ops/op_kernel/dispatch_layout.cpp
               ├─ ops/op_kernel/dispatch_layout.h
               └─ ops/op_kernel/dispatch_layout_a2.h
```

### 5.1 Python 公开接口

```python
def get_dispatch_layout(
    self,
    topk_idx: torch.Tensor,
    num_experts: int,
    previous_event: Optional[EventOverlap] = None,
    async_finish: bool = False,
    allocate_on_comm_stream: bool = False,
)
```

参数：

| 参数 | 形状/类型 | 含义 |
| --- | --- | --- |
| `topk_idx` | `[N,K]`, `int64` | 每个 token 选择的全局 Expert 编号，`-1` 表示丢弃 |
| `num_experts` | Python `int` | EP 通信域内的全局 Expert 总数 |
| `previous_event` | 可选 `EventOverlap` | 当前操作需要等待的前置事件 |
| `async_finish` | `bool` | 是否允许当前计算流不等待通信完成 |
| `allocate_on_comm_stream` | `bool` | 张量是否在通信流上分配和管理 |

其中：

```text
N = 当前 rank 的 token 数
K = Top-K
```

例如：

```python
topk_idx = torch.tensor(
    [
        [0, 2],
        [3, 2],
        [1, 0],
    ],
    dtype=torch.int64,
    device="npu",
)
```

### 5.2 Python 可见返回值

| 返回值 | 形状/类型 | 含义 |
| --- | --- | --- |
| `num_tokens_per_rank` | `[P]`, `int32` | 当前 rank 发往每个 EP rank 的去重 token 数 |
| `num_tokens_per_rdma_rank` | 可选张量 | 分层跨节点通信的 RDMA token 数；intranode 为 `None` |
| `num_tokens_per_expert` | `[round*E]`, `int32` | 当前 rank 发往各全局 Expert 的路径数 |
| `is_token_in_rank` | `[N,P]`, `int32` | 每个 token 是否需要发往某个 EP rank |
| `event` | `EventOverlap` | layout 完成事件；当前 default normal 实际为空 |

前述示例会得到：

```text
num_tokens_per_rank   = [2, 2]
num_tokens_per_expert = [2, 1, 2, 1]

is_token_in_rank =
[
  [1, 1],
  [0, 1],
  [1, 0],
]
```

### 5.3 NPU DispatchLayout 算子参数

C++ 调用：

```cpp
EXEC_NPU_CMD(
    aclnnDispatchLayout,
    topk_idx,
    num_tokens,
    num_ranks,
    num_experts,
    num_topk,
    local_ranksize,
    per_round_tokens,
    rank_id,
    num_tokens_per_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    notify_send_data,
    send_token_idx_small
);
```

参数：

| 参数 | 含义 |
| --- | --- |
| `topk_idx` | `[N,K]` Top-K Expert 编号 |
| `num_tokens` | 当前 rank 的原始 token 数 |
| `num_ranks` | EP world size |
| `num_experts` | 全局 Expert 数 |
| `num_topk` | Top-K |
| `local_ranksize` | 节点内 rank 数；当前常量为 8 |
| `per_round_tokens` | 长序列模式每轮最多处理的 token 数 |
| `rank_id` | 当前 EP rank |
| `num_tokens_per_rank` | 输出：发往各 rank 的 token 数 |
| `num_tokens_per_expert` | 输出：发往各 Expert 的路径数 |
| `is_token_in_rank` | 输出：token 到 rank 的布尔关系 |
| `notify_send_data` | 内部输出：NotifyDispatch 所需元数据 |
| `send_token_idx_small` | 内部输出：每条 Top-K 路径的发送位置 |

### 5.4 layout 的隐藏输出

`get_dispatch_layout()` 还会把下面两个张量保存在 C++ `Buffer`：

```cpp
this->notify_send_data = notify_send_data;
this->send_token_idx_small = send_token_idx_small;
```

它们没有返回给 Python。

`send_token_idx_small` 记录：

```text
x0 → Expert 0 放在发送区域的哪个位置
x0 → Expert 2 放在发送区域的哪个位置
x1 → Expert 3 放在发送区域的哪个位置
...
```

A3/A5 intranode dispatch 会直接读取它。

`notify_send_data` 保存更多 server/Expert 计数和 offset，A2 internode 路径会读取它。

因此当前实现中的 layout 不是可省略的纯辅助接口。

## 6. dispatch 完整调用栈

对于 A3/A5 default normal 路径：

```text
用户
└─ deep_ep.Buffer.dispatch()
   # python/deep_ep/deep_ep/buffer.py
   └─ DefaultNormalCommStrategy.dispatch()
      # python/deep_ep/deep_ep/strategies/normal_strategy.py
      └─ DefaultNormalCommStrategy._intranode_dispatch()
         └─ deep_ep_cpp.Buffer.intranode_dispatch()
            # csrc/deepep/pybind_extension.cpp
            └─ Buffer::intranode_dispatch()
               # csrc/deepep/deep_ep.cpp
               ├─ EXEC_NPU_CMD(aclnnNotifyDispatch)
               │  ├─ ops/op_host/op_api/aclnn_notify_dispatch.{h,cpp}
               │  ├─ ops/op_host/notify_dispatch.cpp
               │  ├─ ops/op_host/notify_dispatch_tiling.cc
               │  ├─ ops/op_kernel/notify_dispatch.cpp
               │  ├─ ops/op_kernel/notify_dispatch.h
               │  └─ ops/op_kernel/notify_dispatch_a5.h
               └─ EXEC_NPU_CMD(aclnnCamMoeDispatchNormal)
                  ├─ ops/op_host/op_api/aclnn_cam_moe_dispatch_normal.{h,cpp}
                  ├─ ops/op_host/cam_moe_dispatch_normal.cpp
                  ├─ ops/op_host/cam_moe_dispatch_normal_tiling.cc
                  ├─ ops/op_kernel/cam_moe_dispatch_normal.cpp
                  ├─ ops/op_kernel/cam_moe_dispatch_normal.h
                  └─ ops/op_kernel/cam_moe_dispatch_normal_a5.h
```

`DefaultNormalCommStrategy.dispatch()` 会检查
`runtime.get_num_rdma_ranks()`：

```text
num_rdma_ranks > 1：_internode_dispatch
其他情况：         _intranode_dispatch
```

Ascend 910B 在 EP rank 超过节点内范围时会走 A2/RDMA 路径。本报告重点分析 A3/A5 的
`_intranode_dispatch()`。

## 7. Python dispatch 接口参数

```python
def dispatch(
    self,
    x,
    handle=None,
    num_tokens_per_rank=None,
    num_tokens_per_rdma_rank=None,
    is_token_in_rank=None,
    num_tokens_per_expert=None,
    topk_idx=None,
    topk_weights=None,
    expert_alignment=1,
    num_worst_tokens=0,
    config=None,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    dispatch_wait_recv_cost_stats=None,
    quant_mode=None,
)
```

| 参数 | 形状/类型 | 含义 | 当前 default intranode 状态 |
| --- | --- | --- | --- |
| `x` | `[N,H]` BF16 或量化 tuple | 原始 token hidden state | BF16 可作为训练首版 |
| `handle` | 可选 tuple | 复用已有通信布局 | 非空会抛 `NotImplementedError` |
| `num_tokens_per_rank` | `[P]` int32 | 发往各 EP rank 的 token 数 | layout 返回；当前 C++ 主要做存在性/形状检查 |
| `num_tokens_per_rdma_rank` | 可选张量 | 跨节点 RDMA 计数 | intranode 不使用 |
| `is_token_in_rank` | `[N,P]` | token 是否发往某 rank | Python 要求非空；当前 intranode C++ 未读取 |
| `num_tokens_per_expert` | `[round*E]` int32 | 发往各 Expert 的路径数 | 传给 NotifyDispatch |
| `topk_idx` | `[N,K]` int64 | 全局 Expert 编号 | C++ 转为 int32 `expert_ids` |
| `topk_weights` | `[N,K]` float32 | Router 权重 | 当前主要保存进 handle，供 combine 使用 |
| `expert_alignment` | Python `int` | Expert token 数对齐要求 | 当前 intranode C++ 未实际使用 |
| `num_worst_tokens` | Python `int` | 预设最大接收 token 数 | 参与 `real_max_bs` 计算 |
| `config` | `deep_ep.Config` | 性能调优配置 | 当前 intranode 明确使用 `num_sms` |
| `previous_event` | 可选 event | 前置流依赖 | 当前路径未实际等待 |
| `async_finish` | `bool` | 是否异步返回 | 当前没有有效 event |
| `allocate_on_comm_stream` | `bool` | 通信流内存管理 | 当前路径未实际使用 |
| `dispatch_wait_recv_cost_stats` | `[P]` int32 | 记录等待各 rank 数据耗时 | 可选诊断输出 |
| `quant_mode` | 字符串 | BF16/INT8/MXFP8 等 | 训练首版建议限制 `"bf16"` |

`Config` 的五个字段：

```python
Config(
    num_sms,
    num_max_nvl_chunked_send_tokens,
    num_max_nvl_chunked_recv_tokens,
    num_max_rdma_chunked_send_tokens,
    num_max_rdma_chunked_recv_tokens,
)
```

当前 intranode C++ 明确使用：

```cpp
num_channels = config.num_sms / 2;
```

其他 chunk 参数没有继续传入 `CamMoeDispatchNormal`。

## 8. Python strategy 层

入口：

```text
python/deep_ep/deep_ep/strategies/normal_strategy.py
  ::DefaultNormalCommStrategy._intranode_dispatch
```

主要逻辑：

```text
检查 quant_mode
检查 handle 必须为空
检查 layout 参数非空
调用 runtime.intranode_dispatch
构造 combine 使用的 handle
返回六个公开结果
```

Python 传给 C++ 的缓存参数目前固定为：

```python
cached_num_recv_tokens = 0
cached_rank_prefix_matrix = None
cached_channel_prefix_matrix = None
```

说明接口预留了复用布局能力，但当前没有完成。

## 9. C++ intranode_dispatch

入口：

```text
csrc/deepep/deep_ep.cpp::Buffer::intranode_dispatch
```

主要分成两个阶段。

### 9.1 NotifyDispatch：先交换元数据

调用：

```cpp
EXEC_NPU_CMD(aclnnNotifyDispatch, ...)
```

这一步不发送完整 hidden state，主要交换每个 rank/Expert 的计数和 offset。

主要输入：

| 参数 | 含义 |
| --- | --- |
| `send_data` | 待交换的 Expert 计数、offset、rank token 数 |
| `num_tokens_per_expert` | 当前 rank 发往各 Expert 的路径数 |
| `send_count` | 元数据元素总数 |
| `num_tokens` | 当前 rank 的原始 token 数 |
| `commGroup` | EP HCCL 通信域 |
| `rankSize/rankId` | EP world size 和当前 rank |
| `round/perRoundTokens` | 长序列分轮参数 |

主要输出：

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| `send_data_offset` | `[round,E]` | 当前 rank 向各 Expert 发送数据的起点 |
| `recv_count` | `[round,E]` | 当前 rank 接收的各 Expert 数据量 |
| `recv_offset` | `[round,E]` | 接收数据写入位置 |
| `expert_global_offset` | `[E_local]` | 本地 Expert 在 `recv_x` 中的起点 |
| `srcrank_in_expert_offset` | `[E_local*P]` | 来源 rank 在某 Expert 区域中的起点 |
| `r_in_srcrank_offset` | `[E_local*P*round]` | 多轮模式的接收偏移 |
| `total_recv_token` | `[1]` | 当前 rank 最终接收的 Expert 路径总数 |
| `max_bs` | `[1]` | 通信所需最大 batch |
| `recv_tokens_per_expert` | `[round*E_local]` | 每个本地 Expert 接收的 token 数 |

NotifyDispatch 完成后，C++ 才能确定：

```text
recv_x 应该分配多少行
每个本地 Expert 占哪一段
每个来源 rank 写到哪个 offset
```

### 9.2 CamMoeDispatchNormal：重排和传输数据

调用：

```cpp
EXEC_NPU_CMD(aclnnCamMoeDispatchNormal, ...)
```

它完成：

```text
读取 x
根据 expert_ids 找出每条 Top-K Expert 路径
根据 layout/notify offset 排列数据
通过 HCCS/HCCL 发送到目标 rank
按本地 Expert 连续写入 recv_x
为 combine 生成来源索引
```

参数分组如下。

原始数据：

| 参数 | 含义 |
| --- | --- |
| `new_x` | `[N,H]` 原始 token |
| `expert_ids` | `[N,K]` int32，由 `topk_idx` 转换 |

DispatchLayout 结果：

| 参数 | 含义 |
| --- | --- |
| `send_token_idx_small` | 每条 Top-K 路径在发送区域中的位置 |

NotifyDispatch 结果：

| 参数 | 含义 |
| --- | --- |
| `send_data_offset` | 各 Expert 发送区域起点 |
| `recv_offset` | 各 Expert 接收区域起点 |
| `recv_count` | 各 Expert 接收数量 |
| `expert_global_offset` | 本地 Expert 在输出中的总起点 |
| `srcrank_in_expert_offset` | 来源 rank 在 Expert 区域中的起点 |
| `r_in_srcrank_offset` | 长序列分轮偏移 |

通信属性：

| 参数 | 含义 |
| --- | --- |
| `hcom_ep_name` | EP HCCL 通信域 |
| `num_ranks/rank` | EP world size 和当前 rank |
| `tp_size/tp_rank` | TP 参数；当前路径固定为 1/0 |
| `num_experts` | 全局 Expert 总数 |

运行配置：

| 参数 | 含义 |
| --- | --- |
| `quant_mode` | 0 为不量化，其他值对应 INT8/FP8 等 |
| `real_max_bs` | 实际预留的最大接收 token 数 |
| `global_bs` | 全局预留规模 |
| `round` | 长序列拆成几轮 |
| `per_round_tokens` | 每轮 token 上限 |

输出：

| 输出 | 含义 |
| --- | --- |
| `expandx_out` | 最终返回的 `recv_x` |
| `dynamic_scales_out` | 量化 scale |
| `expand_idx_out` | combine 使用的来源信息 |
| `dispatch_wait_recv_cost_stats_out` | 等待各来源 rank 的耗时 |

### 9.3 expand_idx_out

每条 dispatch 后的 Expert 路径记录三个 `int32`：

```text
(source_ep_rank, source_token_index, topk_slot)
```

例如：

```text
(0, 2, 1)
```

表示：

```text
这条 Expert 输入来自 EP rank 0 的第 2 个原始 token，
对应该 token 的 Top-K 第 1 个位置。
```

kernel 写入逻辑：

```text
csrc/deepep/ops/op_kernel/cam_moe_dispatch_normal.h::FillTriple
csrc/deepep/ops/op_kernel/cam_moe_dispatch_normal_a5.h::FillTriple
```

它在 Python handle 中被命名为 `recv_src_idx`，是 combine 将 Expert 输出送回原 token 的
关键依据。

## 10. dispatch 返回值

```python
(
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
    num_recv_tokens_per_expert_list,
    handle,
    event,
)
```

| 返回值 | 含义 |
| --- | --- |
| `recv_x` | `[R,H]`，当前 rank 收到并按本地 Expert 排列的输入 |
| `recv_topk_idx` | 设计上是接收侧 Expert 索引 |
| `recv_topk_weights` | 设计上是接收侧 Router 权重 |
| `num_recv_tokens_per_expert_list` | 每个本地 Expert 的 token 数或前缀和 |
| `handle` | combine 和训练 backward 使用的通信上下文 |
| `event` | 异步完成事件 |

其中：

```text
R = 当前 rank 接收的 Expert 路径总数
H = hidden size
```

环境变量：

```bash
MOE_EXPERT_TOKEN_NUMS_TYPE=1
```

返回各 Expert 的原始数量，例如：

```text
[120, 87]
```

设置为 0 时返回前缀和：

```text
[120, 207]
```

## 11. handle 结构

当前 intranode handle：

```python
handle = (
    rank_prefix_matrix,
    channel_prefix_matrix,
    recv_channel_prefix_matrix,
    recv_src_idx,
    is_token_in_rank,
    send_head,
    topk_idx,
    topk_weights,
)
```

当前 combine 真正使用的主要字段：

```text
recv_src_idx
send_head
topk_idx
topk_weights
```

其中：

```text
recv_src_idx = expand_idx_out
send_head = recv_count.sum(dim=0)
```

前三个 prefix matrix 在当前 intranode C++ 中通过 `at::empty()` 分配，但没有传给
dispatch 算子写入，主要是 API 兼容占位字段。

## 12. 当前实现确认的缺口

### 12.1 recv_topk_idx 和 recv_topk_weights 未真正生成

C++ 代码仅执行：

```cpp
recv_topk_idx = at::empty(...);
recv_topk_weights = at::empty(...);
```

随后调用 `aclnnCamMoeDispatchNormal` 时，没有将它们作为输出传给 kernel。

因此当前返回的：

```text
recv_topk_idx
recv_topk_weights
```

属于未初始化内容，不能直接用于训练。现有测试基本使用 `_` 忽略它们。

### 12.2 handle 复用未实现

虽然公开接口支持：

```python
dispatch(handle=handle)
```

default normal strategy 遇到非空 handle 会直接抛：

```python
NotImplementedError
```

### 12.3 多个兼容参数未生效

当前 intranode C++ 中没有实际使用或完整实现：

```text
is_token_in_rank
expert_alignment
cached_num_recv_tokens
cached_rank_prefix_matrix
cached_channel_prefix_matrix
previous_event
async_finish
allocate_on_comm_stream
```

不能只依据 Python docstring 认为这些能力已经具备。

### 12.4 event/stream 尚未完成

当前：

```text
Python EventOverlap.current_stream_wait() 为空操作
C++ normal 路径返回的 event 为 nullopt
```

因此设置 `async_finish=True` 不代表已经能正确进行训练 overlap。

### 12.5 存在 D2H/CPU 同步

C++ 路径包含：

```cpp
max_bs.item<int>()
total_recv_token.item<int>()
recv_tokens_per_expert.to(at::kCPU)
```

第一版正确性 POC 可以接受，但会影响通信计算掩盖、图模式和动态 shape 性能。

### 12.6 没有 autograd

`Buffer.dispatch()` 是普通 Python → pybind → C++ 扩展调用，没有
`torch.autograd.Function` 或 autograd 注册。

直接复制 forward 代码只能让接口跑通，不能自动完成训练反向。

## 13. 目标迁移接口

### 13.1 保留 DeepEP 原始兼容接口

wheel 的 Python 模块名和 Buffer API 建议保持：

```python
import deep_ep

buffer = deep_ep.Buffer(ep_group)
```

Layout：

```python
(
    num_tokens_per_rank,
    num_tokens_per_rdma_rank,
    num_tokens_per_expert,
    is_token_in_rank,
    event,
) = buffer.get_dispatch_layout(
    topk_idx,
    num_experts,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
)
```

Raw dispatch：

```python
(
    recv_x,
    recv_topk_idx,
    recv_topk_weights,
    tokens_per_expert,
    handle,
    event,
) = buffer.dispatch(
    x=x,
    handle=None,
    num_tokens_per_rank=num_tokens_per_rank,
    num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
    is_token_in_rank=is_token_in_rank,
    num_tokens_per_expert=num_tokens_per_expert,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    expert_alignment=1,
    num_worst_tokens=0,
    config=None,
    previous_event=None,
    async_finish=False,
    allocate_on_comm_stream=False,
    dispatch_wait_recv_cost_stats=None,
    quant_mode="bf16",
)
```

首版可限制：

```text
normal
BF16
handle=None
async_finish=False
单机 HCCS
A3/A5
```

但应保留所有参数名称和返回值位置，避免 MindSpeed 集成后再次修改调用点。

### 13.2 增加训练便利接口

在 raw API 之外，可以增加：

```python
recv_x, tokens_per_expert, dispatch_ctx = deep_ep.dispatch_for_training(
    buffer=buffer,
    x=x,
    topk_idx=topk_idx,
    topk_weights=topk_weights,
    num_experts=num_experts,
)
```

内部：

```text
get_dispatch_layout
→ raw dispatch
→ 保存反向需要的 handle
```

但该便利接口不能取代 DeepEP 兼容接口。

## 14. dispatch 的训练反向

若 dispatch 只负责 token 搬运和重排：

```text
正向：
  原始 token
  → 按 Expert 重排并发送到目标 rank

反向：
  Expert 输入梯度
  → 沿来源索引送回原 rank
  → 同一 token 的多条 Expert 路径求和
```

因此：

```text
dispatch.backward 可以复用 unweighted combine
```

伪代码：

```python
class DispatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ...):
        recv_x, _, _, counts, handle, _ = buffer.dispatch(...)
        ctx.handle = handle
        return recv_x, counts

    @staticmethod
    def backward(ctx, grad_recv_x, _):
        grad_x, _, _ = buffer.combine(
            grad_recv_x,
            handle=ctx.handle,
            topk_weights=None,
        )
        return grad_x, ...
```

这里 combine 必须不乘 Router 权重，因为正向 dispatch 本身没有乘权重。

当前 Python `DefaultNormalCommStrategy._intranode_combine()` 会忽略调用者传入的
`topk_weights`，固定使用 handle 中保存的原始 Router 权重。底层 C++ 在权重为空时可以生成
全 1，但 Python 没有把 `None` 传下去。

因此实现 dispatch 反向前，需要先改造 combine：

```text
普通 forward combine：
  使用 Router 权重

dispatch backward：
  使用全 1，不重复乘 Router 权重
```

另外：

```text
topk_idx 是离散索引，没有梯度
Router 权重梯度属于 weighted combine 的反向
```

## 15. 需要迁移的代码

不能只复制 `cam_moe_dispatch_normal`，至少包括以下内容。

### 15.1 Python API 和策略

```text
python/deep_ep/deep_ep/__init__.py
python/deep_ep/deep_ep/buffer.py
python/deep_ep/deep_ep/ep_strategy.py
python/deep_ep/deep_ep/strategies/normal_strategy.py
python/deep_ep/deep_ep/utils.py
```

### 15.2 Python/C++ 绑定和 runtime

```text
csrc/deepep/pybind_extension.cpp
csrc/deepep/deep_ep.hpp
csrc/deepep/deep_ep.cpp
csrc/deepep/config.hpp
csrc/deepep/event.hpp
csrc/deepep/pytorch_npu_helper.hpp
```

### 15.3 DispatchLayout

```text
csrc/deepep/ops/op_host/op_api/aclnn_dispatch_layout.*
csrc/deepep/ops/op_host/dispatch_layout.cpp
csrc/deepep/ops/op_host/dispatch_layout_tiling.cc
csrc/deepep/ops/op_kernel/dispatch_layout*
```

### 15.4 NotifyDispatch

```text
csrc/deepep/ops/op_host/op_api/aclnn_notify_dispatch.*
csrc/deepep/ops/op_host/notify_dispatch.cpp
csrc/deepep/ops/op_host/notify_dispatch_tiling.cc
csrc/deepep/ops/op_kernel/notify_dispatch*
```

### 15.5 CamMoeDispatchNormal

```text
csrc/deepep/ops/op_host/op_api/aclnn_cam_moe_dispatch_normal.*
csrc/deepep/ops/op_host/cam_moe_dispatch_normal.cpp
csrc/deepep/ops/op_host/cam_moe_dispatch_normal_tiling.cc
csrc/deepep/ops/op_kernel/cam_moe_dispatch_normal*
```

### 15.6 构建和发布

```text
公共通信头文件
公共 tiling 头文件
CMake 和算子构建脚本
自定义 OPP 安装目录
op_api 动态库
deep_ep wheel 打包逻辑
MIT LICENSE 和原版权声明
```

## 16. 哪些可以直接复制，哪些需要补写

### 16.1 可以整体迁移

```text
DispatchLayout 的 op_host/op_kernel/op_api
NotifyDispatch 的 op_host/op_kernel/op_api
CamMoeDispatchNormal 的 op_host/op_kernel/op_api
pybind Buffer 框架
Config 与 ACLNN 调用辅助代码
OPP 和 wheel 构建逻辑
```

不建议只挑 kernel 主文件，因为 tiling、注册、公共通信头文件和 op_api 之间存在依赖。

### 16.2 需要修改或新增

```text
有效的 recv_topk_idx/recv_topk_weights 数据通路
训练 autograd Function
dispatch backward 的 unweighted combine
combine backward 的 dispatch
Router 权重梯度
真实 event/stream 等待
handle cache 或明确删除无效缓存语义
减少 total_recv_token/tokens_per_expert 的 D2H 同步
MindSpeed dispatcher/feature flag
forward、backward 和边界场景测试
```

## 17. 第一版实施建议

第一阶段不要同时处理量化、跨机、overlap 和 fused Expert FFN。

建议按以下顺序：

```text
阶段 1：
  抽取并构建独立 deep_ep wheel
  跑通 get_dispatch_layout + dispatch + combine
  限定 BF16、单机 A3/A5

阶段 2：
  对比 DeepEP forward 与 MindSpeed AllToAll forward
  验证 recv_x、tokens_per_expert 和最终 combine 输出

阶段 3：
  增加最小 autograd
  dispatch backward → unweighted combine
  combine backward → dispatch

阶段 4：
  验证 hidden_states、Expert 权重和 Router 权重梯度

阶段 5：
  接入 MindSpeed 普通非 overlap dispatcher

阶段 6：
  补 event/stream 和通信计算 overlap
```

## 18. 当前可形成的汇报结论

> 已完成 SGL DeepEP normal dispatch 的源码级调用栈梳理。该接口并非单一算子，而是由
> DispatchLayout、NotifyDispatch 和 CamMoeDispatchNormal 三个 NPU 算子组成，layout 与
> dispatch 之间还存在 C++ Buffer 内部状态依赖。现有 Python API 基本兼容 DeepEP，但当前
> 实现仍偏 forward/推理使用：handle 复用、有效 recv top-k 数据、异步 event 和 autograd
> 尚未完成。第一阶段可以整体迁移三个算子流水线与最小 adapter，先完成 BF16 forward POC；
> 要用于 MindSpeed 训练，还需要补充无权重 combine 反向、weighted combine 反向和 Router
> 权重梯度。
