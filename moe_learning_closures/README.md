# MoE Learning Closures

This directory contains four small, runnable learning closures for the concepts behind MoE dispatch and the MindSpeed `moe_permute_fusion` issue.

The scripts prefer NPU when `torch_npu` is available, and fall back to CPU otherwise. Distributed examples use `hccl` on NPU and `gloo` on CPU.

## 1. Scalar Backward

Manual backpropagation for:

```text
y = x * w + b
loss = (y - target)^2
```

Run:

```bash
python3 01_scalar_backward.py
```

## 2. MoE Permute / Unpermute

A single-process top-1 MoE toy example:

```text
hidden_states + expert_ids -> permute by expert -> unpermute back
```

Run:

```bash
python3 02_moe_permute_unpermute.py
```

## 3. AllToAll

Two-part example:

- Python list simulation of uneven token exchange.
- Real `torch.distributed.all_to_all_single` with variable split sizes.

Run on 1 process for the list simulation only:

```bash
python3 03_all_to_all.py
```

Run distributed on NPU:

```bash
torchrun --nproc_per_node 2 03_all_to_all.py
```

## 4. MoE + AllToAll Pipeline

Distributed toy MoE dispatch:

```text
local tokens -> router -> permute -> all_to_all_v -> local experts
-> all_to_all_v -> unpermute
```

Run on NPU:

```bash
torchrun --nproc_per_node 2 04_moe_alltoall_pipeline.py
```

You can scale `--nproc_per_node` up to the number of available devices, for example `8` on one 8-card NPU machine.

## Notes

The important invariant in examples 3 and 4 is:

```text
for every pair of ranks i and j:
rank i input_split_sizes[j] == rank j output_split_sizes[i]
```

This is the kind of invariant that can be broken when a fused MoE permute path uses dynamic-token layout while the communication path expects pad-to-capacity layout.
