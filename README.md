# blue_to_yellow

## MoE learning closures

See [moe_learning_closures](./moe_learning_closures) for four small PyTorch examples that can run on NPU with `torch_npu`:

1. scalar backward
2. MoE permute / unpermute
3. AllToAll / AllToAllV
4. MoE dispatch with AllToAll

## TorchTitanTurbo + FSDPTurbo

See [torchtitanturbo_fsdpturbo_qwen35_runbook.md](./torchtitanturbo_fsdpturbo_qwen35_runbook.md)
for the Qwen3.5 environment check, debug baseline, native FSDP baseline, and
FSDPTurbo custom FSDP validation commands.
