# THD RoPE cu_seqlens mismatch fix

This patch set targets the reproduced error:

```text
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 284,
but got split_sizes=[296]
```

## Patch order

1. Apply `20260528-verl-packed-seq-real-cu-seqlens.patch` in the `verl` repository.
2. Apply `20260528-megatron-rope-select-matching-cu-seqlens.patch` in the `Megatron-LM` repository.

## Why

On Megatron-LM dev commit `3714d81d418c9f1bca4594fc35f9e8289f652862`, THD RoPE prefers
`cu_seqlens_q_padded` and `cu_seqlens_kv_padded` whenever they are present. That is wrong for
the reproduced failure when the query/key tensors are unpadded:

```text
query.size(0) = 284
padded local seqlen = 296
```

RoPE calls `torch.split(t, seqlens)`, so its `cu_seqlens` must describe the real local THD tensor
length. The padded fields should remain available for attention kernels that need padded or aligned
starts.

The verl patch makes `cu_seqlens_q` and `cu_seqlens_kv` carry the real unpadded lengths, while still
keeping `cu_seqlens_q_padded` and `cu_seqlens_kv_padded`.

The Megatron-LM patch is intentionally narrow for this reproduced issue: in THD RoPE it uses
`packed_seq_params.cu_seqlens_q` and `packed_seq_params.cu_seqlens_kv` instead of the padded variants.

## Apply

```bash
cd /path/to/verl
git apply /Users/wangjinyi/workspace/blue_to_yellow/patches/20260528-verl-packed-seq-real-cu-seqlens.patch

cd /path/to/Megatron-LM
git apply /Users/wangjinyi/workspace/blue_to_yellow/patches/20260528-megatron-rope-select-matching-cu-seqlens.patch
```
