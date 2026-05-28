# THD RoPE cu_seqlens mismatch fix

This patch set targets errors like:

```text
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 284,
but got split_sizes=[296]
```

## Patch order

1. Apply `20260528-verl-packed-seq-real-cu-seqlens.patch` in the `verl` repository.
2. Apply `20260528-megatron-rope-select-matching-cu-seqlens.patch` in the `Megatron-LM` repository.

## Why

`cu_seqlens_q` and `cu_seqlens_kv` should describe the real THD tensor length used by RoPE.
`cu_seqlens_q_padded` and `cu_seqlens_kv_padded` should remain available for kernels that need padded or aligned starts.

The Megatron-LM patch is defensive: for THD packed sequence RoPE it chooses the `cu_seqlens` variant whose local length matches the current query/key tensor after CP splitting.

## Apply

```bash
cd /path/to/verl
git apply /Users/wangjinyi/workspace/blue_to_yellow/patches/20260528-verl-packed-seq-real-cu-seqlens.patch

cd /path/to/Megatron-LM
git apply /Users/wangjinyi/workspace/blue_to_yellow/patches/20260528-megatron-rope-select-matching-cu-seqlens.patch
```
