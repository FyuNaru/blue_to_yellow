# MindSpeed-only THD RoPE cu_seqlens mismatch fix

This patch targets the reproduced single-sequence error:

```text
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 284,
but got split_sizes=[296]
```

## Patch

Apply `20260528-mindspeed-single-seq-rope-cu-seqlens.patch` in the `MindSpeed` repository.

## Why

On Megatron-LM dev commit `3714d81d418c9f1bca4594fc35f9e8289f652862`, THD RoPE prefers padded
`cu_seqlens` when the padded fields are present. In the reproduced failure, the local query tensor
has 284 tokens, but RoPE receives padded local length 296:

```text
query.size(0) = 284
padded local seqlen = 296
```

MindSpeed can avoid changing Megatron by fixing the `cu_seqlens` passed into MindSpeed's
`apply_rotary_pos_emb` wrapper. For the current single-sequence case, the real RoPE length can be
derived from `t.size(0)` and `cp_group.size()`.

This deliberately does not solve the older multi-sequence packed case because padded lengths alone
are not enough to recover every original per-sample length.

## Apply

```bash
cd /path/to/MindSpeed
git apply /Users/wangjinyi/workspace/blue_to_yellow/patches/20260528-mindspeed-single-seq-rope-cu-seqlens.patch
```
