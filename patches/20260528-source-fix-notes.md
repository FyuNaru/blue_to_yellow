# Source-side THD cu_seqlens fix

This patch is for testing the root-cause hypothesis at the metadata source.

## Patch

Apply `20260528-verl-source-real-thd-cu-seqlens.patch` in the `verl` repository.

```bash
cd /path/to/verl
git apply /path/to/blue_to_yellow/patches/20260528-verl-source-real-thd-cu-seqlens.patch
```

## What It Changes

In `verl/models/mcore/util.py`, `PackedSeqParams` is changed so both normal and padded `cu_seqlens`
fields carry the real THD token lengths:

```text
cu_seqlens_q          = cu_seqlens
cu_seqlens_kv         = cu_seqlens
cu_seqlens_q_padded   = cu_seqlens
cu_seqlens_kv_padded  = cu_seqlens
```

This is intentional for the current failure: Megatron-LM dev commit `3714d81d418c` prefers
`cu_seqlens_q_padded` / `cu_seqlens_kv_padded` for THD RoPE whenever they are present. If verl fills
those fields with padded lengths while the actual THD tensor is unpadded, RoPE and varlen attention
can receive lengths like `296` for a tensor whose local length is `284`.

## Scope

This is a source-side validation patch. It may be too broad if a later path truly needs padded starts
for a padded THD tensor. If this fixes the training run, the production fix should preserve both real
and padded metadata with explicit consumers instead of letting Megatron infer semantics from field
names.
