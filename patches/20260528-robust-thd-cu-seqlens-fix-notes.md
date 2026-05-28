# Robust THD cu_seqlens fix for verl + Megatron 3714d8

Use this pair instead of `20260528-verl-v071-source-real-thd-cu-seqlens.patch`.

## Why the previous source-only patch failed

The source-only patch set both normal and padded fields to real lengths. That can fix a case where
the tensor is unpadded, for example:

```text
tensor size = 284
split sizes = [296]
```

But it breaks a case where the actual THD tensor still contains padding slots:

```text
tensor size = 584
split sizes = [284, 293]
```

Here `[284, 293]` are real local lengths and sum to `577`, while the tensor has `584` padded local
slots.

## Patch order

Apply in `verl release/v0.7.1`:

```bash
git apply /path/to/blue_to_yellow/patches/20260528-verl-v071-preserve-real-and-padded-cu-seqlens.patch
```

If your verl tree has `preprocess_thd_engine` in `verl/models/mcore/util.py`, also apply:

```bash
git apply /path/to/blue_to_yellow/patches/20260528-verl-thd-engine-preserve-real-and-padded-cu-seqlens.patch
```

Apply in `Megatron-LM` dev commit `3714d81d418c`:

```bash
git apply /path/to/blue_to_yellow/patches/20260528-megatron-3714d8-rope-select-matching-cu-seqlens.patch
```

## What changes

The verl patches preserve both meanings at every `PackedSeqParams` construction point used by the
current THD paths:

```text
cu_seqlens_q/kv          = real lengths
cu_seqlens_q/kv_padded   = padded lengths
```

The Megatron patch changes THD RoPE to choose whichever metadata matches the actual current tensor
length after CP splitting. If the tensor is unpadded, it chooses real lengths. If the tensor is
padded, it chooses padded lengths.
