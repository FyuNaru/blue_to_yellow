# verl 最新 main 内存增长排查 patch

本目录存放从 `msverl_patch` 迁移并重新适配到当前 `verl main` 的内存排查 patch。

## 适配基线

- 目标仓库：`verl`
- 目标分支：`main`
- 目标提交：`8a694930`
- 适配日期：`2026-06-22`

原始 patch 不能直接应用到该版本，主要原因是最新 `verl` 已经移除了旧的 `verl/workers/megatron_workers.py`，相关 worker 逻辑迁移到了 `verl/workers/engine_workers.py`；同时 trainer 到 worker 的数据传递已经改为 TensorDict 路径，因此 `global_steps` 需要通过 `tu.assign_non_tensor(...)` 传入 worker。

## 文件说明

- `patches/20260622-verl-mainflow-memtrack-main-8a694930.patch`
  - 全链路排查 patch。
  - 在 trainer 主流程打印 `[mainflow_probe]`。
  - 在 worker 关键位置打印 `[mainflow_mem]`。
  - 用于判断内存增长主要发生在 `compute_log_prob`、`update_actor`、`update_weights` 或 rollout wake up 链路。

- `patches/20260622-verl-update-actor-memtrack-main-8a694930.patch`
  - 最小排查 patch。
  - 只在 `update_actor` 前后打印 `[update_actor_mem]`。
  - 用于减少日志量，只盯 actor 更新阶段。

- `scripts/analyze_mainflow_mem.py`
  - 解析 `[mainflow_mem]` 日志，按 step/rank 聚合各点位 used/free/total，以及相邻点位的增量。
  - 只适用于全链路 patch 的 `[mainflow_mem]` 日志。

## 应用方式

两个 patch 是替代方案，建议一次只应用一个。

全链路排查：

```bash
cd /path/to/verl
git apply /path/to/blue_to_yellow/verl_memtrack_latest_main/patches/20260622-verl-mainflow-memtrack-main-8a694930.patch
```

最小 `update_actor` 排查：

```bash
cd /path/to/verl
git apply /path/to/blue_to_yellow/verl_memtrack_latest_main/patches/20260622-verl-update-actor-memtrack-main-8a694930.patch
```

应用前可先检查：

```bash
git apply --check /path/to/blue_to_yellow/verl_memtrack_latest_main/patches/20260622-verl-mainflow-memtrack-main-8a694930.patch
git apply --check /path/to/blue_to_yellow/verl_memtrack_latest_main/patches/20260622-verl-update-actor-memtrack-main-8a694930.patch
```

回滚：

```bash
git apply -R /path/to/blue_to_yellow/verl_memtrack_latest_main/patches/20260622-verl-mainflow-memtrack-main-8a694930.patch
git apply -R /path/to/blue_to_yellow/verl_memtrack_latest_main/patches/20260622-verl-update-actor-memtrack-main-8a694930.patch
```

## 运行环境变量

全链路排查：

```bash
export RAY_DEDUP_LOGS=0
export VERL_MAIN_FLOW_MEM=1
```

最小 `update_actor` 排查：

```bash
export RAY_DEDUP_LOGS=0
export VERL_UPDATE_ACTOR_MEM=1
```

`RAY_DEDUP_LOGS=0` 用于避免 Ray 去重后吞掉重复格式的 worker 日志。

## 日志过滤

全链路：

```bash
grep -n "\[mainflow_probe\]" train.log
grep -n "\[mainflow_mem\]" train.log
grep -n "\[mainflow_mem_failed\]" train.log
```

最小 `update_actor`：

```bash
grep -n "\[update_actor_mem\]" train.log
grep -n "\[update_actor_mem_failed\]" train.log
```

## 全链路分析脚本

直接打印聚合表：

```bash
python /path/to/blue_to_yellow/verl_memtrack_latest_main/scripts/analyze_mainflow_mem.py train.log
```

只看某个 rank：

```bash
python /path/to/blue_to_yellow/verl_memtrack_latest_main/scripts/analyze_mainflow_mem.py train.log --rank 0
```

导出 CSV：

```bash
python /path/to/blue_to_yellow/verl_memtrack_latest_main/scripts/analyze_mainflow_mem.py train.log --csv-prefix /tmp/mainflow
```

会生成：

- `/tmp/mainflow.points.csv`
- `/tmp/mainflow.intervals.csv`
- `/tmp/mainflow.step_points.csv`
- `/tmp/mainflow.step_intervals.csv`

## 当前点位

trainer 主流程点位：

- `step_begin`
- `after_gen_sleep_replicas`
- `after_reward`
- `before_old_log_prob`
- `after_old_log_prob`
- `before_ref_log_prob`
- `after_ref_log_prob`
- `before_values`
- `after_values`
- `after_adv`
- `after_update_critic`
- `before_update_actor`
- `after_update_actor`
- `after_save_checkpoint`
- `before_checkpoint_manager_update_weights`
- `after_checkpoint_manager_update_weights`

worker 内存点位：

- `compute_log_prob_begin`
- `compute_log_prob_end`
- `update_actor_begin`
- `update_actor_end`
- `update_weights_begin`
- `rollout_mode_begin`
- `before_wake_up_weights`

## 注意事项

- 这些 patch 是排障工具，不建议长期保留在生产训练代码中。
- 内存读取使用 `torch.npu.mem_get_info()`，如果环境没有 `torch.npu`，日志会打印 `*_failed`，不会中断训练。
- 如果后续 `verl main` 继续重构 `ray_trainer.py` 或 `engine_workers.py`，需要重新执行 `git apply --check`，不保证自动适配未来提交。
