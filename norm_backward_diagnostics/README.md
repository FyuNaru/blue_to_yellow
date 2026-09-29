# LayerNorm / RMSNorm 独立反向定位

两个命令入口共用 `norm_common.py`；将三个 Python 文件放在同一目录。无需 ATK，
NPU 需要 torch/torch_npu，GPU 需要 CUDA PyTorch 和 NVIDIA TE 2.17。
本工具仅针对有限值 FP32 独立反向用例；不改 TENPU，不运行其中的补偿求和逻辑。

## 输入准备

正常执行独立反向 ATK task，附加 `--save_data input:bin`。取实际失败用例的
`input.bin`，将同一文件复制到两台机器。必须包含已保存的公共统计量：

- LayerNorm：x, dy, gamma, beta, eps, zero_centered_gamma, means, rstdevs。
- RMSNorm：x, dy, gamma, eps, zero_centered_gamma, rstdevs。

支持 ATK 保存的单元素 bundle 列表、bundle 字典和上述位置参数列表。
不接受仅有前向输入的文件，不重新计算统计量。只加载可信的自有 ATK 输入文件：
ATK protocol-4 格式使用 pickle，需要 `weights_only=False`。

## 运行

GPU 必须在 TENPU 仓库之外运行，PYTHONPATH 不得指向 TENPU。建议两台机器
都将脚本目录放在 `/tmp/norm_backward_diagnostics`，输入放在 `/tmp/input.bin`。

NPU：

```bash
cd /tmp/norm_backward_diagnostics
python npu_vs_cpu.py --op rmsnorm --input /tmp/input.bin --output /tmp/rmsnorm_npu
```

GPU：

```bash
cd /tmp/norm_backward_diagnostics
python gpu_vs_cpu.py --op rmsnorm --input /tmp/input.bin --output /tmp/rmsnorm_gpu
```

LayerNorm 将 `--op rmsnorm` 改为 `--op layernorm`，换成对应的输入文件及结果目录。
`--device` 默认 0；`--chunk-rows` 默认 4096，仅控制 CPU 参考计算的分块，不改变设备
反向调用或 shape。结果目录必须是新目录。返回码 0 表示全部通过，1 表示有精度失败；
异常 traceback 表示未成功完成诊断，不能作为精度失败结论。

## 比较规则与输出

CPU FP64 按反向公式计算，使用输入文件中的 FP32 统计量及有效权重，不重新跑前向。
GPU 使用 NVTE 2.17 的原始反向接口，SM margin=0；NPU 直接调用底层接口：

- RMSNorm: torch_npu.npu_rms_norm_backward。
- LayerNorm: torch.ops.aten.native_layer_norm_backward。

使用本地 ATK bac1aa7 的 mixed_tolerance_bm **FP32 默认值**：

- `abs(actual - reference) <= 1.53e-5 + 9.77e-4 * abs(reference)` 的比例至少 0.99。
- 每个元素还必须满足绝对误差不超过 0.01，或不超过参考值向正无穷方向的 32 ULP。
- 两项同时满足才通过。CPU FP64 参考转 FP32 后应用上述 ATK 单标杆判定；报告同时
  保留相对未舍入 FP64 参考的误差。未加载服务器的 ATK，不自动读取 YAML 自定义阈值。
  如服务器改过默认值或 YAML 配置覆盖了阈值，应先核对，不能称为完全同一标准。

输出包括：

- `report.json`：各输出判定、匹配比例、失败数量、前 32 个失败位置、阈值、输入 SHA256、
  torch/设备/接口信息。提交问题单时另附 torch_npu/CANN/驱动版本、NPU 型号等环境信息。
- `dw_channels.csv`：所有通道的设备 dw、CPU FP64 dw 和绝对误差。
- `native_outputs.pt`：设备原始 dw、db；指定 `--save-dx` 才保存较大的 dx，dx 始终参与比较。
- `cpu_reference_fp64.pt`：CPU FP64 dw、db；FP64 dx 分块计算，不整份保存。

两端先核对 report 的 input_sha256 一致。由于 CPU 软硬件可能不同，FP64 参考末位也可能
有差异；两份参考均保存，必要时核对，不能声称两台 CPU 的结果必然逐位一致。
NPU 对 CPU、GPU 对 CPU 是新增的诊断比较，不等同于原 NPU 对 GPU 的正式结论。
如两端均通过而彼此比较失败，也应保留原问题并分析数值差异。

CPU-only 测试：`python -m pytest -q test_norm_common.py`。本地 CPU 测试不代表真实设备验证。
