# verl + MindSpeed Megatron Engine 注册问题定位

## 背景

报错信息：

```text
AssertionError: Unknown device: npu for model_type: language model and backend: megatron
```

该问题发生在 verl 根据 `model_type + backend + 当前 device` 从 `EngineRegistry` 中查找训练引擎时。当前设备被识别为 `npu`，但 `language_model + megatron + npu` 对应的 engine class 没有成功注册。

本文基于本地 verl main 分支最新代码定位，代码版本：

```text
657cfa5e [trainer] feat: async generation dump with exception propagation and streaming write (#6324)
```

## 关键结论

这个报错不是直接说明“MindSpeed patch 没打上”，而是说明 verl 的 NPU Megatron engine 没有注册成功。

NPU Megatron engine 的注册依赖 `verl.workers.engine.mindspeed` 模块被成功 import。如果该 import 过程中发生 `ImportError`，`verl/workers/engine/__init__.py` 会吞掉异常，并把 `MindspeedEngineWithLMHead` 置为 `None`。后续程序继续运行，直到真正取 engine class 时才报 `Unknown device: npu`。

因此，用户感觉“patch 打不上不报错，然后接着往下跑”，更准确地说是：

```text
MindSpeed engine import 失败被吞掉，导致注册装饰器没有执行，最终 EngineRegistry 查询 NPU Megatron engine 失败。
```

## 代码定位

### 1. EngineRegistry 的注册与查询逻辑

文件：

```text
verl/workers/engine/base.py
```

关键代码：

```python
class EngineRegistry:
    _engines = {}

    @classmethod
    def register(cls, model_type: str, backend: list[str] | str, device: list[str] | str = "cuda"):
        def decorator(engine_class):
            assert issubclass(engine_class, BaseEngine)
            if model_type not in cls._engines:
                cls._engines[model_type] = {}

            backends = backend if isinstance(backend, list) else [backend]
            devices = device if isinstance(device, list) else [device]
            for current_backend in backends:
                for current_device in devices:
                    if current_backend not in cls._engines[model_type]:
                        cls._engines[model_type][current_backend] = {}
                    if current_device not in cls._engines[model_type][current_backend]:
                        cls._engines[model_type][current_backend][current_device] = engine_class

            return engine_class

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        assert model_type in cls._engines, f"Unknown model_type: {model_type}"
        assert backend in cls._engines[model_type], f"Unknown backend: {backend}"
        device = get_device_name()
        assert device in cls._engines[model_type][backend], (
            f"Unknown device: {device} for model_type: {model_type} and backend: {backend}"
        )
        return cls._engines[model_type][backend][device]
```

对应本次报错的触发点是：

```python
assert device in cls._engines[model_type][backend]
```

说明 `model_type="language_model"` 和 `backend="megatron"` 存在，但其下面没有 `device="npu"`。

### 2. MindSpeed engine import 被 ImportError 静默吞掉

文件：

```text
verl/workers/engine/__init__.py
```

关键代码：

```python
# Mindspeed must be imported before Megatron to ensure the related monkey patches take effect as expected
try:
    from .mindspeed import MindspeedEngineWithLMHead, MindspeedEngineWithValueHead, MindSpeedLLMEngineWithLMHead

    __all__ += ["MindspeedEngineWithLMHead", "MindspeedEngineWithValueHead", "MindSpeedLLMEngineWithLMHead"]
except ImportError:
    MindspeedEngineWithLMHead = None
    MindspeedEngineWithValueHead = None
    MindSpeedLLMEngineWithLMHead = None
```

这里有两个重点：

1. 注释明确要求 MindSpeed 必须在 Megatron 前 import，以保证 monkey patch 生效。
2. 如果 `.mindspeed` import 失败，异常会被吞掉，不会立即暴露真实原因。

这就是“看起来没有报 patch 失败，但继续往后跑”的主要原因。

### 3. NPU Megatron engine 的实际注册点

文件：

```text
verl/workers/engine/mindspeed/transformer_impl.py
```

关键代码：

```python
@EngineRegistry.register(model_type="language_model", backend="megatron", device="npu")
class MindspeedEngineWithLMHead(MegatronEngineWithLMHead):
    ...


@EngineRegistry.register(model_type="value_model", backend="megatron", device="npu")
class MindspeedEngineWithValueHead(MegatronEngineWithValueHead):
    ...


@EngineRegistry.register(model_type="language_model", backend="mindspeed_llm", device="npu")
class MindSpeedLLMEngineWithLMHead(MegatronEngineWithLMHead):
    ...
```

如果这个文件没有被成功 import，上面的装饰器就不会执行，`EngineRegistry` 里也就不会出现：

```text
language_model -> megatron -> npu
```

### 4. CUDA Megatron engine 的默认注册点

文件：

```text
verl/workers/engine/megatron/transformer_impl.py
```

关键代码：

```python
@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(...):
    ...


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(...):
    ...
```

`EngineRegistry.register()` 的 `device` 默认值是 `"cuda"`，所以这里注册的是：

```text
language_model -> megatron -> cuda
value_model -> megatron -> cuda
```

如果 MindSpeed import 失败，而 Megatron import 成功，就会形成一种状态：

```text
language_model -> megatron -> cuda 存在
language_model -> megatron -> npu 不存在
```

在 NPU 环境下调用 `get_engine_cls("language_model", "megatron")` 就会触发本次报错。

## 推荐排查命令

在用户运行任务的同一个 Python 环境里执行：

```bash
python - <<'PY'
from verl.workers.engine.base import EngineRegistry

try:
    import verl.workers.engine.mindspeed
    print("import verl.workers.engine.mindspeed: OK")
except Exception as e:
    print("import verl.workers.engine.mindspeed: FAILED")
    print(type(e).__name__, repr(e))

import verl.workers.engine
print("registered language_model/megatron devices:")
print(EngineRegistry._engines.get("language_model", {}).get("megatron", {}).keys())
PY
```

正常情况下应该看到：

```text
dict_keys(['npu', 'cuda'])
```

如果只看到：

```text
dict_keys(['cuda'])
```

则说明 MindSpeed engine 没有注册成功。

## 临时暴露真实错误的方法

可以临时修改 `verl/workers/engine/__init__.py`，将 MindSpeed import 的异常改成直接抛出：

```python
try:
    from .mindspeed import MindspeedEngineWithLMHead, MindspeedEngineWithValueHead, MindSpeedLLMEngineWithLMHead

    __all__ += ["MindspeedEngineWithLMHead", "MindspeedEngineWithValueHead", "MindSpeedLLMEngineWithLMHead"]
except ImportError as e:
    print("Failed to import verl.workers.engine.mindspeed:", repr(e))
    raise
```

这样可以提前看到真正的 import 失败原因，例如 MindSpeed 未安装、Megatron 版本不匹配、依赖模块缺失、NPU 相关包未安装等。

## 结论

本问题的主线不是 `EngineRegistry` 查询逻辑错误，而是 NPU Megatron engine 注册链路没有完成。

排查顺序建议为：

1. 先确认 `verl.workers.engine.mindspeed` 是否能成功 import。
2. 再确认 `EngineRegistry._engines["language_model"]["megatron"]` 下是否有 `npu`。
3. 如果没有 `npu`，优先暴露并修复 `.mindspeed` import 阶段的真实异常。
4. 只有 `npu` 注册成功后，才继续看 MindSpeed patch 是否真正按预期生效。
