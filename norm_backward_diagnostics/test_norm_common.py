"""CPU-only verification of reference math, saved-input parsing and predicates."""
import pytest
import torch

from norm_common import Comparison, load_inputs, reference_and_compare


@pytest.mark.parametrize("op", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("zc", [False, True])
@pytest.mark.parametrize("chunk", [2, 100])
def test_reference_against_cpu_autograd(op, zc, chunk):
    torch.manual_seed(42)
    x = torch.randn(3, 4, 8, dtype=torch.float32)
    dy = torch.randn_like(x)
    w = torch.randn(8)
    bias = torch.randn(8)
    x64 = x.double().detach().requires_grad_()
    # FP32 effective weight mirrors the primitive input arithmetic.
    effective = (w + 1 if zc else w).double().detach().requires_grad_()
    beta = bias.double().detach().requires_grad_()
    eps = 1e-5
    mean = x64.mean(dim=-1, keepdim=True)
    if op == "layernorm":
        rstd = ((x64 - mean).square().mean(dim=-1, keepdim=True) + eps).rsqrt()
        y = (x64 - mean) * rstd * effective + beta
        grads = torch.autograd.grad(y, (x64, effective, beta), dy.double())
    else:
        rstd = (x64.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
        y = x64 * rstd * effective
        grads = torch.autograd.grad(y, (x64, effective), dy.double())
    data = dict(x=x, dy=dy, weight=w, zero_centered=zc,
                mean=mean.detach().float().flatten() if op == "layernorm" else None,
                rstd=rstd.detach().float().flatten())
    reports, refs = reference_and_compare(data, [t.float() for t in grads], chunk)
    assert all(r["result"] for r in reports.values())
    torch.testing.assert_close(refs["dw"], grads[1], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("op", ["layernorm", "rmsnorm"])
@pytest.mark.parametrize("bundle", [False, True])
def test_atk_input_roundtrip(tmp_path, op, bundle):
    x = torch.randn(2, 3)
    values = [x, x.clone(), torch.ones(3)]
    if op == "layernorm":
        values.append(torch.zeros(3))
    values.extend([1e-5, True])
    if op == "layernorm":
        values.append(torch.zeros(2))
    values.append(torch.ones(2))
    saved = [dict(format=f"tenpu_{op}_inputs_v1", phase="backward", values=values)] if bundle else values
    path = tmp_path / "input.bin"
    torch.save(saved, path, pickle_protocol=4)
    data = load_inputs(path, op)
    assert torch.equal(data["x"], x)
    assert data["zero_centered"]
    torch.save(values[:-1], path)
    with pytest.raises(ValueError, match="backward arguments"):
        load_inputs(path, op)


def test_threshold_both_conditions_and_negative_ulp():
    ref = torch.zeros(48, dtype=torch.float64)
    actual = ref.float()
    actual[0] = 1e-3
    c = Comparison()
    c.update(actual, ref)
    assert not c.report()["result"]
    assert c.report()["abs_error_fail_count"] == 0
    assert c.report()["close_fail_count"] == 1
    ref = torch.full((100,), -1e6, dtype=torch.float64)
    actual = ref.float()
    actual[0] += 3.0
    c = Comparison()
    c.update(actual, ref)
    assert c.report()["matched_ratio"] == 1
    assert c.report()["abs_error_fail_count"] == 1
    assert not c.report()["result"]
    actual[0] = ref[0].float() + 1.0
    c = Comparison()
    c.update(actual, ref)
    assert c.report()["result"]


def test_chunk_comparison_equivalence():
    torch.manual_seed(0)
    reference = torch.randn(10000).double()
    actual = reference.float() + torch.randn(10000) * 1e-4
    one, chunks = Comparison(), Comparison()
    one.update(actual, reference)
    for a, r in zip(actual.split(123), reference.split(123)):
        chunks.update(a, r)
    assert one.report() == chunks.report()


def test_gpu_uses_nvte_primitive_without_npu(monkeypatch):
    import sys
    from types import ModuleType
    import norm_common

    calls = []
    te = ModuleType("transformer_engine")
    te.__file__ = "/site-packages/transformer_engine/__init__.py"
    te.__version__ = "2.17.1"
    pytorch = ModuleType("transformer_engine.pytorch")
    te.pytorch = pytorch
    tex = ModuleType("transformer_engine_torch")
    def rms(*args):
        calls.append(args)
        return args[0], args[3]
    tex.rmsnorm_bwd = rms
    for name, module in [("transformer_engine", te), ("transformer_engine.pytorch", pytorch),
                         ("transformer_engine_torch", tex)]:
        monkeypatch.setitem(sys.modules, name, module)
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("cuda:"):
            return self.clone()
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock")
    w = torch.tensor([1., 2., 3.])
    data = dict(x=torch.ones(2, 3), dy=torch.ones(2, 3), weight=w,
                rstd=torch.ones(2), mean=None, zero_centered=True)
    outputs, metadata = norm_common.run_gpu(data, 0)
    assert len(outputs) == 2
    assert torch.equal(calls[0][3], w)  # NVTE receives raw gamma plus the flag.
    assert calls[0][-2:] == (0, True)
    assert metadata["api"] == "transformer_engine_torch.rmsnorm_bwd"
