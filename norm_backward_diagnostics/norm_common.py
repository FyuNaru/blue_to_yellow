"""Standalone FP32 norm backward diagnosis; no ATK or TENPU imports.

Thresholds follow ATK bac1aa7 mixed_tolerance_bm FP32 defaults. Only load
input artifacts you trust: ATK protocol-4 files require weights_only=False.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

THRESHOLDS = dict(rtol=9.77e-4, atol=1.53e-5, required_matched_ratio=0.99,
                  max_abs_error_limit=1e-2, max_ulp_multiple=32)


def load_inputs(path, op):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, (list, tuple)) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict):
        if data.get("format") != f"tenpu_{op}_inputs_v1" or data.get("phase") != "backward":
            raise ValueError("Expected a saved independent BACKWARD input bundle for " + op)
        data = data["values"]
    expected = 8 if op == "layernorm" else 6
    if not isinstance(data, (list, tuple)) or len(data) != expected:
        raise ValueError(f"Expected {expected} saved backward arguments, including statistics. "
                         "Use the independent backward input.bin, not forward/cascade input.bin.")
    x, dy, weight = data[:3]
    if not all(isinstance(t, torch.Tensor) and t.dtype == torch.float32 for t in (x, dy, weight)):
        raise ValueError("This diagnostic targets the FP32 failures; x/dy/gamma must be FP32")
    if x.ndim not in (2, 3, 4) or not x.numel() or dy.shape != x.shape or weight.shape != (x.shape[-1],):
        raise ValueError("Invalid x/dy/gamma shapes")
    bias, eps, zc, mean, rstd = (data[3], data[4], data[5], data[6], data[7]) if op == "layernorm" else (None, data[3], data[4], None, data[5])
    rows = x.numel() // x.shape[-1]
    for stat in [rstd] + ([mean] if mean is not None else []):
        if not isinstance(stat, torch.Tensor) or stat.dtype != torch.float32 or stat.shape != (rows,):
            raise ValueError("Saved statistics must be FP32 vectors with one value per row")
    if bias is not None and (bias.dtype != torch.float32 or bias.shape != weight.shape):
        raise ValueError("Invalid beta")
    for tensor in [x, dy, weight, rstd] + ([mean, bias] if mean is not None else []):
        for chunk in tensor.reshape(-1).split(1048576):
            if not torch.isfinite(chunk).all():
                raise ValueError("Only finite-value accuracy diagnosis is supported, not NaN/Inf cases")
    if not (rstd > 0).all() or not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("Expected positive rstd and eps")
    return dict(x=x, dy=dy, weight=weight, bias=bias, eps=float(eps),
                zero_centered=bool(zc), mean=mean, rstd=rstd)


def run_npu(data, index):
    import torch_npu

    torch.npu.set_device(index)
    device = f"npu:{index}"
    h = data["x"].shape[-1]
    move = lambda t: t.to(device).contiguous()
    x = move(data["x"]).reshape(-1, h)
    dy = move(data["dy"]).reshape_as(x)
    w = move(data["weight"])
    if data["zero_centered"]:
        w = w.float() + 1.0
    rstd = move(data["rstd"]).reshape(-1, 1)
    if data["mean"] is None:
        outputs = torch_npu.npu_rms_norm_backward(dy, x, w, rstd)
        api = "torch_npu.npu_rms_norm_backward"
    else:
        outputs = torch.ops.aten.native_layer_norm_backward(
            dy, x, (h,), move(data["mean"]).reshape(-1, 1), rstd,
            w, move(data["bias"]), [True, True, True],
        )
        api = "torch.ops.aten.native_layer_norm_backward"
    torch.npu.synchronize()
    return [t.detach().cpu() for t in outputs], dict(
        api=api, torch_npu=str(torch_npu.__version__), device=torch.npu.get_device_name(index))


def run_gpu(data, index):
    # Initialize NVIDIA TE first; this branch never imports torch_npu or ATK.
    import transformer_engine
    import transformer_engine.pytorch
    import transformer_engine_torch as tex

    if "transformerenginenpu" in str(transformer_engine.__file__).lower():
        raise RuntimeError("TENPU shadows NVIDIA TE. Run outside the TENPU checkout and fix PYTHONPATH")
    torch.cuda.set_device(index)
    h = data["x"].shape[-1]
    move = lambda t: t.to(f"cuda:{index}").contiguous()
    x = move(data["x"]).reshape(-1, h)
    dy = move(data["dy"]).reshape_as(x)
    w, rstd = move(data["weight"]), move(data["rstd"])
    if data["mean"] is None:
        outputs = tex.rmsnorm_bwd(dy, x, rstd, w, 0, data["zero_centered"])
        api = "transformer_engine_torch.rmsnorm_bwd"
    else:
        outputs = tex.layernorm_bwd(dy, x, move(data["mean"]), rstd, w, 0, data["zero_centered"])
        api = "transformer_engine_torch.layernorm_bwd"
    torch.cuda.synchronize()
    return [t.detach().cpu() for t in outputs], dict(
        api=api, te=str(getattr(transformer_engine, "__version__", "unknown")),
        te_path=str(transformer_engine.__file__), device=torch.cuda.get_device_name(index))


class Comparison:
    """ATK default FP32 predicates; aggregate chunks to avoid full-size temporaries."""

    def __init__(self):
        self.count = self.matched = self.abs_failed = 0
        self.max_abs = 0.0
        self.samples = []

    def update(self, actual, reference64):
        a = actual.reshape(-1)
        r64 = reference64.reshape(-1)
        if a.dtype != torch.float32 or a.shape != r64.shape:
            raise ValueError("Expected matching FP32 device output and reference shapes")
        # ATK single-benchmark FP32 comparison: CPU FP64 reference is rounded
        # to FP32 for the verdict, retained in FP64 for error evidence.
        r = r64.float()
        if not torch.isfinite(a).all() or not torch.isfinite(r).all():
            raise ValueError("Nonfinite result: numerical diagnosis cannot issue a finite-case verdict")
        close = torch.isclose(a, r, rtol=THRESHOLDS["rtol"], atol=THRESHOLDS["atol"])
        error = (a - r).abs()
        spacing = torch.nextafter(r, torch.full_like(r, torch.inf)) - r
        absolute_ok = (error <= THRESHOLDS["max_abs_error_limit"]) | (error <= 32 * spacing)
        failures = (~close | ~absolute_ok).nonzero().flatten()
        for i in failures[:max(0, 32 - len(self.samples))].tolist():
            self.samples.append(dict(index=self.count + i, actual=a[i].item(),
                                     reference_fp64=r64[i].item(), reference_fp32=r[i].item(),
                                     abs_error=error[i].item(), close=close[i].item(),
                                     abs_error_ok=absolute_ok[i].item()))
        self.count += a.numel()
        self.matched += close.sum().item()
        self.abs_failed += (~absolute_ok).sum().item()
        self.max_abs = max(self.max_abs, (a.double() - r64).abs().max().item())

    def report(self):
        ratio = torch.tensor(self.matched / self.count, dtype=torch.float32).item()
        return dict(result=ratio >= THRESHOLDS["required_matched_ratio"] and self.abs_failed == 0,
                    elements=self.count, close_fail_count=self.count - self.matched,
                    matched_ratio=ratio, abs_error_fail_count=self.abs_failed,
                    max_abs_error_to_fp64=self.max_abs, first_failed_elements=self.samples)


def reference_and_compare(data, outputs, chunk_rows):
    h = data["x"].shape[-1]
    x = data["x"].reshape(-1, h)
    dy = data["dy"].reshape_as(x)
    mean = data["mean"]
    # Match effective FP32 gamma used at the device boundary.
    w = (data["weight"] + 1.0 if data["zero_centered"] else data["weight"]).double()
    dw = torch.zeros(h, dtype=torch.float64)
    db = torch.zeros_like(dw)
    dw_correction = torch.zeros_like(dw)
    dx_compare = Comparison()
    native_dx = outputs[0].reshape_as(x)
    for start in range(0, x.shape[0], chunk_rows):
        stop = start + chunk_rows
        xc, dc = x[start:stop].double(), dy[start:stop].double()
        r = data["rstd"][start:stop].double().reshape(-1, 1)
        norm = xc * r if mean is None else (xc - mean[start:stop].double().reshape(-1, 1)) * r
        g = dc * w
        dx = r * (g - norm * (g * norm).mean(dim=1, keepdim=True))
        if mean is not None:
            dx -= r * g.mean(dim=1, keepdim=True)
        dx_compare.update(native_dx[start:stop], dx)
        partial = (dc * norm).sum(dim=0)
        adjusted = partial - dw_correction
        updated = dw + adjusted
        dw_correction = (updated - dw) - adjusted
        dw = updated
        if mean is not None:
            db += dc.sum(dim=0)
    reports = {"dx": dx_compare.report()}
    refs = {"dw": dw}
    if mean is not None:
        refs["db"] = db
    for idx, name in enumerate(refs, 1):
        compare = Comparison()
        compare.update(outputs[idx], refs[name])
        reports[name] = compare.report()
    return reports, refs


def main(backend):
    parser = argparse.ArgumentParser(description="FP32 LayerNorm/RMSNorm native backward vs CPU FP64")
    parser.add_argument("--op", required=True, choices=["layernorm", "rmsnorm"])
    parser.add_argument("--input", required=True, type=Path, help="Trusted ATK backward input.bin")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--save-dx", action="store_true", help="Also save potentially large dx")
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with args.input.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    data = load_inputs(args.input, args.op)
    print(f"{backend}: {args.op}, shape={tuple(data['x'].shape)}, dtype={data['x'].dtype}", flush=True)
    with torch.no_grad():
        outputs, metadata = (run_npu if backend == "npu" else run_gpu)(data, args.device)
        print("Native backward finished; comparing with chunked CPU FP64 reference...", flush=True)
        reports, references = reference_and_compare(data, outputs, args.chunk_rows)
    names = ["dx", "dw"] + (["db"] if args.op == "layernorm" else [])
    torch.save({name: tensor for name, tensor in zip(names, outputs) if name != "dx" or args.save_dx},
               args.output / "native_outputs.pt")
    torch.save(references, args.output / "cpu_reference_fp64.pt")
    with (args.output / "dw_channels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "native_dw", "cpu_fp64_dw", "abs_error_to_fp64"])
        for i, (actual, ref) in enumerate(zip(outputs[1].reshape(-1).tolist(), references["dw"].tolist())):
            writer.writerow([i, actual, ref, abs(actual - ref)])
    report = dict(backend=backend, op=args.op, input_sha256=digest.hexdigest(),
                  shape=list(data["x"].shape), dtype=str(data["x"].dtype),
                  eps=data["eps"], zero_centered_gamma=data["zero_centered"],
                  torch=str(torch.__version__), metadata=metadata, thresholds=THRESHOLDS,
                  reference="CPU FP64 formula using saved FP32 statistics; rounded to FP32 for ATK predicates",
                  chunk_rows=args.chunk_rows, outputs=reports)
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, result in reports.items():
        print(f"{name}: {'PASS' if result['result'] else 'FAIL'}, "
              f"matched_ratio={result['matched_ratio']:.9f}, "
              f"abs_error_fail_count={result['abs_error_fail_count']}")
    print(f"Report: {args.output / 'report.json'}")
    return 0 if all(r["result"] for r in reports.values()) else 1
