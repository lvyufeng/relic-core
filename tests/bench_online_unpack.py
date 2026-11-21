"""Benchmark named online-unpack implementations, never model TPS or native FP8.

Run from the relic-core repository root. On NPU, source a task-private CANN
and Python environment first. Correctness is checked against the independent
Torch bit-table implementation on CPU before synchronized wall timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from relic_core.kernels import unpack


def sync(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def real_sample(root, kind):
    from safetensors import safe_open
    index = json.loads((root / "model.safetensors.index.json").read_bytes())["weight_map"]
    names = sorted(key for key in index if key.startswith("layers.0.") and key.endswith(".weight"))
    expected = torch.int8 if kind == "fp4" else torch.float8_e4m3fn
    for name in names:
        scale_name = name[:-7] + ".scale"
        if scale_name not in index:
            continue
        path, scale_path = root / index[name], root / index[scale_name]
        if not path.resolve().is_relative_to(root.resolve()) or not scale_path.resolve().is_relative_to(root.resolve()):
            raise ValueError("checkpoint index escaped its directory")
        with safe_open(path, framework="pt", device="cpu") as reader:
            shape = reader.get_slice(name).get_shape()
            if len(shape) != 2 or shape[0] * shape[1] > 64 * 1024 * 1024:
                continue
            tensor = reader.get_tensor(name)
            if tensor.dtype != expected:
                continue
            raw = tensor.view(torch.uint8).clone()
        with safe_open(scale_path, framework="pt", device="cpu") as reader:
            scale = reader.get_tensor(scale_name)
            if scale.dtype != getattr(torch, "float8_e8m0fnu", None):
                continue
            scale = scale.view(torch.uint8).clone()
        return raw, scale, {"weight_key": name, "scale_key": scale_name,
                            "source_sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()}
    raise ValueError(f"no bounded original {kind} matrix with E8M0 scales found")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--backend", choices=("torch", "triton", "auto"), required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 3 or args.report.exists():
        raise ValueError("use >=3 iterations and an exclusively new report")
    if args.checkpoint and args.report.resolve().is_relative_to(args.checkpoint.resolve()):
        raise ValueError("report must not be written into the read-only checkpoint")
    device = torch.device(args.device)
    if device.type == "npu":
        import torch_npu
        torch.npu.set_device(device)
    torch.set_num_threads(4)
    result = {"torch": torch.__version__, "device": str(device), "requested_backend": args.backend,
              "timer": "synchronized wall time including Python launch overhead",
              "model_tps_claimed": False, "native_fp4_fp8_arithmetic_claimed": False,
              "checkpoint_rewritten": False, "cases": []}
    for kind in ("fp4", "fp8"):
        if args.checkpoint:
            raw, scale, source = real_sample(args.checkpoint, kind)
        elif kind == "fp4":
            raw = (torch.arange(2304 * 2560, dtype=torch.int64) % 256).to(torch.uint8).reshape(2304, 2560)
            scale = torch.full((2304, 160), 127, dtype=torch.uint8)
            source = {"synthetic_fixture": True}
        else:
            value = torch.arange(1536 * 5120, dtype=torch.int64) % 254
            raw = torch.where(value >= 127, value + 1, value).to(torch.uint8).reshape(1536, 5120)
            scale = torch.full((48, 160), 127, dtype=torch.uint8)
            source = {"synthetic_fixture": True}
        raw_device, scale_device = raw.to(device), scale.to(device)
        if kind == "fp4":
            out = torch.empty((*raw.shape[:-1], raw.shape[-1] * 2), dtype=torch.int8, device=device)
            expected = unpack.unpack_fp4_to_int8(raw, backend="torch")
            operation = lambda: unpack.unpack_fp4_to_int8(raw_device, backend=args.backend, out=out)
        else:
            block = (1, 32) if scale.shape[-2] == raw.shape[-2] and raw.shape[-2] > 1 else (32, 32)
            out = torch.empty(raw.shape, dtype=torch.bfloat16, device=device)
            expected = unpack.unpack_fp8_to_bf16(raw, scale, block_shape=block, backend="torch")
            operation = lambda: unpack.unpack_fp8_to_bf16(raw_device, scale_device, block_shape=block,
                                                         backend=args.backend, out=out)
        provider = "triton" if unpack._select_backend(args.backend, device) is not None else "torch"
        sync(device)
        start = time.perf_counter()
        actual = operation()
        sync(device)
        cold = time.perf_counter() - start
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
        if kind == "fp8":
            observed_bits, expected_bits = actual.cpu().view(torch.int16), expected.view(torch.int16)
            finite = torch.isfinite(expected)
            if not torch.equal(observed_bits[finite], expected_bits[finite]):
                raise RuntimeError("finite BF16 output bits disagree, including zero/subnormal signs")
        for _ in range(3):
            operation()
        sync(device)
        times = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            operation()
            sync(device)
            times.append(time.perf_counter() - start)
        traffic = raw.numel() + out.numel() * out.element_size()
        case = {"kind": kind, "shape": list(raw.shape), "output_shape": list(out.shape),
                "provider": provider, "source": source, "correctness_passed": True,
                "cold_compile_and_first_call_s": cold, "iterations": args.iterations,
                "median_ms": statistics.median(times) * 1000, "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "input_output_GiB_per_s": traffic / statistics.median(times) / 2**30,
                "timings_s": times}
        result["cases"].append(case)
        print(json.dumps(case), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as handle:
        json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
