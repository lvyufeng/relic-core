"""Compiled raw-byte SIMD host-CPU GEMV for the original Engram ``wkv`` rows.

This is the same reduction as :mod:`relic_core.kernels.cpu_block_gemv`, over the
same raw original bytes, and it produces the same bits. What differs is the inner
loop: the weights are consumed **as the checkpoint stores them**, with no
prepacked expansion, and the vector axis is K rather than the output columns.

Why it exists, in one paragraph. The host Engram keeps its projection on the CPU
on the Ascend path, so ``wkv`` is a host GEMV. That kernel is bandwidth-bound:
one call reads ``N * K`` code bytes and nothing else can lower that floor, so an
expanded operand -- four bytes of decoded value per code, or two -- reads more
and loses whatever it saves on decode. Measured: a 4-byte prepack with 96 threads
came out level with the scalar raw kernel at 64. The raw kernel therefore keeps
the bytes untouched and spends the saved traffic on SIMD: four adjacent codes
arrive in one 32-bit load, so a vector of four K positions needs four L1-resident
table gathers instead of four separate scalar decodes, and the 32 products of one
K block stay in registers across **both** halves of the truncation rule.

What is preserved, exactly as in the scalar kernel (and, through it, the accepted
Torch reduction in ``block_gemm.py``):

* the operands stay byte-coded; each byte becomes its exact e4m3 value and its
  **nominal** exponent ``F(v) = max(true exponent, -6)`` in one table word;
* the 32 products are split into two 16-term halves, each truncated toward zero
  onto ``2 ** (anchor - 13)`` with its own anchor; the second half's window is
  ``max(its anchor, floor(log2|first half|))``;
* the block result is renormalised once more onto ``2 ** (floor(log2|total|) - 13)``;
* the activation scale is applied first and the weight scale second, in FP32, and
  the blocks are accumulated sequentially in increasing K order.

The one new degree of freedom is that the 4-lane truncated sums are added as
integers. That is not a numerical relaxation: each truncated term is below
``2 ** 15`` in magnitude -- the extreme is ``240 * 240`` with a nominal sum of 14,
giving 28800, which an exhaustive sweep of all finite code pairs confirms -- so the
per-half sum is below ``2 ** 19`` and cannot overflow int32, and integer addition
is associative -- the 4-lane horizontal sum is the same integer the lane-order sum
is. Everything that rounds (the ``float(whole)`` conversions, the carry, the
renormalisation, the scales, the K accumulation) stays scalar and in the original
order.

The scale grid is read through a 256-entry table of exact powers of two, so a
block's scale costs one L1 load. The scalar kernel instead materialises a float
copy of the weight scale grid on every call (~0.6 MiB for the released square
grid); this kernel builds no such buffer and expands no weight at all.

**Opt-in and never a default**, with its own cache variable and its own build
signature, so this artifact can never be confused with the scalar one: nothing is
compiled, downloaded or written unless ``cache_dir=`` or
``RELIC_CPU_RAW_SIMD_GEMV_CACHE`` names a directory. The operand and scale
validators are the scalar module's own, imported rather than copied, so the two
operators cannot drift apart about what an acceptable input is -- including the
refusal of raw scale byte 255 and of a non-power-of-two FP32 grid.

Non-finite behaviour is the scalar kernel's: a NaN code in the activation row
makes every column NaN, a NaN code in a weight row makes that column NaN.

Example::

    import torch
    from relic_core.kernels.cpu_raw_simd_gemv import fp8_raw_simd_gemv

    out = fp8_raw_simd_gemv(a_codes, a_scale, wkv_codes, wkv_scale, threads=64)
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .cpu_block_gemv import (
    BLOCK,
    _checked_bytes,
    _checked_scales,
    _code_dtypes,
)

__all__ = [
    "ABI_VERSION",
    "BLOCK",
    "CACHE_ENV",
    "build",
    "cache_directory",
    "compiled_library_path",
    "fp8_raw_simd_gemv",
    "fp8_raw_simd_gemv_available",
    "load",
    "source_path",
    "vector_width",
]

#: Version of the C ABI below. A cached artifact built from an older interface is
#: rejected instead of being called with the wrong argument list.
ABI_VERSION = 1

#: Where the compiled kernel is cached. A separate variable from the scalar
#: kernel's, because the two artifacts are different kernels with different
#: sources; sharing a directory would be fine, sharing a *name* would not.
CACHE_ENV = "RELIC_CPU_RAW_SIMD_GEMV_CACHE"

#: Optional override for the C++ compiler; same resolution order as the scalar
#: kernel's loader.
COMPILER_ENV = "RELIC_CPU_RAW_SIMD_GEMV_CXX"

#: Optional extra compiler flags, shell-split, for a toolchain that is not on the
#: compiler's default search path. Part of the artifact identity.
CXXFLAGS_ENV = "RELIC_CPU_RAW_SIMD_GEMV_CXXFLAGS"

#: Part of the artifact identity: a differently compiled kernel is a different
#: kernel. `-ffp-contract=off` and `-fno-fast-math` are load-bearing for the same
#: reason they are in the scalar kernel -- the reduction's separate multiplies and
#: additions are the reference's rounding.
_BUILD_SIGNATURE = (
    b"relic-cpu-raw-simd-gemv;abi=1;std=c++17;-O3;-fno-fast-math;-ffp-contract=off;-DNDEBUG"
)

_COMPILERS = ("c++", "g++", "clang++")

_LOADED: dict[str, ctypes.CDLL] = {}


def source_path() -> Path:
    """The C++ source this module compiles."""
    return Path(__file__).resolve().parents[1] / "csrc" / "cpu_raw_simd_gemv.cpp"


def _library_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def cache_directory(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """The directory a build goes into, from the argument or ``CACHE_ENV``.

    Explicit rather than defaulted: a host that never asked for this operator must
    not have it compiled behind its back.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    configured = os.environ.get(CACHE_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            f"the compiled raw-byte SIMD host GEMV is opt-in: set {CACHE_ENV} to a writable "
            f"directory or pass cache_dir=... (no artifact was built or loaded)"
        )
    return Path(configured).expanduser()


def _extra_flags(explicit: list[str] | tuple[str, ...] | str | None = None) -> list[str]:
    """Extra compiler flags, from the argument or ``CXXFLAGS_ENV``."""
    if explicit is None:
        return shlex.split(os.environ.get(CXXFLAGS_ENV, ""))
    if isinstance(explicit, str):
        return shlex.split(explicit)
    return [str(flag) for flag in explicit]


def compiled_library_path(
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> Path:
    """Where the artifact for the current source and flags lives, built or not.

    Content-addressed over the source, the build signature and the extra flags, so
    a changed toolchain never silently reuses an old binary -- and a test that
    forces the scalar path (it passes ``-DRELIC_RAW_SIMD_FORCE_SCALAR``) gets its
    own artifact instead of overwriting the vector one.
    """
    material = source_path().read_bytes() + _BUILD_SIGNATURE + b"\0"
    material += b"\0".join(flag.encode("utf-8") for flag in _extra_flags(extra_flags))
    digest = hashlib.sha256(material).hexdigest()[:16]
    name = f"cpu_raw_simd_gemv_abi{ABI_VERSION}_{digest}{_library_suffix()}"
    return cache_directory(cache_dir) / name


def _compiler(explicit: str | None = None) -> str:
    """Resolve the C++ compiler, honouring a named one instead of substituting another."""
    configured = os.environ.get(COMPILER_ENV, "").strip() or None
    for named, source in ((explicit, "compiler="), (configured, COMPILER_ENV)):
        if not named:
            continue
        found = shutil.which(named)
        if found:
            return found
        raise RuntimeError(f"the C++ compiler named by {source} was not found: {named!r}")
    for candidate in _COMPILERS:
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        "no C++ compiler found for the compiled raw-byte SIMD host GEMV: set "
        f"{COMPILER_ENV} to a compiler path, or install one of {', '.join(_COMPILERS)}"
    )


def _build_command(compiler: str, output: Path, extra_flags: list[str]) -> list[str]:
    """The compile line, with the rounding-critical flags spelled out.

    ``-ffp-contract=off`` is load-bearing: the reference's per-block scale
    multiplies and its K accumulation are separate rounded operations, and a
    compiler free to contract them into an FMA would produce a different kernel.
    ``-fno-fast-math`` keeps it from reassociating or flushing subnormals.
    """
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-DNDEBUG",
    ]
    command += list(extra_flags)
    if sys.platform == "darwin":
        command += ["-dynamiclib", "-o", str(output)]
    else:
        command += ["-shared", "-pthread", "-o", str(output)]
    command.append(str(source_path()))
    return command


def build(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    compiler: str | None = None,
    force: bool = False,
    verbose: bool = False,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> Path:
    """Compile the kernel into the cache directory and return the artifact path.

    Idempotent: the artifact name carries the source hash, the build signature and
    the extra flags, so an existing file is the same kernel. A concurrent build is
    harmless -- the temporary object is renamed into place atomically.
    """
    flags = _extra_flags(extra_flags)
    target = compiled_library_path(cache_dir, flags)
    if target.is_file() and not force:
        return target
    source = source_path()
    if not source.is_file():
        raise RuntimeError(f"the compiled raw-byte SIMD host GEMV source is missing: {source}")
    compiler_path = _compiler(compiler)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.stem + ".", suffix=_library_suffix()
    )
    os.close(handle)
    temporary = Path(temporary_name)
    command = _build_command(compiler_path, temporary, flags)
    if verbose:
        print(" ".join(command), flush=True)
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not run the C++ compiler {compiler_path!r}: {error}"
        ) from error
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "compiling cpu_raw_simd_gemv.cpp failed with "
            f"exit code {completed.returncode}:\n  " + " ".join(command)
            + ("\n" + completed.stderr.strip() if completed.stderr.strip() else "")
        )
    try:
        os.replace(temporary, target)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> ctypes.CDLL:
    """Load (building if necessary) the compiled kernel and bind the C ABI."""
    target = build(cache_dir=cache_dir, extra_flags=extra_flags)
    key = str(target)
    library = _LOADED.get(key)
    if library is not None:
        return library
    library = ctypes.CDLL(key)
    library.relic_cpu_raw_simd_gemv_abi_version.restype = ctypes.c_int32
    library.relic_cpu_raw_simd_gemv_abi_version.argtypes = []
    library.relic_cpu_raw_simd_gemv_vector_width.restype = ctypes.c_int32
    library.relic_cpu_raw_simd_gemv_vector_width.argtypes = []
    library.relic_cpu_raw_simd_gemv_fp8.restype = ctypes.c_int32
    library.relic_cpu_raw_simd_gemv_fp8.argtypes = [
        ctypes.c_void_p,   # a_codes
        ctypes.c_void_p,   # a_scale
        ctypes.c_void_p,   # b_codes
        ctypes.c_void_p,   # b_scale
        ctypes.c_void_p,   # out
        ctypes.c_int64,    # n
        ctypes.c_int64,    # k
        ctypes.c_int32,    # block_rows
        ctypes.c_int32,    # threads
        ctypes.POINTER(ctypes.c_int32),  # threads_used
    ]
    found = int(library.relic_cpu_raw_simd_gemv_abi_version())
    if found != ABI_VERSION:
        raise RuntimeError(
            f"the cached raw-byte SIMD host GEMV artifact at {target} reports ABI version "
            f"{found}, but this loader speaks {ABI_VERSION}; delete it to rebuild"
        )
    _LOADED[key] = library
    return library


def vector_width(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> int:
    """How many K lanes this artifact's decode handles at once: 4 on NEON, else 1.

    Reported by the artifact, not guessed from the platform, so a caller comparing
    two builds can say which one it timed.
    """
    return int(load(cache_dir=cache_dir, extra_flags=extra_flags)
               .relic_cpu_raw_simd_gemv_vector_width())


def fp8_raw_simd_gemv_available(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> bool:
    """Whether the artifact is already built for this source and these flags."""
    try:
        return compiled_library_path(cache_dir, extra_flags).is_file()
    except RuntimeError:
        return False


def fp8_raw_simd_gemv(
    a_codes,
    a_scale,
    b_codes,
    b_scale,
    *,
    block_rows: int = 32,
    threads: int = 1,
    out=None,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
):
    """Exact ``[1, K] x [N, K]`` block-32 FP8 GEMV over raw original bytes.

    ``a_codes``/``b_codes`` are e4m3 codes as bytes (or as ``float8_e4m3fn``) and
    ``a_scale``/``b_scale`` are E8M0 grids, as raw bytes (``float8_e8m0fnu``) or as
    the exact FP32 powers the CPU activation quantizer returns. ``a_scale`` is
    ``[1, K // 32]``; ``b_scale`` is ``[N // 32, K // 32]`` for the square grid a
    dense FP8 weight is stored with (``block_rows=32``) or ``[N, K // 32]`` for the
    per-row grid (``block_rows=1``). Returns FP32 accumulators of shape ``[1, N]``;
    the caller owns the final narrowing.

    ``threads`` splits the output columns. The result does not depend on it: each
    column's K accumulation is sequential and identical whatever the thread count,
    so this is a pure throughput knob rather than a numerical one.

    ``b_codes`` is read in place. Nothing is prepacked, copied or dequantized, so
    one call touches exactly the file's own bytes.

    ``extra_flags`` participates in the artifact identity, which is how a test
    builds the same source with the scalar path forced and compares the two.

    The compiled kernel is selected by construction -- there is one implementation
    -- and a failed build raises rather than falling back to a slower path that
    would quietly answer a different question about which code ran.
    """
    import torch

    if not isinstance(block_rows, int) or isinstance(block_rows, bool) or block_rows not in (1, BLOCK):
        raise ValueError("block_rows must be 1 or 32")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be an integer >= 1")

    codes = _checked_bytes(a_codes, "activation codes", _code_dtypes())
    activation_scales = _checked_scales(a_scale, "activation scales")
    weights = _checked_bytes(b_codes, "weight codes", _code_dtypes())
    weight_scales = _checked_scales(b_scale, "weight scales")

    m, k = codes.shape
    n, weight_k = weights.shape
    if m != 1:
        raise ValueError(
            f"this is the M = 1 GEMV; the activation codes have {m} rows. Use "
            "relic_core.kernels.block_gemm.fp8_block_gemm for a batched prompt"
        )
    if min(n, k) <= 0 or k % BLOCK or weight_k != k:
        raise ValueError("positive [1,K] and [N,K] operands need the same K, divisible by 32")
    if tuple(activation_scales.shape) != (1, k // BLOCK):
        raise ValueError(f"activation scales must have shape {(1, k // BLOCK)}")
    expected_rows = (n + block_rows - 1) // block_rows
    if tuple(weight_scales.shape) != (expected_rows, k // BLOCK):
        raise ValueError(
            f"weight scales must have shape {(expected_rows, k // BLOCK)} for "
            f"block_rows={block_rows}"
        )

    if out is None:
        out = torch.empty((1, n), dtype=torch.float32)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if out.dtype != torch.float32:
            raise TypeError(f"out has dtype {out.dtype}, expected torch.float32")
        if out.layout != torch.strided or not out.is_contiguous():
            raise ValueError("out must be a contiguous strided matrix")
        if tuple(out.shape) != (1, n) or out.device.type != "cpu":
            raise ValueError(f"out must be a contiguous CPU fp32 tensor of shape {(1, n)}")
        if out.requires_grad:
            raise ValueError("out must not require grad")
        for name, tensor in (
            ("activation codes", codes),
            ("activation scales", activation_scales),
            ("weight codes", weights),
            ("weight scales", weight_scales),
        ):
            if torch._C._overlaps(out, tensor):
                raise ValueError(f"out must not overlap {name}")

    library = load(cache_dir=cache_dir, extra_flags=extra_flags)
    threads_used = ctypes.c_int32(0)
    status = library.relic_cpu_raw_simd_gemv_fp8(
        codes.data_ptr(),
        activation_scales.data_ptr(),
        weights.data_ptr(),
        weight_scales.data_ptr(),
        out.data_ptr(),
        n,
        k,
        block_rows,
        threads,
        ctypes.byref(threads_used),
    )
    if status != 0:
        raise RuntimeError(
            f"the compiled raw-byte SIMD host GEMV refused the call with status {status} "
            "(1 = bad arguments, 2 = out of memory)"
        )
    return out