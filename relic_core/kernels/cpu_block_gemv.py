"""Exact compiled host-CPU FP8 block-32 GEMV, for the original Engram ``wkv``.

The host Engram is deliberately CPU-only on the Ascend path (the release's two
tables are 91.55 GiB each and are read out of the shard mapping), so its ``wkv``
projection is a **host** GEMV over the checkpoint's own bytes, not a tensor op.
The accepted Torch implementation of that reduction lives in ``block_gemm.py``;
this module is the same reduction compiled, because ``M = 1`` makes the general
block GEMM spend most of its time on per-K-tile bookkeeping instead of arithmetic.

What is preserved, and why it is bit-exact rather than approximately equal:

* the operands stay byte-coded. An e4m3 code is read as its exact value and its
  **nominal** exponent ``F(v) = max(true exponent, -6)``; a signed-zero code has
  no nominal exponent and contributes a sentinel to its half. Nothing is
  dequantized whole, requantized, or routed through BF16.
* the 32 products of one K block are cut into two 16-term halves, each truncated
  toward zero onto ``2 ** (anchor - 13)`` with the anchor of its own half; the
  second half's window is ``max(its anchor, floor(log2|first half sum|))``, so
  the running sum is a peer of the second half rather than a prefix.
* the block result is renormalised once more onto ``2 ** (floor(log2|total|) - 13)``.
* the activation scale is applied first and the weight scale second, both in
  FP32, and the K blocks are then accumulated one at a time in increasing K
  order. Fusing either scale multiply, or the K add, would change the rounding.

Every intermediate of that reduction is exactly representable in FP32 (the
argument is written out in ``csrc/cpu_block_gemv.cpp``), so the compiled kernel
produces the same bits as the Torch oracle rather than a tolerance-matched
approximation. The only rounding is the per-K-block FP32 accumulation, which is
where the reference rounds too.

**Opt-in, and never a default.** The kernel is compiled locally from the C++
source beside this file into a cache directory the caller names. Nothing is
built, downloaded or written unless a cache directory is given -- either through
``cache_dir=`` or the ``RELIC_CPU_GEMV_CACHE`` environment variable -- so a
checkout that does not ask for this pays nothing and no package-global state is
touched. There is no OpenBLAS, no float64, no native FP8 cast and no fallback to
a different implementation after the compiled one has been selected: a missing
compiler or a failed build is reported by name.

Non-finite behaviour. An e4m3 byte is a NaN code exactly when ``(byte & 0x7F) ==
0x7F``; a NaN in the activation row makes every output column NaN, and a NaN in a
weight row makes that column NaN. Both are the reference's own propagation -- a
NaN product poisons its half's sum and the block result, and the K accumulation
can never recover from it.

A raw scale byte is a **finite** original-format grid value exactly when it is in
``[0, 254]``: the byte is the exponent, so 255 would mean ``2 ** 128``, which no
binary32 holds, and byte ``0`` means ``2 ** -127``, which is a subnormal. Byte 255
is not a grid value a checkpoint or quantizer produces, and the two existing CPU
references disagree about what it would mean -- the runtime's exponent arithmetic
turns it into an infinity while the Torch oracle maps it to NaN. Rather than pick
one of two contradicting behaviours under an operator named after the original
format, this module **refuses** it by name. An FP32 grid gets the same treatment:
every entry must be an exact power of two in range, and anything else is refused
instead of being rounded onto a neighbouring grid.

Scale inputs. Both scales are accepted in either form the runtime actually
holds them in:

* **raw E8M0 bytes** (``uint8``/``int8``/``float8_e8m0fnu``), which is how a
  checkpoint's weight grid arrives;
* **exact FP32 powers of two**, which is what the CPU activation quantizer
  returns (``2 ** ceil(log2(amax / 448))``) and what an already-decoded grid
  looks like.

An FP32 grid is converted to the byte the kernel reads, after validating every
entry: an original E8M0 grid is exactly ``2 ** (byte - 127)``, so the byte is the
FP32 exponent field, with the one exception that ``2 ** -127`` -- the smallest
grid value -- is a binary32 subnormal (``0x00400000``) rather than a normal power
and maps to byte 0. Anything that is not such a power of two is refused by name
rather than rounded onto a nearby grid, because a scale that does not describe the
weight it scales is a wrong answer, not a slow one.

Example::

    import torch
    from relic_core.kernels.cpu_block_gemv import fp8_block_gemv

    # a_codes [1, K], a_scale [1, K // 32], b_codes [N, K], b_scale [N // 32, K // 32]
    out = fp8_block_gemv(a_codes, a_scale, wkv_codes, wkv_scale, threads=4)
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

__all__ = [
    "ABI_VERSION",
    "BLOCK",
    "CACHE_ENV",
    "COMPILER_ENV",
    "CXXFLAGS_ENV",
    "build",
    "cache_directory",
    "compiled_library_path",
    "fp8_block_gemv",
    "fp8_block_gemv_available",
    "load",
    "source_path",
]

#: Block width of the reference reduction. The two-half rule is tied to it, so
#: any other width is refused rather than silently re-derived.
BLOCK = 32

#: Version of the C ABI below. A cached artifact built from an older interface is
#: rejected instead of being called with the wrong argument list.
ABI_VERSION = 1

#: Where the compiled kernel is cached. Set to a private, writable directory;
#: without it the operator reports that it is not available rather than choosing
#: a location on the caller's behalf.
CACHE_ENV = "RELIC_CPU_GEMV_CACHE"

#: Optional override for the C++ compiler. Falls back to ``c++``, ``g++`` and
#: ``clang++`` on ``PATH``, in that order.
COMPILER_ENV = "RELIC_CPU_GEMV_CXX"

#: Optional extra compiler flags, shell-split. This is how a host whose toolchain
#: is not on the compiler's default search path -- an unusual SDK layout, a
#: cross-compiler sysroot -- states that once. They are part of the artifact
#: identity, because a differently compiled kernel is a different kernel.
CXXFLAGS_ENV = "RELIC_CPU_GEMV_CXXFLAGS"

#: Part of the artifact identity. Any change to the build flags has to invalidate
#: a cached library, because a differently compiled kernel is a different kernel.
_BUILD_SIGNATURE = (
    b"relic-cpu-block-gemv;abi=1;std=c++17;-O3;-fno-fast-math;-ffp-contract=off;-DNDEBUG"
)

_COMPILERS = ("c++", "g++", "clang++")

_LOADED: dict[str, ctypes.CDLL] = {}


def source_path() -> Path:
    """The C++ source this module compiles."""
    return Path(__file__).resolve().parents[1] / "csrc" / "cpu_block_gemv.cpp"


def _library_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def cache_directory(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """The directory a build goes into, from the argument or ``RELIC_CPU_GEMV_CACHE``.

    Explicit rather than defaulted: a host that never asked for this operator must
    not have it compiled behind its back, and the repository does not get to pick a
    location inside someone's home directory.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    configured = os.environ.get(CACHE_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            f"the compiled host CPU GEMV is opt-in: set {CACHE_ENV} to a writable directory "
            f"or pass cache_dir=... (no artifact was built or loaded)"
        )
    return Path(configured).expanduser()


def _extra_flags(
    explicit: list[str] | tuple[str, ...] | str | None = None,
) -> list[str]:
    """Extra compiler flags, from the argument or ``RELIC_CPU_GEMV_CXXFLAGS``."""
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

    The name is content-addressed over the source, the build signature and the
    extra flags, so a changed toolchain never silently reuses an old binary.
    """
    material = source_path().read_bytes() + _BUILD_SIGNATURE + b"\0"
    material += b"\0".join(flag.encode("utf-8") for flag in _extra_flags(extra_flags))
    digest = hashlib.sha256(material).hexdigest()[:16]
    return cache_directory(cache_dir) / f"cpu_block_gemv_abi{ABI_VERSION}_{digest}{_library_suffix()}"


def _compiler(explicit: str | None = None) -> str:
    """Resolve the C++ compiler, honouring a named one instead of substituting another.

    A compiler chosen by the caller -- through the argument or through
    ``RELIC_CPU_GEMV_CXX`` -- is a decision, so failing to find it is an error
    rather than permission to build with something else. Only the implicit
    candidates fall through to the next one.
    """
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
        "no C++ compiler found for the compiled host CPU GEMV: set "
        f"{COMPILER_ENV} to a compiler path, or install one of {', '.join(_COMPILERS)}"
    )


def _build_command(compiler: str, output: Path, extra_flags: list[str]) -> list[str]:
    """The compile line, with the rounding-critical flags spelled out.

    ``-ffp-contract=off`` is load-bearing: the reference's per-block scale
    multiplies and its K accumulation are separate rounded operations, and a
    compiler free to contract them into an FMA would produce a different kernel.
    ``-fno-fast-math`` keeps it from reassociating or flushing subnormals. ``-O3``
    is what makes a one-row GEMV worth compiling at all.
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

    Idempotent: the artifact name carries the source hash and the build signature,
    so an existing file is the same kernel and is reused. A concurrent build is
    harmless -- the temporary object is renamed into place atomically, so a reader
    never observes a half-written library.
    """
    flags = _extra_flags(extra_flags)
    target = compiled_library_path(cache_dir, flags)
    if target.is_file() and not force:
        return target
    source = source_path()
    if not source.is_file():
        raise RuntimeError(f"the compiled host CPU GEMV source is missing: {source}")
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
        raise RuntimeError(f"could not run the C++ compiler {compiler_path!r}: {error}") from error
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "compiling cpu_block_gemv.cpp failed with "
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
    library.relic_cpu_block_gemv_abi_version.restype = ctypes.c_int32
    library.relic_cpu_block_gemv_abi_version.argtypes = []
    library.relic_cpu_block_gemv_fp8.restype = ctypes.c_int32
    library.relic_cpu_block_gemv_fp8.argtypes = [
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
    found = int(library.relic_cpu_block_gemv_abi_version())
    if found != ABI_VERSION:
        raise RuntimeError(
            f"the cached host CPU GEMV artifact at {target} reports ABI version {found}, "
            f"but this loader speaks {ABI_VERSION}; delete it to rebuild"
        )
    _LOADED[key] = library
    return library


def fp8_block_gemv_available(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> bool:
    """Whether the artifact is already built for this source and these flags.

    Reports the cache, not the toolchain: a caller that wants to know whether the
    operator can be *built* should call :func:`build` and handle its error, which
    names the missing compiler.
    """
    try:
        return compiled_library_path(cache_dir, extra_flags).is_file()
    except RuntimeError:
        return False


def _checked_bytes(tensor, name: str, allowed: tuple) -> "torch.Tensor":  # noqa: F821
    """Validate one operand and return it as a contiguous ``uint8`` view.

    Byte-coded operators read encodings, not values, so a dtype view is the right
    conversion and a numeric cast would be a bug. The layout checks are what makes
    handing a raw pointer to C++ sound; nothing here may make a copy.
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in allowed:
        raise TypeError(f"{name} has dtype {tensor.dtype}, expected one of {allowed}")
    if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.ndim != 2:
        raise ValueError(f"{name} must be a contiguous strided matrix")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on the CPU; this kernel is the host path")
    if tensor.requires_grad:
        raise ValueError("byte-coded block GEMV is an inference operator, not an autograd operation")
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


#: The binary32 bit pattern of the smallest original E8M0 grid value, ``2 ** -127``.
#: It is a subnormal, which is why it needs its own case rather than being read as
#: an exponent field.
_GRID_LOWEST_BITS = 0x00400000


def _fp32_grid_to_bytes(tensor, name: str):
    """An exact FP32 E8M0 grid as the raw bytes the kernel reads.

    ``2 ** (byte - 127)`` means the byte *is* the binary32 exponent field of the
    power of two, so the conversion is a field extraction rather than a logarithm.
    The smallest grid value is the one exception and is handled explicitly. Every
    entry is validated, because silently snapping a non-power-of-two onto a
    neighbouring grid would rescale the weight by a factor nobody asked for.
    """
    import torch

    if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.ndim != 2:
        raise ValueError(f"{name} must be a contiguous strided matrix")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on the CPU; this kernel is the host path")
    if tensor.requires_grad:
        raise ValueError("byte-coded block GEMV is an inference operator, not an autograd operation")
    bits = tensor.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    lowest = bits == _GRID_LOWEST_BITS
    normal = (bits >= 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 254)
    valid = lowest | normal
    if not bool(valid.all()):
        raise ValueError(
            f"{name} is not an exact original E8M0 grid: {int((~valid).sum())} of "
            f"{tensor.numel()} entries are not a power of two in [2**-127, 2**127]"
        )
    # The exponent field is the byte for both cases: the subnormal keeps field 0,
    # which is exactly the byte a 2**-127 scale is stored as.
    return exponent.to(torch.uint8)


def _refuse_non_finite_scales(raw, name: str) -> None:
    """Reject the one raw scale byte that is not a finite original-format grid value.

    ``2 ** 128`` has no binary32 representation, so accepting the byte would mean
    answering with an infinity on one path and a NaN on another depending on which
    reference was consulted. The contract here is finite grids only.
    """
    import torch

    offending = int((raw == 255).sum())
    if offending:
        raise ValueError(
            f"{name} contains {offending} raw byte(s) 0xFF, which is not a finite E8M0 "
            "grid value (it would mean 2**128); this operator implements the finite "
            "original-format contract and refuses it rather than choosing an infinity"
        )


def _checked_scales(tensor, name: str):
    """A scale grid as raw E8M0 bytes, from either storage form."""
    import torch

    if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32:
        return _fp32_grid_to_bytes(tensor, name)
    raw = _checked_bytes(tensor, name, _scale_dtypes())
    _refuse_non_finite_scales(raw, name)
    return raw


def _code_dtypes() -> tuple:
    import torch

    allowed = [torch.uint8, torch.int8]
    floating = getattr(torch, "float8_e4m3fn", None)
    if floating is not None:
        allowed.append(floating)
    return tuple(allowed)


def _scale_dtypes() -> tuple:
    import torch

    allowed = [torch.uint8, torch.int8]
    floating = getattr(torch, "float8_e8m0fnu", None)
    if floating is not None:
        allowed.append(floating)
    return tuple(allowed)


def fp8_block_gemv(
    a_codes,
    a_scale,
    b_codes,
    b_scale,
    *,
    block_rows: int = 32,
    threads: int = 1,
    out=None,
    cache_dir: str | os.PathLike[str] | None = None,
):
    """Exact ``[1, K] x [N, K]`` block-32 FP8 GEMV on raw original bytes.

    ``a_codes``/``b_codes`` are e4m3 codes as bytes (or as ``float8_e4m3fn``) and
    ``a_scale``/``b_scale`` are E8M0 grids, as raw bytes (``float8_e8m0fnu``) or as
    the exact FP32 powers the CPU activation quantizer returns. Both scales
    are one exponent per 32-wide K block: ``a_scale`` is ``[1, K // 32]`` and
    ``b_scale`` is ``[N // 32, K // 32]`` for the square grid a dense FP8 weight is
    stored with (``block_rows=32``), or ``[N, K // 32]`` for the per-row grid
    (``block_rows=1``). Returns FP32 accumulators of shape ``[1, N]``; the caller
    owns the final narrowing, exactly as it does around the Torch oracle.

    ``threads`` splits the output columns. The result does not depend on it: each
    column's K accumulation is sequential and identical whatever the thread count,
    so this is a pure throughput knob rather than a numerical one.

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
    if block_rows == BLOCK and n % BLOCK:
        raise ValueError(
            f"a square weight-scale grid needs N divisible by 32, got N = {n}; "
            "use block_rows=1 for a per-row grid"
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

    library = load(cache_dir=cache_dir)
    threads_used = ctypes.c_int32(0)
    status = library.relic_cpu_block_gemv_fp8(
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
            f"the compiled host CPU GEMV refused the call with status {status} "
            "(1 = bad arguments, 2 = out of memory)"
        )
    # The raw pointer writes bypass the autograd graph; nothing here is
    # differentiable, so the return is the caller's own storage, filled in place.
    return out