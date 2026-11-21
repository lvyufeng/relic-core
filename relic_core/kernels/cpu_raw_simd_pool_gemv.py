"""Raw-byte SIMD host GEMV that reuses one process-local worker pool.

This is the same reduction as :mod:`relic_core.kernels.cpu_raw_simd_gemv`, over
the same raw original bytes, producing the same bits -- the frozen kernel's
tables, block rule, plan and range loop are copied into this C++ source byte for
byte, and ``tests/test_cpu_raw_simd_pool_gemv.py`` pins that text and the frozen
file's SHA-256. What changes is only where the threads come from.

The frozen kernel spawned ``threads - 1`` ``std::thread`` objects and joined them
on **every** call. For the host Engram projection that is a repeated 96-thread
GEMV: 95 creations and joins per call, and because a spawned thread starts
claiming columns while the spawn loop is still running, the first columns of a
call are computed by fewer workers than were asked for. This module keeps one
pool per process alive across calls: the workers wait on a condition variable, a
call publishes its plan, every worker starts from that publication, and the call
returns after the last worker finishes.

The arithmetic contract is unchanged, which is the whole point:

* the operands stay byte-coded; each byte becomes its exact e4m3 value and its
  **nominal** exponent ``F(v) = max(true exponent, -6)`` in one table word;
* the 32 products are split into two 16-term halves, each truncated toward zero
  onto ``2 ** (anchor - 13)`` with its own anchor; the second half's window is
  ``max(its anchor, floor(log2|first half|))``;
* the block result is renormalised once more onto ``2 ** (floor(log2|total|) - 13)``;
* the activation scale is applied first and the weight scale second, in FP32, and
  the blocks are accumulated sequentially in increasing K order.

``threads`` only chooses the column partition. Every column's K accumulation is
sequential and independent of it, so the result bits do not depend on the thread
count -- only the number of workers that share the columns does.

Pool semantics, and why they are visible from here:

* one active geometry per process; a call asking for a different ``threads``
  resizes the pool while every worker is idle, so a host never runs two
  96-worker pools at once;
* concurrent callers are serialized, and each call is synchronous: the plan is
  cleared before returning and no operand pointer is retained;
* a forked child inherits pool bookkeeping whose threads do not exist. The child's
  first call refuses by name instead of joining them;
  :func:`reset_after_fork` (registered with :func:`os.register_at_fork` when that
  exists) discards that state so the child builds its own pool;
* the pool is stopped and joined at normal process exit.

**Opt-in and never a default**, with its own cache variable and its own build
signature, so this artifact can never be confused with the frozen one: nothing is
compiled, downloaded or written unless ``cache_dir=`` or
``RELIC_CPU_RAW_SIMD_POOL_CACHE`` names a directory. The operand and scale
validators are :mod:`relic_core.kernels.cpu_block_gemv`'s own, imported rather
than copied, so the two operators cannot drift apart about what an acceptable
input is -- including the refusal of raw scale byte 255 and of a
non-power-of-two FP32 grid.

Example::

    import torch
    from relic_core.kernels.cpu_raw_simd_pool_gemv import fp8_raw_simd_pool_gemv

    out = fp8_raw_simd_pool_gemv(a_codes, a_scale, wkv_codes, wkv_scale, threads=96)
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
    "DEFAULT_THREADS",
    "build",
    "cache_directory",
    "compiled_library_path",
    "fp8_raw_simd_pool_gemv",
    "fp8_raw_simd_pool_gemv_available",
    "load",
    "reset_after_fork",
    "source_path",
    "vector_width",
]

#: Version of the C ABI below. A cached artifact built from an older interface is
#: rejected instead of being called with the wrong argument list.
ABI_VERSION = 1

#: Where the compiled kernel is cached. Its own variable, because this is its own
#: kernel with its own source: an artifact of one is not an artifact of the other.
CACHE_ENV = "RELIC_CPU_RAW_SIMD_POOL_CACHE"

#: Optional override for the C++ compiler; same resolution order as the frozen
#: kernel's loader.
COMPILER_ENV = "RELIC_CPU_RAW_SIMD_POOL_CXX"

#: Optional extra compiler flags, shell-split, for a toolchain that is not on the
#: compiler's default search path. Part of the artifact identity.
CXXFLAGS_ENV = "RELIC_CPU_RAW_SIMD_POOL_CXXFLAGS"

#: Measured worker choice for the released Engram, not a hardware CPU-count claim.
DEFAULT_THREADS = 96

#: Part of the artifact identity: a differently compiled kernel is a different
#: kernel. ``-ffp-contract=off`` and ``-fno-fast-math`` are load-bearing for the
#: same reason they are in the frozen kernel -- the reduction's separate
#: multiplies and additions are the reference's rounding.
_BUILD_SIGNATURE = (
    b"relic-cpu-raw-simd-pool-gemv;abi=1;std=c++17;-O3;-fno-fast-math;"
    b"-ffp-contract=off;-DNDEBUG"
)

_COMPILERS = ("c++", "g++", "clang++")

_LOADED: dict[str, ctypes.CDLL] = {}


def source_path() -> Path:
    """The C++ source this module compiles."""
    return Path(__file__).resolve().parents[1] / "csrc" / "cpu_raw_simd_pool_gemv.cpp"


def _library_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def cache_directory(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """The directory a build goes into, from the argument or ``CACHE_ENV``.

    Explicit rather than defaulted: a host that never asked for this operator must
    not have it compiled behind its back, and no directory is inferred from the
    current working directory.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    configured = os.environ.get(CACHE_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            f"the pooled raw-byte SIMD host GEMV is opt-in: set {CACHE_ENV} to a writable "
            f"directory or pass cache_dir=... (no artifact was built or loaded)"
        )
    return Path(configured).expanduser()


def _extra_flags(explicit: list[str] | tuple[str, ...] | str | None = None) -> list[str]:
    """Extra compiler flags, from the argument or ``CXXFLAGS_ENV``."""
    if explicit is None:
        flags = shlex.split(os.environ.get(CXXFLAGS_ENV, ""))
    elif isinstance(explicit, str):
        flags = shlex.split(explicit)
    else:
        flags = [str(flag) for flag in explicit]
    forbidden = {"-Ofast", "-ffast-math", "-funsafe-math-optimizations",
                 "-fassociative-math", "-freciprocal-math", "-ffinite-math-only",
                 "-fno-signed-zeros", "-ffp-contract=fast", "-ffp-contract=on"}
    if forbidden.intersection(flags):
        raise ValueError("the exact CPU reduction refuses floating-point relaxation flags")
    return flags


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
    name = f"cpu_raw_simd_pool_gemv_abi{ABI_VERSION}_{digest}{_library_suffix()}"
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
        "no C++ compiler found for the pooled raw-byte SIMD host GEMV: set "
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
        # The pool owns threads, so a Linux link needs the thread library that
        # `std::thread` does not pull in by itself.
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
        raise RuntimeError(f"the pooled raw-byte SIMD host GEMV source is missing: {source}")
    compiler_path = _compiler(compiler)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.stem + ".", suffix=_library_suffix()
    )
    os.close(handle)
    temporary = Path(temporary_name)
    resolved_parent = target.parent.resolve()
    if temporary.resolve().parent != resolved_parent or target.resolve().parent != resolved_parent:
        raise RuntimeError("the compiled artifact paths escape the explicitly selected cache")
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
            "compiling cpu_raw_simd_pool_gemv.cpp failed with "
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
    library.relic_cpu_raw_simd_pool_gemv_abi_version.restype = ctypes.c_int32
    library.relic_cpu_raw_simd_pool_gemv_abi_version.argtypes = []
    library.relic_cpu_raw_simd_pool_gemv_vector_width.restype = ctypes.c_int32
    library.relic_cpu_raw_simd_pool_gemv_vector_width.argtypes = []
    library.relic_cpu_raw_simd_pool_gemv_reset.restype = None
    library.relic_cpu_raw_simd_pool_gemv_reset.argtypes = []
    library.relic_cpu_raw_simd_pool_gemv_fp8.restype = ctypes.c_int32
    library.relic_cpu_raw_simd_pool_gemv_fp8.argtypes = [
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
    found = int(library.relic_cpu_raw_simd_pool_gemv_abi_version())
    if found != ABI_VERSION:
        raise RuntimeError(
            f"the cached pooled raw-byte SIMD host GEMV artifact at {target} reports ABI "
            f"version {found}, but this loader speaks {ABI_VERSION}; delete it to rebuild"
        )
    _LOADED[key] = library
    return library


def reset_after_fork() -> None:
    """Discard every loaded pool's inherited state, for a forked child to rebuild.

    A child inherits the pool's bookkeeping but none of its threads, so the C
    kernel refuses rather than joining threads that are not there; calling this
    (which only publishes a null pointer and allocates nothing) makes the child's
    next call build a pool of its own.
    """
    for library in list(_LOADED.values()):
        try:
            library.relic_cpu_raw_simd_pool_gemv_reset()
        except (AttributeError, OSError):
            # An artifact loaded by an older binding has no reset entry point;
            # the child then keeps the named refusal, which is still correct.
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=reset_after_fork)


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
               .relic_cpu_raw_simd_pool_gemv_vector_width())


def fp8_raw_simd_pool_gemv_available(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> bool:
    """Whether the artifact is already built for this source and these flags."""
    try:
        return compiled_library_path(cache_dir, extra_flags).is_file()
    except RuntimeError:
        return False


def fp8_raw_simd_pool_gemv(
    a_codes,
    a_scale,
    b_codes,
    b_scale,
    *,
    block_rows: int = 32,
    threads: int = DEFAULT_THREADS,
    out=None,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
):
    """Exact ``[1, K] x [N, K]`` block-32 FP8 GEMV over raw original bytes.

    The operands and the result are the frozen kernel's, and so is every bit of
    the arithmetic; see the module docstring. ``threads`` is a pure throughput
    knob across the output columns: each column's K accumulation is sequential and
    identical whatever the thread count.

    ``a_codes``/``b_codes`` are e4m3 codes as bytes (or as ``float8_e4m3fn``) and
    ``a_scale``/``b_scale`` are E8M0 grids, as raw bytes (``float8_e8m0fnu``) or as
    the exact FP32 powers the CPU activation quantizer returns. ``a_scale`` is
    ``[1, K // 32]``; ``b_scale`` is ``[N // 32, K // 32]`` for the square grid a
    dense FP8 weight is stored with (``block_rows=32``) or ``[N, K // 32]`` for the
    per-row grid (``block_rows=1``). Returns FP32 accumulators of shape ``[1, N]``;
    the caller owns the final narrowing.

    ``b_codes`` is read in place. Nothing is prepacked, copied or dequantized, so
    one call touches exactly the file's own bytes. The compiled kernel is selected
    by construction -- there is one implementation -- and a failed build raises
    rather than falling back to a slower path that would quietly answer a
    different question about which code ran.
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
    status = library.relic_cpu_raw_simd_pool_gemv_fp8(
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
        if status == 3:
            raise RuntimeError(
                "this process inherited a pooled host GEMV from a fork(): its workers do not "
                "exist here. Call relic_core.kernels.cpu_raw_simd_pool_gemv.reset_after_fork() "
                "in the child (the module registers it with os.register_at_fork) and retry"
            )
        raise RuntimeError(
            f"the pooled raw-byte SIMD host GEMV refused the call with status {status} "
            "(1 = bad arguments, 2 = out of memory)"
        )
    return out