"""Prepacked, column-vectorised exact host-CPU FP8 block-32 GEMV.

This is the second, opt-in implementation of the reduction
``relic_core.kernels.cpu_block_gemv`` (v1) computes, for the same host-side Engram
``wkv`` projection. It is **bit-identical** to v1 and it exists to make that
projection cheap enough for a decode step: v1 re-derives, per call and per lane, a
decode word from the stored e4m3 byte with a 256-entry table lookup and then walks
one output column at a time. This module moves the decode to load time and walks
**four output columns at once** with NEON integer/float intrinsics.

What is prepacked, and why it is the stored bytes rather than a new weight:

* **Decode words.** Every stored e4m3 weight byte becomes the same 32-bit word v1
  builds in a register -- the value's binary32 bits above bit 20 plus the biased
  nominal exponent, i.e. v1's own ``g_packed`` table entry. They are laid out
  ``[K // 32, N, 32]``, so one output column's 32 lanes for one K block are 128
  contiguous bytes: the prepack is a sequential write and the GEMV a sequential
  read, and the 32 lanes of a column become NEON vectors instead of 32 gathers.
* **Weight scale powers.** The raw E8M0 bytes become their exact binary32 powers,
  once, instead of per call.
* **Per-column NaN classification.** A column whose weights contain an e4m3 NaN
  code is resolved at prepack, so the compute loop never converts a NaN to an
  integer.

Nothing else changes, and that is the contract: the halves, the window, the carry,
the final cut, the separate activation-then-weight FP32 scale multiplies and the
sequential K accumulation are the same statements in the same order. The
prepacked words are a private derived form; the original stored bytes remain the
source of truth and the checkpoint is never rewritten.

**Memory is the honest cost.** The words are four bytes per stored code byte, so
the prepacked form is about **4x** the weight's code bytes, plus the scale powers
and one validity flag per column. For the released ``wkv`` geometry
(``K = 6144``, ``N = 25600``) that is 629,145,600 bytes of words for 157,286,400
bytes of codes -- 2520 MiB (decimal) per weight, i.e. the 629 MiB-per-``wkv`` /
1.2 GiB-for-two figure the caller must budget for. :func:`prepacked_nbytes`
computes it without allocating, and ``max_bytes=`` bounds it.

**Opt-in, and never a default.** It is compiled locally from the C++ source beside
this file into a cache directory the caller names -- through ``cache_dir=`` or
``RELIC_CPU_GEMV_PREPACKED_CACHE`` (which falls back to v1's
``RELIC_CPU_GEMV_CACHE`` so one environment variable can cover both). Nothing is
built, downloaded or written unless a cache directory is given, so a checkout that
does not ask for this operator pays nothing.

This module is deliberately **separate from v1**, which stays frozen:

* its own C++ translation unit, its own ABI version (2), its own artifact name and
  its own content fingerprint (source bytes + build signature + flags), so a stale
  or mismatched library can never be loaded in place of the other;
* its own environment variables and its own module-level state;
* it never edits, wraps or monkey-patches ``cpu_block_gemv``. The only thing it
  takes from v1 is the operand-validation contract, imported rather than copied so
  the two cannot drift about which byte is a NaN or which scale is refused.

Threading. The compiled kernel keeps a small process-wide worker pool, so a decode
step does not pay thread creation per token; a host that forks after first use gets
a fresh pool in the child. ``threads`` only splits output columns, and every column
is computed by the same code, so the result does not depend on it.

Example::

    import torch
    from relic_core.kernels.cpu_block_gemv_prepacked import (
        fp8_block_gemv_prepacked, prepack_fp8_weight,
    )

    # a_codes [1, K], a_scale [1, K // 32], b_codes [N, K], b_scale [N // 32, K // 32]
    weight = prepack_fp8_weight(wkv_codes, wkv_scale, block_rows=32)
    out = fp8_block_gemv_prepacked(a_codes, a_scale, weight, threads=8)
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

#: The v1 operand contract (byte views, exact-grid validation, NaN-column and byte
#: 255 refusal) is imported from the frozen module rather than duplicated, so the
#: two operators cannot disagree about which operand is legal.
from relic_core.kernels.cpu_block_gemv import (
    _checked_bytes,
    _checked_scales,
    _code_dtypes,
)

__all__ = [
    "ABI_VERSION",
    "BLOCK",
    "CACHE_ENV",
    "COMPILER_ENV",
    "CXXFLAGS_ENV",
    "PrepackedFP8Weight",
    "build",
    "cache_directory",
    "compiled_library_path",
    "fp8_block_gemv_prepacked",
    "load",
    "prepack_fp8_weight",
    "prepacked_available",
    "prepacked_nbytes",
    "source_path",
]

#: Block width of the reference reduction, unchanged from v1.
BLOCK = 32

#: Version of the C ABI below, distinct from v1's. A cached artifact built from a
#: different interface is rejected instead of being called with the wrong
#: argument list, and the artifact name carries the version besides.
ABI_VERSION = 2

#: Where the compiled kernel is cached. Its own variable, falling back to v1's so a
#: caller can configure both with one setting.
CACHE_ENV = "RELIC_CPU_GEMV_PREPACKED_CACHE"
_FALLBACK_CACHE_ENV = "RELIC_CPU_GEMV_CACHE"

#: Optional override for the C++ compiler (own variable, then v1's, then PATH).
COMPILER_ENV = "RELIC_CPU_GEMV_PREPACKED_CXX"
_FALLBACK_COMPILER_ENV = "RELIC_CPU_GEMV_CXX"

#: Optional extra compiler flags, shell-split. They are part of the artifact
#: identity, because a differently compiled kernel is a different kernel.
CXXFLAGS_ENV = "RELIC_CPU_GEMV_PREPACKED_CXXFLAGS"
_FALLBACK_CXXFLAGS_ENV = "RELIC_CPU_GEMV_CXXFLAGS"

#: Part of the artifact identity. Any change to the build flags has to invalidate a
#: cached library. The ABI number is in both the name and the signature.
_BUILD_SIGNATURE = (
    b"relic-cpu-block-gemv-prepacked;abi=2;std=c++17;-O3;-fno-fast-math;-ffp-contract=off;-DNDEBUG"
)

_COMPILERS = ("c++", "g++", "clang++")

_LOADED: dict[str, ctypes.CDLL] = {}

#: Compute variants, mirroring the C++ `kAuto` / `kScalar`.
_VARIANTS = {"auto": 0, "scalar": 1}


def source_path() -> Path:
    """The C++ source this module compiles (its own, not v1's)."""
    return Path(__file__).resolve().parents[1] / "csrc" / "cpu_block_gemv_prepacked.cpp"


def _library_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def _configured(name: str, fallback: str) -> str:
    """A setting from this module's variable, else the v1 alias, else empty."""
    value = os.environ.get(name, "").strip()
    return value or os.environ.get(fallback, "").strip()


def cache_directory(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """The directory a build goes into, from the argument or the environment.

    Explicit rather than defaulted: a host that never asked for this operator must
    not have it compiled behind its back.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    configured = _configured(CACHE_ENV, _FALLBACK_CACHE_ENV)
    if not configured:
        raise RuntimeError(
            f"the compiled prepacked host CPU GEMV is opt-in: set {CACHE_ENV} "
            f"(or {_FALLBACK_CACHE_ENV}) to a writable directory, or pass "
            "cache_dir=... (no artifact was built or loaded)"
        )
    return Path(configured).expanduser()


def _extra_flags(
    explicit: list[str] | tuple[str, ...] | str | None = None,
) -> list[str]:
    """Extra compiler flags, from the argument or this module's environment."""
    if explicit is None:
        return shlex.split(_configured(CXXFLAGS_ENV, _FALLBACK_CXXFLAGS_ENV))
    if isinstance(explicit, str):
        return shlex.split(explicit)
    return [str(flag) for flag in explicit]


def _fingerprint(extra_flags: list[str]) -> str:
    """Content address of the artifact: source bytes, ABI, flags, this module.

    The source is v2's translation unit only; it has no local includes, so the
    source bytes *are* the dependency set. The build flags and the ABI number are
    appended because the same source compiled differently is a different kernel.
    """
    material = source_path().read_bytes() + _BUILD_SIGNATURE + b"\0"
    material += b"\0".join(flag.encode("utf-8") for flag in extra_flags)
    return hashlib.sha256(material).hexdigest()[:16]


def compiled_library_path(
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> Path:
    """Where the artifact for the current source and flags lives, built or not."""
    digest = _fingerprint(_extra_flags(extra_flags))
    return (
        cache_directory(cache_dir)
        / f"cpu_block_gemv_prepacked_abi{ABI_VERSION}_{digest}{_library_suffix()}"
    )


def _compiler(explicit: str | None = None) -> str:
    """Resolve the compiler, honouring a named one instead of substituting another."""
    configured = _configured(COMPILER_ENV, _FALLBACK_COMPILER_ENV) or None
    for named, source in ((explicit, "compiler="), (configured, "a configured compiler")):
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
        "no C++ compiler found for the compiled prepacked host CPU GEMV: set "
        f"{COMPILER_ENV} to a compiler path, or install one of {', '.join(_COMPILERS)}"
    )


def _build_command(compiler: str, output: Path, extra_flags: list[str]) -> list[str]:
    """The compile line, with the rounding-critical flags spelled out.

    ``-ffp-contract=off`` is load-bearing: the per-block scale multiplies and the K
    accumulation are separate rounded operations, and a compiler free to contract
    them into an FMA would produce a different kernel. ``-fno-fast-math`` keeps it
    from reassociating or flushing subnormals. ``-O3`` is what makes the SIMD path
    exist at all.
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

    Idempotent: the name carries the source hash, the ABI and the build signature,
    so an existing file is the same kernel. A concurrent build is harmless -- the
    temporary object is renamed into place atomically.
    """
    flags = _extra_flags(extra_flags)
    target = compiled_library_path(cache_dir, flags)
    if target.is_file() and not force:
        return target
    source = source_path()
    if not source.is_file():
        raise RuntimeError(f"the compiled prepacked host CPU GEMV source is missing: {source}")
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
            "compiling cpu_block_gemv_prepacked.cpp failed with "
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
    library.relic_cpu_block_gemv_prepacked_abi_version.restype = ctypes.c_int32
    library.relic_cpu_block_gemv_prepacked_abi_version.argtypes = []
    library.relic_cpu_block_gemv_prepacked_prepack.restype = ctypes.c_int32
    library.relic_cpu_block_gemv_prepacked_prepack.argtypes = [
        ctypes.c_void_p,   # b_codes
        ctypes.c_void_p,   # b_scale
        ctypes.c_void_p,   # words
        ctypes.c_void_p,   # scale_powers
        ctypes.c_void_p,   # valid
        ctypes.c_int64,    # n
        ctypes.c_int64,    # k
        ctypes.c_int32,    # block_rows
    ]
    library.relic_cpu_block_gemv_prepacked_fp8.restype = ctypes.c_int32
    library.relic_cpu_block_gemv_prepacked_fp8.argtypes = [
        ctypes.c_void_p,   # a_codes
        ctypes.c_void_p,   # a_scale
        ctypes.c_void_p,   # words
        ctypes.c_void_p,   # scale_powers
        ctypes.c_void_p,   # valid
        ctypes.c_void_p,   # out
        ctypes.c_int64,    # n
        ctypes.c_int64,    # k
        ctypes.c_int32,    # block_rows
        ctypes.c_int32,    # threads
        ctypes.c_int32,    # variant
        ctypes.POINTER(ctypes.c_int32),  # threads_used
    ]
    found = int(library.relic_cpu_block_gemv_prepacked_abi_version())
    if found != ABI_VERSION:
        raise RuntimeError(
            f"the cached prepacked host CPU GEMV artifact at {target} reports ABI "
            f"version {found}, but this loader speaks {ABI_VERSION}; delete it to rebuild"
        )
    _LOADED[key] = library
    return library


def prepacked_available(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> bool:
    """Whether the artifact is already built for this source and these flags."""
    try:
        return compiled_library_path(cache_dir, extra_flags).is_file()
    except RuntimeError:
        return False


def prepacked_nbytes(n: int, k: int, *, block_rows: int = BLOCK) -> dict[str, int]:
    """Exactly how many bytes :func:`prepack_fp8_weight` would allocate.

    Reported rather than guessed, because this is the operator's real cost: the
    decode words are four bytes per stored code byte and the rest is small. The
    caller budgets against these numbers (``max_bytes=`` enforces it) instead of
    discovering them as an out-of-memory error.
    """
    for name, value in (("n", n), ("k", k)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    blocks = k // BLOCK
    if blocks * BLOCK != k:
        raise ValueError(f"k must be a multiple of {BLOCK}")
    scale_rows = _scale_rows(n, block_rows)
    words = blocks * BLOCK * n * 4
    scales = scale_rows * blocks * 4
    valid = n
    source = n * k + scale_rows * blocks
    return {
        "words": words,
        "scale_powers": scales,
        "valid": valid,
        "total": words + scales + valid,
        "source": source,
    }


def _scale_rows(n: int, block_rows: int) -> int:
    if not isinstance(block_rows, int) or isinstance(block_rows, bool):
        raise TypeError("block_rows must be an int")
    if block_rows not in (1, BLOCK):
        raise ValueError("block_rows must be 1 or 32")
    if block_rows == BLOCK and n % BLOCK:
        raise ValueError(
            f"a square weight-scale grid needs N divisible by 32, got N = {n}; "
            "use block_rows=1 for a per-row grid"
        )
    return n if block_rows == 1 else n // block_rows


def _source_version(tensor) -> int | None:
    """A tensor's version counter, or ``None`` when the tensor cannot report one.

    An inference tensor -- one created inside ``torch.inference_mode()`` -- raises
    ``RuntimeError("Inference tensors do not track version counter")``. That is not
    an error here: the prepacked words are already a self-contained copy of the
    bytes the kernel reads, so the source tensors are provenance rather than input.
    Recording ``None`` says "this source could not be checked" instead of claiming
    it was.
    """
    try:
        return int(tensor._version)
    except (AttributeError, RuntimeError):
        return None


class PrepackedFP8Weight:
    """A weight matrix prepared for :func:`fp8_block_gemv_prepacked`.

    Opaque by intent: the three buffers are the kernel's input layout and are
    exposed for inspection and accounting, not for editing. Mutating them makes the
    object inconsistent with its own metadata, which the call checks.
    """

    __slots__ = (
        "n",
        "k",
        "block_rows",
        "words",
        "scale_powers",
        "valid",
        "bytes",
        "_source_codes",
        "_source_scales",
        "_source_versions",
        "_fingerprint",
    )

    def __init__(self, n, k, block_rows, words, scale_powers, valid, source_codes,
                 source_scales, source_versions, fingerprint):
        self.n = n
        self.k = k
        self.block_rows = block_rows
        self.words = words
        self.scale_powers = scale_powers
        self.valid = valid
        self._source_codes = source_codes
        self._source_scales = source_scales
        self._source_versions = source_versions
        self._fingerprint = fingerprint
        self.bytes = {
            "words": words.numel() * words.element_size(),
            "scale_powers": scale_powers.numel() * scale_powers.element_size(),
            "valid": valid.numel() * valid.element_size(),
        }
        self.bytes = dict(self.bytes, total=sum(self.bytes.values()))

    @property
    def valid_columns(self) -> int:
        """How many columns are free of NaN codes (the rest answer NaN)."""
        return int(self.valid.sum())

    @property
    def fingerprint(self) -> str:
        """The source/flags fingerprint this weight was prepared under."""
        return self._fingerprint

    def source_versions(self) -> dict[str, int | None]:
        """The recorded version counters of the source tensors (``None`` if untracked)."""
        return {
            "codes": self._source_versions[0],
            "scales": self._source_versions[1],
        }

    def verify_sources(self) -> None:
        """Raise if a source tensor was mutated after this weight was prepacked.

        This is a provenance check, not a correctness gate: the kernel reads the
        prepacked words, so a mutated source cannot change an answer. It exists so a
        caller that edits weights in place learns that the prepacked form is now
        stale instead of silently getting the old numbers. An inference tensor
        reports no version and is reported as unverifiable rather than passing.
        """
        for name, tensor, recorded in (
            ("codes", self._source_codes, self._source_versions[0]),
            ("scales", self._source_scales, self._source_versions[1]),
        ):
            current = _source_version(tensor)
            if recorded is None or current is None:
                raise RuntimeError(
                    f"the prepacked weight's source {name} tensor does not track a "
                    "version counter (an inference tensor), so its immutability "
                    "cannot be verified; the prepacked words remain the data the "
                    "kernel reads"
                )
            if current != recorded:
                raise RuntimeError(
                    f"the prepacked weight's source {name} tensor was modified after "
                    f"it was prepared (version {recorded} -> {current}); prepack it "
                    "again rather than reusing a stale derived form"
                )

    def __repr__(self) -> str:
        return (
            f"PrepackedFP8Weight(n={self.n}, k={self.k}, block_rows={self.block_rows}, "
            f"bytes={self.bytes['total']}, valid_columns={self.valid_columns}, "
            f"fingerprint={self._fingerprint!r})"
        )


def prepack_fp8_weight(
    b_codes,
    b_scale,
    *,
    block_rows: int = BLOCK,
    max_bytes: int | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
):
    """Prepare one ``[N, K]`` block-32 FP8 weight for the prepacked GEMV.

    ``b_codes`` are e4m3 codes as bytes (or ``float8_e4m3fn``) and ``b_scale`` is
    the E8M0 grid, as raw bytes or as exact FP32 powers -- the same two forms and
    the same validations v1 accepts, because the operand contract is imported from
    it rather than re-derived.

    Returns a :class:`PrepackedFP8Weight`. The source tensors are kept only as
    provenance and are never read by the kernel again: the returned object is
    self-contained, so a host that frees the original bytes can still run the GEMV.

    ``max_bytes=`` refuses a prepack that would exceed it, naming the required and
    allowed sizes, before any allocation happens. See :func:`prepacked_nbytes`.
    """
    import torch

    codes = _checked_bytes(b_codes, "weight codes", _code_dtypes())
    scales = _checked_scales(b_scale, "weight scales")
    n, k = int(codes.shape[0]), int(codes.shape[1])
    if n <= 0 or k <= 0 or k % BLOCK:
        raise ValueError("weight codes must be a positive [N, K] matrix with K divisible by 32")
    scale_rows = _scale_rows(n, block_rows)
    blocks = k // BLOCK
    if tuple(scales.shape) != (scale_rows, blocks):
        raise ValueError(
            f"weight scales must have shape {(scale_rows, blocks)} for "
            f"block_rows={block_rows}, got {tuple(scales.shape)}"
        )

    costs = prepacked_nbytes(n, k, block_rows=block_rows)
    if max_bytes is not None:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer or None")
        if costs["total"] > max_bytes:
            raise ValueError(
                f"prepacking this weight needs {costs['total']} bytes "
                f"({costs['words']} of decode words, 4x its {costs['source']} stored "
                f"bytes) but max_bytes is {max_bytes}"
            )

    library = load(cache_dir=cache_dir, extra_flags=extra_flags)
    fingerprint = _fingerprint(_extra_flags(extra_flags))
    words = torch.empty((blocks, n, BLOCK), dtype=torch.uint32)
    scale_powers = torch.empty((scale_rows, blocks), dtype=torch.float32)
    valid = torch.empty((n,), dtype=torch.uint8)
    status = library.relic_cpu_block_gemv_prepacked_prepack(
        codes.data_ptr(),
        scales.data_ptr(),
        words.data_ptr(),
        scale_powers.data_ptr(),
        valid.data_ptr(),
        n,
        k,
        int(block_rows),
    )
    if status != 0:
        raise RuntimeError(
            f"the compiled prepacked host CPU GEMV refused to prepack with status "
            f"{status} (1 = bad arguments, 2 = out of memory)"
        )
    return PrepackedFP8Weight(
        n=n,
        k=k,
        block_rows=block_rows,
        words=words,
        scale_powers=scale_powers,
        valid=valid,
        source_codes=codes,
        source_scales=scales,
        source_versions=(_source_version(codes), _source_version(scales)),
        fingerprint=fingerprint,
    )


def _checked_weight(weight) -> "PrepackedFP8Weight":
    """Validate a prepacked weight object and its buffers."""
    import torch

    if not isinstance(weight, PrepackedFP8Weight):
        raise TypeError(
            "the prepacked GEMV needs the object prepack_fp8_weight returned, not "
            f"{type(weight).__name__}"
        )
    blocks = weight.k // BLOCK
    expected = {
        "words": ((blocks, weight.n, BLOCK), torch.uint32),
        "scale_powers": ((_scale_rows(weight.n, weight.block_rows), blocks), torch.float32),
        "valid": ((weight.n,), torch.uint8),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(weight, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"prepacked weight {name} is not a tensor")
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
            raise ValueError(
                f"prepacked weight {name} has shape/dtype "
                f"{tuple(tensor.shape)}/{tensor.dtype}, expected {tuple(shape)}/{dtype}"
            )
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"prepacked weight {name} must be a contiguous CPU tensor")
    return weight


def fp8_block_gemv_prepacked(
    a_codes,
    a_scale,
    weight,
    *,
    threads: int = 1,
    out=None,
    variant: str = "auto",
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
):
    """Exact ``[1, K] x [N, K]`` block-32 FP8 GEMV against a prepacked weight.

    Bit-identical to ``cpu_block_gemv.fp8_block_gemv``: same operands, same
    reduction, same rounding. The only differences are the derived weight form and
    how the columns are scheduled.

    ``variant="scalar"`` forces the portable, non-SIMD path on any host. It is a
    test hook for the fallback that non-ARM targets compile -- not a performance
    choice, and the answer is the same either way.

    ``threads`` splits output columns; the result does not depend on it. The
    compiled kernel keeps its worker pool between calls.
    """
    import torch

    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be an integer >= 1")
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_VARIANTS)}, got {variant!r}")

    codes = _checked_bytes(a_codes, "activation codes", _code_dtypes())
    activation_scales = _checked_scales(a_scale, "activation scales")
    weight = _checked_weight(weight)

    m, k = codes.shape
    n = weight.n
    if m != 1:
        raise ValueError(
            f"this is the M = 1 GEMV; the activation codes have {m} rows. Use the "
            "batched block GEMM for a prompt"
        )
    if weight.k != k:
        raise ValueError(
            f"the activation row is K = {k} wide and the prepacked weight is "
            f"K = {weight.k}; they were not read from the same checkpoint"
        )
    if tuple(activation_scales.shape) != (1, k // BLOCK):
        raise ValueError(f"activation scales must have shape {(1, k // BLOCK)}")

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
            ("prepacked words", weight.words),
            ("prepacked scales", weight.scale_powers),
            ("prepacked validity", weight.valid),
        ):
            if torch._C._overlaps(out, tensor):
                raise ValueError(f"out must not overlap {name}")

    library = load(cache_dir=cache_dir, extra_flags=extra_flags)
    threads_used = ctypes.c_int32(0)
    status = library.relic_cpu_block_gemv_prepacked_fp8(
        codes.data_ptr(),
        activation_scales.data_ptr(),
        weight.words.data_ptr(),
        weight.scale_powers.data_ptr(),
        weight.valid.data_ptr(),
        out.data_ptr(),
        n,
        k,
        int(weight.block_rows),
        threads,
        _VARIANTS[variant],
        ctypes.byref(threads_used),
    )
    if status != 0:
        raise RuntimeError(
            f"the compiled prepacked host CPU GEMV refused the call with status "
            f"{status} (1 = bad arguments, 2 = out of memory)"
        )
    return out