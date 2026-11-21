"""Exact compact host-CPU FP8 block-32 GEMV, over a 16-bit-per-weight prepack.

This is the third implementation of the same reduction, and it differs from its two
predecessors only in what it costs:

* ``cpu_block_gemv`` (v1) reads one byte per weight and decodes it through a table of
  binary32 words. It is exact and it is the reference this module is checked against.
* ``cpu_block_gemv_prepacked`` (v2) removed the table with one ``uint32`` per weight.
  That trades 157 MiB of stored codes for 629 MiB of prepacked words, which made the
  resident form the problem instead of the arithmetic.
* this module keeps the table-free inner loop and stores **two bytes** per weight: the
  four fields the reduction reads (significand, sign, power, nominal exponent) fit a
  ``uint16`` exactly, and the reduction itself is integer shifts rather than float
  multiplies.

What is preserved, and why it is bit-exact rather than approximately equal:

* the operands stay byte-coded. Nothing is dequantized whole, requantized, or routed
  through BF16; a signed-zero code keeps its lack of a nominal exponent.
* the 32 products of one K block are cut into two 16-term halves, each truncated
  toward zero onto ``2 ** (anchor - 13)`` with the anchor of its own half, and the
  second half's window is ``max(its anchor, floor(log2|first half sum|))`` so the
  running sum is a peer of the second half rather than a prefix.
* the block result is renormalised once more onto ``2 ** (floor(log2|total|) - 13)``.
* the activation scale is applied first and the weight scale second, both in binary32,
  and the K blocks are accumulated one at a time in increasing K order.

Every intermediate of that reduction is exactly representable: a product of two e4m3
codes has at most eight significant bits, a lane's shift is at most seven, and the
16-term sum stays below ``2**19``. So the integer shifts reproduce the reference's
truncations exactly, and the only roundings left are ones the reference also makes.

The two scale multiplies and the K add stay separate, in that order. They are applied
by bit arithmetic rather than by the FPU, so a host that flushes subnormals answers
the same as one that does not, while a power-of-two multiply with one round-to-nearest-
even is exactly what the reference's own binary32 multiply does.

**Opt-in, and never a default.** The kernel is compiled locally from the C++ source
beside this file into a cache directory the caller names, under this module's own
content-addressed name -- it never reuses a v1 or v2 artifact, because the ABI, the
prepacked layout and the arithmetic are different. Nothing is built, downloaded or
written unless a cache directory is given, either through ``cache_dir=`` or the
``RELIC_CPU_INTEGER_GEMV_CACHE`` environment variable.

Non-finite behaviour. An e4m3 byte is a NaN code exactly when ``(byte & 0x7F) == 0x7F``.
A NaN in the activation row makes every output column NaN; a NaN in a weight column
makes that column NaN. Both are the reference's own propagation.

A raw scale byte is a finite original-format grid value exactly when it is in
``[0, 254]`` -- ``255`` would mean ``2 ** 128``, which no binary32 holds, so it is
refused by name rather than answered with one of two contradicting infinities/NaNs.
An FP32 grid is accepted when every entry is an exact power of two in range.

The weights are the checkpoint's stored bytes. Nothing here rewrites a checkpoint,
expands a weight to BF16/FP32, or quantizes an activation.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

BLOCK = 32
#: Output columns per prepacked group; the vector path advances four at a time.
GROUP = 4
#: ABI of the compiled kernel. Bumped whenever the C signature or the word layout
#: changes, so a stale artifact can never be loaded and called.
ABI_VERSION = 1
CACHE_ENV = "RELIC_CPU_INTEGER_GEMV_CACHE"
COMPILER_ENV = "RELIC_CPU_INTEGER_GEMV_CXX"

_COMPILERS = ("g++", "c++", "clang++")
_BUILD_SIGNATURE = (
    b"cpu_integer_gemv/v1\0"
    b"-O3 -std=c++17 -fPIC -fno-fast-math -ffp-contract=off -fno-math-errno\0"
    b"exact-integer-fixed-point-two-halves-window13-carry-renorm\0"
    b"compact-uint16-sign-sig-power-nominal-group4\0"
    b"neon-vshlq-scalar-oracle\0"
)

#: The scale byte that would mean ``2 ** 128``; refused instead of answered.
_REFUSED_SCALE_BYTE = 255


def source_path() -> Path:
    """The C++ translation unit this module compiles."""
    return Path(__file__).resolve().parent.parent / "csrc" / "cpu_integer_gemv.cpp"


def _library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def cache_directory(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Where artifacts live: the argument, else the environment, else a refused call."""
    configured = cache_dir if cache_dir is not None else os.environ.get(CACHE_ENV)
    if not configured:
        raise RuntimeError(
            "no cache directory for the compiled host CPU integer GEMV; pass cache_dir= "
            f"or set {CACHE_ENV}. Nothing is compiled or written otherwise."
        )
    return Path(configured)


def _extra_flags(extra_flags) -> list[str]:
    if extra_flags is None:
        return []
    if isinstance(extra_flags, str):
        return [extra_flags]
    flags = list(extra_flags)
    for flag in flags:
        if not isinstance(flag, str) or not flag:
            raise ValueError("extra_flags must be non-empty strings")
    return flags


def _fingerprint(extra_flags: list[str]) -> str:
    material = source_path().read_bytes() + _BUILD_SIGNATURE
    material += b"\0".join(flag.encode("utf-8") for flag in extra_flags)
    return hashlib.sha256(material).hexdigest()[:16]


def compiled_library_path(
    cache_dir: str | os.PathLike[str] | None = None,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> Path:
    """Where the artifact for the current source and flags lives, built or not."""
    flags = _extra_flags(extra_flags)
    digest = _fingerprint(flags)
    return cache_directory(cache_dir) / f"cpu_integer_gemv_abi{ABI_VERSION}_{digest}{_library_suffix()}"


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
        "no C++ compiler found for the compiled host CPU integer GEMV: set "
        f"{COMPILER_ENV} to a compiler path, or install one of {', '.join(_COMPILERS)}"
    )


def _build_command(compiler: str, output: Path, extra_flags: list[str]) -> list[str]:
    """The compile line, with the rounding-critical flags spelled out.

    ``-ffp-contract=off`` is load-bearing: the reference's per-block scale multiplies
    and its K accumulation are separate rounded operations, and a compiler free to
    contract them into an FMA would produce a different kernel. ``-fno-fast-math``
    keeps it from reassociating. The integer reduction itself does not care, but the
    K accumulation is binary32 and does.
    """
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-fno-math-errno",
        "-DNDEBUG",
    ]
    command += list(extra_flags)
    if sys.platform == "darwin":
        command += ["-dynamiclib", "-o", str(output)]
        # A CommandLineTools install without the packaged libc++ needs the SDK's
        # headers; asking the toolchain for the path keeps this from being a guess.
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if sdk:
            command += ["-isysroot", sdk, "-nostdinc++", "-isystem", f"{sdk}/usr/include/c++/v1"]
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

    Idempotent: the artifact name carries the source hash and the build signature, so an
    existing file is the same kernel and is reused. A concurrent build is harmless --
    the temporary object is renamed into place atomically.
    """
    flags = _extra_flags(extra_flags)
    target = compiled_library_path(cache_dir, flags)
    if target.is_file() and not force:
        return target
    source = source_path()
    if not source.is_file():
        raise RuntimeError(f"the compiled host CPU integer GEMV source is missing: {source}")
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
            "the compiled host CPU integer GEMV failed to build:\n"
            f"command: {' '.join(command)}\n"
            f"compiler said:\n{completed.stdout}{completed.stderr}"
        )
    os.replace(temporary, target)
    return target


_LOADED: dict[str, ctypes.CDLL] = {}


def _bind(library: ctypes.CDLL) -> ctypes.CDLL:
    """Bind the C entry points, refusing an artifact whose ABI is not this module's."""
    library.relic_cpu_integer_gemv_abi_version.restype = ctypes.c_int32
    if library.relic_cpu_integer_gemv_abi_version() != ABI_VERSION:
        raise RuntimeError(
            f"the loaded host CPU integer GEMV artifact reports ABI "
            f"{library.relic_cpu_integer_gemv_abi_version()}, expected {ABI_VERSION}"
        )
    library.relic_cpu_integer_gemv_words.restype = ctypes.c_int64
    library.relic_cpu_integer_gemv_words.argtypes = [ctypes.c_int64, ctypes.c_int64]
    library.relic_cpu_integer_gemv_pack.restype = ctypes.c_int32
    library.relic_cpu_integer_gemv_pack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    library.relic_cpu_integer_gemv_fp8.restype = ctypes.c_int32
    library.relic_cpu_integer_gemv_fp8.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    return library


def load(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    compiler: str | None = None,
    build_if_missing: bool = True,
    verbose: bool = False,
    extra_flags: list[str] | tuple[str, ...] | str | None = None,
) -> ctypes.CDLL:
    """The compiled kernel for the current source and flags, building it if needed."""
    flags = _extra_flags(extra_flags)
    target = compiled_library_path(cache_dir, flags)
    if build_if_missing and not target.is_file():
        target = build(cache_dir=cache_dir, compiler=compiler, verbose=verbose, extra_flags=flags)
    if not target.is_file():
        raise RuntimeError(
            f"the compiled host CPU integer GEMV artifact is missing: {target}; "
            "build it with relic_core.kernels.cpu_integer_gemv.build()"
        )
    key = str(target)
    library = _LOADED.get(key)
    if library is None:
        library = _bind(ctypes.CDLL(str(target)))
        _LOADED[key] = library
    return library


def packed_bytes(n: int, k: int) -> dict[str, int]:
    """The prepacked footprint of one ``[n, k]`` weight, in bytes."""
    groups = (int(n) + GROUP - 1) // GROUP
    words = groups * GROUP * int(k)
    return {"words": words * 2, "nan_columns": int(n), "total": words * 2 + int(n)}


# ------------------------------------------------------------------------ validation


def _checked_bytes(tensor, name: str, dtypes):
    """A contiguous CPU byte matrix, from either byte storage or a float8 view."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has dtype {tensor.dtype}, expected one of {dtypes}")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided matrix")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on the CPU; this kernel is the host path")
    if tensor.requires_grad:
        raise ValueError("byte-coded integer GEMV is an inference operator, not an autograd operation")
    return tensor.view(torch.uint8)


def _code_dtypes():
    import torch

    allowed = [torch.uint8, torch.int8]
    floating = getattr(torch, "float8_e4m3fn", None)
    if floating is not None:
        allowed.append(floating)
    return tuple(allowed)


def _scale_dtypes():
    import torch

    allowed = [torch.uint8, torch.int8]
    floating = getattr(torch, "float8_e8m0fnu", None)
    if floating is not None:
        allowed.append(floating)
    return tuple(allowed)


_LOWEST_GRID_BITS = 0x00400000  # 2**-127, the one encode that is not a plain exponent


def _fp32_grid_to_bytes(tensor, name: str):
    """Exact powers of two as the E8M0 bytes that mean them."""
    import torch

    if tensor.layout != torch.strided or not tensor.is_contiguous() or tensor.ndim != 2:
        raise ValueError(f"{name} must be a contiguous strided matrix")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on the CPU; this kernel is the host path")
    bits = tensor.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    lowest = bits == _LOWEST_GRID_BITS
    normal = (bits >= 0) & (mantissa == 0) & (exponent >= 1) & (exponent <= 254)
    valid = lowest | normal
    if not bool(valid.all()):
        raise ValueError(
            f"{name} is not an exact original E8M0 grid: {int((~valid).sum())} of "
            f"{tensor.numel()} entries are not a power of two in [2**-127, 2**127]"
        )
    return exponent.to(torch.uint8)


def _refuse_non_finite_scales(raw, name: str) -> None:
    """Reject the one raw scale byte that is not a finite original-format grid value."""
    offending = int((raw == _REFUSED_SCALE_BYTE).sum())
    if offending:
        raise ValueError(
            f"{name} contains {offending} raw byte(s) 0xFF, which is not a finite E8M0 grid "
            "value (it would mean 2**128); this operator implements the finite "
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


def _source_version(tensor) -> int | None:
    """A tensor's version counter, or ``None`` when it cannot report one (inference mode)."""
    try:
        return int(tensor._version)
    except (AttributeError, RuntimeError):
        return None


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


# ------------------------------------------------------------------- prepacked form


class PrepackedIntegerWeight:
    """A weight matrix prepared for :func:`fp8_gemv_integer`.

    Opaque by intent: the two buffers are the kernel's input layout and are exposed for
    inspection and accounting, not for editing. The source tensors are kept as
    provenance so a caller that edits a weight in place learns that the prepacked form
    is stale instead of silently getting the old numbers.
    """

    __slots__ = (
        "n",
        "k",
        "words",
        "nan_columns",
        "bytes",
        "_source_codes",
        "_source_scales",
        "_source_versions",
        "_fingerprint",
    )

    def __init__(self, n, k, words, nan_columns, source_codes, source_scales, source_versions,
                 fingerprint):
        self.n = n
        self.k = k
        self.words = words
        self.nan_columns = nan_columns
        self._source_codes = source_codes
        self._source_scales = source_scales
        self._source_versions = source_versions
        self._fingerprint = fingerprint
        self.bytes = packed_bytes(n, k)

    @property
    def nan_free_columns(self) -> int:
        """How many columns are free of NaN codes (the rest answer NaN)."""
        return int(self.n) - int(self.nan_columns.sum())

    @property
    def fingerprint(self) -> str:
        """The source/flags fingerprint this weight was prepared under."""
        return self._fingerprint

    def source_versions(self) -> dict[str, int | None]:
        """The recorded version counters of the source tensors (``None`` if untracked)."""
        return {"codes": self._source_versions[0], "scales": self._source_versions[1]}

    def verify_sources(self) -> None:
        """Raise if a source tensor was mutated after this weight was prepacked."""
        for name, tensor, recorded in (
            ("codes", self._source_codes, self._source_versions[0]),
            ("scales", self._source_scales, self._source_versions[1]),
        ):
            if tensor is None:
                continue
            current = _source_version(tensor)
            if recorded is None or current is None:
                raise RuntimeError(
                    f"the prepacked weight's source {name} tensor does not track a version "
                    "counter (an inference tensor), so its immutability cannot be verified; "
                    "the prepacked words remain the data the kernel reads"
                )
            if current != recorded:
                raise RuntimeError(
                    f"the prepacked weight's source {name} tensor was modified after it was "
                    f"prepared (version {recorded} -> {current}); prepack it again rather than "
                    "reusing a stale derived form"
                )

    def __repr__(self) -> str:
        return (
            f"PrepackedIntegerWeight(n={self.n}, k={self.k}, "
            f"bytes={self.bytes['total']}, nan_free_columns={self.nan_free_columns}, "
            f"fingerprint={self._fingerprint!r})"
        )


def prepack_fp8_weight(
    b_codes,
    b_scale,
    *,
    block_rows: int = BLOCK,
    threads: int = 1,
    cache_dir: str | os.PathLike[str] | None = None,
    compiler: str | None = None,
    extra_flags=None,
) -> PrepackedIntegerWeight:
    """Prepare a ``[N, K]`` e4m3 weight for :func:`fp8_gemv_integer`.

    The scale grid is validated here and then not needed again: the prepacked words
    carry the weight codes, and the caller passes the scale grid to the GEMV call (its
    rows are indexed by output column, which the prepack does not need to know).

    The prepacked form is a copy of the stored bytes' *encoding*, not of the weights
    themselves: nothing is expanded to BF16/FP32 and the checkpoint is untouched. For
    the released Engram ``wkv`` -- ``[25600, 6144]`` -- the prepack is 315 MB against
    the 157 MB of stored codes, so the accounting is exposed on the object rather than
    left implicit.
    """
    import torch

    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be an integer >= 1")
    codes = _checked_bytes(b_codes, "weight codes", _code_dtypes())
    scales = _checked_scales(b_scale, "weight scales")
    n, k = codes.shape
    if min(n, k) <= 0 or k % BLOCK:
        raise ValueError("a positive [N, K] weight needs K divisible by 32")
    if tuple(scales.shape) != (_scale_rows(n, block_rows), k // BLOCK):
        raise ValueError(
            f"weight scales must have shape {(_scale_rows(n, block_rows), k // BLOCK)} "
            f"for block_rows={block_rows}"
        )

    library = load(cache_dir=cache_dir, compiler=compiler, extra_flags=extra_flags)
    groups = (n + GROUP - 1) // GROUP
    words = torch.empty(groups * GROUP * k, dtype=torch.uint16)
    nan_columns = torch.zeros(n, dtype=torch.uint8)
    status = library.relic_cpu_integer_gemv_pack(
        codes.data_ptr(),
        n,
        k,
        words.data_ptr(),
        nan_columns.data_ptr(),
        threads,
    )
    if status != 0:
        raise RuntimeError(
            f"the compiled host CPU integer GEMV refused the prepack with status {status} "
            "(1 = bad arguments, 2 = out of memory)"
        )
    return PrepackedIntegerWeight(
        n,
        k,
        words,
        nan_columns,
        b_codes,
        b_scale,
        (_source_version(b_codes), _source_version(b_scale)),
        _fingerprint(_extra_flags(extra_flags)),
    )


def _checked_weight(weight) -> PrepackedIntegerWeight:
    if not isinstance(weight, PrepackedIntegerWeight):
        raise TypeError("weight must be a PrepackedIntegerWeight from prepack_fp8_weight")
    if not isinstance(weight.words, __import__("torch").Tensor):
        raise TypeError("the prepacked words are not a tensor")
    return weight


def fp8_gemv_integer(
    a_codes,
    a_scale,
    weight: PrepackedIntegerWeight,
    b_scale,
    *,
    block_rows: int = BLOCK,
    threads: int = 1,
    out=None,
    cache_dir: str | os.PathLike[str] | None = None,
    compiler: str | None = None,
    mode: str = "auto",
    extra_flags=None,
):
    """Exact ``[1, K] x [N, K]`` block-32 FP8 GEMV over a prepacked weight.

    ``a_codes`` is the activation row as e4m3 bytes (or ``float8_e4m3fn``) and
    ``a_scale`` its E8M0 grid, one exponent per 32-wide K block; ``b_scale`` is the
    weight's grid, ``[N // 32, K // 32]`` for the square grid a dense FP8 weight is
    stored with (``block_rows=32``) or ``[N, K // 32]`` for a per-row grid. Returns
    binary32 accumulators of shape ``[1, N]``; the caller owns the final narrowing,
    exactly as it does around the Torch oracle.

    ``threads`` splits the output groups and ``mode`` picks the vector path (``"auto"``,
    the default where the target has NEON) or the scalar path (``"scalar"``), which
    exists as the independent oracle the vector path is checked against. Neither
    changes the result: each column's K accumulation is sequential and identical
    whatever the thread count.
    """
    import torch

    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be an integer >= 1")
    if mode not in ("auto", "scalar"):
        raise ValueError("mode must be 'auto' or 'scalar'")
    codes = _checked_bytes(a_codes, "activation codes", _code_dtypes())
    activation_scales = _checked_scales(a_scale, "activation scales")
    weight_scales = _checked_scales(b_scale, "weight scales")
    prepared = _checked_weight(weight)

    m, k = codes.shape
    if m != 1:
        raise ValueError(
            f"this is the M = 1 GEMV; the activation codes have {m} rows. Use "
            "relic_core.kernels.block_gemm.fp8_block_gemm for a batched prompt"
        )
    n = prepared.n
    if k % BLOCK or prepared.k != k or min(n, k) <= 0:
        raise ValueError(
            "positive [1,K] and [N,K] operands need the same K, divisible by 32; the "
            f"prepacked weight is [N={prepared.n}, K={prepared.k}]"
        )
    if tuple(activation_scales.shape) != (1, k // BLOCK):
        raise ValueError(f"activation scales must have shape {(1, k // BLOCK)}")
    if tuple(weight_scales.shape) != (_scale_rows(n, block_rows), k // BLOCK):
        raise ValueError(
            f"weight scales must have shape {(_scale_rows(n, block_rows), k // BLOCK)} "
            f"for block_rows={block_rows}"
        )
    if not isinstance(prepared.words, torch.Tensor) or prepared.words.dtype != torch.uint16:
        raise TypeError("the prepacked words must be a uint16 tensor")
    total = prepared.words.numel()
    if total % (GROUP * k):
        raise ValueError("the prepacked word buffer does not match its own geometry")

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
            ("weight scales", weight_scales),
            ("prepacked words", prepared.words),
            ("prepacked NaN flags", prepared.nan_columns),
        ):
            if torch._C._overlaps(out, tensor):
                raise ValueError(f"out must not overlap {name}")

    library = load(cache_dir=cache_dir, compiler=compiler, extra_flags=extra_flags)
    threads_used = ctypes.c_int32(0)
    status = library.relic_cpu_integer_gemv_fp8(
        codes.data_ptr(),
        activation_scales.data_ptr(),
        prepared.words.data_ptr(),
        prepared.nan_columns.data_ptr(),
        weight_scales.data_ptr(),
        out.data_ptr(),
        n,
        k,
        block_rows,
        threads,
        1 if mode == "scalar" else 0,
        ctypes.byref(threads_used),
    )
    if status != 0:
        raise RuntimeError(
            f"the compiled host CPU integer GEMV refused the call with status {status} "
            "(1 = bad arguments, 2 = out of memory)"
        )
    # The raw pointer writes bypass the autograd graph; nothing here is differentiable,
    # so the return is the caller's own storage, filled in place.
    return out