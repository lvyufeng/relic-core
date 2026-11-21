"""Acceptance tests for the pooled raw-byte SIMD host CPU FP8 GEMV.

Two claims are made here, and they are checked differently.

**The arithmetic did not change.** The blocked reduction in
``csrc/cpu_raw_simd_pool_gemv.cpp`` is copied verbatim from the frozen
``csrc/cpu_raw_simd_gemv.cpp``, so the first test reads both files and requires
the copied region to be a byte-identical substring of the frozen source, with the
frozen file itself pinned by SHA-256. The numerical tests then require
``torch.equal`` on FP32 bit patterns against both the frozen kernel and the
accepted Torch oracle, never a tolerance.

**Only the workers changed.** The pool keeps one set of threads per process and a
call that asks for a different thread count resizes it while it is idle. So the
same call is run at several counts -- including back-and-forth, so the resize is
exercised in both directions -- and every result must be the same bits; concurrent
callers must serialize and still agree; and a call must not retain any pointer the
previous one handed it.

What these tests deliberately do not claim: a speedup. Timing is the caller's
measurement on its own host (and the target host for the Engram projection is not
this one). The performance-shaped claim is structural and visible in the source:
threads are created when the geometry changes, not per call.

Nothing here needs a GPU, a checkpoint or a network. A C++ compiler has to be
present; without one the module skips with the reason, and a skip is not a pass.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import cpu_block_gemv as scalar  # noqa: E402
from relic_core.kernels import cpu_raw_simd_gemv as frozen  # noqa: E402
from relic_core.kernels import cpu_raw_simd_pool_gemv as pooled  # noqa: E402
from relic_core.kernels.block_gemm import fp8_block_gemm  # noqa: E402

BLOCK = 32
FORCE_SCALAR = ["-DRELIC_RAW_SIMD_FORCE_SCALAR"]

#: The translation unit this kernel's arithmetic is copied from, and its bytes.
FROZEN_SHA256 = "64534aa6ad55a90f80d6711640c5c27bf66f60882f91d4f44e80dd455992111b"
BEGIN_MARKER = "// RELIC_RAW_SIMD_POOL_MATH_BEGIN"
END_MARKER = "// RELIC_RAW_SIMD_POOL_MATH_END"


def _csrc(name: str) -> Path:
    return Path(pooled.__file__).resolve().parents[1] / "csrc" / name


def _math_region(text: str) -> str:
    """The text between this kernel's markers, without the marker lines."""
    start = text.index(BEGIN_MARKER)
    start = text.index("\n", start) + 1
    return text[start:text.index(END_MARKER)]


def _inside_missing_libcxx(compiler_env: str) -> bool:
    """Whether this host's default C++ driver cannot find the standard library headers."""
    compiler = os.environ.get(compiler_env) or "c++"
    driver = shutil.which(compiler.split()[0])
    if driver is None:
        return False
    probe = subprocess.run([driver, "-x", "c++", "-E", "-"], input="#include <algorithm>\n",
                           capture_output=True, text=True, check=False)
    return probe.returncode != 0


@pytest.fixture(scope="session", autouse=True)
def _reference_compiler(tmp_path_factory):
    """Give this checkout's macOS toolchain a driver that can see its own libc++.

    On this checkout's macOS the CommandLineTools install ships an empty libc++ directory,
    so ``#include <algorithm>`` fails while the SDK still holds the headers. The workaround
    is the one the frozen kernel's tests already use: a one-line wrapper named by each
    module's own ``COMPILER_ENV`` hook, so nothing in the tree changes and a host with a
    complete toolchain never creates it. This is a *test* shim, not a global environment
    change: the variables are set for this module's compiler lookups and restored after.
    """
    modules = (pooled, frozen, scalar)
    if not _inside_missing_libcxx(pooled.COMPILER_ENV):
        yield
        return
    sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True,
                         check=False).stdout.strip()
    include = f"{sdk}/usr/include/c++/v1"
    if not sdk or not Path(include).is_dir():
        pytest.skip("the compiler cannot see a C++ standard library on this host")
    wrapper = tmp_path_factory.mktemp("pool_simd_cc") / "cxx"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'exec "{shutil.which("c++")}" -isysroot "{sdk}" -nostdinc++ -isystem "{include}" "$@"\n'
    )
    wrapper.chmod(0o755)
    previous = {module: os.environ.get(module.COMPILER_ENV) for module in modules}
    for module in modules:
        os.environ[module.COMPILER_ENV] = str(wrapper)
    try:
        yield
    finally:
        for module, value in previous.items():
            if value is None:
                os.environ.pop(module.COMPILER_ENV, None)
            else:
                os.environ[module.COMPILER_ENV] = value


@pytest.fixture(scope="session")
def cache():
    """A session-private build directory, or a skip that names the toolchain problem."""
    directory = tempfile.mkdtemp(prefix="relic-cpu-raw-simd-pool-gemv-")
    try:
        pooled.build(cache_dir=directory)
        frozen.build(cache_dir=directory)
    except RuntimeError as error:
        pytest.skip(f"the compiled host CPU GEMV cannot be built here: {error}")
    return directory


@pytest.fixture(scope="session")
def scalar_cache(cache):
    """The frozen source built with the portable scalar path forced."""
    try:
        frozen.build(cache_dir=cache, extra_flags=FORCE_SCALAR)
    except RuntimeError as error:
        pytest.skip(f"the forced-scalar build of the frozen source failed: {error}")
    return cache


def finite_codes(shape, seed, low=0, high=256):
    """Random e4m3 bytes with the two NaN encodings removed.

    A NaN is a separate contract from the arithmetic, so the numerical tests stay
    on finite codes and the NaN tests construct their case explicitly.
    """
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(low, high, shape, generator=generator, dtype=torch.int32)
    values = torch.where((values & 0x7F) == 0x7F, values - 1, values)
    return values.to(torch.uint8)


def grid_bytes(rows, columns, seed, low=100, high=156):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(low, high, (rows, columns), generator=generator, dtype=torch.int32)
    return values.to(torch.uint8)


def run(cache, a_codes, a_scale, b_codes, b_scale, *, block_rows=BLOCK, threads=1, **kwargs):
    return pooled.fp8_raw_simd_pool_gemv(
        a_codes, a_scale, b_codes, b_scale,
        block_rows=block_rows, threads=threads, cache_dir=cache, **kwargs,
    )


def bits_equal(left, right):
    """FP32 bit-pattern equality, with NaN classified rather than compared by bits."""
    finite = torch.isfinite(left) & torch.isfinite(right)
    same = torch.equal(left[finite].view(torch.int32), right[finite].view(torch.int32))
    return same and torch.equal(torch.isnan(left), torch.isnan(right))


def reference(a_codes, a_scale, b_codes, b_scale, block_rows=BLOCK):
    return fp8_block_gemm(a_codes, a_scale, b_codes, b_scale, block_rows=block_rows, backend="torch")


# ---------------------------------------------------------------------------------------------
# The arithmetic is copied, and the copy is pinned
# ---------------------------------------------------------------------------------------------


def test_the_frozen_kernel_is_byte_unchanged():
    """The source this kernel copies must still be the reviewed one."""
    text = _csrc("cpu_raw_simd_gemv.cpp").read_text()
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == FROZEN_SHA256


def test_the_shared_math_is_a_verbatim_substring_of_the_frozen_source():
    """The block rule, the tables and the range loop are the frozen text, not a rewrite."""
    frozen_text = _csrc("cpu_raw_simd_gemv.cpp").read_text()
    region = _math_region(_csrc("cpu_raw_simd_pool_gemv.cpp").read_text())
    assert len(region) > 10_000, "the marked region is not the whole shared implementation"
    assert region in frozen_text, "the pooled kernel's math region is not the frozen text"
    for name in ("BlockFragmentNeon", "BlockFragmentScalar", "RunRange", "CutPowers",
                 "HasNanCode", "FloorExponent", "kWindow", "struct Plan", "Validate"):
        assert name in region, f"the shared region does not carry {name}"


def test_no_scheduling_code_leaked_into_the_shared_math():
    """The copy is the arithmetic only: threads and pool state stay outside it."""
    region = _math_region(_csrc("cpu_raw_simd_pool_gemv.cpp").read_text())
    for token in ("Pool", "condition_variable", "std::thread", "generation"):
        assert token not in region, f"{token!r} appears in the copied arithmetic region"


def test_this_module_ships_its_own_source_and_cache_variable():
    assert pooled.CACHE_ENV != frozen.CACHE_ENV
    assert pooled.source_path() != frozen.source_path()
    assert pooled.source_path().name == "cpu_raw_simd_pool_gemv.cpp"


# ---------------------------------------------------------------------------------------------
# The arithmetic, against the two existing implementations
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("block_rows", [BLOCK, 1])
@pytest.mark.parametrize("shape", [(64, 64), (256, 128), (1024, 256)])
def test_matches_the_frozen_kernel_and_the_torch_oracle(cache, block_rows, shape):
    n, k = shape
    a_codes = finite_codes((1, k), seed=k)
    b_codes = finite_codes((n, k), seed=n)
    a_scale = grid_bytes(1, k // BLOCK, seed=7)
    b_scale = grid_bytes((n + block_rows - 1) // block_rows, k // BLOCK, seed=11)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=block_rows)
    expected = reference(a_codes, a_scale, b_codes, b_scale, block_rows=block_rows)
    old = frozen.fp8_raw_simd_gemv(
        a_codes, a_scale, b_codes, b_scale,
        block_rows=block_rows, cache_dir=cache,
    )
    assert actual.shape == (1, n)
    assert bits_equal(actual, expected)
    assert bits_equal(actual, old)


def test_every_code_product_is_reproduced(cache):
    """All 256 x 256 activation-code / weight-code products, one activation per call."""
    codes = list(range(256))
    k = 64
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((len(codes) // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    b_codes = torch.zeros((len(codes), k), dtype=torch.uint8)
    for row, code in enumerate(codes):
        b_codes[row, 0] = code
        b_codes[row, 1] = code
    for activation in codes:
        a_codes = torch.zeros((1, k), dtype=torch.uint8)
        a_codes[0, 0] = activation
        a_codes[0, 1] = activation
        actual = run(cache, a_codes, a_scale, b_codes, b_scale)
        assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale)), \
            f"activation code {activation} diverged"


def test_cancellation_and_near_cancellation_are_exact(cache):
    """A block whose terms cancel to a tiny or zero total exercises both windows."""
    k = 128
    a_codes = finite_codes((1, k), seed=3)
    b_codes = finite_codes((512, k), seed=5)
    b_codes[0] = a_codes[0]
    b_codes[1] = a_codes[0] ^ 0x80  # sign-flipped codes
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((512 // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale))
    assert actual[0, 1] == -actual[0, 0]


@pytest.mark.parametrize("byte", [0, 1, 126, 127, 128, 253, 254])
def test_scale_endpoints_are_read_as_powers_of_two(cache, byte):
    """Byte 0 is the subnormal 2**-127 and byte 254 the largest finite grid value."""
    k, n = 64, 128
    a_codes = finite_codes((1, k), seed=byte + 1)
    b_codes = finite_codes((n, k), seed=byte + 2)
    a_scale = torch.full((1, k // BLOCK), byte, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), byte, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale))


def test_an_all_zero_activation_row_produces_zero_columns(cache):
    k, n = 64, 64
    a_codes = torch.zeros((1, k), dtype=torch.uint8)
    b_codes = finite_codes((n, k), seed=22)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert torch.equal(actual, torch.zeros_like(actual))
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale))


def test_a_nan_weight_code_nan_classifies_only_its_column(cache):
    k, n = 64, 96
    a_codes = finite_codes((1, k), seed=31)
    b_codes = finite_codes((n, k), seed=32)
    b_codes[5, 0] = 0x7F
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert torch.isnan(actual[0, 5])
    assert torch.isfinite(actual[0, :5]).all()
    assert torch.isfinite(actual[0, 6:]).all()
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale))


def test_a_nan_activation_code_nan_classifies_every_column(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=41)
    a_codes[0, 7] = 0x7F
    b_codes = finite_codes((n, k), seed=42)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale)
    assert torch.isnan(actual).all()


def test_the_released_engram_geometry_matches_the_frozen_kernel(cache):
    """N = 25600, K = 6144: the host Engram projection this exists for."""
    n, k = 25600, 6144
    generator = torch.Generator().manual_seed(1234)
    a_codes = torch.randint(0, 256, (1, k), generator=generator, dtype=torch.uint8)
    a_codes = torch.where((a_codes & 0x7F) == 0x7F, a_codes - 1, a_codes).to(torch.uint8)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    b_codes = torch.zeros((n, k), dtype=torch.uint8)
    # A cheap but non-trivial pattern: every column's first block is drawn from
    # the activation row itself, so the products and both truncation windows are
    # exercised at the real width without allocating random bytes twice.
    b_codes[:, :k] = a_codes
    b_codes[:, k // 2:] = a_codes[:, : k // 2]

    expected = frozen.fp8_raw_simd_gemv(a_codes, a_scale, b_codes, b_scale,
                                        threads=96, cache_dir=cache)
    actual = run(cache, a_codes, a_scale, b_codes, b_scale, threads=96)
    assert bits_equal(actual, expected)
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale))


# ---------------------------------------------------------------------------------------------
# The pool: resizing, reuse and serialization
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [1, 2, 3, 7, 64, 96, 256])
def test_the_result_does_not_depend_on_the_thread_count(cache, threads):
    """Every count resizes the same pool; the bits may not move."""
    k, n = 256, 1000
    a_codes = finite_codes((1, k), seed=51)
    b_codes = finite_codes((n, k), seed=52)
    a_scale = grid_bytes(1, k // BLOCK, seed=53)
    b_scale = grid_bytes(n, k // BLOCK, seed=54)  # per-row grid: N needs no multiple of 32

    baseline = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=1, threads=1)
    actual = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=1, threads=threads)
    assert bits_equal(actual, baseline), f"threads={threads} changed the result"


def test_resizing_back_and_forth_keeps_the_bits_and_the_pool_usable(cache):
    """The pool is joined and respawned on each change; nothing may go stale."""
    k, n = 128, 512
    a_codes = finite_codes((1, k), seed=61)
    b_codes = finite_codes((n, k), seed=62)
    a_scale = grid_bytes(1, k // BLOCK, seed=63)
    b_scale = grid_bytes(n // BLOCK, k // BLOCK, seed=64)
    expected = reference(a_codes, a_scale, b_codes, b_scale)
    for count in (96, 1, 96, 2, 96, 7, 96):
        actual = run(cache, a_codes, a_scale, b_codes, b_scale, threads=count)
        assert bits_equal(actual, expected), f"threads={count} diverged after a resize"


def test_a_repeated_call_reuses_the_pool_and_retains_nothing(cache):
    """Two calls in a row, then a call with different operands: no stale pointer."""
    k, n = 96, 128
    a_codes = finite_codes((1, k), seed=71)
    b_codes = finite_codes((n, k), seed=72)
    a_scale = grid_bytes(1, k // BLOCK, seed=73)
    b_scale = grid_bytes(n // BLOCK, k // BLOCK, seed=74)

    first = run(cache, a_codes, a_scale, b_codes, b_scale, threads=8)
    second = run(cache, a_codes, a_scale, b_codes, b_scale, threads=8)
    assert bits_equal(first, second)

    # Overwrite the operands the first call was handed, then compute on new ones:
    # a retained pointer would answer with the old bytes or crash.
    a_codes.fill_(0x38)
    b_codes.fill_(0x38)
    other = run(cache, a_codes, a_scale, b_codes, b_scale, threads=8)
    assert bits_equal(other, reference(a_codes, a_scale, b_codes, b_scale))
    assert not bits_equal(other, first)


def test_concurrent_callers_serialize_and_all_agree(cache):
    """ctypes releases the GIL, so these really race for the pool's call mutex."""
    k, n = 128, 640
    expected = None
    operands = []
    for seed in range(4):
        a_codes = finite_codes((1, k), seed=80 + seed)
        b_codes = finite_codes((n, k), seed=90 + seed)
        a_scale = grid_bytes(1, k // BLOCK, seed=100 + seed)
        b_scale = grid_bytes(n // BLOCK, k // BLOCK, seed=110 + seed)
        operands.append((a_codes, a_scale, b_codes, b_scale))
    expected = [reference(*item) for item in operands]

    def work(index):
        a_codes, a_scale, b_codes, b_scale = operands[index % len(operands)]
        return index % len(operands), run(
            cache, a_codes, a_scale, b_codes, b_scale,
            threads=(8, 3, 8, 1)[index % 4],
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, range(16)))
    for index, result in results:
        assert bits_equal(result, expected[index]), f"concurrent call {index} diverged"


def test_out_storage_is_used_in_place(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=81)
    b_codes = finite_codes((n, k), seed=82)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    out = torch.empty((1, n), dtype=torch.float32)
    returned = run(cache, a_codes, a_scale, b_codes, b_scale, out=out)
    assert returned is out
    assert bits_equal(out, reference(a_codes, a_scale, b_codes, b_scale))


def test_a_short_final_chunk_and_a_non_multiple_of_32_output_are_exact(cache):
    """N = 1001 with a per-row grid: the last column, and the ragged thread chunk."""
    k, n = 96, 1001
    a_codes = finite_codes((1, k), seed=61)
    b_codes = finite_codes((n, k), seed=62)
    a_scale = grid_bytes(1, k // BLOCK, seed=63)
    b_scale = grid_bytes(n, k // BLOCK, seed=64)

    actual = run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=1, threads=64)
    assert bits_equal(actual, reference(a_codes, a_scale, b_codes, b_scale, block_rows=1))


def test_the_forced_scalar_build_of_the_frozen_source_agrees_bit_for_bit(scalar_cache):
    """The fallback a non-NEON target compiles is the same kernel, not a variant."""
    k, n = 256, 512
    a_codes = finite_codes((1, k), seed=71)
    b_codes = finite_codes((n, k), seed=72)
    a_scale = grid_bytes(1, k // BLOCK, seed=73)
    b_scale = grid_bytes(n // BLOCK, k // BLOCK, seed=74)

    vector = run(scalar_cache, a_codes, a_scale, b_codes, b_scale, threads=8)
    frozen_scalar = frozen.fp8_raw_simd_gemv(
        a_codes, a_scale, b_codes, b_scale, threads=8,
        cache_dir=scalar_cache, extra_flags=FORCE_SCALAR,
    )
    assert bits_equal(vector, frozen_scalar)
    assert pooled.vector_width(cache_dir=scalar_cache) in (1, 4)


def test_the_reported_vector_width_matches_the_build(cache):
    width = pooled.vector_width(cache_dir=cache)
    assert width in (1, 4)
    if hasattr(torch.backends, "cpu") and "aarch64" in (torch.backends.cpu.get_cpu_capability() or "").lower():
        assert width == 4
    assert pooled.fp8_raw_simd_pool_gemv_available(cache_dir=cache)


# ---------------------------------------------------------------------------------------------
# Fork safety and the public contract
# ---------------------------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="this platform has no fork()")
def test_a_forked_child_gets_its_own_pool_and_the_same_bits(cache):
    """The child inherits bookkeeping but no threads; the at-fork hook rebuilds them.

    The operands, the expected result and the output buffer are all allocated
    before the fork, so the child does no tensor work -- it calls the pool and
    compares bytes, which is exactly the boundary under test.
    """
    k, n = 128, 256
    a_codes = finite_codes((1, k), seed=201)
    b_codes = finite_codes((n, k), seed=202)
    a_scale = grid_bytes(1, k // BLOCK, seed=203)
    b_scale = grid_bytes(n // BLOCK, k // BLOCK, seed=204)
    # Warm the pool in the parent so the child really does inherit live state.
    expected = run(cache, a_codes, a_scale, b_codes, b_scale, threads=8)
    child_out = torch.empty_like(expected)
    library = pooled.load(cache_dir=cache)
    pointers = (a_codes.data_ptr(), a_scale.data_ptr(), b_codes.data_ptr(),
                b_scale.data_ptr(), child_out.data_ptr(), n, k, BLOCK, 8)
    expected_bytes = ctypes.string_at(expected.data_ptr(), n * 4)
    used = ctypes.c_int32(0)

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(read_fd)
            status = library.relic_cpu_raw_simd_pool_gemv_fp8(*pointers, ctypes.byref(used))
            ok = status == 0 and ctypes.string_at(pointers[4], n * 4) == expected_bytes
            os.write(write_fd, b"OK" if ok else b"MISMATCH")
            os.close(write_fd)
        except BaseException as error:  # noqa: BLE001 - the parent needs the reason
            try:
                os.write(write_fd, f"ERROR {type(error).__name__}: {error}".encode())
            finally:
                os._exit(1)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as handle:
        message = handle.read()
    _, status = os.waitpid(pid, 0)
    assert status == 0, f"the forked child failed: {message.decode(errors='replace')}"
    assert message == b"OK", f"the forked child diverged: {message.decode(errors='replace')}"
    # The parent's pool is unaffected and still produces the same bits.
    assert bits_equal(run(cache, a_codes, a_scale, b_codes, b_scale, threads=8), expected)


def test_reset_after_fork_is_safe_when_idle(cache):
    """The hook the fork path uses must not disturb a pool that is not in a call."""
    k, n = 64, 128
    a_codes = finite_codes((1, k), seed=211)
    b_codes = finite_codes((n, k), seed=212)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    expected = run(cache, a_codes, a_scale, b_codes, b_scale, threads=4)

    pooled.reset_after_fork()
    assert bits_equal(run(cache, a_codes, a_scale, b_codes, b_scale, threads=4), expected)


def test_a_non_square_grid_needs_block_rows_one(cache):
    k, n = 64, 48
    a_codes = finite_codes((1, k), seed=91)
    b_codes = finite_codes((n, k), seed=92)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    per_row = torch.full((n, k // BLOCK), 127, dtype=torch.uint8)

    with pytest.raises(ValueError, match="weight scales must have shape"):
        run(cache, a_codes, a_scale, b_codes, per_row, block_rows=BLOCK)
    assert run(cache, a_codes, a_scale, b_codes, per_row, block_rows=1).shape == (1, n)


def test_bad_arguments_are_refused(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=101)
    b_codes = finite_codes((n, k), seed=102)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)

    with pytest.raises(ValueError, match="block_rows must be 1 or 32"):
        run(cache, a_codes, a_scale, b_codes, b_scale, block_rows=16)
    with pytest.raises(ValueError, match="threads must be an integer"):
        run(cache, a_codes, a_scale, b_codes, b_scale, threads=0)
    with pytest.raises(ValueError, match="M = 1"):
        run(cache, torch.cat([a_codes, a_codes]), a_scale, b_codes, b_scale)
    with pytest.raises(ValueError, match="contiguous strided matrix"):
        run(cache, a_codes, a_scale, b_codes[:, :-1], b_scale)
    with pytest.raises(ValueError, match="same K, divisible by 32"):
        run(cache, a_codes, a_scale, b_codes[:, :-BLOCK].contiguous(), b_scale)


def test_scale_byte_255_is_refused_and_a_non_power_of_two_grid_too(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=111)
    b_codes = finite_codes((n, k), seed=112)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    bad = b_scale.clone()
    bad[0, 0] = 255
    with pytest.raises(ValueError, match="not a finite E8M0 grid value"):
        run(cache, a_codes, a_scale, b_codes, bad)

    fp32 = torch.full((n // BLOCK, k // BLOCK), 3.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="not an exact original E8M0 grid"):
        run(cache, a_codes, a_scale, b_codes, fp32)


def test_a_non_contiguous_or_wrongly_shaped_out_is_refused(cache):
    k, n = 64, 64
    a_codes = finite_codes((1, k), seed=121)
    b_codes = finite_codes((n, k), seed=122)
    a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // BLOCK, k // BLOCK), 127, dtype=torch.uint8)
    wide = torch.zeros((n, 2 * k), dtype=torch.uint8)
    wide[:, ::2] = b_codes
    with pytest.raises(ValueError, match="contiguous strided matrix"):
        run(cache, a_codes, a_scale, wide[:, ::2], b_scale)

    with pytest.raises(ValueError, match="out must be a contiguous CPU fp32 tensor"):
        run(cache, a_codes, a_scale, b_codes, b_scale, out=torch.zeros((1, n - 1), dtype=torch.float32))
    with pytest.raises(TypeError, match="expected torch.float32"):
        run(cache, a_codes, a_scale, b_codes, b_scale, out=torch.zeros((1, n), dtype=torch.float64))


def test_parent_reset_does_not_abandon_workers_and_child_reinitializes_tables():
    text = _csrc("cpu_raw_simd_pool_gemv.cpp").read_text()
    reset = text[text.index("void ResetPool() {"):text.index("}  // namespace", text.index("void ResetPool() {"))]
    assert "pool->owner_pid == static_cast<int64_t>(::getpid())" in reset
    assert reset.index("return;") < reset.index("current.store(nullptr")
    assert "new (&g_init_once) std::once_flag();" in reset


@pytest.mark.parametrize("flag", ["-Ofast", "-ffast-math", "-funsafe-math-optimizations",
                                  "-fassociative-math", "-ffp-contract=fast"])
def test_exact_pool_refuses_floating_point_relaxation_flags(tmp_path, flag):
    with pytest.raises(ValueError, match="floating-point relaxation"):
        pooled.compiled_library_path(tmp_path, extra_flags=[flag])
