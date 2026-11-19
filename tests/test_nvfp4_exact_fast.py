"""The exact-operand NVFP4 path must be bit-identical to the old decoder path.

Two things are checked, and they are different claims.

*Decoder values.* The new decoders build their operands from register constants; the old ones
loaded a `__constant__` table and called `ldexpf`. The witness entry points expose the new decoders'
float bits, and those bits are compared against the *old* semantics reimplemented on the CPU with
`math.ldexp` / the halved E2M1 table -- all 16 E2M1 codes and all 256 E4M3 bytes, including the two
encodings the format reserves and this path deliberately keeps finite (0x7F, 0xFF are 480) and both
signed zeros.

*MoE results.* A bit-equality test between `nvfp4_moe_rows_forward` and
`nvfp4_moe_rows_forward_exact_fast` for the real per-rank shape and for small and awkward ones,
across valid, duplicated, partly unowned and entirely unowned route sets, adversarial scale bytes,
a global expert offset, and eager plus separately captured graph replays with mutated inputs. The
old entry is unchanged source, so this compares two independently compiled implementations.

Neither part is a speed claim. A tolerance check would not be one either, and is not used: the whole
point of replacing a load with a constant is that the value does not move.
"""

from __future__ import annotations

import math

import pytest
import torch

from relic_core.kernels.cuda_loader import load_cuda_kernel


E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _extension():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ext = load_cuda_kernel()
    required = (
        "nvfp4_moe_rows_forward",
        "nvfp4_moe_rows_forward_exact_fast",
        "nvfp4_exact_decode_e2m1",
        "nvfp4_exact_decode_e4m3",
    )
    missing = [name for name in required if not hasattr(ext, name)]
    if ext is None or missing:
        pytest.skip(f"NVFP4 exact extension is unavailable: {missing}")
    return ext


# ---------------------------------------------------------------------------------------------
# The old decoders, reimplemented on the CPU. These are the specification the new bits must reach.
# ---------------------------------------------------------------------------------------------


def old_e2m1(nibble: int) -> float:
    """`nvfp4_e2m1`: `fp4_lut_x2[code] * 0.5f`, sign applied to the magnitude."""
    magnitude = E2M1_LEVELS[nibble & 0x7]
    return -magnitude if nibble & 0x8 else magnitude


def old_e4m3(byte: int) -> float:
    """`nvfp4_e4m3`: `ldexpf(m/8, -6)` subnormals, `ldexpf(1 + m/8, e - 7)` normals."""
    exponent = (byte >> 3) & 0xF
    mantissa = byte & 0x7
    if exponent == 0:
        value = math.ldexp(mantissa / 8.0, -6)
    else:
        value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return -value if byte & 0x80 else value


def float32_bits(value: float) -> int:
    """The IEEE-754 binary32 bit pattern, as an unsigned integer."""
    return int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item()) & 0xFFFFFFFF


def bits_of(tensor: torch.Tensor) -> torch.Tensor:
    """Float32 bit patterns as unsigned integers, so a `-0.0` and a `+0.0` are not the same value.

    Widened to int64 before masking: a signed int32 view would make every negative float's pattern
    negative too, which compares unequal to the unsigned pattern the CPU oracle produces.
    """
    return tensor.view(torch.int32).to(torch.int64) & 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------
# Decoder witnesses.
# ---------------------------------------------------------------------------------------------


def test_all_sixteen_e2m1_codes_decode_to_the_old_bits():
    ext = _extension()
    codes = torch.arange(16, dtype=torch.int16, device="cuda").to(torch.uint8)
    actual = ext.nvfp4_exact_decode_e2m1(codes)
    expected = [float32_bits(old_e2m1(code)) for code in range(16)]
    assert bits_of(actual).tolist() == expected
    # Named cases, so a failure says which semantic moved rather than only which index.
    assert actual[0].item() == 0.0 and float32_bits(actual[0].item()) == 0x00000000
    assert actual[8].item() == 0.0 and float32_bits(actual[8].item()) == 0x80000000  # -0.0
    assert actual[7].item() == 6.0 and actual[15].item() == -6.0
    assert actual[5].item() == 3.0 and actual[13].item() == -3.0
    assert actual[1].item() == 0.5 and actual[9].item() == -0.5


def test_e2m1_witness_ignores_bits_above_the_nibble():
    ext = _extension()
    codes = torch.tensor([0x11, 0x2F, 0x87, 0xF3], dtype=torch.int16, device="cuda").to(torch.uint8)
    actual = ext.nvfp4_exact_decode_e2m1(codes)
    expected = [old_e2m1(code & 0xF) for code in (0x11, 0x2F, 0x87, 0xF3)]
    assert actual.tolist() == expected


def test_all_256_e4m3_bytes_decode_to_the_old_bits():
    ext = _extension()
    bytes_in = torch.arange(256, dtype=torch.int16, device="cuda").to(torch.uint8)
    actual = ext.nvfp4_exact_decode_e4m3(bytes_in)
    expected = [float32_bits(old_e4m3(byte)) for byte in range(256)]
    assert bits_of(actual).tolist() == expected
    decoded = actual.tolist()
    # The format's reserved encoding, kept as the old path has it: exponent 15 mantissa 7 is 480,
    # and two of the 256 bytes are +-0.0 rather than a value.
    assert decoded[0x7F] == 480.0 and decoded[0xFF] == -480.0
    assert float32_bits(decoded[0x00]) == 0x00000000
    assert float32_bits(decoded[0x80]) == 0x80000000
    # Subnormal range: 0x01 is 2^-9, 0x07 is 7 * 2^-9. Both are fp32-exact, so `==` is the test.
    assert decoded[0x01] == 2.0 ** -9
    assert decoded[0x07] == 7.0 * 2.0 ** -9
    # Largest normal below the reserved encoding, and the smallest normal.
    assert decoded[0x7E] == 448.0 and decoded[0x08] == 2.0 ** -6
    assert all(math.isfinite(value) for value in decoded)


# ---------------------------------------------------------------------------------------------
# The K-row operation against its unchanged oracle.
# ---------------------------------------------------------------------------------------------


def rand_u8(shape, generator, device):
    # int32 then a narrowing cast: `torch.randint` does not accept every integral dtype directly,
    # and the packed payload only cares about the low eight bits.
    return torch.randint(0, 256, shape, dtype=torch.int32, device=device,
                         generator=generator).to(torch.uint8)


def build_case(rows, hidden, inter, experts, topk, seed, swiglu_limit=0.0, start_idx=0,
               scale_bytes=None, routes="valid", device="cuda"):
    """Pack a synthetic NVFP4 expert bank and its activation/route tensors.

    `scale_bytes`, when given, is cycled through every scale tensor so specific E4M3 encodings are
    exercised; otherwise the scales are random bytes. `routes` chooses which route pathologies to
    inject: valid only, with duplicates, with out-of-range ids, or entirely unowned.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(rows, hidden, dtype=torch.float32, device=device, generator=generator)
    w1q = rand_u8((experts, inter, hidden // 2), generator, device)
    w3q = rand_u8((experts, inter, hidden // 2), generator, device)
    w2q = rand_u8((experts, hidden, inter // 2), generator, device)
    if scale_bytes is None:
        w1s = rand_u8((experts, inter, hidden // 16), generator, device)
        w3s = rand_u8((experts, inter, hidden // 16), generator, device)
        w2s = rand_u8((experts, hidden, inter // 16), generator, device)
    else:
        pattern = torch.tensor(scale_bytes, dtype=torch.int16, device=device).to(torch.uint8)
        def cycle(shape):
            count = 1
            for dim in shape:
                count *= dim
            return pattern.repeat((count + pattern.numel() - 1) // pattern.numel())[:count].reshape(shape)
        w1s = cycle((experts, inter, hidden // 16))
        w3s = cycle((experts, inter, hidden // 16))
        w2s = cycle((experts, hidden, inter // 16))
    # Per-expert global scales, the way a ModelOpt checkpoint carries `weight_scale_2`.
    w_scale1 = torch.exp(torch.randn(experts, dtype=torch.float32, device=device,
                                     generator=generator) * 0.25)
    w_scale2 = torch.exp(torch.randn(experts, dtype=torch.float32, device=device,
                                     generator=generator) * 0.25)

    indices = torch.randint(start_idx, start_idx + experts, (rows, topk), dtype=torch.int64,
                            device=device, generator=generator)
    weights = torch.rand(rows, topk, dtype=torch.float32, device=device, generator=generator)
    if routes in {"duplicate", "mixed"} and topk > 1:
        # Repeated experts must accumulate twice rather than once.
        indices[:, 1] = indices[:, 0]
    if routes in {"out_of_range", "mixed"}:
        indices[0, 0] = start_idx - 1 - (topk % 3)
        if topk > 1:
            indices[0, -1] = start_idx + experts + 2
        if rows > 1:
            indices[rows - 1, -1] = start_idx - 7
    if routes == "all_unowned":
        indices.fill_(start_idx - 4)
    return dict(x=x, indices=indices, weights=weights, w1q=w1q, w1s=w1s, w3q=w3q, w3s=w3s,
                w2q=w2q, w2s=w2s, w_scale1=w_scale1, w_scale2=w_scale2,
                experts_start_idx=start_idx, swiglu_limit=swiglu_limit)


def call(ext, name, case):
    return getattr(ext, name)(
        case["x"], case["indices"], case["weights"],
        case["w1q"], case["w1s"], case["w3q"], case["w3s"], case["w2q"], case["w2s"],
        case["experts_start_idx"], case["swiglu_limit"], 1.0, 1.0,
        case["w_scale1"], case["w_scale2"], True,
    )


def assert_bit_identical(actual, expected, label):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual, expected), f"{label}: values differ"
    assert torch.equal(bits_of(actual), bits_of(expected)), f"{label}: bits differ"


REAL_SHAPE = dict(hidden=2560, inter=640, experts=64, topk=10)


@pytest.mark.parametrize("rows", [1, 2, 4, 8])
@pytest.mark.parametrize("routes", ["valid", "duplicate", "out_of_range", "all_unowned", "mixed"])
def test_real_per_rank_shape_matches_old_entry(rows, routes):
    """hidden 2560 / inter 640 / 64 local experts / top 10 is what an 8-way TP rank runs."""
    ext = _extension()
    case = build_case(rows, seed=4100 + rows, routes=routes, **REAL_SHAPE)
    assert_bit_identical(call(ext, "nvfp4_moe_rows_forward_exact_fast", case),
                         call(ext, "nvfp4_moe_rows_forward", case),
                         f"rows={rows} routes={routes}")


@pytest.mark.parametrize("shape", [
    dict(hidden=64, inter=32, experts=8, topk=3),
    dict(hidden=272, inter=48, experts=5, topk=7),   # inter/16 and hidden/16 not round numbers
    dict(hidden=128, inter=16, experts=4, topk=1),   # single route
    dict(hidden=4096, inter=256, experts=16, topk=8),
])
@pytest.mark.parametrize("rows", [1, 3, 4])
def test_other_shapes_match_old_entry(shape, rows):
    ext = _extension()
    case = build_case(rows, seed=7300 + rows + shape["hidden"], routes="mixed", **shape)
    assert_bit_identical(call(ext, "nvfp4_moe_rows_forward_exact_fast", case),
                         call(ext, "nvfp4_moe_rows_forward", case),
                         f"{shape} rows={rows}")


@pytest.mark.parametrize("rows", [1, 4])
def test_global_expert_offset_matches_old_entry(rows):
    """A rank that owns experts 1000..1063 sees global ids; only its slice may be read."""
    ext = _extension()
    case = build_case(rows, seed=9100 + rows, start_idx=1000, routes="mixed", **REAL_SHAPE)
    assert_bit_identical(call(ext, "nvfp4_moe_rows_forward_exact_fast", case),
                         call(ext, "nvfp4_moe_rows_forward", case),
                         f"rows={rows} global offset")


@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0, 1.5])
def test_swiglu_clamping_matches_old_entry(swiglu_limit):
    ext = _extension()
    case = build_case(4, seed=5200, swiglu_limit=swiglu_limit, routes="mixed", **REAL_SHAPE)
    assert_bit_identical(call(ext, "nvfp4_moe_rows_forward_exact_fast", case),
                         call(ext, "nvfp4_moe_rows_forward", case),
                         f"swiglu_limit={swiglu_limit}")


@pytest.mark.parametrize("scale_bytes", [
    [0x7F, 0xFF],                                   # the reserved-as-NaN encodings, finite here
    [0x00, 0x80, 0x01, 0x7F],                       # signed zeros and the subnormal edge
    [0x08, 0x38, 0x3C, 0x7E],                       # 2^-6, 1.0, 2.0, 448
    [0x7F, 0xFF, 0x7E, 0x80, 0x00, 0x37, 0x01, 0x03],
])
def test_adversarial_scale_bytes_match_old_entry(scale_bytes):
    """Every scale the kernel reads comes from these bytes; the reserved ones must not move."""
    ext = _extension()
    for rows in (1, 4):
        case = build_case(rows, seed=6100 + len(scale_bytes) + rows, scale_bytes=scale_bytes,
                          routes="valid", **REAL_SHAPE)
        assert_bit_identical(call(ext, "nvfp4_moe_rows_forward_exact_fast", case),
                             call(ext, "nvfp4_moe_rows_forward", case),
                             f"rows={rows} scale_bytes={scale_bytes}")


def test_every_scale_byte_participates_and_agrees():
    """Cycle all 256 encodings through the scale tensors so no byte pattern is left untested."""
    ext = _extension()
    case = build_case(8, seed=6800, scale_bytes=list(range(256)), routes="mixed", **REAL_SHAPE)
    assert_bit_identical(call(ext, "nvfp4_moe_rows_forward_exact_fast", case),
                         call(ext, "nvfp4_moe_rows_forward", case),
                         "all 256 scale bytes")


def test_captured_graphs_agree_after_input_mutations():
    """Capture each entry separately, then mutate every input in place and replay.

    A decode round runs under a captured graph, where the tensors' addresses are fixed; a decoder
    that read anything outside them, or that depended on a value baked in at capture time, would
    only show up on a replay with new data.
    """
    ext = _extension()
    case = build_case(4, seed=8800, routes="mixed", **REAL_SHAPE)
    arguments = [case[name] for name in (
        "x", "indices", "weights", "w1q", "w1s", "w3q", "w3s", "w2q", "w2s",
    )]
    scalars = [case["experts_start_idx"], case["swiglu_limit"], 1.0, 1.0]
    tails = [case["w_scale1"], case["w_scale2"]]

    def run(name):
        return getattr(ext, name)(*arguments, *scalars, *tails, True)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run("nvfp4_moe_rows_forward")
        run("nvfp4_moe_rows_forward_exact_fast")
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graphs, statics = {}, {}
    for name in ("nvfp4_moe_rows_forward", "nvfp4_moe_rows_forward_exact_fast"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            statics[name] = run(name)
        graphs[name] = graph

    generator = torch.Generator(device="cuda").manual_seed(4200)
    for step in range(4):
        with torch.no_grad():
            case["x"].copy_(torch.randn_like(case["x"], generator=generator))
            case["weights"].copy_(torch.rand(case["weights"].shape, dtype=torch.float32,
                                             device="cuda", generator=generator))
            # Re-draw the packed weights and scales: the nibbles are what the decoder consumes.
            for name in ("w1q", "w3q", "w2q"):
                case[name].copy_(rand_u8(case[name].shape, generator, "cuda"))
            for name in ("w1s", "w3s", "w2s"):
                case[name].copy_(rand_u8(case[name].shape, generator, "cuda"))
        graphs["nvfp4_moe_rows_forward"].replay()
        reference = statics["nvfp4_moe_rows_forward"].clone()
        graphs["nvfp4_moe_rows_forward_exact_fast"].replay()
        assert_bit_identical(statics["nvfp4_moe_rows_forward_exact_fast"], reference,
                             f"captured replay {step}")


def test_eager_and_captured_results_agree_with_each_other():
    """The captured path must not disagree with eager, in either entry."""
    ext = _extension()
    case = build_case(2, seed=9300, routes="mixed", **REAL_SHAPE)
    eager_old = call(ext, "nvfp4_moe_rows_forward", case).clone()
    eager_new = call(ext, "nvfp4_moe_rows_forward_exact_fast", case).clone()
    assert_bit_identical(eager_new, eager_old, "eager before capture")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call(ext, "nvfp4_moe_rows_forward_exact_fast", case)
    graph.replay()
    assert_bit_identical(captured, eager_old, "captured new against eager old")


def test_guards_are_the_old_entry_s_guards():
    """The new entry accepts and rejects exactly what the old one does, so they are comparable."""
    ext = _extension()
    case = build_case(2, seed=9900, routes="valid", **REAL_SHAPE)
    bad = dict(case)
    bad["x"] = case["x"].t().contiguous().t()  # non-contiguous view of the same values
    for name in ("nvfp4_moe_rows_forward", "nvfp4_moe_rows_forward_exact_fast"):
        with pytest.raises(RuntimeError):
            call(ext, name, bad)

    mismatched = dict(case)
    mismatched["indices"] = case["indices"][:1]
    for name in ("nvfp4_moe_rows_forward", "nvfp4_moe_rows_forward_exact_fast"):
        with pytest.raises(RuntimeError):
            call(ext, name, mismatched)