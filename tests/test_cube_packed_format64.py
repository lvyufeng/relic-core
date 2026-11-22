"""Independent scalar mapping and byte roundtrips for the distinct N64 format."""
from dataclasses import replace

import pytest
import torch

from relic_core.kernels import cube_packed_format64 as cube
from relic_core.kernels import cube_packed_format as old


def source(n=64, k=128, *, signed=False):
    codes = ((torch.arange(n)[:, None] * 133 + torch.arange(k // 2)[None] * 17) % 256).to(torch.uint8)
    scales = torch.arange(n * k // 32).remainder(256).to(torch.uint8).reshape(n, k // 32)
    return codes.view(torch.int8) if signed else codes, scales


def oracle_codes(codes, n, k):
    output = [0] * (n * k // 2)
    values = codes.view(torch.uint8).reshape(-1).tolist()
    for row in range(n):
        nb, n16, local_row = row // 64, row % 64 // 16, row % 16
        for byte in range(k // 2):
            kb, k16, pair = byte // 64, byte % 64 // 8, byte % 8
            target = (((((nb * (k // 128) + kb) * 8 + k16) * 4 + n16) * 16 + local_row) * 8 + pair)
            output[target] = values[row * (k // 2) + byte]
    return output


def oracle_scales(scales, n, k):
    output = [0] * (n * k // 32)
    values = scales.reshape(-1).tolist()
    for row in range(n):
        nb, local_row = row // 64, row % 64
        for block in range(k // 32):
            kb, group = block // 4, block % 4
            target = ((nb * (k // 128) + kb) * 4 + group) * 64 + local_row
            output[target] = values[row * (k // 32) + block]
    return output


@pytest.mark.parametrize("n,k", [(64, 128), (128, 384), (192, 256), (1152, 5120), (5120, 1152)])
def test_geometry_is_distinct_complete_and_same_size(n, k):
    geometry = cube.cube_geometry64(n, k)
    assert geometry.codes_shape == (n // 64, k // 128, 8, 4, 16, 8)
    assert geometry.scales_shape == (n // 64, k // 128, 4, 64)
    assert geometry.payload_bytes == n * k // 2 + n * k // 32 == geometry.source_bytes
    assert geometry.padding_bytes == 0 and geometry.report()["format"] == "relic.fp4.cube-n64-k128.v1"


@pytest.mark.parametrize("n,k", [(32, 128), (96, 128), (64, 192), (0, 128), (-64, 128), (64, 0)])
def test_no_padding_or_partial_tile(n, k):
    with pytest.raises(ValueError):
        cube.cube_geometry64(n, k)


@pytest.mark.parametrize("n,k", [(True, 128), (64, False), (64.0, 128), (64, "128")])
def test_geometry_refuses_non_integral_or_boolean_extents(n, k):
    with pytest.raises(TypeError):
        cube.cube_geometry64(n, k)


@pytest.mark.parametrize("n,k", [(64, 128), (128, 384), (192, 256)])
@pytest.mark.parametrize("signed", [False, True])
def test_scalar_oracle_and_fresh_byte_roundtrip_include_every_encoding(n, k, signed):
    codes, scales = source(n, k, signed=signed)
    originals = codes.clone(), scales.clone()
    payload = cube.convert_fp4_cube64(codes, scales)
    assert payload.codes.reshape(-1).tolist() == oracle_codes(codes, n, k)
    assert payload.scales.reshape(-1).tolist() == oracle_scales(scales, n, k)
    assert set(codes.view(torch.uint8).reshape(-1).tolist()) == set(range(256))
    assert set(scales.reshape(-1).tolist()) == set(range(256))
    recovered = cube.recover_fp4_cube64(payload)
    assert recovered[0].dtype == codes.dtype and recovered[1].dtype == scales.dtype
    assert all(torch.equal(value, expected) for value, expected in zip(recovered, originals))
    assert all(torch.equal(value, expected) for value, expected in zip((codes, scales), originals))
    for left in (payload.codes, payload.scales):
        assert left.device.type == "cpu" and left.dtype == torch.uint8 and left.is_contiguous()
        assert not any(torch._C._overlaps(left, right) for right in (codes, scales))
    for left in recovered:
        assert not any(torch._C._overlaps(left, right) for right in (payload.codes, payload.scales))


def test_n32_and_n64_payload_types_and_tags_are_not_interchangeable():
    codes, scales = source()
    n64, n32 = cube.convert_fp4_cube64(codes, scales), old.convert_fp4_cube(codes, scales)
    with pytest.raises(TypeError):
        cube.recover_fp4_cube64(n32)
    with pytest.raises(TypeError):
        old.recover_fp4_cube(n64)
    with pytest.raises(ValueError, match="distinct"):
        replace(n64, format=old.FORMAT)
    object.__setattr__(n64, "format", old.FORMAT)
    with pytest.raises(ValueError, match="distinct"):
        cube.recover_fp4_cube64(n64)


@pytest.mark.parametrize("device", ["npu:0", "cuda:0", "meta"])
def test_converter_refuses_non_cpu_output_without_allocating_it(device):
    with pytest.raises(ValueError, match="CPU"):
        cube.convert_fp4_cube64(*source(), device=device)


@pytest.mark.parametrize("field", ["codes", "scales"])
def test_converter_refuses_numeric_dtype_casts(field):
    codes, scales = source()
    if field == "codes":
        codes = codes.float()
    else:
        scales = scales.view(torch.int8)
    with pytest.raises(TypeError):
        cube.convert_fp4_cube64(codes, scales)


def test_converter_refuses_strided_source_and_wrong_scale_extent():
    codes, scales = source()
    with pytest.raises(ValueError, match="contiguous"):
        cube.convert_fp4_cube64(codes.t(), scales)
    with pytest.raises(ValueError, match="extent"):
        cube.convert_fp4_cube64(codes, scales[:, :2].contiguous())


def test_dtype_override_changes_only_the_byte_spelling():
    codes, scales = source(signed=True)
    recovered, recovered_scales = cube.recover_fp4_cube64(cube.convert_fp4_cube64(codes, scales), codes_dtype="uint8")
    assert torch.equal(recovered, codes.view(torch.uint8)) and torch.equal(recovered_scales, scales)
    with pytest.raises(ValueError, match="codes_dtype"):
        cube.recover_fp4_cube64(cube.convert_fp4_cube64(codes, scales), codes_dtype=torch.float32)
