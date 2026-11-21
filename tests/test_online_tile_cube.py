"""Original-byte layout and device-selected tile-consumer interface controls."""
import pytest
import torch

from relic_core.kernels.online_tile_cube import pack_expert_tile_bank, selected_tile_cube_fp4_gemv
from relic_core.kernels.tile_major_packed_gemv import unpack_tile_major
from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale


def fixture():
    generator = torch.Generator().manual_seed(212)
    codes = torch.randint(0, 256, (3, 35, 64), dtype=torch.uint8, generator=generator)
    scales = torch.randint(118, 123, (3, 35, 4), dtype=torch.uint8, generator=generator)
    x = torch.randn((3, 128), generator=generator).bfloat16()
    b, s = pack_expert_tile_bank(codes, scales)
    return x, codes, scales, b, s


def test_expert_layout_is_a_byte_exact_permutation_and_retains_scale_bytes():
    _, codes, scales, b, s = fixture()
    for expert in range(3):
        restored_code, restored_scale = unpack_tile_major(b[expert], s[expert], n_output=35)
        assert torch.equal(restored_code, codes[expert])
        assert torch.equal(restored_scale, scales[expert])
    assert b.shape == (3, 3, 4, 16, 16) and s.shape == (3, 3, 4, 16)


@pytest.mark.parametrize("group_k", [1, 2, 4])
@pytest.mark.parametrize("shared", [True, False])
def test_original_i8x2_half_scale_operand_and_selected_sources(group_k, shared):
    x, codes, scales, b, s = fixture()
    selected = torch.tensor([4, 0, -1], dtype=torch.int32)
    value = x[:1] if shared else x
    actual = selected_tile_cube_fp4_gemv(value, b, s, selected, world=2,
                                        n_output=35, group_k=group_k, backend="torch")
    expected = torch.zeros((3, 35), dtype=torch.bfloat16)
    for slot, expert in enumerate((2, 0)):
        decoded = (unpack_fp4_to_int8(codes[expert], backend="torch").float()
                   * fp4_int8_scale(scales[expert], backend="torch").repeat_interleave(32, dim=-1)).bfloat16()
        partial = torch.zeros(35, dtype=torch.float32)
        width = group_k * 32
        row = value[0 if shared else slot]
        for start in range(0, 128, width):
            partial += (row[start:start+width].float()[None] @ decoded[:, start:start+width].float().t())[0]
        expected[slot] = partial.bfloat16()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_invalid_sources_are_poisoned_and_empty_slots_are_zero():
    x, _, _, b, s = fixture()
    selected = torch.tensor([-2, 6, -1], dtype=torch.int32)
    result = selected_tile_cube_fp4_gemv(x, b, s, selected, world=2, n_output=35, backend="torch")
    assert torch.isnan(result[:2]).all() and torch.equal(result[2], torch.zeros_like(result[2]))


def test_output_buffer_is_used_and_input_alias_is_refused():
    x, _, _, b, s = fixture()
    selected = torch.tensor([0, 2, 4], dtype=torch.int32)
    output = torch.empty((3, 35), dtype=torch.bfloat16)
    assert selected_tile_cube_fp4_gemv(x, b, s, selected, world=2, n_output=35, backend="torch", out=output) is output
    with pytest.raises(ValueError, match="alias"):
        selected_tile_cube_fp4_gemv(x, b, s, selected, world=2, n_output=35, backend="torch", out=x.reshape(-1)[:105].view(3, 35))


def test_widths_and_selected_metadata_are_checked_before_execution():
    x, _, _, b, s = fixture()
    selected = torch.tensor([0, 2, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="selected IDs"):
        selected_tile_cube_fp4_gemv(x, b, s, selected.long(), world=2, backend="torch")
    with pytest.raises(ValueError, match="BF16"):
        selected_tile_cube_fp4_gemv(x.float(), b, s, selected, world=2, backend="torch")
    with pytest.raises(ValueError, match="group_k"):
        selected_tile_cube_fp4_gemv(x, b, s, selected, world=2, group_k=8, backend="torch")
