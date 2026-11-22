# Scale-aware paired-byte LUT Ascend910B tile consumer

This is an **explicit numerical experiment**, not a default provider and not a production path. It
consumes the original lossless packed E2M1 bytes and the original per-K32 E8M0 scale grid, and
decodes each tile to the exact BF16 value the accepted `online_tile_cube` consumer produces. The
checkpoint is unchanged: same FP4 codes, same scale bytes, same 4.25-bit-per-value budget, no INT4
recode, no expanded BF16 weight bank, no regrouped or averaged scale grid.

It is a sibling of `online_tile_cube.cpp`, not a modification of it. Both directories build
independently and neither is registered as a default.

## Why a paired-byte LUT

The accepted consumer reconstructs each value with floating weight arithmetic: widen the packed
code, select a scale factor per K32 row, multiply in FP32, narrow to BF16. That is correct but it
spends a multiply plus two width changes per value on the Vector unit.

The measured checkpoint makes a cheaper exact form possible. Across all 40 x 384 x 3 = 46,080
matrices — 16,986,931,200 E8M0 bytes — every scale byte is in **117..127**, eleven values. So the
scale is not an arbitrary per-row quantity; it is one of eleven slots. With the scale fixed per
slot, the pair of BF16 values a packed byte decodes to is a pure function of `(slot, byte)`:

```
code_lut[slot][packed_byte] -> uint32
    low  16 bits = OriginalFp4Bits(packed_byte & 15, slot_scale)
    high 16 bits = OriginalFp4Bits(packed_byte >> 4, slot_scale)
```

`OriginalFp4Bits` is the accepted kernel's own helper (`online_tile_cube.cpp:92-99`), including its
zero fold: the accepted INT8 x2 widening has a single integer zero, so both E2M1 zero codes decode to
`+0`. Slot 11 is not a scale; it is the poison slot, and every pair in it is `0x7fc07fc0`.

That table is 12 x 256 words = 12 KiB. Decoding a K128 tile then costs two index widenings, two
gathers to get the offsets, one add and one final gather that writes the BF16 pairs straight into
the output slot. There is no FP32 weight value, no multiply and no BF16 narrowing on this path.

### Exactness

The result is bit-identical to the accepted consumer's BF16 operand, not an approximation:

- `OriginalFp4Bits(code, s)` puts `s + delta` in the BF16 exponent field, where `delta` is the
  E2M1 magnitude's exponent offset in `[-1, 2]`. With `s` in `117..127` the field stays in
  `116..129`, so every value is normal and the helper's `scale < 2 || scale > 252` guard cannot fire.
- The accepted consumer instead multiplies the code's `s = 127` BF16 bit pattern by the power of two
  `2**(s - 127)`. That product is exact in FP32 (a value with at most four significand bits times a
  power of two), and narrowing it back to BF16 is also exact, so both paths land on the same bits.
- Both paths fold `-0` to `+0` for the same reason: the accepted widening is integral.
- The Cube side is untouched. Activation admission, K16-major/N16-minor B order, `Mmad` with
  `m=16`, `n=N`, `k=128`, the ascending block order and the FP32 `Fixpipe` accumulation are all the
  accepted implementation, so neither the accumulation order nor the FP32 output changes.

The one behavioural difference is deliberate: a scale byte outside `117..127` maps to the poison
slot and yields NaN rather than a decoded value. That is safe for any input and cannot occur in this
checkpoint, but a caller must still certify the bank before trusting it (see below).

## Prepared tables

Three immutable regions, concatenated in this exact word order. This order is the ABI; the binding
writes it and the kernel reads it, and neither may reorder it.

| Region | Offset (words) | Length (words) | Content |
|---|---|---|---|
| code LUT | 0 | 3072 | 12 slots x 256 packed bytes, two BF16 per word |
| scale map | 3072 | 256 | raw scale byte -> LUT slot byte offset |
| pair repeat | 3328 | `BYTES` | per-pair byte offset into the scale-offset vector |

`BYTES = N * 128 / 2`, so the total is **5376 words for N32** and **7424 words for N64**. The tables
contain no checkpoint weights, no expanded values and no routing state.

Generation formulas, which the Python side must follow exactly:

```
code_lut[slot][byte] = OriginalFp4Bits(byte & 15, 117 + slot)
                     | OriginalFp4Bits(byte >> 4, 117 + slot) << 16      for slot 0..10
code_lut[11][byte]   = 0x7fc07fc0                                        for all 256 bytes

scale_map[raw]       = (raw - 117) * 1024    if 117 <= raw <= 127
                     = 11 * 1024             otherwise

pair_repeat[p]       = ((p / (N * 16)) * N + (p / 8) % N) * 4
```

`1024` is one slot's byte length (256 words x 4 bytes). The `pair_repeat` formula maps a destination
position in Cube K16-major/N16-minor order back to its original `(K32 block, N row)` scale:
`p / (N * 16)` is the K32 block, and within a block the fractal order is

```
[ n16 panel ][ row ][ kpair ]        panel = N/16 panels, row = 16, kpair = 8 pairs
```

so `p = block * N * 16 + panel * 128 + row * 8 + kpair` and `(p / 8) % N` is therefore the
**compact nrow, `panel * 16 + row`, not a row inside one panel.** That is exactly the scale bank's
own row order, because a wider tile is two or four adjacent original N16 banks and the panel index
runs outermost inside the tile, so the composite index is contiguous across panels.

This distinction is load-bearing and easy to "simplify" wrongly. For N32 the concrete case is
`p = 128`: it is panel 1, row 0, pair 0, so the formula yields nrow 16 (`pair_repeat[128] = 64`),
whereas a plausible row-major reading `(p % 512) // 16` yields 8 (`32`). Both are self-consistent
arithmetic; only the first matches the layout. A scale written with the row-major reading decodes
plausible but wrong weights, so any oracle for this table must pin `p = 128` and the within-fragment
panel shift (`p + 128` raises the nrow by exactly 16, i.e. 64 bytes of `pair_repeat`).

One scale covers a whole K32 row, that is sixteen packed bytes, and this table fans it out to one
offset per pair. Because the gather already writes in fractal order, the N32 tile needs no separate
panel rearrangement.

## Geometry

Fixed: `M = 16`, `GROUP_K = 4` (K128), `K_BLOCK = 32`, `layout_id = 1` for N32 and `2` for N64.
`RELIC_LUT_OUTPUT_ROWS` selects 32 or 64 at build time; 32 is the first candidate and 64 exists only
so the wider layout can be built, not as a measured configuration.

Unified Buffer byte addresses, all computed from `UL` constants and asserted to fit 128 KiB:

| Region | N32 | N64 |
|---|---|---|
| code LUT + scale map + pair repeat | 21.0 KiB | 29.0 KiB |
| raw packed bytes + scale bytes | 2.1 KiB | 4.25 KiB |
| half/int32 operands and offsets | 21.25 KiB | 42.5 KiB |
| two BF16 output slots | 16.0 KiB | 32.0 KiB |
| invalid-tile staging | 128 B | 256 B |
| **total** | **60.5 KiB** | **108.0 KiB** |

N64 fits 128 KiB only because no region is shared with a separate value tensor: the decoded BF16 is
written directly into the output slot by the final gather. Adding an intermediate value tensor would
push N64 over the limit.

L1 holds the two B1 slots, 16 KiB each at N64 (32 KiB total) plus one 4 KiB A1 activation slot. The
GM ring is `[block_dim, 2, ELEMENTS]` BF16, two slots per worker, and retains the accepted
`CACHE_MODE_DISABLE` hint.

## ABI

Exported from `librelic_scale_lut_tile_cube.so`:

```
relic_scale_lut_tile_cube_launch_tables(blockDim, stream,
    x, b, s, selected, out, stage, tables,
    tiles, blocks, groups, owned, world, n, shared)
```

That is the only launch entry point: two leading arguments, seven pointers and seven `uint32`
geometry values. The geometry meanings are the accepted ones and must not be rescaled by the caller:

| argument | meaning | example (K=5120, width 2304) |
|---|---|---|
| `tiles` | **source N16 bank count, `n / 16`** | 144 |
| `blocks` | original K32 block count, `K / 32` | 160 |
| `n` | output width in rows | 2304 |
| `groups`, `owned`, `world`, `shared` | unchanged from the accepted consumer | — |

`tiles` is the source count, not the output-tile count: one output tile of `N` rows consumes
`N_TILES = N / 16` source banks, so both `Init` functions derive `tiles_ = tiles / N_TILES` and the
bank stride is the compact `(local * tiles_ + tile) * blocks_` in `N`-row units. Passing the output
count here, or forgetting the division, would walk two or four times too many tiles and read past
the bank — this is why the derived value is stated in one place rather than recomputed per use. There is deliberately no table-less variant, because this format cannot decode
without its prepared table and a silently all-NaN tile is a worse failure than a missing symbol.
`Init` still poisons the LUT to NaN if a C caller passes a null table, so misuse cannot read
uninitialised Unified Buffer contents.

Geometry queries, all literal constants from the compiled configuration:

```
relic_scale_lut_tile_cube_output_rows()            32 or 64
relic_scale_lut_tile_cube_layout_id()              1 (N32) or 2 (N64)
relic_scale_lut_tile_cube_k_blocks_per_tile()      4
relic_scale_lut_tile_cube_group_k()                4   (same value, second spelling)
relic_scale_lut_tile_cube_table_words()            5376 or 7424
relic_scale_lut_tile_cube_scale_min()              117
relic_scale_lut_tile_cube_scale_max()              127
relic_scale_lut_tile_cube_prepared_tables()        1
relic_scale_lut_tile_cube_unsigned_byte_cast()     1
relic_scale_lut_tile_cube_lut_slots()              12
relic_scale_lut_tile_cube_poison_slot()            11
relic_scale_lut_tile_cube_stage_elements()         rows * 128 (BF16 per ring slot)
```

`shared` keeps the accepted meaning, and it selects a **GM row, not an L1 slot**: a `shared=true`
invocation reads activation row 0 for the whole launch, otherwise it reads this group's row. L1 holds
exactly one A1 slot (`L1_A_OFFSET`, `M * K`); every block's `DataCopy` writes it at base 0 and the
`LoadData` into L0A reads it at base 0, so there is no per-group L1 base to index. This kernel has no
activation-preload mode, and the accepted kernel's preload-only `row * M * fullK` offset must not
reappear here: with one slot allocated, indexing it by `group` reads past the allocation whenever
`shared` is false and `group` is non-zero. Empty (`-1`) and unowned slots write zeros and NaN into
the FP32 output exactly as before.

The unsigned query uses the repository's established spelling, `_unsigned_byte_cast`, as the
accepted compact consumers do, and exactly one such symbol is exported so a binder that
requires a single match cannot see two. `stage_elements()` reports the BF16 extent of one GM ring
slot, so the ring is `[block_dim, 2, stage_elements()]` and can be sized and checked by the caller
rather than recomputed.

The byte indices use the proven A2 unsigned widening, `uint8 -> half -> int32`. Both casts are
supported on Atlas A2 under `CAST_NONE` and `CAST_RINT`, and neither can round: a byte in `0..255`
is exact in half, and the widening only ever moves an index, never a decoded weight. The signed
`int8` form the accepted kernel needs on its default path, with its `+128` compensation, is not used
here at all. Note that `int8 -> int16` and `uint8 -> int16` are **not** A2 combinations — that pair
exists only on Ascend 950 — so a direct one-cast widening is not available and is not attempted.

## Build

Build only in a writable private directory; never modify the installed CANN or a checkpoint, and
never rebuild an accepted library in place:

```bash
cmake -S relic_core/csrc/ascend/scale_lut_tile_cube -B /private/build/scale-lut-tile-cube \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DSOC_VERSION=Ascend910B3 -DRUN_MODE=npu -DRELIC_LUT_OUTPUT_ROWS=32 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /private/build/scale-lut-tile-cube --parallel 2
```

Produces `lib/librelic_scale_lut_tile_cube.so` under that build root. `cmake` writes a one-line
translation unit that embeds `RELIC_LUT_OUTPUT_ROWS` and includes the kernel source, so the host
queries and the device compilation see the same literal. Hash the library before loading it.

## Validation obligations

None of the following has been executed for this directory. It is written, not measured, and no
performance or model-quality claim is made here.

- **Exactness gate**: for all 256 packed bytes and all 11 real slots, each LUT word must equal the
  accepted consumer's decoded pair bit for bit; then a whole-tile gate against the accepted
  `CUBE_PACKED` control on real W1/W3/down operands with zero differing BF16 bits.
- **Scale-range certificate**: a load-time obligation that every scale byte is in `117..127`, not an
  inference from hot-loop reads. An out-of-range byte poisons rather than decodes, so a bank that
  breaks the assumption is visible as NaN, never as a silently wrong value.
- **Table identity**: the binding must write the word order above and the kernel must read it; a
  reordered region is a silent mis-scale, not a crash. Check `table_words()` against the buffer
  length before the first launch.
- **Ring and fences**: the READY/FREE cross-core flags and the GM ring are the accepted ones, with
  two UB output slots and no per-copy `MTE3_V`. Two different lifetimes share this loop and they need
  opposite fences: the two decoded BF16 output slots are released through FREE after two blocks,
  while the single `raw_`/`scaleBytes_` UB slot is overwritten by the *next* iteration's MTE2, so the
  end-of-block fence must be `V_MTE2` (Vector finished reading before MTE2 writes again). The
  post-load fence stays `MTE2_V`. Fencing the end of the loop with `MTE2_V` leaves the raw/scale
  hazard open; this is the same pairing the accepted kernel uses at its lines 242 and 309. Guard capture/replay by the same no-alias and
  scratch-lifetime rules the accepted consumer uses.
- **Single A1 slot**: the probe corpus must cover `shared=false` with more than one active group.
  K128 with `shared=true` and dynamic `shared=true` both pass even when the A1 base is wrong, because
  those paths only ever use group 0; the failure appears only at `shared=false` with two or more
  active groups. This is how the first hardware gate caught it, so it belongs in the corpus rather
  than being inferred from the shared cases.
- **Numerical scope**: this is a native-consumer parity experiment. It is not equivalence to the
  window-13 production reduction, which remains the exact default.

## Status

Built and measured on Ascend910B3 during the 2026-11-22 same-provider experiment. Both N32 and
N64 passed codec, changing-route graph and real-projection gates. The selected N64/K128 path
measured 10.680570 single-request decode TPS versus its 8.562436 compact control (+24.7375%);
complete saved FP32 tensors and output IDs matched on all eight ranks. The matching RelicLLM
performance guide carries the frozen receipts and benchmark limitations.

This is an explicit, load-certified opt-in consumer, not an exact-default or general model-quality
acceptance. It neither reaches 50 TPS nor delivers a full persistent converted checkpoint. The
paired and wider-K experiments are not part of this implementation.
