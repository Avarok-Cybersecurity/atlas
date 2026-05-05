# NVFP4 Quantization

NVFP4 is Atlas's flagship format on GB10 — 4-bit weights with FP8 block scales. Most Qwen and Nemotron checkpoints ship in it, and it's the KV-cache dtype that hits the best compression/quality balance for the Qwen3.5 family.

## The numeric format

Each NVFP4 tensor is stored as two pieces:

- **`weight`** — `[N, K/2]` bytes, two E2M1 nibbles packed per byte.
- **`weight_scale`** — `[N, K/16]` bytes, one FP8 E4M3 scale per 16-element block along K.

E2M1 encoding (per nibble):

| Bits | Decoded | Bits | Decoded |
|---|---:|---|---:|
| `0000` | +0.0 | `1000` | -0.0 |
| `0001` | +0.5 | `1001` | -0.5 |
| `0010` | +1.0 | `1010` | -1.0 |
| `0011` | +1.5 | `1011` | -1.5 |
| `0100` | +2.0 | `1100` | -2.0 |
| `0101` | +3.0 | `1101` | -3.0 |
| `0110` | +4.0 | `1110` | -4.0 |
| `0111` | +6.0 | `1111` | -6.0 |

Eight values after sign. Per 16-element block, a scalar FP8 scale is stored — so the reconstructed value is `scale[block] * e2m1_lut[nibble]`.

## Why blocks of 16

Two reasons that both land on 16:

1. **Tensor-core fragment sizes.** SM121's MMA instructions work on `16 × k` fragments along the K dim. Block-aligning the scale to 16 lets a single tile load reach all the scales it needs without a second indexed load.
2. **Accuracy.** 16 is small enough that outliers inside a block are rare; dequantizing against a scale computed over 16 elements is close enough to per-element calibration for modern transformer activation statistics. Larger blocks (64, 128) lose perplexity; smaller blocks (4, 8) waste scale bytes.

16 is the community standard for NVFP4 and what every HF checkpoint we load uses.

## SM121's E2M1 conversion problem — and the software fix

Later Blackwell parts have a single-instruction conversion: `cvt.rn.satfinite.e2m1x2.f32` takes two floats and emits two E2M1 nibbles. SM121 **does not have this instruction**. It is the headline hardware limitation of GB10 NVFP4.

Atlas's software fix — `e2m1_branchless.cu` — does the conversion in 7 ALU ops using the IEEE-754 bit pattern:

```cuda
// Simplified sketch: convert a positive f32 to a 3-bit magnitude nibble.
// (sign bit is handled separately)
//
// The E2M1 value set for positive magnitude is:
//   {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// corresponding to (exponent_field, mantissa) pairs:
//   (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)
//
// 7 compares against the mid-points of adjacent E2M1 levels map each
// f32 to one of the 8 values without a single branch.

uint32_t bits = __float_as_uint(x);
uint32_t abs  = bits & 0x7fffffff;
uint32_t sign = (bits >> 31) << 3;

// Thresholds (bit patterns of mid-points in f32)
bool ge_025 = abs >= 0x3e800000;  // 0.25
bool ge_075 = abs >= 0x3f400000;  // 0.75
bool ge_125 = abs >= 0x3fa00000;  // 1.25
bool ge_175 = abs >= 0x3fe00000;  // 1.75
bool ge_250 = abs >= 0x40200000;  // 2.5
bool ge_350 = abs >= 0x40600000;  // 3.5
bool ge_500 = abs >= 0x40a00000;  // 5.0
// Accumulate the magnitude index in 0..=7
uint32_t idx = ge_025 + ge_075 + ge_125 + ge_175 + ge_250 + ge_350 + ge_500;

uint32_t nibble = sign | idx;  // final 4-bit E2M1
```

Seven compares, seven adds, one bit shift, one OR. Fully branchless. Two nibbles per byte are produced by running the same sequence on `lo` and `hi` halves of a 64-bit pair and packing.

The payoff: NVFP4 round-trip (`dequant → compute → requant`) runs at full pipeline speed on SM121, *despite* the missing instruction. Exhaustive testing on 19 experiments (logged in `trtllm-ngram-experiments.csv`) established 29.6 tok/s as the TRT-LLM ceiling on the same model; Atlas's vLLM-path approach running NVFP4 through software-E2M1 CUTLASS hits **36.4 tok/s (CUTLASS MoE)** and **59.9 tok/s (Marlin + MTP)** on the same hardware. Software E2M1 is a 32× speedup vs the first "enable E2M1" build.

## Dequantization in the GEMM kernel

Atlas does not pre-materialize BF16 from NVFP4. Instead, the GEMM kernel loads NVFP4 directly into shared memory and dequantizes *on the fragment boundary* just before the MMA:

1. `cp.async` a tile of `weight` (packed nibbles) + its `weight_scale` into shared memory.
2. On the consumer warp, unpack a 16×8 fragment of weights: convert each pair of nibbles to two BF16s using the E2M1 LUT in shared/constant memory, multiply by the block scale (broadcast from the scale tile).
3. Feed the BF16 fragment to `mma.sync.aligned.m16n8k16.row.col.bf16.bf16.f32`.
4. Accumulate in FP32, downcast to BF16 on output.

Pre-materializing BF16 would require `~8×` more shared memory and throw away most of the compression benefit. The fragment-time dequant is the key to NVFP4 actually being fast.

## NVFP4 KV cache

The same E2M1 format is available as a KV-cache dtype (`--kv-cache-dtype nvfp4`). K and V tensors are stored as NVFP4 with per-block FP8 scales; the paged cache allocator budgets `0.5 bytes + scale_bytes` per element.

When a request hits prefix caching, the cached K/V is in NVFP4; the attention kernel reads directly and dequantizes at the MMA boundary, same pattern as the weight GEMM.

For coherence at long context, `--kv-high-precision-layers N` keeps the first and last N attention layers at BF16 — those layers' KVs are the most sensitive to precision loss. The default is `0`; production deployments of the 122B model use `2`.

## When not to use NVFP4

- **When a model's weights don't ship in it.** A few checkpoints (Qwen3.6-35B-FP8) are FP8-native. Don't re-quantize — load them FP8 and use the FP8 KV cache. See the [FP8 chapter](./fp8.md).
- **When you have only one GB10 and the model fits in FP8.** FP8 weights require no software-E2M1 path, so the prefill/decode hot loops are slightly simpler and slightly faster per kernel-launch. The trade is the 2× memory increase vs NVFP4.
- **When you're debugging a coherence regression.** Fall back to BF16 first, then FP8, then NVFP4 — narrows the bug source quickly.

## Why Atlas is not pursuing native FP4 MMA on SM121

Discovering this was the point of a multi-week research dive (logged as the "FP4 MMA GB10" project in the repo). The short version:

- SM121 silicon does not expose the relevant MMA or conversion instructions.
- CUTLASS 4.3's SM120 builders enforce cooperative-pipeline scheduling via `static_assert` — you cannot swap in pingpong mode where it would help.
- Every alternate MoE backend (TRTLLM, CuteDSL, DeepGEMM, Triton) fails with `NotImplementedError: SM120 and above` or crashes outright.
- The community benchmarks that claim native FP4 throughput on "Blackwell" are on SM100a / SM101a parts and do not apply.

Atlas's 131 tok/s on Qwen3.5-35B and 104 tok/s on Qwen3-Next-80B are, in this sense, **the real GB10 ceiling** — achieved through software E2M1, Marlin-style dequant-to-BF16, and MTP speculative decoding. New NVIDIA silicon would unlock another axis of improvement; on today's GB10, the numbers in the README are the answer.

## Files to read

- `kernels/gb10/<model>/nvfp4/e2m1_branchless.cu` — the conversion.
- `kernels/gb10/<model>/nvfp4/moe_prefill.cu`, `dense_gemm_nvfp4.cu` — the GEMM tiles + fragment-time dequant.
- `kernels/gb10/<model>/nvfp4/paged_decode_attn_nvfp4.cu` — the NVFP4 KV attention.
- `crates/atlas-quant/src/nvfp4.rs` — the Rust trait binding.
- `docs/design/NVFP4_COHERENCE.md` — the design note on `--kv-high-precision-layers`.
