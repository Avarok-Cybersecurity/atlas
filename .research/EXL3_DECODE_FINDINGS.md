# EXL3 decode — nailed-down findings (2026-08-31)

Researched against turboderp/exllamav3 @ master (files snapshotted in
`.research/exllamav3_ref/`), specifically to de-risk native decode support
for `unsloth/Qwen3.8-Flash-Next-exl3` (Atlas's `qwen4_exp`).

## The "trellis" is not a decode-time state machine

Despite the QTIP/"trellis-coded quantization" name, decoding one weight is
O(1) and stateless: extract a fixed-width bit-window from a packed stream,
feed it through a tiny procedural function. The trellis search happens only
at ENCODE time (choosing index sequences that reconstruct well); nothing
about decode depends on neighboring weights or carries state between them.

## Bit extraction (`exl3_dq.cuh`)

Per-weight codes are packed contiguously at a fixed `bits` width (1-8) in a
`uint32_t[]` stream. Extraction is funnel-shift + mask against a 16-bit
window (`fshift`/`__funnelshift_r`), dispatched by bit-width via
`dq_dispatch<bits, cb>` to one of several unrolled variants
(`dq8_aligned_1/2/4bits`, `dq8`, `dq4`, `dq2x2`) chosen for register/ALU
efficiency at each width. All are pure arithmetic on the packed words — no
lookup table, no branching on data.

## Codebook = procedural PRNG, not stored data (`codebook.cuh`)

Three codebook variants (`cb` = 0/1/2), each 2-3 CUDA instructions,
verbatim:

- **cb=0 ("3inst")**: `x = code*89226354u + 64248484u`, then
  `lop3.b32 x,x,0x8fff8fff,0x3b603b60,0x6a` (bitwise 3-input LUT: keeps
  sign+mantissa bits from the scrambled integer, forces a fixed exponent —
  the standard "scrambled-int-to-bounded-float" trick), sum the two `half`
  lanes of the resulting `half2`.
- **cb=1 ("mcg")**: `x *= 0xCBAC1FEDu` (pure multiplicative congruential
  generator, no additive constant), same `lop3` trick as cb=0.
- **cb=2 ("mul1")**: `x *= 0x83DCD12Du`, then `__dp4a(x, 0x01010101u, 0x6400u)`
  (hardware 4-way byte-sum + bias), converted to `half` via a fixed affine
  scale (`k_inv_h=0.00677`, `k_bias_h=-10.39`).

No embedded table of any size — this is 100% arithmetic, trivially portable
bit-for-bit to any target that has 32-bit integer multiply and can do the
equivalent bit-select (a 3-input LUT is one `SEL`/`AND`+`OR`+`OR` sequence
if a target lacks `lop3` natively — CUDA-to-CUDA this is a non-issue).

## Blackwell is not a porting risk, it's already the tuned target

`codebook.cuh`'s own comment: `__dp4a` "native on Blackwell where vabsdiff4
is emulated" — i.e. this exact code was already optimized FOR Blackwell,
not just Ampere/Ada/Hopper. `exl3_kernel_map.cu`'s `select_gemm_shape` has
an explicit `CC_BLACKWELL` case in its dispatch switch. GB10 is a first-
class supported target upstream already.

## Kernel/PyTorch coupling

The actual `__global__` kernels (`exl3_gemv_kernel`, etc.) take raw
pointers, not `at::Tensor`. ATen-typed wrappers exist only as a thin outer
layer (e.g. `reconstruct()` in `reconstruct.cu`) that immediately calls
`.data_ptr()` and hands off to the pointer-based kernel. Porting means
writing Atlas's own `KernelLaunch`-based launcher against the same
kernel body — not detangling PyTorch from the math.

## What's NOT yet nailed down (still ahead of writing the kernel, but no
## longer "unknowns" in the sense of hidden algorithmic risk)

- Hadamard sandwich transform exact tiling (`reconstruct_had_kernel` in
  `reconstruct.cu`, ~200 lines, `diag(suh)·H128·W_hat·H128·diag(svh)` per
  128x128 tile) — real work, but a known, well-defined linear-algebra
  operation, not a research risk.
- EXL3 tensor-naming/metadata scheme for THIS model's checkpoint (trellis
  codes, `su`/`sv` scale vectors, Hadamard seed, per-layer bits-per-weight)
  needs mapping into Atlas's weight_map for qwen4_exp specifically —
  mechanical, same shape as the existing NVFP4 loader.
- License: MIT (turboderp, 2025) — permissive, compatible with vendoring
  into Atlas's AGPL-3.0 codebase (permissive-into-copyleft is the
  compatible direction).

## Revised estimate

Native EXL3 decode kernel for qwen4_exp: ~1-1.5 weeks (short end of the
prior 1-3 week range), since the single biggest risk — an exotic,
hard-to-verify decode algorithm — turned out not to exist. Remaining
effort is mechanical porting + the Hadamard tile math + weight-map wiring,
not algorithm research.

---

# IMPLEMENTED (2026-09-01, branch wip/exl3-research)

## Landed and verified

- **`kernels/gb10/common/exl3_reconstruct.cu`** — self-contained port of
  upstream's fused reconstruct+Hadamard (`reconstruct_had_slice`): all 24
  (K=1..8 x cb=0/1/2) `extern "C"` instances + `exl3_f16_to_bf16_t`
  layout conversion to Atlas's `[out, in]` BF16. MIT attribution carried.
- **`crates/spark-runtime/src/weights/exl3.rs`** — launch wrappers
  (`reconstruct_had_f16_device`, `reconstruct_had_bf16`), an INDEPENDENT
  CPU reference (`cpu_ref`, written from the format spec, not transcribed
  from the kernel), `Exl3Weight::from_store`/`to_bf16`, detection helpers
  (`is_exl3_linear`, `store_has_exl3`, `is_exl3_f16_aux`), and
  `Exl3Codebook::from_flag_scalar`.
- **Loader plumbing** — new store dtypes `F16`/`UInt16`/`Int32` with
  safetensors mappings in all three ingest paths (mmap loader,
  fast-weights O_DIRECT header, RDMA wire strings); the blanket
  F16->BF16-at-load conversion now EXEMPTS `.suh`/`.svh` (their exact f16
  bits are decode inputs — BF16 rounding would silently change every
  reconstructed weight).
- **Parity gate** (`spark-model/examples/exl3_reconstruct_parity.rs`):
  GPU vs CPU BYTE-IDENTICAL at every leg — 3 shapes x {K=2,3,4,5,6 mul1;
  K=4,6 mcg; K=4 3inst} x both stages (raw f16 + transposed bf16), with
  1-bit negative controls, PLUS a REAL tensor from
  turboderp/Qwen3.8-Flash-Next-exl3 4.05bpw (layer-0 expert-0 gate_proj,
  fetched via HTTP range): bit-identical, healthy weight stats
  (mean -3e-6, std 0.0135 ~ 1/sqrt(2560), all finite).

## Format facts pinned down while implementing

- Bitstream is MSB-FIRST WITHIN EACH u32 (u16 pairs little-endian into
  u32s), ascending u32 order; code `t`'s window = stream bits
  `[(t+1)K-16, (t+1)K)` mod 256K, window value read MSB-first (bit 15 =
  oldest). My first LSB-first model produced ~100% mismatch; this one is
  parity-proven.
- Tile position map (m16n8k16 B-fragment order): `l=t/8, j=t%8`,
  `row = (l%4)*2 + (j&1) + ((j>>1)&1)*8`,
  `col = ((l&~4)/8)*2 + ((l>>2)&1) + ((j>>2)&1)*8`.
- The `.mul1` scalar stores the CODEBOOK'S MULTIPLIER CONSTANT
  (0x83DCD12D = mul1, 0xCBAC1FED = mcg) — the checkpoint self-describes.
- The 4.05bpw branch mixes K per tensor family: attention/GDN projections
  K=6, MoE experts K=4. K is derivable per-tensor from the trellis shape.
- `ngram_embedding.safetensors` is its OWN row-wise format
  (`exl3_ngram_trellis` v1: [320M rows, 61 u16], K=6 mul1, per-row
  decodable for sparse PLE gather) — needs a separate row decoder, NOT
  covered by the tile kernel.
- Tensors with dims not divisible by 128 (e.g. GDN in_proj_a/b [48,2560])
  ship UNQUANTIZED (f16) — the 128-multiple constraint never bites.

## Still ahead (the rest of the ~1-1.5 week native path)

1. qwen4_exp loader wiring: route `.trellis`-present prefixes through
   `Exl3Weight` (reconstruct->BF16->existing runtime NVFP4/FP8 requant,
   tensor-at-a-time to bound transients), F16 dense tensors already
   convert via the existing loader path.
2. The ngram row-format decoder (PLE gather path).
3. Vision shard (`vision_k6.safetensors`) mapping.
4. The NATIVE fused trellis-GEMM/GEMV port (exl3_gemm/exl3_gemv) — the
   actual memory-win path; the reconstruct route above serves at
   requantized quality/footprint, native keeps 2-6 bpw resident.
