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
