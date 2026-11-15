# Tensor-core chunked GDN prefill for Ada, from NInfer via Cinference (lane L16)

**What it is.** A three-launch Gated DeltaNet prefill for sm_89
(`kernels/ada/common/gdn_chunk_tc_ada.cu`) that puts every per-chunk product
on `mma.sync`: the K K^T Gram and Q K^T, Q S on BF16 m16n8k16, the triangular
solve products and the state update on TF32 m16n8k8, with the recurrent state
held in FP32 MMA accumulators across chunks. It replaces, behind the existing
`[defaults] gdn_prefill_tc` row, the FLA route Atlas runs on Ada today
(`gated_delta_rule_recompute_wu` -> `gated_delta_rule_chunk_delta_h_vfused` ->
`gated_delta_rule_chunk_fwd_o`), whose state spine is scalar FP32 FMA work that
issues zero MMAs and is latency bound on its dependent chain
(GDN-PREFILL-ATTRIBUTION.md, the H100 receipt; the porting guide's section 10
item 4 estimates 30 to 40 ms off the 948 ms 4B TTFT at 3.9K tokens on the RTX
4060 Ti).

**Status.** Written, CPU-tested, not yet run on a GPU. The Ada value of
`gdn_prefill_tc` stays `false`; no target's default changes. The receipt that
flips it is listed at the end.

## Source

| | |
|---|---|
| Upstream | Cinference, https://github.com/satellitedown/cinference, commit `383e5db` |
| Derived from | NInfer by Neroued and contributors, https://github.com/Neroued/ninfer |
| License | Apache License, Version 2.0 (root `LICENSE`, no copyright line); text in `LICENSES/Apache-2.0.txt` |
| NOTICE | the first paragraph of Cinference's `NOTICE`, reproduced in Atlas's `NOTICE` |
| Kind | code copy with adaptation (CUDA), plus a translation of the launch geometry (C++ to Rust) |

Files and line ranges read and copied or translated:

| Upstream path | Lines | Lands in |
|---|---|---|
| `src/ops/linear_attention/gated_delta_net/chunked/common.cuh` | 1-21 | `gdn_chunk_tc_ada.cu` constants |
| `src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh` | 1-724 | `gdn_tc_ada::prep` (stage 1) |
| `src/ops/linear_attention/gated_delta_net/chunked/state_passing.cuh` | 1-538 | `gdn_tc_ada::sp` (stage 2) |
| `src/ops/linear_attention/gated_delta_net/chunked/output.cuh` | 1-370 | `gdn_tc_ada::out` (stage 3) |
| `src/ops/linear_attention/gated_delta_net/chunked/{prepare_wy_wu,state_passing,output}.cu` | 1-46, 1-52, 1-57 | geometry: the entry points and `layers/qwen3_ssm/gdn_chunk_tc_ada.rs` |
| `src/ops/linear_attention/gated_delta_net/common.cuh` | 52-104 | `SmemTile`, the BF16 to FP32 panel load |
| `src/ops/common/mma.cuh` | 10-43, 91-104 | `ldmatrix_x2/x4/x2_t`, `mma_bf16`, `mma_tf32(_bits)` |
| `src/ops/common/memory.cuh` | 9-137 | `smem_addr`, `cp.async` (with zero fill), `load_vec`, `store_vec` |
| `src/ops/common/bf16_vector.cuh` | 13-26 | `Bf16x4Pack` |
| `docs/maintainer/op-development.md` | section 6.3 | the acceptance criterion shape (idea; no text copied) |

The four `chunked/*.cuh` files carry no Cinference modification line, so they
are NInfer's original code under the root Apache-2.0 LICENSE. `mma.cuh` and
`memory.cuh` carry satellitedown's modification notice; only their generic
PTX wrappers are used. Nothing under `third_party/` was read or used.

## Changes for Atlas (by Atlas contributors)

1. **One translation unit, Atlas ABI.** Three `extern "C"` entries
   (`gdn_chunk_tc_ada_prepare`, `_state`, `_output`) with plain pointer and
   `unsigned int` arguments; each template is bound to one sm_89 geometry; the
   C++20 template lambdas (`kkt_strip`, `wave_compute_store`) are template
   functions, because Atlas's kernel build uses the nvcc default dialect.
2. **Atlas layouts.** q, k and v are read from the packed conv output at the
   `qk_stride` and `v_stride` token pitches (`conv_dim`), gate and beta at
   `gb_stride` (`2 * nv`, interleaved). The FP32 state is Atlas's
   `[nv][k][v]` layout, updated in place, contiguous or through a per-request
   pointer table (upstream: a separate `[v][k]` input and output). The
   workspace is Atlas's FLA scratch carved in the FLA order (W, U, S, uc, gc),
   every region blocked by `(b * num_chunks + c) * nv + h_v`, so the two
   routes use the same bytes for the same call (upstream: token-major
   `[T][H][128]` regions).
3. **Atlas's gate.** Atlas passes the LINEAR decay. The kernel takes it to log
   space as `logf(fmaxf(gate, 1e-30))` and sums the cumulative gate on one
   thread in token order, the exact arithmetic of Atlas's FLA
   `gated_delta_rule_recompute_wu` (whose comment records that a
   re-associated tree scan of this sum cost 2.6 BFCL points). Upstream reads a
   log-space gate and scans it with a Hillis-Steele warp scan.
4. **Partial last chunk.** Upstream requires `T % 64 == 0` and routes tail
   tokens elsewhere. Here rows past the end are zero-filled (`cp.async` with a
   zero source size, no global read), their beta is 0 and their log gate is 0,
   so they hold the last real cumulative gate: they contribute nothing, add no
   decay, and keep every `exp` defined; output stores are masked.
5. **Batch.** A grid dimension for the uniform co-dispatched prefill.
6. **`expf` for every decay** instead of `ex2.approx` (`exp2_approx`), as
   Atlas's FLA route computes them.
7. **TF32 rounding.** Operands fed to TF32 MMAs from FP32 values (T_inv, the
   scaled V and K panels, the state snapshot in v_new = U - W S, v_new times its
   decay, and the decayed Q K^T) are rounded to nearest with
   `cvt.rna.tf32.f32` where they are produced, instead of truncated by the MMA,
   which drops the low 13 bits with a consistent bias toward zero. The BF16
   copy of S_c that stage 3 reads is rounded from the unrounded FP32 value, as
   the FLA route rounds it.
8. **sm_89, 34 SMs.** Prepare: 4 warps, 23.75 KiB, 4 CTAs per SM (upstream's
   H_v = 32 geometry, used for every head count). State passing: 16-column
   value strips, 8 warps, 44.25 KiB, 2 CTAs per SM (upstream's H_v >= 48
   geometry). Output: 4 warps, 24 KiB, 4 CTAs per SM, with the grid-stride chunk
   loop sized from the device's SM count (upstream: a constant 170 for the
   RTX 5090). All three stay under the 48 KiB default, so no opt-in is needed;
   they still go through `prefill_dyn_smem.cuh` and export
   `<entry>_dyn_smem_bytes`, so the registry raises any launch that asks for
   less.

## Selection and safety

- `layers/qwen3_ssm/gdn_chunk_tc_ada.rs` resolves the three entries lazily,
  once per layer, with `try_target_kernel`, and only after the row resolved
  on. The module exists only in the `kernels/ada` tree, so on Hopper (which
  declares `gdn_prefill_tc = true` for its own spine) the module is absent,
  no lookup is issued and nothing is logged. GB10, B200, B300, Metal and
  Strix never compile the file.
- The arm is tried after FlashInfer (an explicit opt-in) and before the FLA
  ladder in the three GDN prefill entry points (`trait_prefill_recur.rs`,
  `trait_prefill_gdn.rs` and its batched child). It declines, naming the
  guard once per reason, on: zero tokens, warm exact replay, an active
  mid-chunk SSM tail capture, head dims other than 128, a head ratio that is
  not whole, unaligned strides or pointers, a missing or too small FLA scratch,
  or a grid limit. Declining runs today's ladder unchanged.

## Verification

- CPU (`cargo test -p spark-model --lib gdn_chunk_tc_ada`): the FP64 chunk
  algebra against the FP64 recurrence at T = 1 to 4096 (normwise below 1e-9),
  chunk-size invariance, the gate floor; both route profiles (emulated TC
  rounding points, FP32 scalar) meet the criteria and agree within them;
  corruption fails them; the eligibility grammar, the scratch carving against
  the FLA formula, the shared-memory budgets, the output grid, the launch ABI on
  the mock backend, lazy resolution; and a host simulation proving every
  fragment, store and swizzle map of the kernel a bijection over its tile.
- GPU (`examples/gdn_chunk_tc_ada_microtest.rs`): the kernel's output and
  final state against the FP64 recurrence under `oracle::{OUTPUT, STATE}`
  (normwise 2e-2 with a 0.25 gross pointwise cap, non-finite fails), its
  agreement with the scalar spine within the same bound, a sentinel tail
  after a partial chunk, batch 2 through a state pointer table, and timings
  against the FLA and scalar routes.

**Receipt that flips the Ada row:** the microtest PASS at T 64, 150, 1024,
3921, 4096; `longctx.py` recall unchanged at 3.9K and 32K; `qual.py` KL against
the FLA spine within the numerics gate; TTFT ABBA at 3.9K, 32K and 128K tokens
showing a gain. The FLA route's own history on this kernel (a cosine of
1.0000 that still cost 1.4 BFCL points) applies: the ssm-poisoning tripwire
belongs in the receipt before the row moves.
