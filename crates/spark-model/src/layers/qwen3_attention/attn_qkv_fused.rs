// SPDX-License-Identifier: AGPL-3.0-only

//! The FUSED attention Q/K/V DECODE GEMM — one block-scaled FP8 cuBLASLt call
//! at `N = q_proj_dim + 2*kv_dim` in place of three.
//!
//! # WHY (#927)
//!
//! nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3, `Qwen/Qwen3.8-27B-FP8` @
//! `3717cb05e`, round 13 cell V, median `n = 16` decode step **19.887 ms** of
//! kernel busy (`h100-r13-attribution.md` §§C.2–C.4). The 16 full-attention
//! layers run THREE cuBLASLt W8A8 GEMMs each:
//!
//! | arm | K | N | nodes | µs (µs/node) | GB/s | % HBM |
//! |---|---|---|---|---|---|---|
//! | attn `q_proj` (gated `[Q\|gate]`) | 5120 | 12288 | 16 | 460.0 (28.75) | 2 189 | 65.3 % |
//! | attn `k_proj` + `v_proj` | 5120 | 1024 | 32 | 510.9 (15.97) | **328** | **9.8 %** |
//! | (control) FFN `down`, same step | 17408 | 5120 | 64 | 2 384.0 (37.25) | 2 393 | 71.4 % |
//!
//! `k`/`v` are the worst-utilised GEMMs in the step by a factor of six. The
//! byte model at `M = 16` is `K·N + (K/128)·(N/128)·4`, so one `k`/`v` node
//! moves **5.24 MB** — a weight read that cannot amortise a launch. At a
//! 128-wide N tile that node is `8 × ceil(16/128) = 8` tiles on **132 SMs**:
//! 124 SMs idle for the whole launch, twice per layer. Appended onto
//! `q_proj`'s 96 tiles they ride a wave that is already running, and the two
//! launches disappear.
//!
//! At the table's 60 % target the three cost
//! `10.72 GB / (0.60 × 3.35 TB/s) = 543 µs` against a measured 970.9 µs →
//! **428 µs/step, 2.2 % of the step**, rank **5** of the round-13 decode
//! table (`§C.4`), behind gate+up (1 476 µs, shipped), paged-decode split-K
//! (986 µs), GDN state decode (945 µs) and `o_proj` (400 µs).
//!
//! # Numerics: a bit claim, not a tolerance
//!
//! The fused weight is the three `[N_i, K]` E4M3 blocks appended along N, and
//! its `[N/128, K/128]` FP32 block-scale grid is the three grids appended the
//! same way. Splitting N produces **independent output columns over the same
//! K with the same scales**: fused element `(m, j)` is the same dot product,
//! in the same order, as `q (m, j)` for `j < q_proj_dim`, `k (m, j −
//! q_proj_dim)` next and `v` above that. Same cuBLASLt op, same `ldc`, same
//! FP32 epilogue. `examples/native_fp8_attn_qkv_fused_microtest.rs` asserts
//! **byte equality** of the three slices at `M ∈ {5, 8, 16}` — and of the two
//! consumers that read them, `deinterleave_qg` and the KV-cache write.
//!
//! # Layout: N-concatenation, and it needs no new kernel at all
//!
//! `qkv_output` is ALREADY `[n, per_seq_qkv]` with Q at column 0, K at column
//! `q_proj_dim` and V after it, and `per_seq_qkv / 2 == q_proj_dim +
//! 2*kv_dim` by construction (`multi_seq::ctx`). So the fused GEMM's
//! `[m, fused_n]` output at `ldc = per_seq_qkv / 2` IS that layout byte for
//! byte, and `fused_n == ldc`:
//!
//! * `deinterleave_qg` already takes the row stride and already runs over
//!   `qkv_buf` — no strided variant, unlike the gate+up fusion's
//!   `silu_mul_strided`;
//! * the KV-cache write (`reshape_and_cache_flash*`, and the FP8 KV path with
//!   the #919 calibration window) already reads K and V at their column
//!   offsets inside the same rows;
//! * the paged decode kernel already reads Q at column 0.
//!
//! That is the whole layout argument, and it is why the fused arm adds ZERO
//! kernels to any hardware tree. An interleaved layout would have needed all
//! three consumers changed. The NVFP4 wide-verify path in
//! `trait_impl::multi_seq::qkv` has fused on exactly this identity since #915
//! (`fused_n`, `ATLAS_NO_FUSED_QKV`); this is its W8A8 twin.
//!
//! # Residency: net zero, by construction
//!
//! A second copy of q/k/v is 73.4 MB × 16 full-attention layers = **1.17 GB**.
//! The loader does not make one: it builds the fused buffer, re-points
//! `q_weight`/`k_weight`/`v_weight` at VIEWS inside it, and
//! `Qwen35DenseWeightLoader::prune_after_load` releases the three source store
//! tensors the copy consumed. Steady-state delta is zero and
//! `predicted_residency` prices it as a difference, so the preflight ring fit
//! is unmoved.
//!
//! # Band, and what is NOT in it
//!
//! 5..=[`ATTN_QKV_FUSED_MAX_M`] rows — the W8A8 cuBLASLt arm's own band
//! ([`ops::DECODE_W8A8_ROWS`]), matching gate+up.
//!
//! * **`n = 1`** keeps the W8A16 GEMVs. §C.5 measures that arm at
//!   **2 567 GB/s = 76.6 % of HBM** across all 24.327 GB of weights — there is
//!   no launch headroom there to buy.
//! * **PREFILL** keeps its three GEMMs. Its W8A8 arm writes Q into
//!   `qkv_output` CONTIGUOUS `[m, q_proj_dim]` while K and V go to `ssm_qkvz`
//!   (`prefill_qkv_w8a8.rs`), so a fused N would need a scatter and a new
//!   capacity argument — a separate measurement, not a free ride. At M=4576
//!   these GEMMs are compute-bound anyway (§A.4).
//!
//! The lever is `[defaults] attn_qkv_fused`: **hopper `true`**, gb10 and b200
//! `false`. `ATLAS_ATTN_QKV_FUSED=0` kills it; on a target that declares
//! `false` the loader never builds the weight, so `=1` there arms an arm with
//! no `qkv_fp8_fused` and it declines.

use spark_runtime::buffers::ATTN_QKV_FUSED_MAX_M;

use crate::layers::ops;

/// Whether the compiled target arms the fused Q/K/V decode GEMM.
///
/// The target declares it (`kernels/<hw>/HARDWARE.toml` `[defaults]
/// attn_qkv_fused`); `ATLAS_ATTN_QKV_FUSED` overrides it under the 2026-09-11
/// grammar, so `=0`/`=off`/`=false` turn it off and anything else turns it on.
/// There is no `ATLAS_NO_*` legacy spelling: the lever is new, so no script
/// predates the grammar and none can be surprised by it.
///
/// Read ONCE per layer at construction (`Qwen3AttentionLayer::attn_qkv_fused`)
/// and once per model in the residency prediction — never per step, so it
/// cannot vary across CUDA-graph replays.
pub fn attn_qkv_fused() -> bool {
    ops::target_defaults::resolved().attn_qkv_fused.value
}

/// The fused output width: `q_proj_dim + 2 * kv_dim` BF16 columns.
///
/// SSOT for the loader's concat extent, the dispatch site's plan and the
/// microtest's slice offsets. `q_proj_dim` is already doubled when the layer
/// is gated (`[Q|gate]`), which is what makes this 14336 and not 8192 on
/// Qwen3.8-27B.
pub fn fused_n(q_proj_dim: u32, kv_dim: u32) -> u32 {
    q_proj_dim + 2 * kv_dim
}

/// Every clause of the fused-arm shape rule that is knowable WITHOUT a
/// checkpoint — so the loader, the residency prediction and the dispatch site
/// ask one question.
///
/// Clauses, each load-bearing:
///
/// * `q_proj_dim % 128 == 0` and `kv_dim % 128 == 0` — the concat appends the
///   `[N/128, K/128]` scale grids, which is only the fused grid when BOTH
///   seams fall on a block boundary (`ceil` of a sum is not the sum of the
///   `ceil`s). It is also what `decode_w8a8_selected` will demand of the fused
///   `n` itself, transitively.
/// * `k % 128 == 0` — the same grid, on the contract side.
/// * both widths non-zero — a degenerate head count would make the concat a
///   no-op and the views alias.
pub fn qkv_fused_shape_ok(q_proj_dim: u32, kv_dim: u32, k: u32) -> bool {
    q_proj_dim > 0
        && kv_dim > 0
        && k > 0
        && q_proj_dim.is_multiple_of(128)
        && kv_dim.is_multiple_of(128)
        && k.is_multiple_of(128)
}

/// The whole fused-arm selection rule, as a pure function.
///
/// Split out from the layer for the reason [`ops::decode_w8a8_selected`] is:
/// the CPU tests pin every clause without a `ForwardContext`, and `lever` is
/// injected because the process-global `OnceLock` behind [`attn_qkv_fused`]
/// cannot be toggled per test.
///
/// Clauses beyond [`qkv_fused_shape_ok`]:
///
/// * `lever` — the target's declaration, environment-overridable.
/// * `fused_installed` — the loader built the `[fused_n, K]` weight. Absent on
///   any checkpoint or route the fusion was not built for, which is the only
///   thing that makes this arm optional at runtime.
/// * `w8a8_selected` — the caller's THREE-GEMM answer for this step. The fused
///   arm IS that arm with a wider N, so if any of q/k/v would not have taken
///   it, fusing would change that projection's ARITHMETIC and not just its
///   launch count. It carries every clause of `decode_w8a8_selected`
///   transitively: family, kill switch, band, format, `%128`, the scratch and
///   the write extent.
/// * `ldc == fused_n` — the identity the whole layout argument rests on. If a
///   config ever made `per_seq_qkv` wider than `[Q|K|V]`, the fused GEMM's
///   contiguous columns would no longer be the slot layout and the K/V
///   consumers would read the wrong offsets. An equality and not a `>=`: too
///   wide is as wrong as too narrow here, and declining is always sound.
/// * `5..=ATTN_QKV_FUSED_MAX_M` — the band, see the module docs. Redundant
///   with `w8a8_selected`'s own band today and stated anyway, because the two
///   constants live in different crates and a future widening of one must not
///   silently widen this arm.
pub fn attn_qkv_fused_selected(
    rows: usize,
    q_proj_dim: u32,
    kv_dim: u32,
    k: u32,
    ldc: u32,
    lever: bool,
    fused_installed: bool,
    w8a8_selected: bool,
) -> bool {
    lever
        && fused_installed
        && w8a8_selected
        && qkv_fused_shape_ok(q_proj_dim, kv_dim, k)
        && ldc == fused_n(q_proj_dim, kv_dim)
        && (5..=ATTN_QKV_FUSED_MAX_M).contains(&rows)
}

#[cfg(test)]
#[path = "attn_qkv_fused_tests.rs"]
mod tests;
