// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 serving predicates (`ATLAS_EXL3_NATIVE=1`): the gate, the
//! natively-served prefix set and the compiled-kernel envelope. Child module
//! of `exl3_materialize.rs`, split out for the ≤500 LoC cap; re-exported from
//! there so the public paths (`weight_map::exl3_native_*`) are unchanged.

use spark_runtime::weights::exl3::{Exl3Codebook, Exl3Weight};

// ── Native EXL3 serving (ATLAS_EXL3_NATIVE=1) ────────────────────────
//
// Milestone 1 keeps a SELECTED set of non-expert trellis linears packed in
// the store (skip the BF16 rewrite AND the frees) and serves them through
// the fused `exl3_matmul` kernels. Everything outside that set — experts,
// GDN projections, attention, the ViT tower — still materializes exactly as
// before, because their consumers (concat/shard/quantize pipelines, the ViT
// loader) read BF16 `.weight` tensors and have not been routed yet.
//
// The three functions below are the SINGLE source of truth for the decision;
// `factory/build.rs` re-derives "was this kept?" through the same predicates,
// so the materialize pass and the model builder can never disagree.

/// `ATLAS_EXL3_NATIVE=1`: serve supported trellis linears natively instead
/// of materializing them. Read per call — this only runs on load paths.
pub fn exl3_native_enabled() -> bool {
    std::env::var("ATLAS_EXL3_NATIVE").as_deref() == Ok("1")
}

/// The natively-served set: the LM head, plus — when `ATLAS_EXL3_NATIVE_MOE=1`
/// — the routed experts (`.mlp.experts.N.{gate,up,down}_proj`; see
/// `exl3_materialize_moe.rs` for the exclusions: `mtp.*` and the shared
/// expert keep materializing), plus — when `ATLAS_EXL3_NATIVE_DENSE=1` — the
/// GDN (`linear_attn.{in_proj_qkv,in_proj_z,out_proj}`) and attention
/// (`self_attn.{q,k,v,o}_proj`) dense families (see
/// `exl3_materialize_dense.rs`; `mtp.*`, the QSA indexer and the shared
/// expert keep materializing).
///
/// `lm_head` is the single largest dense tensor (`[vocab, hidden]`, ~1.27 GB
/// BF16 on Qwen3.8-Flash-Next vs ~325 MB packed at K=4) and its dispatch is
/// concentrated in three model-level functions, so it is the narrow path
/// that exercises the full native stack (bf16->f16 ingress, fused
/// suh/H128 rotation, trellis GEMV/GEMM, svh epilogue, f16->bf16 egress)
/// end to end. ViT expansion is tracked follow-up work: it needs its
/// layer-site dispatch routed before its prefix can join this set (a prefix
/// listed here without a serving path would fail at load with the
/// `dense_auto`/`quantized_any` native-mode probe error).
pub fn exl3_native_serves(prefix: &str) -> bool {
    exl3_native_serves_with(
        prefix,
        crate::weight_map::exl3_native_moe_enabled(),
        crate::weight_map::exl3_native_dense_families(),
    )
}

/// Env-independent body of [`exl3_native_serves`] (tests and the
/// materialize-impl thread the gates explicitly — `set_var` in parallel unit
/// tests races).
pub fn exl3_native_serves_with(
    prefix: &str,
    native_moe: bool,
    dense: crate::weight_map::Exl3DenseFamilies,
) -> bool {
    prefix == "lm_head"
        || (native_moe && crate::weight_map::exl3_native_serves_moe(prefix))
        || crate::weight_map::exl3_native_serves_dense(prefix, dense)
}

/// Whether the vendored kernel set can serve this tensor: a compiled
/// codebook (cb0/"3inst" is not instantiated — no shipped checkpoint uses
/// it) and the 128-divisible geometry the fused rotation assumes
/// (guaranteed by the format for quantized tensors, but checked rather
/// than trusted).
///
/// K is restricted to {2, 4} — the set the GEMV path serves at ANY
/// `rows <= 8` call. This is a CONCURRENCY invariant, not a kernel gap:
/// K in {3, 5, 6, 8} would route small-row projections to the split-K GEMM,
/// whose shared locks buffer is only safe because today's `rows > 8`
/// callers never overlap across streams — but small-row projections DO run
/// concurrently under co-dispatched prefill finalize. Kernels for the other
/// K are compiled and parity-proven; widening this set requires either
/// per-stream locks buffers or a GEMV envelope that covers those K at
/// small rows (review finding, 2026-09-01).
pub fn exl3_native_supported(w: &Exl3Weight) -> bool {
    matches!(w.k_bits, 2 | 4)
        && matches!(w.cb, Exl3Codebook::Mcg | Exl3Codebook::Mul1)
        && w.in_dim.is_multiple_of(128)
        && w.out_dim.is_multiple_of(128)
}
