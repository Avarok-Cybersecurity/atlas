// SPDX-License-Identifier: AGPL-3.0-only

//! Which derived weight copies the native-FP8 dense route actually needs, and
//! a tally of the ones it still builds.
//!
//! **WHY (#915 root cause; O9 of the 2026-09-05 rental; refs #916, #917, #736).**
//! Measured on 1xH100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8` at `5f78270dc`,
//! native-FP8 profile (`ATLAS_DENSE_FP8=1`, `--lm-head-dtype bf16`):
//!
//! * the checkpoint itself is **28.75 GB** — `WeightStore after prune: 1606
//!   tensors, 28.747 GiB still resident`, one ledger site
//!   (`fast_weights/mod.rs:434`, 29,436.7 MB x1606);
//! * the ledger nevertheless reported **58.1 GB live before the KV cache was
//!   sized**, and `ATLAS_MEM_PROFILE` recorded GPU-free falling **437 MB per
//!   layer over all 64 layers (28.0 GB)** *after* the checkpoint was resident;
//! * the teardown sweep reclaimed **28.01 GB across 1,980 allocations that no
//!   `ModelResource` owned**.
//!
//! The six sites the ledger named, and what each holds per layer:
//!
//! | site | bytes x count | tensor |
//! |---|---|---|
//! | `weight_map/loaders_fp8.rs:229` | 8,960 MB x256 | `quantize_to_nvfp4` packed `[N,K/2]` |
//! | `weight_map/quantized.rs:261` | 8,960 MB x256 | the transposed twin of the same |
//! | `weight_loader/qwen35_dense.rs:135` | 3,840 MB x48 | SSM fused `[QKV\|Z]` FP8 concat |
//! | `weight_map/quantized.rs:643` | 1,600 MB x64 | attention `Fp8WeightTransposed::weight_t` |
//! | `weight_map/loaders_fp8.rs:230` | 1,120 MB x256 | the NVFP4 per-16 group scales |
//! | `weight_map/quantized.rs:262` | 1,120 MB x256 | the transposed twin of those scales |
//!
//! The 256 counts are `64 layers x 3` dense-FFN projections (gate/up/down,
//! 42.5 MiB packed each) plus `16 attention layers x 4` (q/k/v/o, 25/6.25/
//! 6.25/12.5 MiB) — 8,160 + 800 MB, which is exactly the 8,960 reported.
//!
//! **None of the NVFP4 copies is reachable once the native FP8 overlay is
//! installed.** `DenseFfnLayer::forward` (`dense_ffn.rs:1044`) and
//! `forward_prefill_inner` (`dense_ffn.rs:1985`) both return from inside their
//! `if let Some(ref fp8w) = self.fp8_weights` arm, `forward_k2`/`forward_k3`/
//! `forward_km` redirect to `forward_prefill` via
//! `native_small_batch_uses_prefill`, and the `w8_gemm!` macro binds
//! `gate_t`/`up_t`/`down_t` to a literal `None`, so even the W8A16 fallback
//! rungs read the original `[N,K]` E4M3 bytes. The NVFP4 gate/up/down and
//! their transposed twins are pure load-time waste: 18.4 GiB of the 28.
//!
//! **The loader runs before dispatch exists**, so the route has to be derived
//! from the same resolvers the forward pass uses rather than from a
//! `ForwardContext`:
//!
//! * [`crate::layers::ops::GemmDispatch::from_env`] — the exact constructor
//!   `model/impl_a1.rs:836` calls to build the context. It is a pure function
//!   of the environment and a serve never rewrites its own environment, so the
//!   loader's answer and the context's answer cannot disagree.
//!
//! `ATLAS_FFN_W8A16_ONLY` is deliberately NOT an input: it steers the dense FFN
//! from the W8A8 arm onto rung 5 of the same `w8_gemm!` match, whose transposed
//! operand is that literal `None` — so it selects between two kernels that both
//! read the original FP8 bytes, and cannot resurrect an NVFP4 reader.
//!
//! That is the contract: **any new lever that can route a native-FP8 layer back
//! onto an NVFP4 kernel must be added to [`DenseFp8Plan::resolve`] as well as
//! to the dispatch site**, or the loader will have freed the weight that site
//! wants. `ATLAS_DENSE_FP8_KEEP_NVFP4` is the escape hatch that restores the
//! pre-#915 behaviour wholesale while such a gap is diagnosed.

use crate::layers::ops::GemmDispatch;
use crate::layers::qwen3_attention::Fp8TwinSet;

/// The per-layer kill switch that restores the pre-#915 behaviour: build every
/// NVFP4 fallback copy even where the dispatch cannot reach it.
///
/// PRESENCE (any value, including empty), matching `ATLAS_FFN_W8A16_ONLY` —
/// this is an escape hatch an operator reaches for while a serve is
/// misbehaving, and `ATLAS_DENSE_FP8_KEEP_NVFP4=0` meaning "on" is a trap.
pub fn keep_nvfp4_fallback() -> bool {
    static KEEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEEP.get_or_init(|| std::env::var_os("ATLAS_DENSE_FP8_KEEP_NVFP4").is_some())
}

/// What a native-FP8 dense layer must materialise beyond the checkpoint bytes.
///
/// Every field is "build this derived copy": `false` means the selected
/// kernels provably never read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseFp8Plan {
    /// NVFP4 gate/up/down **and** their `w4a16_gemm_t_m128` transposed twins.
    pub ffn_nvfp4: bool,
    /// NVFP4 q/k/v/o, their transposed twins, and the fused `[q|k|v]` twin.
    pub attn_nvfp4: bool,
    /// Which `Fp8WeightTransposed` twins `transpose_fp8_for_prefill` builds.
    pub attn_fp8_twins: Fp8TwinSet,
}

/// The inputs [`DenseFp8Plan::resolve`] decides from. Taken as a struct so the
/// CPU decision-table test can pin every clause without touching the process
/// environment (the `OnceLock` resolvers cannot be toggled per test).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseFp8Inputs {
    /// The native FP8 dense-FFN overlay will be installed on this layer
    /// (`ATLAS_DENSE_FP8=1`, tp_size 1, `Fp8Dequanted`, gate_proj native FP8).
    pub ffn_fp8: bool,
    /// The native FP8 attention overlay will be installed on this layer.
    pub attn_fp8: bool,
    /// `ATLAS_DENSE_FP8_KEEP_NVFP4` — restore the pre-#915 behaviour.
    pub keep_nvfp4: bool,
    /// Resolved exactly as `model/impl_a1.rs:836` resolves it.
    pub dispatch: GemmDispatch,
    /// Whether the target's `per_token_group_quant_fp8` + `fp8_gemm_t_blockscaled`
    /// kernels are both loaded. Both are required by the W8A8 prefill arm in
    /// `qwen3_attention/prefill/paged_qkv.rs:220` and
    /// `prefill/paged_oproj.rs:94`; without them those two chains fall through
    /// to the transposed W8A16 kernels, which read the FP8 twins.
    pub w8a8_kernels: bool,
    /// `ATLAS_ATTN_W4A4` is set. `prefill/paged_oproj.rs:38-42` builds its W4A4
    /// arm with NO weight-type predicate and then feeds it
    /// `&self.attn.o_proj` — the NVFP4 o_proj — so this one lever keeps the
    /// NVFP4 attention weights alive even under a full FP8 overlay. (The QKV
    /// side at `paged_qkv.rs:51` does check `as_nvfp4()`, so it is already
    /// closed; the o_proj asymmetry is not.)
    pub attn_w4a4: bool,
    /// `ATLAS_ATTN_PREFILL_Q_T=1`. `prefill/cache_skip_qkv.rs:142` reads it per
    /// projection per prefill and, when set, dispatches Q through `q_fp8w_t`.
    pub attn_prefill_q_t: bool,
}

impl DenseFp8Plan {
    /// The decision table. Pure — no environment reads, no allocation.
    ///
    /// * **FFN NVFP4** — unreachable the moment `set_fp8_weights` runs (see the
    ///   module docs: every entry point returns from inside the FP8 arm and the
    ///   `w8_gemm!` transposed operands are a literal `None`). Built only under
    ///   the escape hatch.
    /// * **Attention NVFP4** — `set_fp8_weights` *overwrites*
    ///   `q_weight`/`k_weight`/`v_weight`/`o_weight` with `QuantWeight::Fp8`,
    ///   so the base NVFP4 weights are orphaned at load. The transposed and
    ///   fused twins survive only behind the `ATLAS_CUTLASS_NVFP4_*` levers,
    ///   which default off.
    /// * **Attention FP8 twins, K and V** — KEPT UNCONDITIONALLY. The
    ///   first prefill chunk (`seq_len_start == 0`, the default for every
    ///   request) does not go through `paged_qkv.rs` at all: it goes through
    ///   `prefill/cache_skip_qkv.rs`, whose dispatch chain has **no W8A8 arm**,
    ///   so `k_fp8w_t`/`v_fp8w_t` are dereferenced at `cache_skip_qkv.rs:218`
    ///   / `:235` on every request regardless of `fp8_blockscaled_prefill`.
    ///   Freeing them is a NULL-pointer kernel launch on the first token.
    /// * **Attention FP8 twins, Q and O** — Q on that same chain is behind
    ///   `ATLAS_ATTN_PREFILL_Q_T=1` (`cache_skip_qkv.rs:142`) and O is routed
    ///   to `paged_oproj.rs` from both chains, so both are reachable only after
    ///   the W8A8 arm declines — block-scaled prefill off, or a target missing
    ///   one of the two kernels.
    pub fn resolve(i: DenseFp8Inputs) -> Self {
        if i.keep_nvfp4 {
            return Self {
                ffn_nvfp4: true,
                attn_nvfp4: true,
                attn_fp8_twins: if i.attn_fp8 {
                    Fp8TwinSet::ALL
                } else {
                    Fp8TwinSet::NONE
                },
            };
        }
        let cutlass_nvfp4_attn = i.dispatch.cutlass_nvfp4_gemm
            || i.dispatch.cutlass_nvfp4_attn_q
            || i.dispatch.cutlass_nvfp4_attn_kv
            || i.dispatch.cutlass_nvfp4_attn_o;
        // `transpose_fp8_for_prefill` already refuses to build anything under
        // the umbrella NVFP4 flag (`prefill_weights.rs:357`); mirror that here
        // so the plan and the builder cannot drift.
        let w8a8_covers_prefill = i.dispatch.fp8_blockscaled_prefill && i.w8a8_kernels;
        let fp8_twins = if !i.attn_fp8 || i.dispatch.cutlass_nvfp4_gemm {
            Fp8TwinSet::NONE
        } else {
            Fp8TwinSet {
                q: i.attn_prefill_q_t || !w8a8_covers_prefill,
                k: true,
                v: true,
                o: !w8a8_covers_prefill,
            }
        };
        Self {
            ffn_nvfp4: !i.ffn_fp8,
            attn_nvfp4: !i.attn_fp8 || cutlass_nvfp4_attn || i.attn_w4a4,
            attn_fp8_twins: fp8_twins,
        }
    }
}

/// The environment-resolved half of [`DenseFp8Inputs`], read ONCE per load.
///
/// Hoisted out of the per-layer decision because `GemmDispatch::from_env`
/// walks the environment block for a dozen variables and a 64-layer model
/// would otherwise do it 64 times — and because resolving once is what makes
/// "every layer of this model took the same route" a property of the type
/// rather than of the environment holding still.
#[derive(Clone, Copy, Debug)]
pub struct RouteEnv {
    pub keep_nvfp4: bool,
    pub dispatch: GemmDispatch,
    /// Mirrors `prefill/paged_oproj.rs:42` and `prefill/paged_qkv.rs:53`,
    /// which read this per projection per prefill and are NOT memoised.
    pub attn_w4a4: bool,
    /// Mirrors `prefill/cache_skip_qkv.rs:142`, same caveat.
    pub attn_prefill_q_t: bool,
}

impl RouteEnv {
    pub fn from_env() -> Self {
        Self {
            keep_nvfp4: keep_nvfp4_fallback(),
            dispatch: GemmDispatch::from_env(),
            // Same predicates as the dispatch sites, character for character:
            // `is_ok()` (presence) for W4A4, `== "1"` for the Q-transpose.
            attn_w4a4: std::env::var("ATLAS_ATTN_W4A4").is_ok(),
            attn_prefill_q_t: std::env::var("ATLAS_ATTN_PREFILL_Q_T").ok().as_deref() == Some("1"),
        }
    }

    /// This layer's plan. `w8a8_kernels` is a property of the *layer* (its
    /// resolved kernel handles), which is why it is not part of `RouteEnv`.
    pub fn plan(&self, ffn_fp8: bool, attn_fp8: bool, w8a8_kernels: bool) -> DenseFp8Plan {
        DenseFp8Plan::resolve(DenseFp8Inputs {
            ffn_fp8,
            attn_fp8,
            keep_nvfp4: self.keep_nvfp4,
            dispatch: self.dispatch,
            w8a8_kernels,
            attn_w4a4: self.attn_w4a4,
            attn_prefill_q_t: self.attn_prefill_q_t,
        })
    }

    /// The NVFP4 half of the plan. Asked BEFORE the layer exists, so it may
    /// not depend on any layer-local kernel handle — pinned by
    /// `attn_nvfp4_does_not_depend_on_the_w8a8_kernels`.
    pub fn attn_nvfp4(&self, attn_fp8: bool) -> bool {
        self.plan(false, attn_fp8, true).attn_nvfp4
    }

    /// Which FP8 prefill twins this layer needs. `w8a8_kernels` is read off
    /// the constructed layer (`Qwen3AttentionLayer::has_w8a8_prefill_kernels`).
    pub fn attn_fp8_twins(&self, attn_fp8: bool, w8a8_kernels: bool) -> Fp8TwinSet {
        self.plan(false, attn_fp8, w8a8_kernels).attn_fp8_twins
    }
}

/// Bytes one NVFP4 `QuantizedWeight` costs: packed `[N, K/2]` E2M1 nibbles
/// plus the `[N, K/16]` per-group scale byte. Mirrors `quantize_to_nvfp4`
/// (`weight_map/loaders_fp8.rs:229`-`230`) and `transpose_for_gemm_gs`
/// (`weight_map/quantized.rs:261`-`262`), which allocate the same two sizes.
pub fn nvfp4_bytes(n: usize, k: usize) -> usize {
    n * k / 2 + n * k / 16
}

/// What the pre-#915 loader spent per dense-FFN layer on NVFP4: gate, up and
/// down, each with a transposed twin of identical size.
pub fn dense_ffn_nvfp4_bytes(hidden: usize, inter: usize) -> usize {
    // gate/up are [inter, hidden] and down is [hidden, inter] — the same
    // element count, so all three cost the same.
    3 * 2 * nvfp4_bytes(inter, hidden)
}

/// What the pre-#915 loader spent per full-attention layer on NVFP4: q/k/v/o,
/// each with a transposed twin, plus the fused `[q|k|v]` transposed twin
/// (`transpose_concat_for_gemm`).
///
/// `q_n` is `num_attention_heads * head_dim`, doubled when `attn_gated`;
/// `kv_n` is `num_key_value_heads * head_dim`; `o_k` is the o_proj contraction
/// width `num_attention_heads * head_dim`.
pub fn attn_nvfp4_bytes(q_n: usize, kv_n: usize, o_k: usize, hidden: usize) -> usize {
    let qkv = nvfp4_bytes(q_n, hidden) + 2 * nvfp4_bytes(kv_n, hidden);
    let o = nvfp4_bytes(hidden, o_k);
    // base + per-projection twin + the fused q|k|v twin
    2 * (qkv + o) + nvfp4_bytes(q_n + 2 * kv_n, hidden)
}

/// Bytes one FP8 `[K, N]` transposed twin costs: the E4M3 bytes plus the
/// transposed `[K/128, N/128]` FP32 block-scale grid. Mirrors
/// `Fp8Weight::transpose_for_gemm` (`weight_map/quantized.rs:643`/`:660`).
pub fn fp8_twin_bytes(n: usize, k: usize) -> usize {
    n * k + n.div_ceil(128) * k.div_ceil(128) * 4
}

/// The FP8 prefill twins `want` selects, for one full-attention layer.
pub fn attn_fp8_twin_bytes(
    want: Fp8TwinSet,
    q_n: usize,
    kv_n: usize,
    o_k: usize,
    hidden: usize,
) -> usize {
    let mut b = 0;
    if want.q {
        b += fp8_twin_bytes(q_n, hidden);
    }
    if want.k {
        b += fp8_twin_bytes(kv_n, hidden);
    }
    if want.v {
        b += fp8_twin_bytes(kv_n, hidden);
    }
    if want.o {
        b += fp8_twin_bytes(hidden, o_k);
    }
    b
}

/// What ONE fused dense-FFN gate+up weight costs: the two `[inter, hidden]`
/// E4M3 blocks appended along N, plus their two `[inter/128, hidden/128]` FP32
/// block-scale grids appended the same way (#927).
///
/// The sum of the two grids and NOT one grid over the fused N, for the reason
/// `predicted_residency::ssm_concat_bytes` gives: `ceil` of a sum is not the
/// sum of the `ceil`s, and the concat copies the two grids side by side.
///
/// RESIDENCY-NEUTRAL: `prune_after_load` releases the two `[inter, hidden]`
/// store tensors this copied, so the same number is also what the checkpoint
/// gives back. Both sides are priced from this one function.
pub fn ffn_gateup_fused_bytes(hidden: usize, inter: usize) -> usize {
    let (w, s) = ffn_gateup_fused_parts(hidden, inter);
    w + s
}

/// [`ffn_gateup_fused_bytes`] split into `(weight bytes, scale-grid bytes)` —
/// the loader adopts the two buffers separately, so it needs the terms rather
/// than the sum, and taking them from here is what keeps the prediction and
/// the tally one arithmetic.
pub fn ffn_gateup_fused_parts(hidden: usize, inter: usize) -> (usize, usize) {
    (
        2 * inter * hidden,
        2 * (inter.div_ceil(128) * hidden.div_ceil(128) * 4),
    )
}

/// Running tally of the derived (non-checkpoint) device bytes this loader
/// allocated, and of the ones it decided not to build.
///
/// Counted in the loader rather than read back from the ledger because the
/// ledger cannot tell a derived copy from a checkpoint tensor — both are plain
/// `gpu.alloc` — and because the "not built" number is the one the fix is
/// judged on and has no allocation to read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DerivedResidency {
    /// Derived bytes still resident at the end of load.
    pub kept: u64,
    /// Derived bytes the plan declined to build, versus the pre-#915 loader.
    pub skipped: u64,
    /// Transient derived bytes allocated and freed during load.
    pub freed: u64,
    /// Which twin families were built, for the one-line summary.
    pub twins: TwinsBuilt,
}

/// Which derived twin families the plan built. Reported by name so the serve
/// log says *which* copies are resident rather than only how many bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TwinsBuilt {
    pub ffn_nvfp4: bool,
    pub attn_nvfp4: bool,
    pub attn_fp8: bool,
    pub ssm_fp8_concat: bool,
    /// The dense-FFN `[2*inter, hidden]` gate+up concat (#927). Named in the
    /// summary like the others, and worth naming even though it is residency-
    /// NEUTRAL: its bytes appear in `kept` while the two store tensors they
    /// replace disappear from `WeightStore::resident_bytes` at the prune, so a
    /// reader comparing two serve logs needs to know which of the two numbers
    /// moved and why.
    pub ffn_gateup_fused: bool,
}

impl TwinsBuilt {
    /// `none`, or a comma-separated list, for the summary line.
    pub fn describe(self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.ffn_nvfp4 {
            parts.push("ffn-nvfp4+t");
        }
        if self.attn_nvfp4 {
            parts.push("attn-nvfp4+t");
        }
        if self.attn_fp8 {
            parts.push("attn-fp8-t");
        }
        if self.ssm_fp8_concat {
            parts.push("ssm-qkvz-fp8");
        }
        if self.ffn_gateup_fused {
            parts.push("ffn-gateup-fp8");
        }
        if parts.is_empty() {
            "none".to_owned()
        } else {
            parts.join(", ")
        }
    }
}

impl DerivedResidency {
    pub fn keep(&mut self, bytes: usize) {
        self.kept += bytes as u64;
    }

    pub fn skip(&mut self, bytes: usize) {
        self.skipped += bytes as u64;
    }

    pub fn free(&mut self, bytes: usize) {
        self.freed += bytes as u64;
    }

    /// The line the next H100 run is read against.
    ///
    /// Emitted at the end of `load_layers` so a serve log proves the residency
    /// without an `ATLAS_MEM_PROFILE` rerun: `weights` is the checkpoint,
    /// `derived` is everything this loader built on top of it, and `skipped`
    /// is what the pre-#915 loader would have built and this one did not.
    pub fn summary(&self, weight_bytes: usize) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        format!(
            "native FP8 dense residency: weights {:.2} GB, derived {:.2} GB \
             (twins: {}), freed {:.2} GB, not built {:.2} GB",
            weight_bytes as f64 / 1e9,
            gb(self.kept),
            self.twins.describe(),
            gb(self.freed),
            gb(self.skipped),
        )
    }
}

#[cfg(test)]
#[path = "fp8_residency_tests.rs"]
mod tests;
