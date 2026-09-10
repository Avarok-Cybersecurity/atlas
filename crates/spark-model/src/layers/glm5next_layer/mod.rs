// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextLayer` — the composite GLM-5.3 decoder layer that implements `TransformerLayer`.
//!
//! This is the piece that makes the model *bind*. Everything it dispatches to already existed
//! and was numerically gated in Slices 1–13; what did not exist was a single type the loader can
//! return 45 of, dispatching mixer (KDA | DSA) + MLP (dense | routed MoE) + mHC in the order
//! [`crate::layers::glm5next_skeleton`] records as data.
//!
//! # The residual plan, executed
//!
//! Per site, exactly `ResidualStep`'s order:
//!
//! ```text
//! layer 0 only:  hc_expand(hidden) -> streams        [hc_mult, hidden] FP32 highway
//!
//! attention site:  hc_pre(streams) -> y, post, comb
//!                  rms_norm_vanilla(y, input_layernorm) -> normed
//!                  mixer(normed) -> block_out
//!                  hc_post(block_out, residual = streams, post, comb) -> streams
//!
//! FFN site:        the same, with post_attention_layernorm and the MLP
//!
//! last layer:    hc_head_mean(streams) -> hidden     UNWEIGHTED mean, no parameters
//! ```
//!
//! 🪤 **`hc_pre` does not modify `streams`.** That is what makes the skeleton's
//! `ResidualStep::SaveResidual` free here — `hc_post` reads the same buffer as its residual and
//! writes back over it. Snapshotting is only needed if something overwrites the highway between
//! the two calls; nothing here does, and the ordering below is the guard.
//!
//! # 🪤 The traps this file holds
//!
//! * **GLM's norms are PLAIN RMSNorm.** `rms_norm_vanilla` is `x * rms * w`; the other
//!   `rms_norm` is `x * rms * (1 + w)`. Identical signatures, identical shapes, and picking the
//!   wrong one is silent. Every norm here takes the vanilla entry point.
//! * **The mHC head collapse is an UNWEIGHTED MEAN.** GLM's `Glm5NextTextHyperHead` has no
//!   parameters and the checkpoint carries zero `hc_head` tensors, unlike DeepSeek-V4's learned
//!   sigmoid-weighted sum. Reaching for `ops::hc_head` would look for weights that do not exist.
//! * **The highway is indexed by TOKEN.** Prefill is overridden rather than left to the trait's
//!   sequential default, because that default runs every token through layer 0 before layer 1 —
//!   which with a single-slot highway would leave only the LAST token's streams alive. See
//!   `Glm5NextLayer::prefill`.
//! * **Both MLP arms leave a PARTIAL SUM** whenever TP or EP is on. The single `all_reduce` at
//!   the end of the FFN site covers both, and it must happen *before* `hc_post` mixes the output
//!   back into the highway.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{AttnMetadataDev, ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::glm5next_dsa;
use crate::layers::glm5next_dsa::layer::Glm5NextDsaLayer;
use crate::layers::glm5next_dsa::state::Glm5NextDsaState;
use crate::layers::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaLayer, Glm5NextKdaWorkspace, KdaSeqState,
};
use crate::layers::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_dense, forward_moe};
use crate::layers::glm5next_mlp::weights::{Glm5NextDenseMlpWeights, Glm5NextMoeWeights};
use crate::layers::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
// 🪤 Through `ops`'s glob re-export, and by the GLM-prefixed names only: `ops` also exports
// DeepSeek-V4's `hc_pre`/`hc_post`, which use a different mixing law and a different weight set.
// The rename is what makes reaching for the wrong one a compile error instead of a silent
// architecture swap.
use crate::layers::ops::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_expand, glm_hc_post, glm_hc_pre,
    hc_head_mean,
};

pub mod state;

pub mod profile;
pub use state::alloc_kda_ssm_state;

/// Which mixer this layer runs. Both halves already exist and are GPU-gated; this enum is the
/// dispatch, not new math.
///
/// 🪤 KDA's `decode`/`prefill` are **inherent** methods with their own signature, not the
/// `TransformerLayer` ones — they take a `&KdaSeqState` and a workspace and leave the result in
/// `ws.final_out`. DSA's `decode` *is* the trait method and writes its result back into the
/// buffer it was handed. The two conventions differ; this is where they are reconciled.
pub enum Glm5NextMixer {
    Kda {
        layer: Box<Glm5NextKdaLayer>,
        /// Shared across every KDA layer — all 34 have identical geometry, so one workspace
        /// serves them all. `Arc` because the layers are independent owners.
        ws: Arc<Glm5NextKdaWorkspace>,
        cfg: Glm5NextKdaConfig,
    },
    Dsa(Box<Glm5NextDsaLayer>),
}

/// Which MLP this layer runs. Layers `0..first_k_dense_replace` are dense; the rest route.
pub enum Glm5NextMlpSite {
    Dense(Glm5NextDenseMlpWeights),
    Moe(Box<Glm5NextMoeWeights>),
}

/// This layer's hyper-connection: both sites' weights plus the kernels and the two scalars.
pub struct Glm5NextMhc {
    pub kernels: Glm5NextMhcKernels,
    pub attn: Glm5NextMhcSiteWeights,
    pub ffn: Glm5NextMhcSiteWeights,
    pub hc_mult: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
}

/// One bound GLM-5.3 decoder layer.
pub struct Glm5NextLayer {
    pub layer_idx: usize,
    pub mixer: Glm5NextMixer,
    pub mlp: Glm5NextMlpSite,
    pub mlp_cfg: Glm5NextMlpConfig,
    pub mlp_kernels: Glm5NextMlpKernels,
    pub mlp_ws: Glm5NextMlpWorkspace,
    /// `None` only for a layer with no hyper-connection — i.e. the MTP layer, which carries zero
    /// `hc_*` tensors. Every text layer has one.
    pub mhc: Option<Glm5NextMhc>,
    /// `input_layernorm.weight` / `post_attention_layernorm.weight`, both plain RMSNorm.
    pub input_norm: DevicePtr,
    pub post_attn_norm: DevicePtr,
    /// 🪤 `rms_norm_vanilla`, never `rms_norm`. See the module header.
    pub rms_norm_k: KernelHandle,
    /// `bf16_add_inplace`, the residual add the MTP layer needs and the text layers do not:
    /// a text layer's residual lives in the mHC highway and `hc_post` folds the block output
    /// into it. `0` on a target without the kernel, which the MTP path refuses.
    pub add_k: KernelHandle,
    pub rms_eps: f32,
    pub hidden: usize,
    /// 🔴 Whether the MIXER output is a partial sum. Both mixers end in a **row-parallel**
    /// `o_proj` (`KdaShard::ChannelCols` / `DsaShard::HeadCols`), so at TP>1 each rank holds
    /// only part of the attention output and it must be all-reduced **before** `hc_post` folds
    /// it into the highway. Reducing after would mix a half-answer into every later layer's
    /// residual stream; not reducing at all is a plausible, wrong output with no shape error.
    pub mixer_all_reduce: bool,
    /// Expand the highway here. True for layer 0 only.
    pub is_first: bool,
    /// Collapse the highway here. True for the last TEXT layer only.
    pub is_last: bool,
}

/// Tokens per batched prefill sub-chunk — the width `Glm5NextLayer::prefill` hands
/// [`Glm5NextLayer::forward_k`].
///
/// 🔴 **16, and the ceiling is still a KERNEL boundary, not a bandwidth knee.** Every dense
/// projection on this path goes through [`ops::dense_mm_bf16`], whose batched-GEMV arm
/// (`dense_gemv_bf16_batchm`) is **bit-identical to M serial GEMVs** and stops at
/// `DENSE_GEMV_BATCHM_MAX_M`. Past it the same call falls to the tile GEMM, which both
/// reassociates (so prefill stops being bit-identical to the per-token walk) and is the slower
/// kernel at these widths — Atlas measured it 3.6x slower than the batched GEMV at M <= 8.
///
/// 🔴 WIDENED 8 -> 16 (2026-09-02). The A65 measurement below — "R = 32 is no faster" — was
/// TRUE AND MISATTRIBUTED. R = 32 lost because it left the batched GEMV for the tile GEMM, not
/// because row batching stops paying. With the tier itself widened to 16, the same 12 GLM
/// prefill shapes cost **1.36-1.98x less per token at M = 16 than at M = 8** with cold weights
/// (11 of 12; shallow-K N4096 K128 is the one loser at 0.77x), worth a modelled **-17.6 s of a
/// 173.9 s 9K TTFT**, bit-identical (`scripts/glm53-dense-bf16/bench_m16.cu`, spark-bench).
///
/// 🪤 The routed MoE does NOT follow the width up. `glm5next_mlp::forward::forward_moe` splits a
/// wider row group into even sub-groups of at most `MOE_ROW_BATCH_MAX_ROWS` — exact, because a
/// row's expert sum depends only on its own top-k, never on which rows share the sweep — so the
/// routed experts still amortize over 8 rows, not 16. Widening THAT is a separate and unmeasured
/// question: the union kernel is a single block with an O(T^3) scan, and the tier's register
/// cost was already 80 at R = 8.
///
/// 🔴 MEASURED 2026-08-31, one image, one control, ~1,950-token prompt: control TTFT 125.3 s;
/// R = 8 **43.4 s (2.89x) and byte-identical on all four probes**; R = 32 44.3 / 51.8 s — no
/// faster, and it moves two of the four completions. The per-token-bytes model that first
/// picked 32 (predicting 4.5x at R = 32 against 3.0x at R = 8) is REFUTED as a width law: it
/// modelled weight traffic only, and above R = 8 the traffic saved is handed to a slower kernel.
/// The routed experts do not amortize past R = 4 either way (`forward_moe`'s union arm caps
/// there), so R = 8 takes the whole available win. ANOMALIES A65.
pub(crate) const PREFILL_ROWS: usize = 16;

/// Route GLM prefill's KDA mixer through the chunked scan instead of the per-token
/// recurrent walk.
///
/// 🔴 DEFAULT ON. `ATLAS_GLM_KDA_CHUNK_PREFILL=0` is the kill-switch.
///
/// 🪤 THIS DEFAULT WAS FLIPPED ON EVIDENCE THAT REVERSED. When the arm first landed it
/// measured 90.5/85.5 tok/s against 94.2/88.6 — 4% SLOWER — and was kept default-off as a
/// recorded negative result. That verdict was real but it was measuring the LAUNCH, not
/// the algorithm: `kda_chunk_scan` ran at `BLOCK = 128` on a `grid(H)` of 32 blocks, i.e.
/// 4,096 threads for a whole layer on a 48-SM part. Splitting the `kda_mixer` bucket
/// three ways made it legible — chunked `kda_recur` was 227.84 ms/tok against the
/// recurrent walk's 123.68, so the chunked arm was 1.84x slower at the one thing it
/// exists to make cheaper. Widening that block to 1,024 (bit-identically — see
/// `CHUNK_SCAN_BLOCK`) took it to 87.86 ms/tok, below the recurrent walk, and the arm
/// flipped from a loss to a win:
///
///     recurrent            314.8 / 299.1 tok/s   (5.4K / 21K prompt)
///     chunked, BLOCK=128   280.1 / 267.6         -11%
///     chunked, BLOCK=1024  337.0 / 319.3         +7.1% / +6.8%
///
/// The lesson worth keeping: an arm measured slow while its launch geometry is wrong has
/// not been measured. Two separate "chunking is slower" results here were both artifacts.
///
/// 🪤 STILL NOT BIT-IDENTICAL to the recurrent walk — the chunked scan reassociates the
/// recurrence. That is why it stays confined to prefill (`is_prefill && k > 1 &&
/// snaps.is_empty()`, see the call site): decode and the speculative verify keep the
/// per-token path, so an accepted token still matches what decode would have produced.
/// Judged on long-context recall, not just throughput — a 700-row ledger needle at row
/// 431 is recalled exactly on this arm.
fn kda_chunk_prefill() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("ATLAS_GLM_KDA_CHUNK_PREFILL").as_deref() != Ok("0");
        tracing::warn!(
            "ATLAS_GLM_KDA_CHUNK_PREFILL: GLM prefill KDA uses the {}",
            if on {
                "CHUNKED scan (kda_chunk_prepare + kda_chunk_scan)"
            } else {
                "per-token recurrent walk"
            }
        );
        on
    })
}

/// Route GLM projections WIDER than the bit-exact tier through cuBLASLt BF16 instead of
/// the scalar tile GEMM. Shared by the KDA and DSA blocks.
///
/// 🔴 DEFAULT ON. `ATLAS_GLM_CUBLAS_PROJ=0` is the kill-switch.
///
/// `dense_gemm_bf16` is, by its own doc, a "scalar strict-order BF16 GEMM, no
/// reassociation" — correctness-first, one thread per output element, not a tensor-core
/// kernel. With `kda_mixer` split three ways it was the largest leaf in prefill by a wide
/// margin: `kda_front` 24.4% + `kda_back` 7.7% + `dsa_proj` 14.9% = 47% of prefill in
/// projections alone, against the per-token KDA recurrence — the assumed culprit — at
/// 4.6%. At 256 rows `kda_front` cost 19.8 ms per layer-call to move ~268 MB of weights,
/// ~20x off a 273 GB/s part.
///
/// 🪤 The M threshold is a NUMERICS boundary, not a tuning knob. `dense_mm_bf16` sends
/// M=1 to the GEMV and M=2..=16 to the batched single-sweep kernel; both accumulate per
/// row in FP32 in a fixed K order and are BIT-IDENTICAL. Only above
/// `DENSE_GEMV_BATCHM_MAX_M` does it fall to the tile GEMM, which already reassociates.
/// So this replaces one non-bit-exact path with another and leaves every bit-exact tier
/// alone: decode is M=1 and a speculative verify is M<=16, so neither can reach here, and
/// a verify still matches what decode produced.
pub(crate) fn cublas_wide_proj() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("ATLAS_GLM_CUBLAS_PROJ").as_deref() != Ok("0");
        tracing::warn!(
            "ATLAS_GLM_CUBLAS_PROJ: wide GLM projections (M > {}) use {}",
            crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M,
            if on {
                "cuBLASLt BF16"
            } else {
                "the scalar tile GEMM"
            }
        );
        on
    })
}

/// Batch the DSA indexer `wq_b` projection across all prefill rows in one cuBLASLt GEMM
/// instead of one M=1 GEMV per row. `ATLAS_GLM_DSA_BATCH_QIDX=0` is the kill-switch.
///
/// 🔴 DEFAULT ON. Prefill-only (see the call site in `glm5next_dsa::layer`), so decode and
/// the speculative verify keep their bit-exact M=1 GEMV.
pub(crate) fn dsa_batch_qidx() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| {
        let on = std::env::var("ATLAS_GLM_DSA_BATCH_QIDX").as_deref() != Ok("0");
        tracing::warn!(
            "ATLAS_GLM_DSA_BATCH_QIDX: prefill DSA indexer q uses {}",
            if on {
                "ONE batched cuBLASLt GEMM"
            } else {
                "one M=1 GEMV per row"
            }
        );
        on
    })
}

/// `PREFILL_ROWS`, overridable at launch with `ATLAS_GLM_PREFILL_ROWS`.
///
/// 🔬 Kept as the A/B lever it was built as. It found A65's real defect (the DSA attend read
/// `seq_lens[row]` / `block_tables[row]` out of a single-row buffer) by sweeping width against a
/// fixed control in ONE serve instead of one image per width. `1` restores the per-token walk
/// exactly — the `rows > 1` gate in `prefill` falls through to `forward_one`.
///
/// 🪤 Above `DENSE_GEMV_BATCHM_MAX_M` the dense projections leave the bit-identical batched-GEMV
/// arm for the tile GEMM, so a width past it is a NUMERICS change as well as a speed one. Keep
/// this lever at or below that constant.
///
/// 🪤 Read once and cached: an env read per layer per sub-chunk would sit in the hot loop.
/// `ATLAS_GLM_VERIFY_FUSED=1` selects the fused R-row verify body.
///
/// DEFAULT OFF, and the default is the per-sequence `forward_k` sweep, which is
/// the BIT-IDENTITY REFERENCE: each call sees exactly the rows, state, length
/// and page table the serial verify gave it.
///
/// 🔴 The fused body is FASTER and NOT YET CORRECT. Measured on GLM-5.3-Flash at
/// C=4/K=2 (R=8): the verify forward drops 313 ms -> 148.6 ms and C=4 throughput
/// rises 19.5 -> 23.1 tok/s, but a long-context needle probe that the
/// per-sequence sweep passes 4/4 scores 3/4 with it, and cross-contamination
/// goes 6/6 identical -> 5/6. Short-context known-answer probes are 4/4 either
/// way, so the defect needs the LONG-context probe to see — which is why this
/// ships opt-in rather than as the default it is fast enough to deserve.
fn verify_fused_rows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_VERIFY_FUSED").as_deref() == Ok("1"))
}

pub(crate) fn prefill_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let r = std::env::var("ATLAS_GLM_PREFILL_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|r| *r >= 1)
            .unwrap_or(PREFILL_ROWS);
        if r != PREFILL_ROWS {
            tracing::warn!("GLM prefill sub-chunk overridden to {r} rows (default {PREFILL_ROWS})");
        }
        r
    })
}

impl Glm5NextLayer {
    /// RMSNorm over `rows` contiguous `[hidden]` rows.
    ///
    /// 🪤 `rms_norm_vanilla`'s grid IS the token axis (`token = blockIdx.x`), so `rows > 1` is
    /// one launch doing exactly what `rows` launches would do, block for block — bit-identical,
    /// which is what lets a K-token verify take it.
    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([rows as u32, 1, 1])
            .block([(self.hidden.min(1024)) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(self.hidden as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;
        Ok(())
    }

    /// Run the mixer on `normed`, returning the pointer that holds its output.
    #[allow(clippy::too_many_arguments)]
    fn mixer_forward(
        &self,
        normed: DevicePtr,
        residual: DevicePtr,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                let ssm = self.kda_state(st)?;
                let kda = KdaSeqState {
                    conv: ssm.conv_state,
                    recurrent: ssm.h_state,
                };
                let t = profile::start();
                layer.decode(ctx.gpu, normed, &kda, ws, stream)?;
                profile::end(profile::KDA, t, ctx.gpu, stream);
                Ok(ws.final_out)
            }
            Glm5NextMixer::Dsa(layer) => {
                let dsa: &mut Glm5NextDsaState = self.dsa_state(st)?;
                layer.decode(
                    normed,
                    residual,
                    dsa,
                    kv_cache,
                    seq_len,
                    block_table,
                    disk_block_ids,
                    disk_offloaded,
                    ctx,
                    stream,
                )?;
                // 🪤 DSA writes its `o_proj` output back over the buffer it was handed.
                Ok(normed)
            }
        }
    }

    /// PROFILING ONLY: a 2-byte collective that both ranks must reach before either leaves.
    /// Charged to `bar`, it drains the per-call arrival jitter so the real reduce that follows
    /// measures network + kernel rather than "network + how late the other rank was".
    fn reduce_probe(&self, bar: usize, site: &str, ctx: &ForwardContext, stream: u64) {
        if !profile::on() {
            return;
        }
        let Some(comm) = ctx.comm else { return };
        let t = profile::start_hot();
        let p = profile::probe_buf(ctx.gpu);
        if p != 0 {
            let _ = comm.all_reduce_async(p, 2, stream);
        }
        let us = profile::end_us(bar, t, ctx.gpu, stream);
        profile::trace_bar(
            site,
            self.layer_idx,
            matches!(self.mlp, Glm5NextMlpSite::Moe(_)),
            us,
        );
    }

    /// `all_reduce(SUM)` a `[rows, hidden]` BF16 partial, when one is needed and a comm exists.
    ///
    /// 🪤 One collective over `rows` contiguous rows, not `rows` collectives: `all_reduce(SUM)`
    /// is linear and the rows are adjacent, so the result is identical and a K-token verify
    /// pays the latency once.
    fn reduce_partial(
        &self,
        p: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(comm) = ctx.comm {
            let bytes = rows * self.hidden * 2;
            if ctx.graph_capture {
                comm.all_reduce(p.0, bytes)?;
            } else {
                comm.all_reduce_async(p.0, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// Run the MLP on `rows` rows of `normed` into `out`, then reduce once if this rank holds
    /// only part of the result.
    ///
    /// The dense FFN and the shared expert sweep their weights ONCE for all rows; only the
    /// routed experts stay per-row, because their weight traffic genuinely scales with K (the
    /// measured expert union over K consecutive tokens is 8.00 / 13.74 / 18.76 at K = 1..3).
    fn mlp_forward(
        &self,
        normed: DevicePtr,
        out: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let t_dense = matches!(self.mlp, Glm5NextMlpSite::Dense(_))
            .then(profile::start)
            .flatten();
        match &self.mlp {
            Glm5NextMlpSite::Dense(w) => forward_dense(
                ctx.gpu,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                self.mlp_cfg.local_dense_intermediate,
                normed,
                out,
                rows,
                &self.mlp_ws,
                stream,
            )?,
            Glm5NextMlpSite::Moe(w) => forward_moe(
                ctx.gpu,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                normed,
                out,
                rows,
                &self.mlp_ws,
                stream,
            )?,
        }
        profile::end(profile::MLP_DENSE, t_dense, ctx.gpu, stream);
        // 🔴 ONE collective for both partials: the routed experts are EP-sharded and the
        // dense/shared half is TP-sharded, and `all_reduce(SUM)` is linear. It must land here,
        // before `hc_post` folds the output into the highway — reducing afterwards would mix a
        // half-answer into the residual stream of every later layer.
        if self.mlp_cfg.needs_all_reduce() {
            self.reduce_probe(profile::REDUCE_MLP_BAR, "mlp", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(out, rows, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_MLP_ENQ, t);
            // Second span times ONLY the sync: device + network + rank skew.
            let t = profile::start_hot();
            profile::end(profile::REDUCE_MLP, t, ctx.gpu, stream);
        }
        Ok(())
    }

    /// `dst += src` over `n` BF16 elements.
    fn add_inplace(
        &self,
        gpu: &dyn GpuBackend,
        dst: DevicePtr,
        src: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        if self.add_k.0 == 0 {
            bail!(
                "GLM layer {}: bf16_add_inplace is not loaded on this target; the MTP layer's \
                 plain residual path needs it",
                self.layer_idx
            );
        }
        KernelLaunch::new(gpu, self.add_k)
            .grid([(n as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dst)
            .arg_ptr(src)
            .arg_i32(n as i32)
            .launch(stream)
    }

    /// One token through a layer with NO hyper-connection — a plain pre-norm residual block.
    ///
    /// 🔴 This is the MTP layer, and it is the only GLM-5.3 block shaped this way. Every text
    /// layer carries `hc_*` tensors and its residual lives in the mHC highway, where `hc_post`
    /// folds the block output back in; `layers.45` carries none, so it is an ordinary
    /// `x = x + attn(norm(x))` / `x = x + mlp(norm(x))` block. That asymmetry is why
    /// [`Glm5NextLayer::mhc`] is an `Option` rather than a field.
    #[allow(clippy::too_many_arguments)]
    fn forward_one_plain(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        self.norm(gpu, hidden, self.input_norm, normed, 1, stream)?;
        let attn_out = self.mixer_forward(
            normed,
            residual,
            st,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_offloaded,
            ctx,
            stream,
        )?;
        // 🔴 Row-parallel `o_proj` ⇒ a PARTIAL SUM at TP>1. Reduce before it joins the
        // residual, exactly as the mHC path reduces before `hc_post`.
        if self.mixer_all_reduce {
            self.reduce_partial(attn_out, 1, ctx, stream)?;
        }
        self.add_inplace(gpu, hidden, attn_out, h, stream)?;

        self.norm(gpu, hidden, self.post_attn_norm, normed, 1, stream)?;
        self.mlp_forward(normed, ffn_out, 1, ctx, stream)?;
        self.add_inplace(gpu, hidden, ffn_out, h, stream)
    }

    /// One token through this layer for the MTP drafter.
    ///
    /// Only valid on a `mhc: None` block — the drafter's layer. `hidden` is read and written in
    /// place (the plain residual path accumulates into it), and there are no disk tiers because
    /// the drafter's KV pool is small, private and fully resident.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_one_for_drafter(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mhc.is_some() {
            bail!(
                "GLM layer {}: decode_one_for_drafter is the MTP block's path; this layer has a \
                 hyper-connection",
                self.layer_idx
            );
        }
        let (mut disk_a, mut disk_b) = (Vec::new(), Vec::new());
        self.forward_one_plain(
            hidden,
            hidden,
            state,
            kv_cache,
            seq_len,
            block_table,
            &mut disk_a,
            &mut disk_b,
            ctx,
            stream,
        )
    }

    /// One drafter CONTEXT row: `input_norm` then the DSA caches only.
    ///
    /// `x` is the block input (post `eh_proj`) for a row whose OUTPUT is discarded — a prompt
    /// or catch-up row. See [`Glm5NextDsaLayer::write_kv_row`] for why that is enough.
    #[allow(clippy::too_many_arguments)]
    pub fn drafter_write_kv_row(
        &self,
        x: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let layer = match &self.mixer {
            Glm5NextMixer::Dsa(l) => l,
            _ => bail!(
                "GLM layer {}: drafter_write_kv_row is the MTP block's path; this layer is not \
                 a DSA layer",
                self.layer_idx
            ),
        };
        let normed = ctx.buffers.norm_output();
        self.norm(ctx.gpu, x, self.input_norm, normed, 1, stream)?;
        layer.write_kv_row(normed, state, kv_cache, seq_len, block_table, ctx, stream)
    }

    /// One token through the whole layer, using highway slot `slot`.
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        slot: usize,
        st: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_offloaded: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            // No hyper-connection: the MTP layer, a plain pre-norm residual block.
            return self.forward_one_plain(
                hidden,
                residual,
                st,
                kv_cache,
                seq_len,
                block_table,
                disk_block_ids,
                disk_offloaded,
                ctx,
                stream,
            );
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                1,
                h as u32,
                hc as u32,
                stream,
            )?;
        }

        // ── attention site ──
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.attn,
            hidden,
            post,
            comb,
            1,
            h as u32,
            hc as u32,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.input_norm, normed, 1, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        let attn_out = self.mixer_forward(
            normed,
            residual,
            st,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_offloaded,
            ctx,
            stream,
        )?;
        // 🔴 Row-parallel `o_proj` ⇒ `attn_out` is a PARTIAL SUM at TP>1. Reduce it here,
        // before it enters the highway.
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, 1, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            // Second span times ONLY the sync: device + network + rank skew.
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            attn_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            hc as u32,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

        // ── FFN site ──
        let t_mhc = profile::start();
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.ffn,
            hidden,
            post,
            comb,
            1,
            h as u32,
            hc as u32,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.post_attn_norm, normed, 1, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        self.mlp_forward(normed, ffn_out, 1, ctx, stream)?;
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            hc as u32,
            stream,
        )?;

        // 🪤 UNWEIGHTED mean, no weights. Not DeepSeek-V4's learned collapse.
        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                1,
                h as u32,
                hc as u32,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            profile::step();
        }
        Ok(())
    }

    /// ONE weight sweep for a whole batched VERIFY: N sequences x `ks[i]` rows.
    ///
    /// [`Self::forward_n_seqs`] on the sequence axis with `ks[i]` rows per
    /// sequence instead of one. Every row-parallel site — both mHC sites, both
    /// norms, the MLP/MoE, the TP all-reduce — runs ONCE over R = Σ`ks` rows,
    /// which is the amortisation a per-sequence `forward_k` sweep cannot give:
    /// that one re-reads every layer's weights once per sequence, and the
    /// weights are the whole cost.
    ///
    /// Only the mixer stays per-sequence, and only where it must: KDA walks each
    /// sequence's recurrence in order against its own state (with the per-row
    /// rewind points a partial accept needs), and DSA attends each sequence with
    /// its own page table, length and metadata rows.
    ///
    /// 🪤 R MUST STAY INSIDE THE BATCHED-GEMV BAND. Past
    /// `DENSE_GEMV_BATCHM_MAX_M` the dense sites reassociate, and an accepted
    /// draft is then no longer necessarily the token the unspeculated engine
    /// would have emitted — which is the one thing a verify may not do. The
    /// caller checks it; this refuses what the workspaces cannot hold.
    #[allow(clippy::too_many_arguments)]
    fn forward_verify_n_seqs<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        slot_base: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let n = ks.len();
        let r_total: usize = ks.iter().sum();
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        let (rt, ht, hct) = (r_total as u32, h as u32, hc as u32);

        // Row offsets, seq-major.
        let mut off = Vec::with_capacity(n);
        let mut acc = 0usize;
        for &k in ks {
            off.push(acc);
            acc += k;
        }

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                rt,
                ht,
                hct,
                stream,
            )?;
        }
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.attn,
            hidden,
            post,
            comb,
            rt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.input_norm, normed, r_total, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);

        let t = profile::start();
        let attn_out = match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                // Per-sequence state and per-row rewind points, gathered before
                // the call so the recurrence walk itself is a plain loop.
                let mut seq_states: Vec<KdaSeqState> = Vec::with_capacity(n);
                let mut snaps: Vec<Vec<(DevicePtr, DevicePtr)>> = Vec::with_capacity(n);
                for (i, state) in states.iter_mut().enumerate().take(n) {
                    let st = self.kda_state(*state)?;
                    // Same guard `decode_batched` makes, per sequence: the pool
                    // tier is per slot and `verify_rows_n_seqs` indexes it
                    // unguarded. Bail rather than skip — skipping leaves the
                    // state advanced past the accepted boundary with no error.
                    if ks[i] > 1
                        && (st.h_state_intermediates.len() + 1 < ks[i]
                            || st.conv_state_intermediates.len() + 1 < ks[i])
                    {
                        bail!(
                            "GLM layer {}: seq {i} of a batched {}-token verify needs {} \
                             snapshots but its slot's pool has h={} conv={}",
                            self.layer_idx,
                            ks[i],
                            ks[i] - 1,
                            st.h_state_intermediates.len(),
                            st.conv_state_intermediates.len(),
                        );
                    }
                    seq_states.push(KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                    });
                    snaps.push(
                        (0..ks[i].saturating_sub(1))
                            .map(|t| (st.h_state_intermediates[t], st.conv_state_intermediates[t]))
                            .collect(),
                    );
                }
                layer.verify_rows_n_seqs(gpu, normed, ks, &seq_states, &snaps, ws, stream)?;
                ws.final_out
            }
            Glm5NextMixer::Dsa(layer) => {
                // DSA writes o_proj back over the buffer it is handed, so the
                // rows land contiguous in `normed` on their own.
                for (i, state) in states.iter_mut().enumerate().take(n) {
                    // num_seqs MUST be this sequence's row count: DSA admits a
                    // K-row metadata block only when it describes exactly `k`
                    // rows, and a plain row_view reports the rows REMAINING.
                    let row_ctx = ForwardContext {
                        attn_metadata: ctx.attn_metadata.as_ref().map(|m| AttnMetadataDev {
                            num_seqs: ks[i] as u32,
                            ..m.row_view(off[i])
                        }),
                        ..*ctx
                    };
                    let mut bt = block_tables[i].clone();
                    layer.decode_k(
                        normed.offset(off[i] * h * 2),
                        ks[i],
                        *state,
                        kv_cache,
                        seq_lens[i],
                        &mut bt,
                        &row_ctx,
                        stream,
                        false,
                        0,
                    )?;
                }
                normed
            }
        };
        profile::end(profile::KDA, t, gpu, stream);

        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, r_total, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            attn_out,
            streams,
            post,
            comb,
            streams,
            rt,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

        // ── FFN site ──
        let t_mhc = profile::start();
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.ffn,
            hidden,
            post,
            comb,
            rt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.post_attn_norm, normed, r_total, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        // THE amortised site: one grouped MoE over every row in the batch.
        self.mlp_forward(normed, ffn_out, r_total, ctx, stream)?;
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            rt,
            ht,
            hct,
            stream,
        )?;
        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                rt,
                ht,
                hct,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            profile::step();
        }
        Ok(())
    }

    /// One weight sweep for N sequences' decode rows.
    ///
    /// This is [`Self::forward_k`]'s body with the token axis reinterpreted as
    /// the SEQUENCE axis. Everything that is row-parallel — the mHC highway,
    /// both norms, the MLP/MoE, the TP all-reduce — runs ONCE over all N rows,
    /// which is the whole point: a grouped MoE GEMM at 4 rows costs about what
    /// it costs at 1, so those reads are amortized N ways instead of paid N
    /// times.
    ///
    /// Only the MIXER stays per-sequence, and it has to: KDA carries a
    /// recurrent state per sequence, and DSA needs each row's own page table
    /// and length. That is the same split `decode_verify_multi` describes —
    /// "projections/FFN batch across all rows; the stateful recurrence runs
    /// per-sequence" — applied to the sequence axis instead of the draft axis.
    ///
    /// 🪤 The highway slots must be CONTIGUOUS from `slot_base`, because the mHC
    /// kernels read `[N, ...]` from that base (`forward_k`: "K contiguous slots
    /// from there ARE the `[K, ...]` the kernels want"). Row `i` therefore owns
    /// slot `slot_base + i` — legal only because a decode step REBUILDS the
    /// highway at the first layer and collapses it at the last, so it never has
    /// to agree with a scheduler slot or survive to the next step.
    #[allow(clippy::too_many_arguments)]
    fn forward_n_seqs<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        n: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        slot_base: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        let (nt, ht, hct) = (n as u32, h as u32, hc as u32);

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                nt,
                ht,
                hct,
                stream,
            )?;
        }

        // ── attention site ──
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.attn,
            hidden,
            post,
            comb,
            nt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.input_norm, normed, n, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);

        let t = profile::start();
        let attn_out = match &self.mixer {
            Glm5NextMixer::Kda { layer, ws, .. } => {
                // ONE weight sweep for all N rows. The recurrences are independent
                // across sequences, so only the state update stays per-row — and it
                // reads row `i` of the workspace `front_end` just filled, which is
                // why the rows land contiguous in `ws.final_out` with no copy.
                let seq_states: Vec<KdaSeqState> = states
                    .iter_mut()
                    .take(n)
                    .map(|state| {
                        let st = self.kda_state(*state)?;
                        Ok(KdaSeqState {
                            conv: st.conv_state,
                            recurrent: st.h_state,
                        })
                    })
                    .collect::<Result<_>>()?;
                layer.decode_n_seqs(gpu, normed, n, &seq_states, ws, stream)?;
                ws.final_out
            }
            Glm5NextMixer::Dsa(layer) => {
                // DSA writes `o_proj` back over the buffer it is handed, so the
                // rows land contiguous in `normed` on their own.
                for (i, state) in states.iter_mut().enumerate().take(n) {
                    let row_ctx = ForwardContext {
                        attn_metadata: ctx.attn_metadata.as_ref().map(|m| m.row_view(i)),
                        ..*ctx
                    };
                    let mut bt = block_tables[i].clone();
                    layer.decode_k(
                        normed.offset(i * h * 2),
                        1,
                        *state,
                        kv_cache,
                        seq_lens[i],
                        &mut bt,
                        &row_ctx,
                        stream,
                        false,
                        0,
                    )?;
                }
                normed
            }
        };
        profile::end(profile::KDA, t, gpu, stream);

        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, n, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            attn_out,
            streams,
            post,
            comb,
            streams,
            nt,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

        // ── FFN site ──
        let t_mhc = profile::start();
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.ffn,
            hidden,
            post,
            comb,
            nt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.post_attn_norm, normed, n, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        // THE amortized site: one grouped MoE over all N rows.
        self.mlp_forward(normed, ffn_out, n, ctx, stream)?;
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            nt,
            ht,
            hct,
            stream,
        )?;
        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                nt,
                ht,
                hct,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            profile::step();
        }
        Ok(())
    }

    /// K tokens of one sequence through a KDA layer, with ONE sweep over the weights.
    ///
    /// The site order is exactly [`Self::forward_one`]'s — `hc_pre -> norm -> mixer -> hc_post`
    /// per site, the mHC highway collapsed at the last layer — but every stage runs over all K
    /// rows at once instead of K times over one. The mHC kernels, `rms_norm_vanilla` and
    /// `Glm5NextKdaLayer::decode_k` are each grid-parallel or batched over the token axis and
    /// each is bit-identical to the K serial calls it replaces, so an accepted draft token is
    /// the token the unspeculated engine would have emitted.
    ///
    /// 🪤 Highway slots are `0..K` and MUST stay per-token — the mHC streams are a per-token
    /// activation that has to survive across layers, so a shared slot would leave every layer
    /// past the first reading the last token's highway for all K rows.
    ///
    /// Both mixers come here: each has its own `decode_k` that batches its projections and
    /// keeps its per-token part (KDA's recurrence, DSA's selection and gather-attend) serial.
    #[allow(clippy::too_many_arguments)]
    fn forward_k(
        &self,
        hidden: DevicePtr,
        k: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        take_snapshots: bool,
        slot_base: usize,
        // TRUE only when this call is a PREFILL sub-chunk. `forward_k` is shared by prefill
        // and by the speculative verify, and nothing in `ForwardContext` separates them:
        // `decode_step` is false for both, and `graph_capture` is false for prefill AND for
        // an eager verify. The DSA layer's batched selector is qualified on prefill only, so
        // the distinction is carried explicitly rather than re-derived downstream.
        is_prefill: bool,
        // ABSOLUTE position below which DSA must not rewrite the MLA latent — the Marconi
        // replay window `[snap_tok, matched)` of an intermediate prefix-cache hit, whose
        // latents already sit in shared radix blocks. `prefill` derives it from the trait's
        // `kv_write_start`; the verify passes 0. KDA ignores it (no KV). See
        // `Glm5NextDsaLayer::decode_k`.
        kv_write_floor: usize,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let hc = mhc.hc_mult;
        // `slot_base`'s base: the per-slot strides are exactly these, so K contiguous slots
        // from there ARE the `[K, ...]` the kernels want.
        //
        // 🔴 The highway is a PER-TOKEN activation that must survive across layers, so the slot
        // is the token's index within the whole forward, NOT its row within this call. A verify
        // is one call at `slot_base = 0`; a batched prefill is `ceil(N / PREFILL_ROWS)` calls
        // that must land on disjoint slots, or sub-chunk 1 overwrites sub-chunk 0's streams and
        // every later layer reads the wrong token's highway. ANOMALIES A65.
        let streams = ctx.buffers.hc_streams().offset(slot_base * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot_base * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot_base * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();
        let (kt, ht, hct) = (k as u32, h as u32, hc as u32);

        // Per-token state snapshots a partial accept rewinds to; row `t` writes slot `t`, and
        // the last row needs none because a full accept never rolls back. KDA only — DSA's
        // per-sequence state is the indexer cache, which rewinds by a host counter.
        let kda_ctx = match &self.mixer {
            Glm5NextMixer::Kda { ws, .. } => {
                if k > ws.max_tokens() {
                    bail!(
                        "GLM layer {}: a {k}-token verify exceeds the KDA workspace built for {}",
                        self.layer_idx,
                        ws.max_tokens()
                    );
                }
                let st = self.kda_state(state)?;
                // 🔴 PREFILL TAKES NONE. A verify needs a per-row rewind point, so it snapshots
                // rows `0..k-1`; prefill is never rolled back, and the intermediates pool holds
                // only `num_spec` slots — indexing it for a 32-row prefill chunk would run off
                // the end. `decode_k` reads these with `snapshots.get(row)`, so an empty slice
                // is a clean "take none", not a special case.
                let snaps: Vec<(DevicePtr, DevicePtr)> = if take_snapshots {
                    (0..k.saturating_sub(1))
                        .map(|t| (st.h_state_intermediates[t], st.conv_state_intermediates[t]))
                        .collect()
                } else {
                    Vec::new()
                };
                Some((
                    KdaSeqState {
                        conv: st.conv_state,
                        recurrent: st.h_state,
                    },
                    snaps,
                ))
            }
            Glm5NextMixer::Dsa(_) => None,
        };

        let t_mhc = profile::start();
        if self.is_first {
            glm_hc_expand(
                gpu,
                mhc.kernels.hc_expand,
                hidden,
                streams,
                kt,
                ht,
                hct,
                stream,
            )?;
        }

        // ── attention site ──
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.attn,
            hidden,
            post,
            comb,
            kt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.input_norm, normed, k, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        let t = profile::start();
        let attn_out = match (&self.mixer, &kda_ctx) {
            (Glm5NextMixer::Kda { layer, ws, .. }, Some((kda, snaps))) => {
                // 🔴 The CHUNKED scan, on prefill only. `decode_k` walks `stateful_row` once
                // per token in a host loop: at `ATLAS_GLM_PREFILL_ROWS=256` x 34 KDA layers
                // that is 8,704 conv + 8,704 recurrent launches per sub-chunk, each a
                // 1-warp-per-block kernel that re-reads and rewrites the WHOLE recurrent
                // state (~4 MB/layer at 64 heads), fully serialised by the state dependency.
                // A corrected profile puts `kda_mixer` at 53.5% of prefill — the largest
                // single bucket by 5x. `kda_chunk_prepare` + `kda_chunk_scan` compute the
                // same recurrence chunk-wise, touching the state once per `cfg.chunk` (32)
                // tokens instead of once per token.
                //
                // The path was already written, numerically gated against the CPU reference,
                // and its workspace buffers (`chunk_gc/u/w`, sized `t_pad`) are ALLOCATED in
                // production — it simply had no caller outside a microtest harness. The
                // loader's stale note ("prefill still runs token-by-token ... the mHC highway
                // forces per-token anyway") predates `forward_k`, which already does K rows
                // of mHC in one call.
                //
                // 🪤 NOT bit-identical to `decode_k`: the chunked scan reassociates the
                // recurrence, so it is refused for a speculative VERIFY, whose accepted
                // tokens must match what decode would have produced. Three conditions keep
                // it to prefill, and all three are stated rather than inferred:
                //   * `is_prefill` — the caller's own flag, not `!decode_step` (a verify
                //     also clears that) and not `k > 1` (a verify is K rows too).
                //   * `snaps.is_empty()` — belt and braces. Prefill takes no per-row
                //     snapshots (see `take_snapshots` above), and the chunked path CANNOT
                //     produce them: it never materialises the state at interior rows. If a
                //     caller ever asks for them here, fall back rather than silently drop
                //     the rewind points.
                //   * `k > 1` — a 1-row chunk has nothing to batch.
                //
                // 🪤 Prefix caching: a Marconi replay of `[snap_tok, matched)` must not be
                // recomputed by a DIFFERENT association than the pass that produced the
                // cached blocks, or the state drifts and ratchets across turns. That holds
                // here because the choice keys on `is_prefill` alone, so the original pass
                // and its replay both take this arm.
                if is_prefill && k > 1 && snaps.is_empty() && kda_chunk_prefill() {
                    layer.prefill(gpu, normed, k, kda, ws, stream)?;
                } else {
                    layer.decode_k(gpu, normed, k, kda, ws, snaps, stream)?;
                }
                ws.final_out
            }
            (Glm5NextMixer::Dsa(layer), _) => {
                // 🪤 DSA writes its `o_proj` output back over the buffer it was handed.
                layer.decode_k(
                    normed,
                    k,
                    state,
                    kv_cache,
                    seq_len,
                    block_table,
                    ctx,
                    stream,
                    is_prefill,
                    kv_write_floor,
                )?;
                normed
            }
            (Glm5NextMixer::Kda { .. }, None) => {
                bail!("GLM layer {}: KDA mixer without KDA state", self.layer_idx)
            }
        };
        profile::end(profile::KDA, t, gpu, stream);
        if self.mixer_all_reduce {
            self.reduce_probe(profile::REDUCE_ATTN_BAR, "attn", ctx, stream);
            let t = profile::start_hot();
            self.reduce_partial(attn_out, k, ctx, stream)?;
            profile::end_nosync(profile::REDUCE_ATTN_ENQ, t);
            let t = profile::start_hot();
            profile::end(profile::REDUCE_ATTN, t, ctx.gpu, stream);
        }
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            attn_out,
            streams,
            post,
            comb,
            streams,
            kt,
            ht,
            hct,
            stream,
        )?;
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);

        // ── FFN site ──
        let t_mhc = profile::start();
        glm_hc_pre(
            gpu,
            &mhc.kernels,
            streams,
            &mhc.ffn,
            hidden,
            post,
            comb,
            kt,
            ht,
            hct,
            mhc.sinkhorn_iters as u32,
            self.rms_eps,
            mhc.hc_eps,
            stream,
        )?;
        profile::end(profile::MHC, t_mhc, gpu, stream);
        let t_norm = profile::start();
        self.norm(gpu, hidden, self.post_attn_norm, normed, k, stream)?;
        profile::end(profile::NORM, t_norm, gpu, stream);
        self.mlp_forward(normed, ffn_out, k, ctx, stream)?;
        let t_mhc_post = profile::start();
        glm_hc_post(
            gpu,
            mhc.kernels.hc_post,
            ffn_out,
            streams,
            post,
            comb,
            streams,
            kt,
            ht,
            hct,
            stream,
        )?;
        if self.is_last {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                streams,
                hidden,
                kt,
                ht,
                hct,
                stream,
            )?;
        }
        profile::end(profile::MHC_POST, t_mhc_post, gpu, stream);
        if self.is_last {
            profile::step();
        }
        Ok(())
    }

    /// This KDA layer's recurrent + conv state.
    ///
    /// 🔴 It is an [`SsmLayerState`] — the SAME type Qwen's GDN layers carry — and that is
    /// deliberate, not incidental. `rollback_ssm_states_dispatch` walks every
    /// `LayerType::LinearAttention` layer and downcasts to exactly this type to restore a
    /// rejected speculative draft; GLM's KDA blocks ARE `linear_attention` in `layer_types`,
    /// so carrying anything else means the first rejected draft is a hard error. The shapes
    /// line up with the pool's own math: `h = nv·vd·kd·4` and
    /// `conv = (nk·kd·2 + nv·vd)·d_conv·4` are byte-for-byte GLM's
    /// `recurrent_state_elems()·4` and `conv_state_elems()·4`, because the parser fills the
    /// `linear_*` fields from `linear_attn_config` and they are already TP-local.
    ///
    /// 🪤 A mixer/state mismatch means the scheduler handed this layer another layer's slot.
    /// Refuse loudly: allocating a fresh state here would decode from a zero recurrent state.
    fn kda_state<'a>(&self, state: &'a mut dyn LayerState) -> Result<&'a mut SsmLayerState> {
        let st = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a KDA mixer was handed state that is not an SsmLayerState",
                    self.layer_idx
                )
            })?;
        // 🔴 HF casts the KDA recurrent state to float32 and vLLM hardcodes `kda_state_dtype`.
        // A narrowed h slot (`--ssm-h-dtype f16`/`f16-pool`) is a deviation from the
        // reference, not a memory setting, and every KDA kernel reads FP32.
        if st.h_is_f16 || st.h_prefill_stage.is_some() {
            bail!(
                "GLM layer {}: KDA recurrent state is FP32-only; --ssm-h-dtype f16 narrowed it",
                self.layer_idx
            );
        }
        Ok(st)
    }

    /// This DSA layer's indexer key cache.
    fn dsa_state<'a>(&self, state: &'a mut dyn LayerState) -> Result<&'a mut Glm5NextDsaState> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a DSA mixer was handed state that is not a Glm5NextDsaState",
                    self.layer_idx
                )
            })
    }
}

impl TransformerLayer for Glm5NextLayer {
    /// EXL3 routed experts VETO decode-graph capture.
    ///
    /// The trellis mGEMMs launch cooperatively (`cuLaunchCooperativeKernel`),
    /// and a cooperative launch is not stream-capturable. The failure is not a
    /// clean refusal at the launch either: the shared `Exl3LaunchState`'s
    /// cross-stream fence trips first, and `cuStreamWaitEvent` returns 905
    /// (`STREAM_CAPTURE_ISOLATION`) mid-request — which is exactly how this
    /// surfaced, as an inference error rather than a capture error.
    ///
    /// Scoped to the EXL3 arm on purpose: an NVFP4 GLM layer captures as it
    /// always did.
    fn decode_graph_unsupported(&self) -> bool {
        match &self.mlp {
            Glm5NextMlpSite::Moe(w) => w.exl3.is_some(),
            _ => false,
        }
    }

    /// Decode N sequences, each contributing one token, through this layer.
    ///
    /// The trait default loops `decode` per row, and `decode` pins highway slot
    /// 0 and metadata row 0 — the two structural aliases that made
    /// [`Self::decode_multi_seq_unsupported`] true. This override is the same
    /// loop with both aliases removed:
    ///
    ///   * **D1 — highway.** Row `i` takes highway slot `ctx.hc_row_offset + i`
    ///     instead of 0. The highway is a PER-TOKEN activation, and a decode
    ///     step rebuilds it from scratch (`hc_expand` at the first layer) and
    ///     collapses it at the last, so it only has to be consistent ACROSS
    ///     LAYERS within one step — never across steps. That is what lets row
    ///     `i` own slot `i` regardless of which scheduler slot the sequence
    ///     holds, and it is why the slots are contiguous, which is in turn what
    ///     a future batched mHC call needs (`forward_k` already documents that
    ///     "K contiguous slots from there ARE the `[K, ...]` the kernels want").
    ///   * **D2 — DSA metadata.** Row `i` gets `attn_metadata.row_view(i)`, so
    ///     it attends with its OWN positions, slot, seq_len and page table.
    ///
    /// 🪤 Rows are still executed ONE AT A TIME. This is correctness, not the
    /// throughput win: the weights are still swept once per row, so aggregate
    /// tok/s is unchanged and only the wrong-answer hazard is gone. Batching
    /// the projections/MoE across the N rows — the part that actually pays,
    /// because a grouped GEMM at 4 rows costs about what it costs at 1 — is the
    /// follow-up, and it needs the per-sequence KDA recurrence and DSA page
    /// tables split out from the row-batched sites first.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Bail rather than truncate: a short slice means the caller and this
        // layer disagree about the batch, and serving the rows we happen to
        // have would answer some sequences with another's activations — the
        // exact failure this override exists to remove.
        if states.len() < num_seqs || seq_lens.len() < num_seqs || block_tables.len() < num_seqs {
            bail!(
                "GLM layer {}: decode_multi_seq of {num_seqs} seqs got states={} seq_lens={} \
                 block_tables={}",
                self.layer_idx,
                states.len(),
                seq_lens.len(),
                block_tables.len()
            );
        }
        // No hyper-connection (the MTP layer) has no highway to alias, so the
        // trait's per-row default is already correct for it.
        if self.mhc.is_none() || num_seqs == 1 {
            let h = self.hidden;
            for (i, state) in states.iter_mut().enumerate().take(num_seqs) {
                let off = i * h * 2;
                let row_ctx = ForwardContext {
                    attn_metadata: ctx.attn_metadata.as_ref().map(|m| m.row_view(i)),
                    ..*ctx
                };
                let mut bt = block_tables[i].clone();
                let mut stub_disk = Vec::<u32>::new();
                let mut stub_last_offloaded = Vec::<u32>::new();
                self.forward_one(
                    hidden.offset(off),
                    residual.offset(off),
                    ctx.hc_row_offset + i,
                    *state,
                    kv_cache,
                    seq_lens[i],
                    &mut bt,
                    &mut stub_disk,
                    &mut stub_last_offloaded,
                    &row_ctx,
                    stream,
                )?;
            }
            return Ok(());
        }
        self.forward_n_seqs(
            hidden,
            num_seqs,
            states,
            kv_cache,
            seq_lens,
            block_tables,
            ctx.hc_row_offset,
            ctx,
            stream,
        )
    }

    /// 🔴 GLM-5.3 CANNOT serve a batched multi-sequence decode step. Two
    /// independent row-0 aliases, both structural, either one sufficient:
    ///
    ///   * **D1 — mHC highway slot.** `Glm5NextLayer::forward_one` pins highway
    ///     slot 0. `decode_multi_seq`'s default loop shares one
    ///     `ForwardContext` across the batch, so every sequence would write and
    ///     then read the SAME highway stream, and each layer past the first
    ///     reads the last sequence's mHC state for all rows. This is the
    ///     sequence-axis twin of the token-axis argument already written on
    ///     [`Self::decode_batched`] above — the trait default is WRONG there
    ///     for exactly the same reason.
    ///   * **D2 — DSA `attn_metadata` row.** The DSA mixer reads metadata row 0
    ///     (`glm5next_dsa/layer.rs`: slot, positions, block_table, seq_len), so
    ///     every sequence in the batch would attend with sequence 0's page
    ///     table and length.
    ///
    /// Answering `true` does NOT cost concurrency: the caller routes GLM onto
    /// #753 item B's per-sequence highway loop, which serves C>1 correctly at
    /// C=1-equivalent per-request throughput.
    ///
    /// ✅ BOTH ALIASES ARE GONE, which is why this now answers `false`. The
    /// `Glm5NextLayer::decode_multi_seq` directly above is that Stage 1 commit:
    /// row `i` runs `forward_one` at highway slot `ctx.hc_row_offset + i` (D1)
    /// against `attn_metadata.row_view(i)` (D2). The description below is kept
    /// because it is the reason the override has to exist at all — delete the
    /// override and both aliases come straight back.
    ///
    /// 🪤 It buys CORRECTNESS, not throughput. Rows are still served one at a
    /// time, so this is what makes a batched step safe, not fast.
    fn decode_multi_seq_unsupported(&self) -> bool {
        false
    }

    /// GLM's batched multi-sequence decode runs the DSA mixer PER SEQUENCE,
    /// against that sequence's own `Glm5NextDsaState` — `k_normed`, `gate` and
    /// `valid` are its own allocations, and its `len` is its own — with its own
    /// page table, its own `seq_len` and its own `attn_metadata` row. Selection
    /// is therefore already per-sequence on this path, and a batch does not
    /// have to leave it when some sequence passes the index budget.
    ///
    /// That budget is `index_topk + index_compress_ratio - 1` = 2047 here, and
    /// before this the whole batch dropped to the per-sequence loop the moment
    /// ANY sequence crossed it — so concurrency paid on short contexts and
    /// silently stopped paying on long ones.
    ///
    /// 🪤 True because the mixer LOOPS, not because anything about selection was
    /// made batch-aware. If `forward_n_seqs`'s DSA arm is ever collapsed into a
    /// single batched attention call, this must go back to `false` in the same
    /// commit unless that call carries per-row indexer state.
    fn decode_multi_seq_selection_per_seq(&self) -> bool {
        true
    }

    /// Batched MTP verify: `n_seqs` sequences x `ks[i]` draft rows, seq-major.
    ///
    /// The rows are already contiguous in `hidden` (`r = off[i] + j`), so this
    /// runs `forward_k` once per SEQUENCE at that sequence's row base. It is
    /// the shape the note below prescribes, and it is deliberately NOT a fused
    /// R-row body: the per-sequence sweep is BIT-IDENTICAL to the serial verify
    /// this replaces, because each call sees exactly the rows, the state, the
    /// length and the page table the serial path gave it. What it buys is not
    /// weight amortisation inside the layer but the right to run on the batched
    /// driver at all — one metadata block, one lm_head, one argmax, one D2H for
    /// the whole batch instead of `n_seqs` of each.
    ///
    /// 🪤 `slot_base = off[i]`, never 0. The mHC highway slot is the row's index
    /// within the WHOLE forward (A65), and the driver hands every sequence one
    /// `hidden` with `hc_row_offset: 0`; a sequence writing from slot 0 would
    /// scribble sequence 0's highway and every later layer would read it back.
    ///
    /// 🪤 Each sequence gets `num_seqs: ks[i]` on its metadata view, not the
    /// count `row_view` leaves behind. DSA admits a K-row metadata block ONLY
    /// when it describes exactly `k` rows; a plain `row_view(off[i])` reports
    /// the rows REMAINING in the batch, which is right only for the last
    /// sequence and silently drops every earlier one onto the per-row
    /// `copy_h2d` path — capture-unsupported, and a stream drain per row.
    ///
    /// `residual` is ignored: a GLM text layer's residual lives in the mHC
    /// highway. `wy_tables` is ignored: those are the GDN family's WY tables,
    /// and KDA carries its own workspace.
    #[allow(clippy::too_many_arguments)]
    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        block_tables: &[Vec<u32>],
        _wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Bail rather than serve the rows we happen to have: a short slice
        // means the caller and this layer disagree about the batch, and a
        // verify that advances the wrong sequence's recurrent state is a
        // silent, compounding wrong answer that survives as fluent text.
        if states.len() < n_seqs
            || ks.len() < n_seqs
            || seq_lens.len() < n_seqs
            || block_tables.len() < n_seqs
        {
            bail!(
                "GLM layer {}: decode_verify_multi of {n_seqs} seqs got states={} ks={} \
                 seq_lens={} block_tables={}",
                self.layer_idx,
                states.len(),
                ks.len(),
                seq_lens.len(),
                block_tables.len()
            );
        }
        // Per-sequence sweep by DEFAULT — it is the bit-identity reference and
        // the one that passes the long-context probe. `ATLAS_GLM_VERIFY_FUSED=1`
        // selects the fused R-row body, which is ~2x faster on the verify
        // forward and has an open long-context defect; see `verify_fused_rows`.
        let r_total: usize = ks[..n_seqs].iter().sum();
        if verify_fused_rows() && r_total <= crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize {
            return self.forward_verify_n_seqs(
                hidden,
                &ks[..n_seqs],
                states,
                kv_cache,
                seq_lens,
                block_tables,
                ctx.hc_row_offset,
                ctx,
                stream,
            );
        }

        let h = self.hidden;
        let mut base = 0usize;
        for (i, state) in states.iter_mut().enumerate().take(n_seqs) {
            let k = ks[i];
            if k == 0 {
                continue;
            }
            // The per-row rewind points a PARTIAL accept restores. `forward_k`
            // indexes `h_state_intermediates[t]` unguarded, and the pool tier is
            // per slot, so a sequence whose slot came from a narrow tier would
            // run off the end mid-sweep. Checked HERE, before any launch, for
            // the same reason `decode_batched` checks it: bailing is recoverable,
            // a short pool is not.
            if matches!(self.mixer, Glm5NextMixer::Kda { .. }) && k > 1 {
                let st = self.kda_state(*state)?;
                if st.h_state_intermediates.len() + 1 < k
                    || st.conv_state_intermediates.len() + 1 < k
                {
                    bail!(
                        "GLM layer {}: seq {i} of a batched {k}-token verify needs {} \
                         per-token state snapshots but its slot's pool has h={} conv={}",
                        self.layer_idx,
                        k - 1,
                        st.h_state_intermediates.len(),
                        st.conv_state_intermediates.len(),
                    );
                }
            }
            let row_ctx = ForwardContext {
                attn_metadata: ctx.attn_metadata.as_ref().map(|m| AttnMetadataDev {
                    num_seqs: k as u32,
                    ..m.row_view(base)
                }),
                ..*ctx
            };
            let mut bt = block_tables[i].clone();
            self.forward_k(
                hidden.offset(base * h * 2),
                k,
                *state,
                kv_cache,
                seq_lens[i],
                &mut bt,
                &row_ctx,
                stream,
                true, // take_snapshots: a partial accept must have a rewind point
                base, // slot_base: this sequence's highway rows
                // A BATCHED speculative verify, NOT a prefill sub-chunk. Same
                // answer as the single-sequence verify caller and for the same
                // reason: DSA's batched selector is qualified on prefill only,
                // and a verify that claimed prefill would take it.
                false,
                0, // kv_write_floor: a verify never replays a cached prefix
            )?;
            base += k;
        }
        Ok(())
    }

    /// 🔴 GLM-5.3 implements no `decode_verify_multi`, so the batched verify
    /// sweep must not be selected for it. The trait default already `bail!`s,
    /// but that is a mid-request abort; declaring it here makes
    /// `can_batch_verify_dispatch` route around it instead, leaving spec-on
    /// C>1 on the per-sequence verify loop — the sealed K=3 path.
    ///
    /// GLM's batched verify is asserted to be bit-identical to the serial one —
    /// the per-sequence sweep by construction, and the fused body by staying
    /// inside the batched-GEMV band — so it takes the band as its cap rather
    /// than the caller's much wider `VERIFY_ROW_CAP`.
    ///
    /// At the production ladder this is not restrictive: K=2 admits 4 sequences,
    /// K=4 admits 2, and D-Cut is free to spend the 8 rows unevenly across them
    /// (deeper where drafts survive) because `ks` is ragged end to end.
    ///
    /// 🪤 8, NOT `DENSE_GEMV_BATCHM_MAX_M` (16). Two different bands: 16 is the
    /// GEMV kernel's own ceiling, 8 is where the DECODE sites — including the
    /// BF16 lm_head this verify's argmax comes out of — stop choosing the
    /// non-reassociating kernel. The argmax is what the accept walk compares,
    /// so the lm_head band is the one that binds.
    fn decode_verify_multi_row_cap(&self) -> Option<usize> {
        Some(crate::layers::ops::DENSE_GEMV_BATCHM_DECODE_MAX_M as usize)
    }

    /// ✅ NOW FALSE — `Glm5NextLayer::decode_verify_multi` above is that commit:
    /// a per-sequence `forward_k` sweep with `slot_base = row_base` and a
    /// per-sequence metadata view. The text below is kept because it is why the
    /// override has to exist; delete the override and the bail comes back.
    ///
    /// 🪤 The LAYER answering `false` is necessary, not sufficient. Under EP the
    /// batched sweep is still refused by `can_batch_verify_dispatch`'s
    /// `comm.is_none()` conjunct, which has no worker-side command behind it —
    /// so this alone changes nothing on a multi-rank serve.
    fn decode_verify_multi_unsupported(&self) -> bool {
        false
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(match &self.mixer {
            // Pool-free fallback. `uses_ssm_pool()` is true for KDA, so the model hands these
            // layers a pool slot and never calls this — but the paths that build states
            // directly still need a correctly shaped, ZEROED one. A fresh sequence starts from
            // a zero recurrent state and an empty conv window; inheriting the previous
            // sequence's residue is a wrong answer that decays over a few tokens rather than
            // crashing.
            Glm5NextMixer::Kda { cfg, .. } => Box::new(alloc_kda_ssm_state(gpu, cfg)?),
            Glm5NextMixer::Dsa(l) => Box::new(Glm5NextDsaState::alloc(gpu, &l.cfg)?),
        })
    }

    /// Release what `alloc_state` allocated — ANOMALIES A76. The DSA indexer cache is
    /// sized by `--max-seq-len`, not by the prompt (513 B/token/layer), so leaking one
    /// per request walks a unified-memory host into the ground.
    ///
    /// 🔴 Type-driven on purpose. A KDA layer's state on the model path is POOL-owned
    /// (`uses_ssm_pool()` is true for `Kda`, so `alloc_sequence` hands it pool addresses
    /// and `free_sequence` skips it entirely) — but refusing by TYPE as well means a
    /// pool address can never reach `gpu.free` even if a future call site forgets the
    /// skip. `SsmLayerState` is therefore left alone here, always.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(dsa) = state.as_any_mut().downcast_mut::<Glm5NextDsaState>() {
            dsa.free(gpu)?;
        }
        Ok(())
    }

    /// The DSA mixer allocates its per-sequence state with `gpu.alloc` in `alloc_state`, so
    /// the addresses a capture bakes belong to THAT sequence, not to the slot.
    ///
    /// 🪤 Only DSA. The KDA mixer is POOL-backed on the model path — `uses_ssm_pool()` is true
    /// for `Kda`, so `meta.rs` hands it pool addresses and never calls its `alloc_state`. The
    /// `true` below is still correct (one owned mixer is enough); the previous wording claimed
    /// both mixers own their state, and that was wrong.
    fn graph_stale_on_new_sequence(&self) -> bool {
        true
    }

    /// 🔴 A CUDA-graph replay runs kernels and nothing else, so this layer's one piece of
    /// HOST-side per-sequence bookkeeping — the DSA indexer cache length — has to be advanced
    /// here. The inner `Glm5NextDsaLayer` implements this too, but the model's layer vec holds
    /// the COMPOSITE, so the inner impl is never reached and the default no-op left the
    /// counter frozen at its capture-time value.
    ///
    /// 🪤 That was invisible on the spec-off path: after capture, `decode` never runs again, so
    /// nothing compared the counter to `seq_len`. The first EAGER step after a run of replays —
    /// which is exactly what a speculative verify is — then failed with "indexer cache holds 5
    /// tokens but the sequence is at 12". The rows were there; only the counter was stale.
    ///
    /// KDA keeps nothing on the host: its recurrent and conv state are device-resident and the
    /// replayed kernels update them in place.
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => self.dsa_state(state)?.sync_to(seq_len, k),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }

    /// The model's layer vec holds the COMPOSITE, so — exactly as with `sync_replayed_step`
    /// — the inner `Glm5NextDsaLayer` impl is never reached and this one is what runs. A62.
    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => state
                .as_any()
                .downcast_ref::<Glm5NextDsaState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM layer {}: a DSA mixer was handed state that is not a \
                         Glm5NextDsaState",
                        self.layer_idx
                    )
                })?
                .ensure_room_through(seq_len + k)
                // 🔴 Both A62 routes raise the SAME refusal, so without a tag a log cannot
                // tell a pre-launch replay refusal from `indexer_forward`'s prefill one —
                // and "we proved the replay route" would rest on inference. Name it.
                .with_context(|| {
                    format!(
                        "DSA replay pre-check (layer {}, before launch_graph, seq_len \
                         {seq_len} + k {k})",
                        self.layer_idx
                    )
                }),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }

    /// The DSA indexer cache is the one piece of GLM per-sequence state that neither the KV
    /// blocks nor the Marconi SSM slot carries, so it is what the aux blob exists for.
    ///
    /// 🔴 KDA answers FALSE, and that is not an omission. Its state IS the pool's
    /// `SsmLayerState` (`uses_ssm_pool()` below; `state.rs` for why the shapes line up), and
    /// `SsmSnapshotPool::save`/`restore` already D2D-copies every pool slot — all 34 KDA
    /// layers — into and out of the Marconi region. Answering true here would make the
    /// restore gate demand a blob KDA can never produce and decline every snapshot.
    ///
    /// The model's layer vec holds the COMPOSITE, so this is the impl that runs — the inner
    /// `Glm5NextDsaLayer` deliberately has NO aux overrides (pinned in `tests.rs`), because an
    /// inner impl is unreachable from here and would only look like coverage.
    fn has_aux_state(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Dsa(_))
    }

    /// Image this DSA layer's indexer rows for a Marconi snapshot; `Ok(None)` above the
    /// `ATLAS_GLM_DSA_AUX_MAX_TOKENS` cap (all 11 layers decline together — see
    /// `glm5next_dsa::aux`) and always `Ok(None)` for KDA, whose state rides the pool slot.
    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => {
                let st = state
                    .as_any()
                    .downcast_ref::<Glm5NextDsaState>()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "GLM layer {}: a DSA mixer was handed state that is not a \
                             Glm5NextDsaState",
                            self.layer_idx
                        )
                    })?;
                glm5next_dsa::aux::snapshot(st, self.layer_idx as u32, gpu, stream)
            }
            Glm5NextMixer::Kda { .. } => Ok(None),
        }
    }

    /// Put a `snapshot_aux` blob back on a prefix-cache hit, before the resumed prefill. The
    /// codec validates the blob against THIS layer (index, row width, reservation, exact
    /// length) before the first upload, so a wrong blob is a request error, never a
    /// half-written cache. A KDA layer can only reach here through a corrupted blob set —
    /// `apply_aux_states` routes by the layer index the blob was saved under — so it bails.
    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => glm5next_dsa::aux::restore(
                self.dsa_state(state)?,
                self.layer_idx as u32,
                blob,
                gpu,
                stream,
            ),
            Glm5NextMixer::Kda { .. } => bail!(
                "GLM layer {}: restore_aux on a KDA layer — KDA state rides the Marconi SSM \
                 slot and never produces an aux blob, so this blob set is mis-indexed",
                self.layer_idx
            ),
        }
    }

    /// GLM's KDA blocks are `linear_attention` in `layer_types` AND carry the pool's
    /// `SsmLayerState`, so they take pool slots like any other recurrent layer. That is what
    /// buys the speculative-verify checkpoints and per-token intermediates for free —
    /// `meta.rs` only wires `h_state_checkpoint` / `h_state_intermediates` for layers that
    /// answer true here, and `rollback_ssm_states_dispatch` needs both.
    fn uses_ssm_pool(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_one(
            hidden,
            residual,
            0,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// Prefill, one token at a time — but with the highway indexed by token.
    ///
    /// 🔴 The trait's default fallback would be WRONG here, not merely slow. It runs every token
    /// through this layer before the next layer sees any of them, so a single-slot highway would
    /// hold only the last token's streams by the time layer `n+1` reads it. The mHC highway is a
    /// per-token activation that must survive across layers, so each token gets its own slot.
    ///
    /// Per-token (rather than chunked) is deliberate for this slice: KDA's recurrence is
    /// sequential anyway, and the chunked `Glm5NextKdaLayer::prefill` needs a workspace sized
    /// for the chunk. That is an optimisation, explicitly out of scope.
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        // Chunk-relative floor below which attention must not rewrite K/V — the Marconi
        // replay window on an intermediate prefix-cache hit (`prefill_a.rs` /
        // `forward_layers.rs` `layer_kv_write_start`). Was `_kv_write_start` (ignored) until
        // the GLM prefix cache opened; with it ignored, a warm hit's replay rewrote radix
        // blocks other live sequences read. Honoured by the DSA `latent_write` only — KDA
        // has no KV, and the indexer rows are per-sequence, never shared.
        kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let cap = ctx.buffers.max_batch_tokens();
        if num_tokens > cap {
            bail!(
                "GLM layer {}: prefill of {num_tokens} tokens exceeds the {cap}-token mHC \
                 highway the buffer arena was sized for; each token needs its own slot",
                self.layer_idx
            );
        }
        // ── Batched sub-chunks: ONE weight sweep per PREFILL_ROWS tokens ──
        //
        // 🔴 ANOMALIES A65. The per-token walk below pays a full sweep of this layer's weights
        // for EVERY token, which is why prefill ran at decode speed (15.2 tok/s measured, and
        // 97 % linear in the token count: TTFT 579.1 s at 9,000 tokens and 1,187.6 s at 18,000,
        // a ratio of 2.051 against 2.000 for pure-linear).
        //
        // `forward_k` is the SAME body the speculative verify uses and already sweeps once for
        // all its rows, so this is reuse, not a new path. What it does NOT yet amortize is the
        // routed MoE: `forward_moe`'s expert-union arm is capped at 4 rows (the union kernel
        // resolves `rows * top_k <= 64` ids in one block), so above that each row still pays its
        // own 8 experts. That caps the win here at ~5x — Atlas's own sizing note puts KDA at
        // 9,366 MB/token against ~2.1 GB/token of routed-expert traffic, so amortizing the
        // former is most of the prize and the latter needs a grouped MoE GEMM (separate lane).
        //
        // 🪤 `mhc: None` is the MTP drafter block, whose plain residual path `forward_k` does
        // not implement — it bails on a missing highway. That block keeps the per-token walk.
        let rows = if self.mhc.is_some() {
            prefill_rows().min(cap)
        } else {
            1
        };
        if rows > 1 {
            let mut t = 0usize;
            while t < num_tokens {
                let k = rows.min(num_tokens - t);
                self.forward_k(
                    hidden.offset(t * self.hidden * 2),
                    k,
                    state,
                    kv_cache,
                    seq_len_start + t,
                    block_table,
                    ctx,
                    stream,
                    // Prefill is never rolled back, so it takes no per-row KDA snapshots.
                    false,
                    // Absolute slot within this prefill, so sub-chunks never share a slot.
                    t,
                    // This IS the prefill sub-chunk caller.
                    true,
                    // The trait's floor is chunk-relative; DSA positions are absolute.
                    seq_len_start + kv_write_start,
                )?;
                t += k;
            }
            return Ok(());
        }
        for t in 0..num_tokens {
            let off = t * self.hidden * 2;
            self.forward_one(
                hidden.offset(off),
                residual.offset(off),
                t,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// K tokens of ONE sequence in a single call — the speculative-verify body.
    ///
    /// The same per-token walk as [`Self::prefill`] — same highway slots, same
    /// `seq_len + t` positions, same KV writes — plus the per-token KDA state snapshots that
    /// only a verify needs. Prefill is never rolled back, so it does not pay for them.
    ///
    /// 🔴 The trait's default would be WRONG, not merely slow: it calls `decode` per token, and
    /// `decode` pins highway slot 0. K tokens would then overwrite each other's mHC streams and
    /// every layer past the first would read the last token's highway for all K rows.
    ///
    /// ✅ **Batched.** This delegates to `Self::forward_k`, which sweeps the weights ONCE for
    /// all K rows. (An earlier revision of this comment said "still one `forward_one` per row";
    /// that was stale — `forward_k` has been the body since the batched-verify work, and the
    /// measured K=3 step of ~101 ms against a ~63 ms single-row step is only explicable by it.)
    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // A KDA layer must leave behind the state it held after EACH verify token, or a
        // partially-accepted draft has nothing to rewind to. `rollback_ssm_states_dispatch`
        // restores `h_state_intermediates[num_accepted - 1]`, so row `t` writes slot `t` and
        // the LAST row needs none (a full accept never rolls back).
        let kda_bytes = matches!(self.mixer, Glm5NextMixer::Kda { .. }).then_some(());
        if kda_bytes.is_some() && num_tokens > 1 {
            let st = self.kda_state(state)?;
            // 🔴 Bail rather than skip. Skipping leaves `h_state` ADVANCED past the accepted
            // boundary with no error and no log line, which corrupts every subsequent decode
            // and surfaces much later as gibberish. Same check the Qwen verify arms make.
            if st.h_state_intermediates.len() + 1 < num_tokens
                || st.conv_state_intermediates.len() + 1 < num_tokens
            {
                bail!(
                    "GLM layer {}: a {num_tokens}-token verify needs {} per-token state \
                     snapshots but the pool has h={} conv={}. With none, this is the \
                     self-speculative / ngram path on a model whose MTP pool was never \
                     sized; with too few, --num-drafts exceeds the pool's tier.",
                    self.layer_idx,
                    num_tokens - 1,
                    st.h_state_intermediates.len(),
                    st.conv_state_intermediates.len(),
                );
            }
        }

        // ONE sweep over the weights for all K rows — the whole reason speculation pays.
        self.forward_k(
            hidden,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            true,
            0,
            // A speculative verify, NOT a prefill sub-chunk — true here would hand an eager
            // verify the prefill-only batched DSA selector.
            false,
            // A verify writes fresh positions only; no replay window, no floor. Also what
            // keeps the floor branch inert under the graphed verify's capture.
            0,
        )
    }

    /// KDA layers carry recurrent state; DSA layers do not.
    fn is_ssm_layer(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }
}

#[cfg(test)]
mod tests;
