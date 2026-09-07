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

use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
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
/// 🔴 **8, and the ceiling is a KERNEL boundary, not a bandwidth knee.** Every dense projection
/// on this path goes through [`ops::dense_mm_bf16`], whose batched-GEMV arm
/// (`dense_gemv_bf16_batchm`) is **bit-identical to M serial GEMVs** and stops at
/// `DENSE_GEMV_BATCHM_MAX_M = 8`. At `R > 8` the same call falls to the tile GEMM, which both
/// reassociates (so prefill stops being bit-identical to the per-token walk) and is the slower
/// kernel at these widths — Atlas measured it 3.6x slower than the batched GEMV at M <= 8.
///
/// 🔴 MEASURED 2026-08-31, one image, one control, ~1,950-token prompt: control TTFT 125.3 s;
/// R = 8 **43.4 s (2.89x) and byte-identical on all four probes**; R = 32 44.3 / 51.8 s — no
/// faster, and it moves two of the four completions. The per-token-bytes model that first
/// picked 32 (predicting 4.5x at R = 32 against 3.0x at R = 8) is REFUTED as a width law: it
/// modelled weight traffic only, and above R = 8 the traffic saved is handed to a slower kernel.
/// The routed experts do not amortize past R = 4 either way (`forward_moe`'s union arm caps
/// there), so R = 8 takes the whole available win. ANOMALIES A65.
pub(crate) const PREFILL_ROWS: usize = 8;

/// `PREFILL_ROWS`, overridable at launch with `ATLAS_GLM_PREFILL_ROWS`.
///
/// 🔬 Kept as the A/B lever it was built as. It found A65's real defect (the DSA attend read
/// `seq_lens[row]` / `block_tables[row]` out of a single-row buffer) by sweeping width against a
/// fixed control in ONE serve instead of one image per width. `1` restores the per-token walk
/// exactly — the `rows > 1` gate in `prefill` falls through to `forward_one`.
///
/// 🪤 Above 8 the dense projections leave the bit-identical batched-GEMV arm for the tile GEMM,
/// so a width > 8 is a NUMERICS change as well as a speed one. Measured: it is not faster.
///
/// 🪤 Read once and cached: an env read per layer per sub-chunk would sit in the hot loop.
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
                layer.decode_k(gpu, normed, k, kda, ws, snaps, stream)?;
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

    /// Both mixers allocate their per-sequence state with `gpu.alloc` in `alloc_state`, so
    /// the addresses a capture bakes belong to THAT sequence, not to the slot.
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
        _kv_write_start: usize,
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
    /// ✅ **Batched.** This delegates to [`Self::forward_k`], which sweeps the weights ONCE for
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
        )
    }

    /// KDA layers carry recurrent state; DSA layers do not.
    fn is_ssm_layer(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }
}

#[cfg(test)]
mod tests;
