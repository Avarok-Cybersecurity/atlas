// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextLayer` — the composite GLM-5.3 decoder layer that implements [`TransformerLayer`].
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
//!   [`Glm5NextLayer::prefill`].
//! * **Both MLP arms leave a PARTIAL SUM** whenever TP or EP is on. The single `all_reduce` at
//!   the end of the FFN site covers both, and it must happen *before* `hc_post` mixes the output
//!   back into the highway.

use std::sync::Arc;

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{ForwardContext, LayerState, TransformerLayer};
use crate::layers::glm5next_dsa::layer::Glm5NextDsaLayer;
use crate::layers::glm5next_dsa::state::Glm5NextDsaState;
use crate::layers::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaLayer, Glm5NextKdaWorkspace};
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

pub use state::{Glm5NextLayerState, OwnedKdaState};

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

impl Glm5NextLayer {
    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([1, 1, 1])
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
        st: &mut Glm5NextLayerState,
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
                let kda = st.kda()?;
                layer.decode(ctx.gpu, normed, &kda.inner, ws, stream)?;
                Ok(ws.final_out)
            }
            Glm5NextMixer::Dsa(layer) => {
                // The DSA layer's own state type IS a `LayerState`, so handing it straight
                // through downcasts cleanly — no adapter, no second allocation.
                let dsa: &mut Glm5NextDsaState = st.dsa()?;
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

    /// `all_reduce(SUM)` a `[1, hidden]` BF16 partial, when one is needed and a comm exists.
    fn reduce_partial(&self, p: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<()> {
        if let Some(comm) = ctx.comm {
            let bytes = self.hidden * 2;
            if ctx.graph_capture {
                comm.all_reduce(p.0, bytes)?;
            } else {
                comm.all_reduce_async(p.0, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// Run the MLP on `normed` into `out`, then reduce if this rank holds only part of it.
    fn mlp_forward(
        &self,
        normed: DevicePtr,
        out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match &self.mlp {
            Glm5NextMlpSite::Dense(w) => forward_dense(
                ctx.gpu,
                &self.mlp_kernels,
                &self.mlp_cfg,
                w,
                self.mlp_cfg.local_dense_intermediate,
                normed,
                out,
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
                &self.mlp_ws,
                stream,
            )?,
        }
        // 🔴 ONE collective for both partials: the routed experts are EP-sharded and the
        // dense/shared half is TP-sharded, and `all_reduce(SUM)` is linear. It must land here,
        // before `hc_post` folds the output into the highway — reducing afterwards would mix a
        // half-answer into the residual stream of every later layer.
        if self.mlp_cfg.needs_all_reduce() {
            self.reduce_partial(out, ctx, stream)?;
        }
        Ok(())
    }

    /// One token through the whole layer, using highway slot `slot`.
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        slot: usize,
        st: &mut Glm5NextLayerState,
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
            bail!(
                "GLM layer {}: no hyper-connection bound. Every text layer of GLM-5.3 has one; \
                 only the MTP layer does not, and the MTP layer is not part of this stack.",
                self.layer_idx
            );
        };
        let hc = mhc.hc_mult;
        let streams = ctx.buffers.hc_streams().offset(slot * hc * h * 4);
        let post = ctx.buffers.hc_post().offset(slot * hc * 4);
        let comb = ctx.buffers.hc_comb().offset(slot * hc * hc * 4);
        let normed = ctx.buffers.norm_output();
        let ffn_out = ctx.buffers.moe_output();

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
            mhc.kernels.hc_pre,
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
        self.norm(gpu, hidden, self.input_norm, normed, stream)?;
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
            self.reduce_partial(attn_out, ctx, stream)?;
        }
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

        // ── FFN site ──
        glm_hc_pre(
            gpu,
            mhc.kernels.hc_pre,
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
        self.norm(gpu, hidden, self.post_attn_norm, normed, stream)?;
        self.mlp_forward(normed, ffn_out, ctx, stream)?;
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
        Ok(())
    }

    fn downcast<'a>(&self, state: &'a mut dyn LayerState) -> Result<&'a mut Glm5NextLayerState> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextLayerState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: state is not a Glm5NextLayerState",
                    self.layer_idx
                )
            })
    }
}

impl TransformerLayer for Glm5NextLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(match &self.mixer {
            Glm5NextMixer::Kda { cfg, .. } => {
                Glm5NextLayerState::Kda(OwnedKdaState::alloc(gpu, cfg)?)
            }
            Glm5NextMixer::Dsa(l) => Glm5NextLayerState::Dsa(Glm5NextDsaState::alloc(gpu, &l.cfg)?),
        }))
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
        let st = self.downcast(state)?;
        self.forward_one(
            hidden,
            residual,
            0,
            st,
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
        let st = self.downcast(state)?;
        for t in 0..num_tokens {
            let off = t * self.hidden * 2;
            self.forward_one(
                hidden.offset(off),
                residual.offset(off),
                t,
                st,
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

    /// KDA layers carry recurrent state; DSA layers do not.
    fn is_ssm_layer(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }
}

#[cfg(test)]
mod tests;
