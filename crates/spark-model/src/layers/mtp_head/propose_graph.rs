// SPDX-License-Identifier: AGPL-3.0-only

//! CUDA-graph capture/replay for the grammarless MTP `forward_one` GPU body.
//!
//! MTP propose is a single-layer decoder + NVFP4 lm_head. On GB10 that is
//! ~2.4 ms/step eager — mostly launch overhead, not LPDDR5X weight traffic.
//! Target verify is already graphed; this module graphs the leftover propose
//! slice. Token-dependent embed D2D, metadata H2D, and KV block alloc stay
//! *outside* the graph (varying src pointer / host buffer / CPU).
//!
//! BF16/FP8 generic MoE D2Hs expert ids and launches per-token weight
//! pointers, so it cannot be captured. Production Qwen3.6 MTP is BF16: we
//! capture a pre-MoE graph and a post-MoE (lm_head) graph around that eager
//! slice. NVFP4 fused MoE is device-side and captures as one graph.

use super::{MtpHead, MtpProposerState};
use crate::layer::ForwardContext;
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

/// Kill switch `ATLAS_NO_MTP_PROPOSE_GRAPH`. ON unless the value is exactly
/// `"1"`. `=0` / empty / unset do **not** disable — same `== "1"` reading as
/// `ATLAS_NO_GEMV_SW`.
pub const DISABLE_ENV: &str = "ATLAS_NO_MTP_PROPOSE_GRAPH";

pub fn mtp_propose_graph_from(no_graph: Option<&str>) -> bool {
    no_graph != Some("1")
}

/// Process-lifetime read of [`DISABLE_ENV`].
pub fn mtp_propose_graph_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| mtp_propose_graph_from(std::env::var(DISABLE_ENV).ok().as_deref()))
}

/// Whether this `forward_one` call may enter a CUDA graph.
///
/// Ineligible paths D2H inside the GPU body (grammar CPU argmax, debug
/// norms, draft-conf softmax, shadow top-k). Profile mode syncs after every
/// kernel. Host-dispatched MoE is still graphable: it runs *between* two
/// captured slices rather than vetoing capture.
pub fn propose_graphable(
    grammar: bool,
    shadow_topk: usize,
    draft_conf_tau: f32,
    profile: bool,
    debug_norms: bool,
) -> bool {
    !grammar && shadow_topk == 0 && draft_conf_tau <= 0.0 && !profile && !debug_norms
}

pub(super) struct ProposeKvView {
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub cache_stride: u64,
    pub block_size: u32,
    pub max_blocks: u32,
    pub meta_base: DevicePtr,
}

impl MtpHead {
    /// KV block alloc + metadata H2D. CPU + host pointer — never captured.
    pub(super) fn prepare_propose_kv(
        &self,
        state: &mut MtpProposerState,
        ctx: &ForwardContext<'_>,
        position: usize,
        stream: u64,
    ) -> Result<ProposeKvView> {
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }

        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let max_blocks = state.block_table.len() as u32;
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        let actual_seq_len = (state.seq_len + 1) as i32;
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            actual_seq_len,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

        Ok(ProposeKvView {
            k_pool: kv_cache.k_pool_ptr(self.attn_layer_idx),
            v_pool: kv_cache.v_pool_ptr(self.attn_layer_idx),
            cache_stride: kv_cache.cache_stride() as u64,
            block_size: bs as u32,
            max_blocks,
            meta_base,
        })
    }

    /// True when MoE/FFN kernel pointers are process-lifetime (NVFP4 fused
    /// or dense FFN) and can live inside the same graph as attention.
    fn propose_moe_in_graph(&self) -> bool {
        self.moe_experts_generic.is_none() || self.bf16_moe_fused.is_some()
    }

    /// Replay cached propose graph(s), or capture them around the GPU body.
    pub(super) fn replay_or_capture_propose(
        &self,
        ctx: &ForwardContext<'_>,
        stream: u64,
        kv: &ProposeKvView,
    ) -> Result<bool> {
        let graph_ctx = ctx_for_graph(ctx);
        if self.propose_moe_in_graph() {
            let mut slot = self.propose_graph.lock();
            replay_or_capture_body(ctx.gpu, &mut slot, stream, "full", || {
                self.propose_gpu_to_argmax(&graph_ctx, stream, kv)
            })?;
            return Ok(true);
        }

        let mut pre = self.propose_graph.lock();
        let mut post = self.propose_graph_post.lock();
        replay_or_capture_body(ctx.gpu, &mut pre, stream, "pre-MoE", || {
            self.propose_gpu_pre_moe(&graph_ctx, stream, kv)
        })?;
        let ffn_out = self.propose_gpu_ffn(ctx, stream)?;
        replay_or_capture_body(ctx.gpu, &mut post, stream, "post-MoE", || {
            self.propose_gpu_post_moe(&graph_ctx, stream, ffn_out)
        })?;
        Ok(true)
    }
}

fn replay_or_capture_body(
    gpu: &dyn GpuBackend,
    slot: &mut Option<GraphHandle>,
    stream: u64,
    label: &'static str,
    mut body: impl FnMut() -> Result<()>,
) -> Result<()> {
    if let Some(graph) = *slot
        && graph.0 != 0
    {
        gpu.launch_graph(graph, stream)?;
        return Ok(());
    }

    let mut capture_active = false;
    match gpu.begin_capture(stream) {
        Ok(()) => capture_active = true,
        Err(e) => {
            tracing::warn!(
                "MTP propose CUDA graph begin_capture failed ({label}, {e:#}) — running eagerly"
            );
        }
    }

    if let Err(e) = body() {
        if capture_active {
            gpu.abort_capture_if_active(stream);
        }
        return Err(e);
    }
    if !capture_active {
        return Ok(());
    }

    match gpu.end_capture(stream) {
        Ok(graph) if graph.0 != 0 => {
            tracing::info!("Captured CUDA graph for MTP propose ({label})");
            gpu.launch_graph(graph, stream)?;
            *slot = Some(graph);
            Ok(())
        }
        Ok(_) => {
            tracing::warn!(
                "MTP propose CUDA graph capture returned null handle ({label}) — running eagerly"
            );
            Ok(())
        }
        Err(e) => {
            tracing::warn!(
                "MTP propose CUDA graph end_capture failed ({label}, {e:#}) — \
                 re-running propose eagerly"
            );
            body()
        }
    }
}

fn ctx_for_graph<'a>(ctx: &ForwardContext<'a>) -> ForwardContext<'a> {
    ForwardContext {
        buffers: ctx.buffers,
        gpu: ctx.gpu,
        config: ctx.config,
        dispatch: ctx.dispatch,
        derived: ctx.derived,
        levers: ctx.levers,
        stats: ctx.stats,
        attn_metadata: ctx.attn_metadata,
        profile: false,
        comm: ctx.comm,
        graph_capture: true,
        gdn_exact_replay: ctx.gdn_exact_replay,
        token_ids: ctx.token_ids,
        routed_lora_layers: ctx.routed_lora_layers,
        midchunk_capture: None,
        moe_lora_route: ctx.moe_lora_route,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_switch_is_exactly_one() {
        assert!(mtp_propose_graph_from(None), "unset → ON");
        assert!(mtp_propose_graph_from(Some("0")), "`=0` is NOT off");
        assert!(mtp_propose_graph_from(Some("")), "empty is NOT off");
        assert!(!mtp_propose_graph_from(Some("1")), "`=1` is the kill");
    }

    #[test]
    fn production_bf16_mtp_is_graphable() {
        // Qwen3.6-35B-A3B-NVFP4 ships a BF16 MTP head. Host MoE must not veto.
        assert!(propose_graphable(false, 0, 0.0, false, false));
    }

    #[test]
    fn d2h_observability_paths_are_not_graphable() {
        assert!(!propose_graphable(true, 0, 0.0, false, false));
        assert!(!propose_graphable(false, 4, 0.0, false, false));
        assert!(!propose_graphable(false, 0, 0.5, false, false));
        assert!(!propose_graphable(false, 0, 0.0, true, false));
        assert!(!propose_graphable(false, 0, 0.0, false, true));
    }
}
