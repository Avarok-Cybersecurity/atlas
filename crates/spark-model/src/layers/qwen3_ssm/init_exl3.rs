// SPDX-License-Identifier: AGPL-3.0-only

//! Native EXL3 GDN projections (`ATLAS_EXL3_NATIVE_DENSE=1`) on
//! [`Qwen3SsmLayer`]: install + the two dispatch funnels every QKVZ /
//! out_proj site calls. Split out of `init.rs` (500-LoC cap), sibling of
//! `init_fp8.rs`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::exl3_dense::Exl3GdnWeights;

/// Widest `m` the chunked GEMV out_proj is used for. Above it the GEMM tier
/// is the right arm again; the crossover is a measurement, not a guess, and
/// prefill (m in the thousands) must never reach the chunked path.
const EXL3_OPROJ_CHUNK_MAX_M: usize = 32;

/// `ATLAS_EXL3_OPROJ_CHUNKED=0` restores the single GEMM-tier call.
fn exl3_oproj_chunked() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_EXL3_OPROJ_CHUNKED").as_deref() != Ok("0"))
}

impl Qwen3SsmLayer {
    /// Install the packed GDN family. Every materialized slot of BOTH the
    /// fused in-projection (BF16 dense, NVFP4 quant + transposed twin, FP8
    /// block-scaled / per-row / single-scale, Q2_0) and `out_proj` (BF16
    /// dense, NVFP4 twin, FP8 variants) must be EMPTY: a layer that carried
    /// both a packed and a materialized copy would double-hold memory and
    /// make "which arm ran" ambiguous — refused rather than resolved by
    /// priority. The layer must be `sequential_qkvz` (the pair writes the
    /// `[Q|K|V|Z]` row directly; no deinterleave exists for it) and its
    /// fused row width must be exactly what the pair produces.
    pub fn set_exl3_gdn_weights(&mut self, w: Exl3GdnWeights) -> Result<()> {
        ensure!(
            self.ssm.in_proj_qkvz.weight.is_null()
                && self.qkvz_nvfp4.is_none()
                && self.qkvz_nvfp4_t.is_none()
                && self.qkvz_fp8w.is_none()
                && self.qkvz_fp8w_rowwise.is_none()
                && self.qkvz_fp8.is_none()
                && self.qkvz_q2.is_none(),
            "EXL3 native GDN: the layer already carries a materialized in_proj_qkvz \
             copy — the loader must leave every dense/quantized QKVZ slot null when \
             it keeps the pair packed"
        );
        ensure!(
            self.ssm.out_proj.is_null()
                && self.out_proj_dense.is_none()
                && self.out_proj_nvfp4_t.is_none()
                && self.out_proj_fp8w.is_none()
                && self.out_proj_fp8w_rowwise.is_none()
                && self.out_proj_fp8.is_none(),
            "EXL3 native GDN: the layer already carries a materialized out_proj \
             copy — the loader must leave every dense/quantized out_proj slot \
             null when it keeps the projection packed"
        );
        ensure!(
            self.sequential_qkvz,
            "EXL3 native GDN: only the sequential [Q|K|V|Z] layout (separate \
             in_proj_qkv / in_proj_z on disk) is served natively"
        );
        self.exl3_gdn = Some(w);
        Ok(())
    }

    /// The installed packed GDN projections, if any.
    pub fn exl3_gdn_weights(&self) -> Option<&Exl3GdnWeights> {
        self.exl3_gdn.as_ref()
    }

    /// Refuse a capturing caller: cooperative launches are not
    /// graph-capturable. `exl3_graph_veto` keeps capture off this layer in
    /// the first place; this is the loud backstop.
    fn ensure_not_capturing(ctx: &ForwardContext, what: &str) -> Result<()> {
        ensure!(
            !ctx.graph_capture,
            "Qwen3SsmLayer: native EXL3 {what} reached under CUDA-graph capture — \
             cooperative launches are not capturable (exl3_graph_veto must veto this layer)"
        );
        Ok(())
    }

    /// The native QKVZ arm: `arena[m, qkvz_size] = [a @ in_proj_qkv | a @
    /// in_proj_z]` into the fused sequential `[Q|K|V|Z]` rows every
    /// downstream consumer reads (row stride `qkvz_size`) — the ONE funnel
    /// for the M=1 decode, batched-decode, multi-seq and prefill QKVZ sites.
    /// `arena_bf16` is the `ssm_deinterleaved` row block (sequential layout:
    /// no deinterleave follows). Checks the model's fused row width against
    /// the pair's so a geometry drift fails here, not in conv1d.
    pub(super) fn exl3_in_proj(
        &self,
        g: &Exl3GdnWeights,
        ctx: &ForwardContext,
        a_bf16: DevicePtr,
        arena_bf16: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        Self::ensure_not_capturing(ctx, "in_proj_qkv/in_proj_z")?;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        ensure!(
            g.qkvz_row_elems() == qkvz_size,
            "Qwen3SsmLayer: native EXL3 in_proj pair writes {}-wide rows but the model's \
             fused QKVZ row is {qkvz_size} — geometry drift between the packed weights and \
             the config",
            g.qkvz_row_elems(),
        );
        g.in_proj_linear(ctx.gpu, a_bf16, arena_bf16, m, stream)
    }

    /// The native `out_proj` arm: `dst[m, hidden] = a[m, value_dim] @ W`,
    /// contiguous BF16 both sides — the ONE funnel for the M=1 decode,
    /// batched-decode, multi-seq and prefill sites (so a site cannot drift
    /// from the arm the parity example proves).
    pub(super) fn exl3_out_proj(
        &self,
        g: &Exl3GdnWeights,
        ctx: &ForwardContext,
        a_bf16: DevicePtr,
        dst_bf16: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        Self::ensure_not_capturing(ctx, "out_proj")?;
        // ── M > 8: chunk onto the GEMV tier ──
        // `exl3_dense_linear` branches `m <= EXL3_GEMV_MAX_M` (8) onto GEMV,
        // else the GEMM tier. The cross-sequence verify runs m = sum(ks) = 11
        // at C=4 and falls off — the same cliff that cost the NVFP4 out_proj
        // ~8x its bandwidth floor, where chunking 8+3 was worth +4.1%.
        //
        // Projection rows are INDEPENDENT (each its own GEMV against the
        // shared weight), so the split is arithmetically identical to one
        // call. Bounded: past `EXL3_OPROJ_CHUNK_MAX_M` the GEMM tier wins
        // again, and prefill (m in the thousands) must never come here — which
        // is why this lives at the verify site and not inside
        // `exl3_dense_linear`.
        let gemv_max = crate::layers::ops::EXL3_GEMV_MAX_M;
        if m > gemv_max && m <= EXL3_OPROJ_CHUNK_MAX_M && exl3_oproj_chunked() {
            let h = ctx.config.hidden_size;
            let vd = ctx.config.linear_value_head_dim * ctx.config.linear_num_value_heads;
            let bf16 = 2usize;
            let mut off = 0usize;
            while off < m {
                let take = (m - off).min(gemv_max);
                g.out_proj_linear(
                    ctx.gpu,
                    a_bf16.offset(off * vd * bf16),
                    dst_bf16.offset(off * h * bf16),
                    take,
                    stream,
                )?;
                off += take;
            }
            {
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::info!(
                        rows = m,
                        "EXL3 out_proj: CHUNKED onto the GEMV tier \
                         (ATLAS_EXL3_OPROJ_CHUNKED=0 restores the single GEMM call)"
                    );
                });
            }
            return Ok(());
        }
        g.out_proj_linear(ctx.gpu, a_bf16, dst_bf16, m, stream)
    }
}
