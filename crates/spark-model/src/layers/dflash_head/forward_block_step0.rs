// SPDX-License-Identifier: AGPL-3.0-only

//! Step 0 (fc projection of captured target hiddens) of
//! `BlockDiffusionDraftHead::forward_block`. Extracted from
//! `forward_block.rs` so the parent file fits the 500-LoC budget.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;

/// Inputs the fc-projection step needs from the surrounding `forward_block`
/// body, so the helper can run without re-deriving them.
pub(super) struct Step0Args {
    pub last_token: u32,
    pub position: usize,
    pub stream: u64,
    pub ctx_base_ptr: Option<DevicePtr>,
    pub ctx_total: usize,
    pub eff_ctx: usize,
    pub ctx_slot_bytes: usize,
    pub h: u32,
    pub target_hidden_dim: usize,
    pub debug_dump: bool,
}

impl BlockDiffusionDraftHead {
    /// For each of the `eff_ctx` most-recent ctx positions, run a GEMM
    /// through `self.fc` (input: 10240 BF16 → output: 2048 BF16) and then
    /// per-row RMSNorm through `self.hidden_norm`. Results land contiguously
    /// in `scratch.fc_proj` shaped `[eff_ctx, hidden]`. No-op if
    /// `ctx_base_ptr` is `None` (no ctx conditioning this step).
    pub(super) fn forward_block_step0_fc_projection(
        &self,
        ctx: &crate::layer::ForwardContext,
        args: &Step0Args,
    ) -> Result<()> {
        use crate::layers::ops;

        let Step0Args {
            last_token,
            position,
            stream,
            ctx_base_ptr,
            ctx_total,
            eff_ctx,
            ctx_slot_bytes,
            h,
            target_hidden_dim,
            debug_dump,
        } = *args;
        let gpu = ctx.gpu;

        let dump_bf16 = |label: &str, ptr: DevicePtr, n: usize| -> Result<()> {
            if !debug_dump {
                return Ok(());
            }
            let mut buf = vec![0u8; n * 2];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
            Ok(())
        };

        let Some(base) = ctx_base_ptr else {
            return Ok(());
        };

        // Walk the LAST `eff_ctx` slots of the accumulator.
        let start_slot = ctx_total.saturating_sub(eff_ctx);
        // ATLAS_DFLASH_DEBUG_FORCE_PATTERN=1 overwrites the captured
        // target_hidden_stack with a deterministic test pattern so a
        // PyTorch reference run on the same input produces directly
        // comparable intermediates. Pattern: row i, col j contains
        // `0.01 * (i+1) * (j+1) / target_hidden` BF16. Mirrors
        // `dflash_pytorch_reference.py:make_input_target_hidden_stack`.
        let force_pattern = std::env::var("ATLAS_DFLASH_DEBUG_FORCE_PATTERN")
            .ok()
            .as_deref()
            == Some("1");
        if force_pattern && eff_ctx > 0 {
            let n_rows = self.target_layer_ids.len();
            let n_cols = self.target_hidden_size;
            let mut bytes = Vec::with_capacity(n_rows * n_cols * 2);
            for i in 0..n_rows {
                for j in 0..n_cols {
                    let v = 0.01_f32 * ((i + 1) as f32) * ((j + 1) as f32) / (n_cols as f32);
                    // f32 → bf16 (truncate-to-zero of low 16 bits).
                    let bits = v.to_bits();
                    let bf16_bits = (bits >> 16) as u16;
                    bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                }
            }
            gpu.copy_h2d(&bytes, base.offset(start_slot * ctx_slot_bytes))?;
        }
        // Dump the FIRST ctx slot's input target_hidden_stack (first 10 floats).
        if eff_ctx > 0 {
            dump_bf16(
                "step0.input.target_hidden_stack[0]",
                base.offset(start_slot * ctx_slot_bytes),
                10,
            )?;
        }
        // ATLAS_DFLASH_DEBUG_DUMP_FULL=1: write the full 10240-element
        // target_hidden_stack (one ctx slot) to /tmp/atlas_target_hidden.bin
        // so a Python reference can run dflash.py forward on the same
        // input and compare predicted draft tokens vs Atlas drafts.
        // Also dumps last_token + drafter outputs separately for the
        // bisect script. ONE-SHOT: writes only the first propose() call.
        static FULL_DUMP_DONE: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if eff_ctx > 0
            && !FULL_DUMP_DONE.load(std::sync::atomic::Ordering::Relaxed)
            && std::env::var("ATLAS_DFLASH_DEBUG_DUMP_FULL")
                .ok()
                .as_deref()
                == Some("1")
        {
            // Dump ALL eff_ctx slots — needed to reproduce the
            // multi-token ctx in PyTorch reference. Layout:
            // contiguous BF16, eff_ctx slots × 5 layers × 2048 dims.
            let n_bytes = eff_ctx * ctx_slot_bytes;
            let mut buf = vec![0u8; n_bytes];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(base.offset(start_slot * ctx_slot_bytes), &mut buf)?;
            if let Err(e) = std::fs::write("/tmp/atlas_target_hidden.bin", &buf) {
                tracing::warn!("DFLASH DUMP_FULL: target_hidden write failed: {e}");
            } else {
                tracing::info!(
                    "DFLASH DUMP_FULL: wrote {} bytes ({} ctx slots × {} BF16 elements) to /tmp/atlas_target_hidden.bin (last_token={}, position={}, eff_ctx={})",
                    n_bytes,
                    eff_ctx,
                    ctx_slot_bytes / 2,
                    last_token,
                    position,
                    eff_ctx,
                );
            }
            FULL_DUMP_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        // Batched GEMM over all eff_ctx slots in one call — reads fc
        // weight matrix once instead of eff_ctx sequential GEMVs.
        // Previous loop caused O(eff_ctx) weight re-reads: each GEMV
        // loaded 262 MB (fc: [5120×25600] BF16) → ~1.5ms × eff_ctx
        // sequential, growing to 300ms+ at eff_ctx=200. Single GEMM
        // reads the weight once regardless of eff_ctx.
        if eff_ctx > 0 {
            let src_all = base.offset(start_slot * ctx_slot_bytes);
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                src_all,
                &self.fc,
                self.scratch.fc_proj,
                eff_ctx as u32,
                h,
                target_hidden_dim as u32,
                stream,
            )?;
            dump_bf16("step0.fc_proj.pre_norm[0]", self.scratch.fc_proj, 10)?;
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.fc_proj,
                &self.hidden_norm,
                self.scratch.fc_proj,
                eff_ctx as u32,
                h,
                self.rms_norm_eps,
                stream,
            )?;
            dump_bf16(
                "step0.fc_proj.post_hidden_norm[0]",
                self.scratch.fc_proj,
                10,
            )?;
        }
        Ok(())
    }
}
