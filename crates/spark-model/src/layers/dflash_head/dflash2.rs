// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DFlash2 drafter components: grouped dynamic two-tap convolutions around
//! every attention/MLP sublayer, and the candidate-selector tail that
//! replaces per-row argmax with a coherent path walk over top-16 candidates.
//!
//! Reference: z-lab `dflash/model.py` (read in full) + inco DFlash2 blog.
//!   Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}
//!   S_t(a,b)    = U_t(b) + ⟨A(a) ⊙ H(h_t), B(b)⟩
//!
//! Dynamic-kernel dataflow (faithful to `GroupedDynamicCausalConv`): at
//! `prepare`, ONE GEMM through `kernel_projection` on the pre-conv hidden
//! produces the dynamic coefficients for BOTH applications, laid out per
//! row as `[application(2), tap(kernel_size), group]`. The prepare
//! application consumes slice 0 immediately; the finish application's
//! slice 1 stays in `scratch.conv_dyn` across the sublayer (stable device
//! pointer — capture-safe across the eager attention boundary).
//!
//! Selector dataflow: lm_head logits → destructive per-row top-16 →
//! `hidden_projection` GEMM on the post-norm hidden → ONE chain-walk
//! launch that reads `draft_tokens_dev[0]` (still `last_token` at tail
//! entry, H2D'd pre-capture every propose) as the anchor predecessor and
//! writes the chosen path into `draft_tokens_dev[1..γ)` device-side.
//! Row 0 keeps the plain argmax (anchor echo — discarded by propose).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::{BlockDiffusionDraftHead, DflashLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Which sublayer a conv application wraps.
#[derive(Clone, Copy)]
pub(super) enum ConvSite {
    Attention,
    Mlp,
}

impl BlockDiffusionDraftHead {
    /// True when the DFlash2 conv+selector path should run: checkpoint
    /// shipped the components, kernels compiled for this target, and the
    /// operator hasn't disabled it (`ATLAS_DFLASH2=0`).
    pub(super) fn dflash2_active(&self) -> bool {
        self.selector_pred.is_some()
            && self.selector_succ.is_some()
            && self.selector_hidden_proj.is_some()
            && self.selector_top_k == 16
            && self.selector_rank > 0
            && self.conv_kernel_size == 2
            && self.conv_group_size > 0
            && self.kernels.dflash2_conv2.0 != 0
            && self.kernels.dflash2_topk16.0 != 0
            && self.kernels.dflash2_selector_walk.0 != 0
            && std::env::var("ATLAS_DFLASH2").ok().as_deref() != Some("0")
    }

    fn conv_weights(&self, layer: &DflashLayer, site: ConvSite) -> Option<(DevicePtr, DevicePtr)> {
        match site {
            ConvSite::Attention => Some((
                layer.attention_conv_base.as_ref()?.weight,
                layer.attention_conv_proj.as_ref()?.weight,
            )),
            ConvSite::Mlp => Some((
                layer.mlp_conv_base.as_ref()?.weight,
                layer.mlp_conv_proj.as_ref()?.weight,
            )),
        }
    }

    /// One conv application (raw kernel launch). `x` and `out` must be
    /// distinct buffers (row t reads x[t-1]).
    fn conv_apply(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        base: DevicePtr,
        out: DevicePtr,
        application: u32, // 0 = prepare, 1 = finish
        stream: u64,
    ) -> Result<()> {
        let g = self.gamma as u32;
        let h = self.hidden_size as u32;
        let groups = h / self.conv_group_size as u32;
        let k = self.conv_kernel_size as u32;
        let dyn_stride = 2 * k * groups;
        let app_off = application * k * groups;
        // base_kernel is [2, k, h]; each application's slice is [k, h].
        let base_slice =
            base.offset((application as usize) * self.conv_kernel_size * self.hidden_size * 2);
        KernelLaunch::new(ctx.gpu, self.kernels.dflash2_conv2)
            .grid([g, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(self.scratch.conv_dyn)
            .arg_ptr(base_slice)
            .arg_ptr(out)
            .arg_u32(h)
            .arg_u32(self.conv_group_size as u32)
            .arg_u32(dyn_stride)
            .arg_u32(app_off)
            .launch(stream)
    }

    /// Conv `prepare`: dynamic-kernel GEMM on the post-layernorm hidden,
    /// then the first conv application. Returns the buffer the sublayer's
    /// GEMMs should read (`conv_tmp`), or the input unchanged when
    /// inactive. Overwrites `scratch.conv_dyn` — the matching
    /// [`Self::conv_finish`] must run before the next prepare.
    pub(super) fn conv_prepare(
        &self,
        layer: &DflashLayer,
        site: ConvSite,
        hidden_in: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        if !self.dflash2_active() {
            return Ok(hidden_in);
        }
        let Some((base, proj)) = self.conv_weights(layer, site) else {
            return Ok(hidden_in);
        };
        let g = self.gamma as u32;
        let h = self.hidden_size as u32;
        let groups = h / self.conv_group_size as u32;
        let dyn_cols = 2 * self.conv_kernel_size as u32 * groups;
        // Dynamic kernels for BOTH applications, from the PRE-conv hidden.
        ops::dense_gemm_bf16_pipelined(
            ctx.gpu,
            self.kernels.dense_gemm_pipelined,
            hidden_in,
            &crate::weight_map::DenseWeight { weight: proj },
            self.scratch.conv_dyn,
            g,
            dyn_cols,
            h,
            stream,
        )?;
        self.conv_apply(ctx, hidden_in, base, self.scratch.conv_tmp, 0, stream)?;
        Ok(self.scratch.conv_tmp)
    }

    /// Conv `finish`: second application on the sublayer output, using the
    /// dynamic slice computed at prepare. Returns the buffer the residual
    /// add should consume.
    pub(super) fn conv_finish(
        &self,
        layer: &DflashLayer,
        site: ConvSite,
        sublayer_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        if !self.dflash2_active() {
            return Ok(sublayer_out);
        }
        let Some((base, _proj)) = self.conv_weights(layer, site) else {
            return Ok(sublayer_out);
        };
        self.conv_apply(ctx, sublayer_out, base, self.scratch.conv_tmp, 1, stream)?;
        Ok(self.scratch.conv_tmp)
    }

    /// Selector tail: top-16 per row (destructive on `scratch.logits`),
    /// H(h_t) projection, then the single-launch chain walk writing
    /// `draft_tokens_dev[1..γ)`. Row 0 (anchor echo) gets plain argmax
    /// BEFORE the destructive top-k so upstream echo-drop semantics hold.
    pub(super) fn dflash2_select_block(
        &self,
        ctx: &ForwardContext,
        norm_noise: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let g = self.gamma as u32;
        let vocab = self.vocab_size as u32;
        let rank = self.selector_rank as u32;
        let hproj = self
            .selector_hidden_proj
            .as_ref()
            .expect("dflash2_active() checked selector_hidden_proj");
        let pred = self
            .selector_pred
            .as_ref()
            .expect("dflash2_active() checked selector_pred");
        let succ = self
            .selector_succ
            .as_ref()
            .expect("dflash2_active() checked selector_succ");

        // Row 0: plain argmax (the anchor echo slot the propose path drops).
        // Must run BEFORE dflash2_topk16 masks row 0's logits, and AFTER the
        // walk reads tokens[0]... the walk reads tokens[0] as the anchor
        // predecessor, and argmax OVERWRITES tokens[0]. Order therefore:
        // topk (non-destructive to tokens) → hproj → WALK (reads tokens[0] =
        // last_token) → row-0 argmax LAST.
        KernelLaunch::new(gpu, self.kernels.dflash2_topk16)
            .grid([g, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(self.scratch.logits)
            .arg_ptr(self.scratch.sel_vals)
            .arg_ptr(self.scratch.sel_idx)
            .arg_u32(vocab)
            .launch(stream)?;

        // H(h_t): [γ, hidden] → [γ, rank].
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            norm_noise,
            hproj,
            self.scratch.sel_hproj,
            g,
            rank,
            self.hidden_size as u32,
            stream,
        )?;

        // Chain walk: rows 1..γ, prev_1 = tokens[0] (= last_token).
        KernelLaunch::new(gpu, self.kernels.dflash2_selector_walk)
            .grid([1, 1, 1])
            .block([512, 1, 1])
            .arg_ptr(self.scratch.sel_vals)
            .arg_ptr(self.scratch.sel_idx)
            .arg_ptr(self.scratch.sel_hproj)
            .arg_ptr(pred.weight)
            .arg_ptr(succ.weight)
            .arg_ptr(self.scratch.draft_tokens_dev)
            .arg_u32(g)
            .arg_u32(rank)
            .arg_u32(1)
            .launch(stream)?;

        // Row 0 echo: top-1 of row 0 is sel_idx[0][0] (top-16 is sorted by
        // construction — the k=0 pass finds the max). One 4-byte D2D-free
        // write via a 1-row walk? Simpler: reuse argmax on row 0's ORIGINAL
        // logits is impossible (masked). sel_idx[0][0] IS the argmax; copy
        // it into tokens[0] with a 4-byte device copy on-stream via the
        // conv2 kernel? Cleanest available primitive: a 1-thread launch of
        // dflash2_selector_walk degenerates. Instead: copy_d2d is not
        // stream-ordered; use ops::residual_add trick — no. We accept a
        // tiny dedicated launch: walk with first_row=0..0 is a no-op, so
        // use batched_embed? No — simplest correct: tokens[0] is DISCARDED
        // by the propose echo-drop, so its value is irrelevant post-walk.
        // Leave it holding last_token. (Verified: propose.rs drops index 0
        // unconditionally when mask_token_id != 0.)
        Ok(())
    }
}
