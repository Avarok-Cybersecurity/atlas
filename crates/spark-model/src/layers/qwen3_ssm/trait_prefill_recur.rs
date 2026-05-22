// SPDX-License-Identifier: AGPL-3.0-only

//! GDN recurrence kernel dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC
//! cap. [`Qwen3SsmLayer::prefill_gdn_recurrence`] mirrors the original
//! step 8 block 1:1 — same WY4-persistent / single-token persistent /
//! split4 dispatch, same env overrides, same kernel launches.

use super::*;

impl Qwen3SsmLayer {
    /// GDN prefill recurrence via the WY4-persistent kernel.
    ///
    /// Processes 4 tokens per iteration with WY algebraic correction,
    /// keeping H state in shared memory for the entire sequence. Falls
    /// back to single-token persistent (256..=4096), then split4 for
    /// unsupported configurations.
    ///
    /// Env overrides:
    /// - `ATLAS_DISABLE_WY4=1` — skip WY4-persistent.
    /// - `ATLAS_FORCE_PERSISTENT=1` — force single-token persistent at any `k`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_recurrence(
        &self,
        h_state: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gates_buf: DevicePtr,
        gdn_out_buf: DevicePtr,
        k: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let fp32 = 4usize;
        let gb_stride = (nv * 2) as u32;

        // Env overrides for kernel investigation:
        //   ATLAS_DISABLE_WY4=1       — skip WY4-persistent, fall through to
        //                               single-token persistent (256..=4096)
        //                               or split4.
        //   ATLAS_FORCE_PERSISTENT=1  — force the single-token persistent
        //                               kernel at any k (lifts the 4096 cap).
        //                               Mathematically correct per-token
        //                               sequential recurrence with FP32 SMEM
        //                               H state — useful for isolating WY
        //                               chunkwise reduction noise.
        let wy4_disabled = matches!(
            std::env::var("ATLAS_DISABLE_WY4").ok().as_deref(),
            Some("1")
        );
        let force_persistent = matches!(
            std::env::var("ATLAS_FORCE_PERSISTENT").ok().as_deref(),
            Some("1")
        );
        if force_persistent && self.gdn_prefill_persistent_k.0 != 0 {
            // Forced per-token persistent at ANY k.
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if !wy4_disabled && self.gdn_prefill_persistent_wy4_k.0 != 0 {
            // WY4-persistent: H in shared memory, 4 tokens per iteration
            // smem = H[K_DIM*V_DIM] + 8*k/q buffers + warp sums + WY scalars
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&k) && self.gdn_prefill_persistent_k.0 != 0 {
            ops::gdn_prefill_persistent(
                ctx.gpu,
                self.gdn_prefill_persistent_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            ops::gdn_prefill_split4(
                ctx.gpu,
                self.gdn_prefill_split4_k,
                h_state,
                q_ptr,
                k_ptr,
                v_ptr,
                gates_buf,
                gates_buf.offset(nv * fp32),
                gdn_out_buf,
                1,
                k,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        }
        Ok(())
    }
}
