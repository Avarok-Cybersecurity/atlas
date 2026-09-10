// SPDX-License-Identifier: AGPL-3.0-only

//! Write-on-accept for the batched K=4 GDN verify (`gated_delta_rule_wy4_woa`
//! plus `gated_delta_rule_wy4_fold`): the layer-side state, the ENGAGE
//! decision as a pure function, the one-time stash binding and the
//! post-verdict fold.
//!
//! ## The engaged word
//!
//! The batched verify replays as a CUDA graph, so the only reliable record
//! of WHICH kernel ran for a layer is a device word written inside the graph:
//! the woa twin sets it to 1. It is CLEARED at the top of every batched
//! verify that requests write-on-accept, inside the same capture, so the
//! word describes THAT launch only. A step that ran the twin and then never
//! folded (verify error, short result) cannot leak its stash into the next
//! step's fold: the next verify clears the word before anything runs.
//!
//! ## Who asks
//!
//! Write-on-accept is never a layer default. The caller asks per verify
//! (`ForwardContext::gdn_write_on_accept`, set from
//! `VerifyBatchedOpts::write_on_accept` by the DFlash batched step only).
//! The MTP batched K-row verify shares this layer code and commits through
//! `verify_k4_verdict`, which never folds; with the request off, it always
//! gets the parent wy4 (every intermediate written) exactly as before.
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use std::sync::atomic::Ordering;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layers::ops;

/// The three kernel handles, `KernelHandle(0)` each when the module is
/// absent (`try_kernel`); the decision below gates on all three.
pub(super) struct WoaKernels {
    pub woa_k: KernelHandle,
    pub fold_k: KernelHandle,
    pub clear_k: KernelHandle,
}

pub(super) fn woa_kernels(gpu: &dyn GpuBackend) -> WoaKernels {
    let m = "gated_delta_rule_wy4_woa";
    WoaKernels {
        woa_k: crate::layers::try_kernel(gpu, m, "gated_delta_rule_wy4_woa"),
        fold_k: crate::layers::try_kernel(gpu, m, "gated_delta_rule_wy4_fold"),
        clear_k: crate::layers::try_kernel(gpu, m, "gated_delta_rule_wy4_flag_clear"),
    }
}

/// Per-sequence stash width in f32: `vn[4][nv][vd] | g[4][nv] | sk[4][nk][kd]`.
/// ~131 KB per sequence at 48 v-heads x 16 k-heads x 128.
pub(super) const fn stash_seq_floats(nk: usize, nv: usize, kd: usize, vd: usize) -> usize {
    4 * (nv * vd + nv + nk * kd)
}

/// Everything the engage decision depends on, so it can be pinned by tests
/// without a GPU. `kd`/`vd` are the head dims the kernels are built for
/// (128 x 128: the host OWNS this check, the kernels do not re-check).
#[derive(Clone, Copy, Debug)]
pub(super) struct WoaRequest {
    /// `ForwardContext::gdn_write_on_accept`: the caller will fold.
    pub requested: bool,
    /// Verify width of this batched call.
    pub kk: usize,
    /// `ATLAS_NO_GDN_WOA` not set.
    pub enabled: bool,
    /// `--ssm-h-dtype f16`: the twins are FP32 readers.
    pub h_f16: bool,
    pub kd: usize,
    pub vd: usize,
    /// All three kernel handles resolved.
    pub kernels_linked: bool,
    /// The model bound a stash for this layer (`seqs` > 0).
    pub bound_seqs: usize,
    /// Batch width of this call.
    pub n: usize,
}

/// The ONE place that decides whether the K=4 write-on-accept twin runs.
pub(super) const fn woa_decision(r: WoaRequest) -> bool {
    r.requested
        && r.kk == 4
        && r.enabled
        && !r.h_f16
        && r.kd == 128
        && r.vd == 128
        && r.kernels_linked
        && r.bound_seqs > 0
        && r.n <= r.bound_seqs
}

impl Qwen3SsmLayer {
    fn woa_flag(&self) -> DevicePtr {
        DevicePtr(self.woa_flag.load(Ordering::Acquire))
    }
    fn woa_stash(&self) -> DevicePtr {
        DevicePtr(self.woa_stash.load(Ordering::Acquire))
    }
    pub(super) fn woa_kernels_linked(&self) -> bool {
        self.gdn_wy4_woa_k.0 != 0 && self.gdn_wy4_fold_k.0 != 0 && self.gdn_wy4_clear_k.0 != 0
    }

    /// `TransformerLayer::gdn_woa_stash_seq_floats`.
    pub(super) fn woa_stash_seq_floats_impl(&self) -> Option<usize> {
        let [nk, nv, kd, vd] = self.woa_dims;
        (self.woa_kernels_linked() && super::gdn_flags::gdn_woa_enabled() && kd == 128 && vd == 128)
            .then(|| stash_seq_floats(nk, nv, kd, vd))
    }

    /// `TransformerLayer::gdn_woa_bind`: pre-capture, once.
    pub(super) fn woa_bind_impl(&self, flag: DevicePtr, stash: DevicePtr, seqs: usize) {
        self.woa_flag.store(flag.0, Ordering::Release);
        self.woa_stash.store(stash.0, Ordering::Release);
        self.woa_seqs.store(seqs, Ordering::Release);
    }

    /// The engage decision for THIS batched call.
    pub(super) fn woa_now(&self, requested: bool, kk: usize, n: usize) -> bool {
        let [_, _, kd, vd] = self.woa_dims;
        woa_decision(WoaRequest {
            requested,
            kk,
            enabled: super::gdn_flags::gdn_woa_enabled(),
            h_f16: super::ssm_h_fp16_enabled(),
            kd,
            vd,
            kernels_linked: self.woa_kernels_linked(),
            bound_seqs: if self.woa_stash().is_null() {
                0
            } else {
                self.woa_seqs.load(Ordering::Acquire)
            },
            n,
        })
    }

    /// Clear the engaged word at the top of a batched verify that requested
    /// write-on-accept: one one-thread node per layer per step, inside the
    /// same capture, so the word describes this launch only. No-op when the
    /// layer is not bound (nothing could have set the word).
    pub(super) fn woa_clear_at_entry(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let flag = self.woa_flag();
        if flag.is_null() || self.gdn_wy4_clear_k.0 == 0 {
            return Ok(());
        }
        ops::gdn_wy4_flag_clear(gpu, self.gdn_wy4_clear_k, flag, stream)
    }

    /// Launch the K=4 write-on-accept twin. A launch failure is an error,
    /// not a fallback: the 64 KB dynamic-smem opt-in is set by the launch
    /// path for every launch over 48 KB (`registry.rs`), so a refusal here
    /// is a build or device mismatch that should stop the step, and a
    /// silent kernel swap inside a graph capture is exactly what the review
    /// of PR #844 asked to remove.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn woa_launch(
        &self,
        gpu: &dyn GpuBackend,
        wy_tables: DevicePtr,
        q_ptr: DevicePtr,
        k_ptr: DevicePtr,
        v_ptr: DevicePtr,
        gate_ptr: DevicePtr,
        beta_ptr: DevicePtr,
        gdn_out_buf: DevicePtr,
        n: usize,
        conv_dim: usize,
        stream: u64,
    ) -> Result<()> {
        let [nk, nv, kd, vd] = self.woa_dims;
        ops::gdn_decode_wy4_woa(
            gpu,
            self.gdn_wy4_woa_k,
            wy_tables,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_out_buf,
            self.woa_stash(),
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            (nv * 2) as u32,
            stash_seq_floats(nk, nv, kd, vd) as u32,
            self.woa_flag(),
            stream,
        )?;
        static WOA_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !WOA_LOGGED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "batched-verify GDN WRITE-ON-ACCEPT ENGAGED (n={n}, k=4): wy4_woa + post-verdict fold \
                 (state read once, written once per step)"
            );
        }
        Ok(())
    }

    /// `TransformerLayer::gdn_fold_accepted`. The fold kernel reads the
    /// engaged word: 1 folds the stash (rows 0..na), 0 performs the parent's
    /// partial-accept restore from the Hi tables. Either way h is committed
    /// here and the caller skips its h restore. No clear after the fold:
    /// the next requesting verify clears at entry.
    pub(super) fn fold_accepted_impl(
        &self,
        gpu: &dyn GpuBackend,
        h_table: DevicePtr,
        na_tab: DevicePtr,
        k_rows: usize,
        n: usize,
        stream: u64,
    ) -> Result<bool> {
        let flag = self.woa_flag();
        let stash = self.woa_stash();
        if !super::gdn_flags::gdn_woa_enabled()
            || self.gdn_wy4_fold_k.0 == 0
            || flag.is_null()
            || stash.is_null()
            || h_table.is_null()
            || n > self.woa_seqs.load(Ordering::Acquire)
        {
            return Ok(false);
        }
        let [nk, nv, kd, vd] = self.woa_dims;
        let hi_tables = h_table.offset(crate::layer::VERIFY_WY_TABLE_STRIDE_BYTES);
        ops::gdn_wy4_fold(
            gpu,
            self.gdn_wy4_fold_k,
            h_table,
            stash,
            na_tab,
            hi_tables,
            crate::layer::VERIFY_WY_TABLE_SEQS as u32,
            flag,
            k_rows as u32,
            n as u32,
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            stash_seq_floats(nk, nv, kd, vd) as u32,
            stream,
        )?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{WoaRequest, stash_seq_floats, woa_decision};

    const GO: WoaRequest = WoaRequest {
        requested: true,
        kk: 4,
        enabled: true,
        h_f16: false,
        kd: 128,
        vd: 128,
        kernels_linked: true,
        bound_seqs: 16,
        n: 16,
    };

    /// The DFlash batched step at K=4 with everything linked and bound.
    #[test]
    fn dflash_k4_request_engages() {
        assert!(woa_decision(GO));
        assert!(woa_decision(WoaRequest { n: 2, ..GO }));
    }

    /// THE BLOCKER, pinned: a caller that did not ask never gets the twin,
    /// whatever else is true. The MTP batched K-row verify (n=2..4, K=4 by
    /// the ladder) passes `requested: false` and keeps the parent wy4, whose
    /// intermediates its verdict path restores from.
    #[test]
    fn unrequested_never_engages() {
        for n in 2..=4 {
            assert!(!woa_decision(WoaRequest {
                requested: false,
                n,
                ..GO
            }));
        }
    }

    /// Every other gate declines on its own: width, kill switch, f16 pool,
    /// head dims (host-owned check), missing kernels, unbound stash, and a
    /// batch wider than the bound stash.
    #[test]
    fn each_gate_declines_alone() {
        assert!(!woa_decision(WoaRequest { kk: 3, ..GO }));
        assert!(!woa_decision(WoaRequest { kk: 5, ..GO }));
        assert!(!woa_decision(WoaRequest {
            enabled: false,
            ..GO
        }));
        assert!(!woa_decision(WoaRequest { h_f16: true, ..GO }));
        assert!(!woa_decision(WoaRequest { kd: 64, ..GO }));
        assert!(!woa_decision(WoaRequest { vd: 256, ..GO }));
        assert!(!woa_decision(WoaRequest {
            kernels_linked: false,
            ..GO
        }));
        assert!(!woa_decision(WoaRequest {
            bound_seqs: 0,
            ..GO
        }));
        assert!(!woa_decision(WoaRequest {
            bound_seqs: 8,
            n: 9,
            ..GO
        }));
    }

    /// The stash width the model sizes from matches the kernel's layout
    /// macros: vn[4][nv][vd] | g[4][nv] | sk[4][nk][kd].
    #[test]
    fn stash_width_matches_kernel_layout() {
        assert_eq!(
            stash_seq_floats(16, 48, 128, 128),
            4 * (48 * 128 + 48 + 16 * 128)
        );
        assert_eq!(stash_seq_floats(16, 48, 128, 128) * 4, 131_840);
    }
}
