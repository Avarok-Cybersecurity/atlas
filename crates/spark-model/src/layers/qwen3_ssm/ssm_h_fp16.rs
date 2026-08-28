// SPDX-License-Identifier: AGPL-3.0-only

//! FP16 h-state storage for the GDN decode scan (`ATLAS_SSM_H_FP16`).
//!
//! The decode scan is pure state traffic — it moves 2.0 DRAM passes over h and
//! already runs at 90% of GB10's row-strided ceiling — so its time is set by
//! the state footprint and by nothing else. Storing h as FP16 halves that
//! footprint and halves the time: 183 -> 84 ms/step at n=128, measured on a
//! replica faithful to the in-serve kernel within 2.4%.
//!
//! **Stage 1 keeps the pool FP32-sized.** Prefill still writes FP32 through six
//! kernel families, so the slot must stay large enough for FP32; the FP16 state
//! occupies the first half of the same region. Nothing about allocation,
//! preflight arithmetic, snapshot sizing, spill layout or the tier fingerprint
//! changes, and every byte-wise copier (snapshot save/restore, decode ring,
//! swap file, slot migration) stays correct without knowing the dtype. The one
//! consequence the batched kernel must be told about is that consecutive slots
//! are then `h_state_bytes` apart, i.e. TWICE the dense FP16 footprint — hence
//! its explicit `h_seq_stride` parameter.
//!
//! The invariant is: **a slot holds FP32 while its sequence is prefilling and
//! FP16 while it is decoding**, and `SsmLayerState::h_is_f16` is the single
//! source of truth for which. The flip happens in exactly one place —
//! `TransformerModel::ssm_h_to_f16_dispatch`, at the top of each decode entry
//! point.
//!
//! ★ It CANNOT happen inside the layer. Decode runs under a captured CUDA
//! graph, and a conversion launched from the layer is captured into that graph
//! and then replayed on every subsequent step — re-reading the already-FP16
//! state as FP32. That produced fluent-but-degenerate output (`"Reducing!!!!!!"`)
//! while the host-side flag correctly said "already converted": the host was
//! right, the graph did not care. The layer therefore only ever SELECTS a
//! kernel, and refuses loudly if it is handed an unconverted state.
//!
//! The reverse edge is closed by writing decode-produced Marconi snapshots back
//! as FP32 (`ssm_snapshot::save`), which keeps every snapshot FP32 and leaves
//! the restore path — always into a prefill — untouched.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, SsmLayerState};

/// Refuse to run an FP16 decode kernel over a state that was never converted.
///
/// A missed conversion hook is the one failure mode of this design, and it is
/// silent by nature — an FP32 bit pattern read as two halves is a plausible
/// number, not a fault. This turns it into an error at the first decode step.
pub(super) fn require_h_f16(state: &SsmLayerState) -> Result<()> {
    if !state.h_is_f16 {
        bail!(
            "ATLAS_SSM_H_FP16: decode reached an SSM layer whose h-state is still FP32. \
             `ssm_h_to_f16_dispatch` must run at the top of every decode entry point, \
             OUTSIDE the CUDA-graph region."
        );
    }
    Ok(())
}

// ── Stage 3: the f16-SIZED pool ────────────────────────────────────────────
//
// Stage 1/2 above keep the pool FP32-sized because prefill writes the running
// h-state FP32 IN PLACE through six GDN kernel families. Stage 3
// (`--ssm-h-dtype f16-pool`) halves the pool instead, and pays for it with a
// per-slot FP32 STAGING blob (`SsmStatePool::h_prefill_stage`) that prefill
// runs over:
//
//   widen (f16 slot -> FP32 stage) -> the unchanged FP32 kernels -> narrow
//   (FP32 stage -> f16 slot)
//
// so not one prefill kernel changes and the pool holds FP16 at every moment
// OUTSIDE a layer's prefill call. That last property is the whole invariant:
// `SsmLayerState::h_is_f16` is then simply `true` for a sequence's entire
// life, the decode mixer is a no-op, snapshot save always widens, snapshot
// restore always narrows, and slot zeroing is format-agnostic.
//
// ★ Unlike the decode conversion, this pair is SAFE to launch from inside the
// layer: it is issued once per prefill call and is self-cancelling, so a
// replay would re-widen the current slot rather than re-narrow an already
// narrow one. Prefill is not CUDA-graph captured; if it ever is, this pair
// captures correctly because both halves are inside the captured region.
//
// The cost is 1.5 extra DRAM passes over one layer's h per prefill pass
// (widen reads 2 bytes/elem and writes 4, narrow the reverse): ~3.1 MB per
// layer per pass on the 27B GDN shape, ~150 MB per 48-layer pass — a few
// percent of a chunk's prefill time, against halving the per-step per-
// sequence decode state traffic that dominates concurrency.

/// A prefill pass's h-state pointer, plus what has to happen when it is done.
///
/// `Narrowed` carries BOTH ends so [`prefill_h_end`] cannot be handed a
/// mismatched pair, and so the caller never has to re-derive which pointer
/// the kernels actually ran over.
#[derive(Debug)]
pub(super) enum PrefillH {
    /// FP32-sized pool: the kernels ran over the slot itself and there is
    /// nothing to do. Every configuration before stage 3 takes this arm, and
    /// it moves zero bytes.
    InPlace(DevicePtr),
    /// f16-SIZED pool: the kernels ran over `stage`, which must be narrowed
    /// back into `slot`.
    Staged { slot: DevicePtr, stage: DevicePtr },
}

impl PrefillH {
    /// The pointer the FP32 prefill kernels must be given.
    pub(super) fn ptr(&self) -> DevicePtr {
        match self {
            Self::InPlace(p) => *p,
            Self::Staged { stage, .. } => *stage,
        }
    }
}

/// Widen this sequence's h slot into its FP32 staging blob, if the pool is
/// f16-SIZED. Returns the pointer the prefill kernels must run over.
///
/// `h_f32_bytes` is the FP32 width of one layer's h blob — the ELEMENT-count
/// authority (`n = h_f32_bytes / 4`), never a duplicated shape literal.
pub(super) fn prefill_h_begin(
    gpu: &dyn GpuBackend,
    f16_to_f32_k: KernelHandle,
    state: &SsmLayerState,
    h_f32_bytes: usize,
    stream: u64,
) -> Result<PrefillH> {
    let Some(stage) = state.h_prefill_stage else {
        return Ok(PrefillH::InPlace(state.h_state));
    };
    if !state.h_is_f16 {
        bail!(
            "--ssm-h-dtype f16-pool: prefill reached an SSM layer whose h-state is flagged FP32 \
             over a 2-byte-sized pool slot. The slot cannot physically hold FP32; \
             `h_is_f16` must be set at slot allocation under the f16-sized pool."
        );
    }
    if f16_to_f32_k.0 == 0 {
        bail!(
            "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f16_to_f32 did not resolve on this \
             target — refusing to run the FP32 GDN prefill kernels over a 2-byte-sized pool slot \
             (that is an out-of-bounds write into the neighbouring slot, not a precision loss)."
        );
    }
    crate::layers::ops::ssm_h_state_f16_to_f32(
        gpu,
        f16_to_f32_k,
        state.h_state,
        stage,
        (h_f32_bytes / 4) as u64,
        stream,
    )?;
    Ok(PrefillH::Staged {
        slot: state.h_state,
        stage,
    })
}

/// Narrow the FP32 staging blob back into the h slot. No-op on the
/// FP32-sized pool.
///
/// ★ MUST run on the SAME stream as the prefill kernels — it is ordered
/// against them by stream order alone, and the staging blob is reused by the
/// next layer of the same pass.
pub(super) fn prefill_h_end(
    gpu: &dyn GpuBackend,
    f32_to_f16_k: KernelHandle,
    h: PrefillH,
    h_f32_bytes: usize,
    stream: u64,
) -> Result<()> {
    let PrefillH::Staged { slot, stage } = h else {
        return Ok(());
    };
    if f32_to_f16_k.0 == 0 {
        bail!(
            "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f32_to_f16 did not resolve on this \
             target — the prefill h-state cannot be narrowed back into its 2-byte-sized slot."
        );
    }
    crate::layers::ops::ssm_h_state_f32_to_f16(
        gpu,
        f32_to_f16_k,
        stage,
        slot,
        (h_f32_bytes / 4) as u64,
        stream,
    )
}

impl Qwen3SsmLayer {
    /// The h pool's per-SLOT byte pitch — `h_state_bytes` on an FP32-sized
    /// pool, half that under the stage-3 f16-SIZED pool.
    ///
    /// SSOT with `SsmStatePool::h_stored_bytes` through
    /// [`crate::ssm_reserve::ssm_h_stored_bytes`], which is the point: the
    /// layer's contiguity checks and strided-kernel arguments and the
    /// allocator cannot disagree about how far apart two slots are.
    pub(super) fn h_slot_stride_bytes(&self) -> usize {
        crate::ssm_reserve::ssm_h_stored_bytes(
            self.h_state_bytes,
            super::gdn_flags::ssm_h_f16_pool_enabled(),
        )
    }

    /// [`Self::prefill_gdn_recurrence`] with the stage-3 h-state width
    /// conversion wrapped around it (`--ssm-h-dtype f16-pool`).
    ///
    /// On an FP32-sized pool (`h_prefill_stage == None` — every config before
    /// stage 3) this is EXACTLY `prefill_gdn_recurrence(ssm_state.h_state,
    /// ..)`: `prefill_h_begin` returns the slot pointer unchanged and
    /// `prefill_h_end` moves nothing. On the f16-SIZED pool it widens the
    /// 2-byte slot into the sequence's FP32 staging blob, runs the unchanged
    /// FP32 recurrence over that, and narrows the result back.
    ///
    /// ★ On a FAILED recurrence the narrowing is SKIPPED, which leaves the
    /// slot holding its pre-pass f16 value rather than a partially written
    /// one. That is the whole reason the staging blob is separate memory:
    /// under an FP32-sized pool a failed prefill scribbles the slot itself
    /// and a later snapshot save would cache the scribble.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gdn_recurrence_staged(
        &self,
        ssm_state: &SsmLayerState,
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
        midcap_idx: Option<usize>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = prefill_h_begin(
            ctx.gpu,
            self.ssm_h_f16_to_f32_k,
            ssm_state,
            self.h_state_bytes,
            stream,
        )?;
        self.prefill_gdn_recurrence(
            h.ptr(),
            q_ptr,
            k_ptr,
            v_ptr,
            gates_buf,
            gdn_out_buf,
            k,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            midcap_idx,
            ctx,
            stream,
        )?;
        prefill_h_end(
            ctx.gpu,
            self.ssm_h_f32_to_f16_k,
            h,
            self.h_state_bytes,
            stream,
        )
    }
}

#[cfg(test)]
mod prefill_narrowing_tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// One layer's h blob at the 27B GDN shape (48 v-heads × 128 × 128 FP32).
    const H_F32: usize = 48 * 128 * 128 * 4;
    const K: KernelHandle = KernelHandle(0xDEAD);

    fn state(stage: Option<DevicePtr>, h_is_f16: bool) -> SsmLayerState {
        SsmLayerState {
            h_state: DevicePtr(0x1000),
            conv_state: DevicePtr(0x2000),
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            h_is_f16,
            h_prefill_stage: stage,
            ple: None,
        }
    }

    /// FLAG OFF is byte-identical: the prefill kernels are handed the slot
    /// itself and NOT ONE conversion is launched, at either end. This is the
    /// assertion that protects every currently-serveable config from this
    /// change — a stray unconditional widen would show up here as a launch.
    #[test]
    fn fp32_sized_pool_launches_nothing_and_hands_back_the_slot() {
        let gpu = MockGpuBackend::new();
        let st = state(None, false);
        let h = prefill_h_begin(&gpu, K, &st, H_F32, 0).unwrap();
        assert_eq!(h.ptr().0, st.h_state.0);
        assert!(matches!(h, PrefillH::InPlace(_)));
        prefill_h_end(&gpu, K, h, H_F32, 0).unwrap();
        assert!(
            gpu.launches_snapshot().is_empty(),
            "the FP32-sized pool must not launch a conversion"
        );
        // ...and a NULL converter handle is not even consulted, so a target
        // without `ssm_h_dtype.cu` still serves the default mode.
        let gpu2 = MockGpuBackend::new();
        let st2 = state(None, false);
        let h2 = prefill_h_begin(&gpu2, KernelHandle(0), &st2, H_F32, 0).unwrap();
        prefill_h_end(&gpu2, KernelHandle(0), h2, H_F32, 0).unwrap();
        assert!(gpu2.launches_snapshot().is_empty());
    }

    /// Stage 3: exactly ONE widen on the way in and ONE narrow on the way
    /// out, the kernels run over the STAGING blob, and both launches carry
    /// the FP32 element count (`h_bytes / 4`) — not the byte count, and not
    /// the halved storage width.
    #[test]
    fn f16_sized_pool_widens_in_and_narrows_out_over_the_stage() {
        let gpu = MockGpuBackend::new();
        let stage = DevicePtr(0x9000);
        let st = state(Some(stage), true);

        let h = prefill_h_begin(&gpu, K, &st, H_F32, 0).unwrap();
        assert_eq!(h.ptr().0, stage.0, "the kernels must run over the stage");
        assert!(matches!(
            h,
            PrefillH::Staged { slot, stage: s } if slot.0 == st.h_state.0 && s.0 == stage.0
        ));
        let after_begin = gpu.launches_snapshot();
        assert_eq!(after_begin.len(), 1, "one widen");

        prefill_h_end(&gpu, K, h, H_F32, 0).unwrap();
        let all = gpu.launches_snapshot();
        assert_eq!(all.len(), 2, "one widen + one narrow, no more");

        // Geometry: BLOCK 256 over `n = h_bytes / 4` FP32 elements, capped at
        // 4096 blocks (grid-stride). Both halves convert the SAME element
        // count — a widen sized off the storage width would halve this.
        let n = (H_F32 / 4) as u32;
        assert_eq!(n, 786_432);
        for l in &all {
            assert_eq!(l.block, [256, 1, 1]);
            assert_eq!(l.grid, [n.div_ceil(256).clamp(1, 4096), 1, 1]);
        }
    }

    /// A missing converter is a HARD ERROR at both ends, never a silent
    /// pass-through: pointing an FP32 prefill kernel at a 2-byte-sized slot
    /// is an out-of-bounds write into the neighbouring sequence, which no
    /// later check can detect.
    #[test]
    fn a_null_converter_refuses_rather_than_running_fp32_over_a_narrow_slot() {
        let gpu = MockGpuBackend::new();
        let st = state(Some(DevicePtr(0x9000)), true);
        let e = prefill_h_begin(&gpu, KernelHandle(0), &st, H_F32, 0).unwrap_err();
        assert!(e.to_string().contains("ssm_h_state_f16_to_f32"), "{e}");
        assert!(e.to_string().contains("out-of-bounds"), "{e}");
        assert!(gpu.launches_snapshot().is_empty());

        let staged = PrefillH::Staged {
            slot: DevicePtr(0x1000),
            stage: DevicePtr(0x9000),
        };
        let e2 = prefill_h_end(&gpu, KernelHandle(0), staged, H_F32, 0).unwrap_err();
        assert!(e2.to_string().contains("ssm_h_state_f32_to_f16"), "{e2}");
    }

    /// The invariant "a staged slot is ALWAYS f16" is checked, not assumed.
    /// An FP32-flagged state over a narrowed slot means the slot never got
    /// the f16 tag at allocation, and widening it would read 2x the bytes
    /// the slot owns.
    #[test]
    fn an_fp32_flagged_state_over_a_narrow_slot_is_refused() {
        let gpu = MockGpuBackend::new();
        let st = state(Some(DevicePtr(0x9000)), false);
        let e = prefill_h_begin(&gpu, K, &st, H_F32, 0).unwrap_err();
        assert!(e.to_string().contains("flagged FP32"), "{e}");
        assert!(gpu.launches_snapshot().is_empty());
    }
}
