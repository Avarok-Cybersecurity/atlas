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

use crate::layer::SsmLayerState;

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
