// SPDX-License-Identifier: AGPL-3.0-only

//! The model-wide FP8-KV calibration-frozen probe. Split out of `decode_a.rs`
//! (500-LoC cap); consumed by the graph-unsuppress gates in `decode_a.rs` and
//! `verify_b.rs`.

use super::super::types::TransformerModel;

impl TransformerModel {
    /// Whether the online FP8-KV calibration has frozen its scale, model-wide.
    ///
    /// Every calibrating attention layer freezes on ITS first observe within
    /// the same first forward pass, so the first layer that reports a state
    /// speaks for all of them. `true` when NO layer runs online calibration —
    /// there is nothing to wait for, and the graph-suppression gate below is
    /// additionally guarded by `fp8_kv_calibration_tokens > 0`.
    pub(in crate::model) fn fp8_calibration_frozen(&self) -> bool {
        self.layers
            .iter()
            .find_map(|l| l.fp8_calibration_frozen())
            .unwrap_or(true)
    }
}
