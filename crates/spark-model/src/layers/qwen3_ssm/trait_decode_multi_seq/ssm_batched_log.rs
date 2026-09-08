// SPDX-License-Identifier: AGPL-3.0-only

//! The one-shot "why did the batched SSM mixer decline" log, split out of
//! `ssm_batched.rs` (500-LoC cap).

use super::super::*;

impl Qwen3SsmLayer {
    /// Say WHY the batched projections declined, once. Declining is silent
    /// otherwise, and the fallback re-streams the ~50 MB QKVZ/out_proj
    /// weights once per sequence — the difference between decode that scales
    /// with N and decode that does not. A whole campaign phase was spent
    /// measuring the symptom (SSM time linear in N) without knowing which
    /// condition failed. `flags` = `[f32_conv, f32_gdn, qkvz_ok, out_ok]`.
    pub(super) fn log_batched_proj_declined(&self, n: usize, flags: [bool; 4], tc_wide_ok: bool) {
        static WHY: std::sync::Once = std::sync::Once::new();
        let [fc, fg, qk, op] = flags;
        let sq = self.sequential_qkvz;
        let nvfp4 = self.qkvz_nvfp4.is_some();
        let b4 = self.w4a16_batchm.has_base();
        let tct = self.qkvz_nvfp4_t.is_some() && self.out_proj_nvfp4_t.is_some();
        let exl3 = self.exl3_gdn.is_some();
        WHY.call_once(|| {
            tracing::info!(
                "SSM batched projections DECLINED (n={n}): sequential_qkvz={sq} \
                 f32_conv={fc} f32_gdn={fg} qkvz_ok={qk} out_ok={op} \
                 [qkvz_nvfp4={nvfp4} w4a16_gemv_batch4={b4} tc_twins={tct} \
                 tc_wide_ok={tc_wide_ok} exl3_gdn={exl3}] — falling back to the \
                 per-seq loop, which re-reads QKVZ/out_proj weights n times"
            );
        });
    }
}
