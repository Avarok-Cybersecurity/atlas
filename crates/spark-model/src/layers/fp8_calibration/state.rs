// SPDX-License-Identifier: AGPL-3.0-only

//! Pure, GPU-free state machine behind [`super::Fp8KvCalibration`].
//!
//! SSOT for "when does the FP8 KV scale freeze, and on what data". Split out
//! of `fp8_calibration.rs` so the decision can be unit-tested without a GPU:
//! the glue module owns the absmax kernel launch and the BF16 staging, this
//! module owns nothing but arithmetic.
//!
//! Atlas #919: the serve log line
//! `FP8 KV cache with online calibration (checkpoint ships no k/v scales):
//!  freezing per-tensor scales on the first observed tokens.`
//! meant exactly that — the freeze fired on the FIRST `observe`, so a 24k
//! serve got its per-tensor scales from whatever the readiness probe sent
//! (13 tokens). `--fp8-kv-calibration-tokens N` now means what it reads like:
//! accumulate the running amax over the first N observed tokens, ACROSS
//! requests, and freeze on the observe that reaches N.

/// FP8 E4M3 max representable magnitude.
pub(super) const FP8_E4M3_MAX: f32 = 448.0;

/// Minimum scale to prevent division by zero or denormalized values.
pub(super) const MIN_SCALE: f32 = 1e-12;

/// Post-freeze re-observation period, in tokens. Only used by the opt-in EMA
/// recalibration path (`ATLAS_FP8_KV_EMA_RECAL=1`).
pub(super) const POST_FREEZE_OBSERVE_PERIOD: usize = 128;

/// What the caller must do with the batch it is about to write.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum CalibrationStep {
    /// Still inside the calibration window. Write this batch with the current
    /// (provisional) scales AND stage its BF16 K/V + slot mapping, so the
    /// freeze can rewrite those cache entries at the final scale.
    Stage,
    /// This batch reached the window. `k_scale`/`v_scale` are final and were
    /// computed over every token observed so far, this batch included. The
    /// caller replays the staged batches at these scales, then writes this
    /// batch at them too.
    Freeze {
        k_scale: f32,
        v_scale: f32,
        tokens_seen: usize,
    },
    /// Already frozen — nothing to stage, nothing to replay.
    Frozen,
}

/// Mutable calibration state. One per attention layer.
#[derive(Debug)]
pub(super) struct CalibrationState {
    /// Running max of |K| over every token observed so far.
    pub(super) k_running_max: f32,
    /// Running max of |V| over every token observed so far.
    pub(super) v_running_max: f32,
    /// Total tokens observed (calibration window + post-freeze probes).
    pub(super) tokens_seen: usize,
    /// Whether calibration is complete (scales frozen).
    pub(super) frozen: bool,
    /// Live k_scale: provisional before the freeze, final after it.
    pub(super) k_scale: f32,
    /// Live v_scale: provisional before the freeze, final after it.
    pub(super) v_scale: f32,
    /// Tokens that must be observed before the freeze (`>= 1`).
    pub(super) window_tokens: usize,
    /// Headroom multiplier on the accumulated amax (`--fp8-kv-headroom`).
    pub(super) headroom: f32,
    /// Tokens actually staged for requantization (`<= window_tokens`).
    pub(super) staged_tokens: usize,
    /// Staged batch count, for the freeze log line.
    pub(super) staged_batches: usize,
}

impl CalibrationState {
    /// `window_tokens` is clamped to `>= 1`: `--fp8-kv-calibration-tokens 0`
    /// never constructs a calibrator at all (checkpoint/static scales win), and
    /// a 1-token window reproduces the pre-#919 freeze-on-first-observe
    /// behaviour exactly — nothing is ever staged, so nothing is ever rewritten.
    pub(super) fn new(window_tokens: usize, headroom: f32, provisional_scale: f32) -> Self {
        Self {
            k_running_max: 0.0,
            v_running_max: 0.0,
            tokens_seen: 0,
            frozen: false,
            k_scale: provisional_scale,
            v_scale: provisional_scale,
            window_tokens: window_tokens.max(1),
            headroom,
            staged_tokens: 0,
            staged_batches: 0,
        }
    }

    /// Whether this batch needs an absmax reduction at all.
    ///
    /// Every pre-freeze batch is observed (that is the whole point of the
    /// window). After the freeze only the periodic probe runs, and only to feed
    /// the opt-in EMA path.
    pub(super) fn should_observe(&self, num_tokens: usize) -> bool {
        !self.frozen || self.tokens_seen % POST_FREEZE_OBSERVE_PERIOD < num_tokens
    }

    /// Fold one observation into the window and decide what happens to the
    /// batch it came from.
    ///
    /// The freeze uses the amax over EVERY token in the window, this batch
    /// included, so a single 300-token first request freezes on all 300 rather
    /// than on a prefix of them — and a 13-token readiness probe followed by a
    /// 243-token request freezes on the max of both.
    pub(super) fn record(&mut self, k_max: f32, v_max: f32, num_tokens: usize) -> CalibrationStep {
        self.k_running_max = self.k_running_max.max(k_max);
        self.v_running_max = self.v_running_max.max(v_max);
        self.tokens_seen += num_tokens;

        if self.frozen {
            return CalibrationStep::Frozen;
        }
        if self.tokens_seen < self.window_tokens {
            self.staged_tokens += num_tokens;
            self.staged_batches += 1;
            return CalibrationStep::Stage;
        }

        // `headroom >= 1.0` (CLI-validated, clamped again in the glue), so the
        // frozen scale can never quantize the window's own amax into clipping:
        // `k_scale * 448 == k_running_max * headroom >= k_running_max`.
        self.k_scale = (self.k_running_max * self.headroom / FP8_E4M3_MAX).max(MIN_SCALE);
        self.v_scale = (self.v_running_max * self.headroom / FP8_E4M3_MAX).max(MIN_SCALE);
        self.frozen = true;
        CalibrationStep::Freeze {
            k_scale: self.k_scale,
            v_scale: self.v_scale,
            tokens_seen: self.tokens_seen,
        }
    }

    /// Opt-in post-freeze EMA nudge (`ATLAS_FP8_KV_EMA_RECAL=1`).
    ///
    /// Default OFF and deliberately so: moving a frozen scale re-bases every
    /// already-written cache entry. Kept because the flag is documented.
    pub(super) fn ema_recalibrate(&mut self, k_max: f32, v_max: f32) {
        let new_k = (k_max / FP8_E4M3_MAX).max(MIN_SCALE);
        let new_v = (v_max / FP8_E4M3_MAX).max(MIN_SCALE);
        let k_shift = (new_k - self.k_scale).abs() / self.k_scale.max(MIN_SCALE);
        let v_shift = (new_v - self.v_scale).abs() / self.v_scale.max(MIN_SCALE);
        let alpha = if k_shift > 0.2 || v_shift > 0.2 {
            0.3
        } else {
            0.1
        };
        self.k_scale = (1.0 - alpha) * self.k_scale + alpha * new_k;
        self.v_scale = (1.0 - alpha) * self.v_scale + alpha * new_v;
        self.k_running_max = k_max;
        self.v_running_max = v_max;
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod state_tests;
