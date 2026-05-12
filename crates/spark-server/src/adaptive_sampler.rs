// SPDX-License-Identifier: AGPL-3.0-only

//! Adaptive sampling: per-token temperature and sampling adjustments based on
//! generation context. Implements zone-based temperature control, greedy-threshold
//! gating (arXiv:2510.05987), entropy-based diversity injection, and LZ compression
//! ratio monitoring.
//!
//! The adaptive system operates between logit computation and final sampling.
//! It piggybacks on existing `ActiveSeq` state (tool_call_opened, inside_thinking,
//! grammar_state) rather than scanning token text.
//!
//! Disabled during MTP verify (argmax-only acceptance). Applied to bootstrap
//! decode and non-MTP decode paths.

use std::collections::VecDeque;

/// What the model is currently generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationZone {
    /// Free text output (highest diversity).
    FreeText,
    /// Inside `<think>...</think>` (model reasoning, moderate diversity).
    Thinking,
    /// Inside `<tool_call>...</tool_call>` (JSON structure, lowest diversity).
    ToolCall,
    /// Grammar-constrained output (reduced diversity, grammar handles structure).
    StructuredOutput,
}

/// Per-sequence adaptive sampling state machine.
pub struct AdaptiveSamplingState {
    /// Current generation zone.
    pub zone: GenerationZone,
    /// Rolling entropy window (last 32 tokens).
    entropy_window: VecDeque<f32>,
    /// Consecutive tokens below low-entropy threshold.
    consecutive_low_entropy: u32,
    /// LZ compression ratio of recent output (updated every 16 tokens).
    lz_ratio: f32,
    /// Last LZ check position (to avoid recomputing every token).
    lz_last_check: usize,
    /// Base temperature from the request.
    base_temperature: f32,
}

impl AdaptiveSamplingState {
    /// Create a new adaptive state from the request's base temperature.
    pub fn new(base_temperature: f32) -> Self {
        Self {
            zone: GenerationZone::FreeText,
            entropy_window: VecDeque::with_capacity(32),
            consecutive_low_entropy: 0,
            lz_ratio: 1.0,
            lz_last_check: 0,
            base_temperature,
        }
    }

    /// Update zone from ActiveSeq state flags.
    pub fn update_zone(
        &mut self,
        tool_call_opened: bool,
        inside_thinking: bool,
        grammar_active: bool,
    ) {
        self.zone = if tool_call_opened {
            GenerationZone::ToolCall
        } else if inside_thinking {
            GenerationZone::Thinking
        } else if grammar_active {
            GenerationZone::StructuredOutput
        } else {
            GenerationZone::FreeText
        };
    }

    /// Compute effective temperature for this token.
    pub fn effective_temperature(&self) -> f32 {
        let base = self.base_temperature;
        if base == 0.0 {
            return 0.0; // Greedy request — adaptive doesn't override
        }

        let zone_temp = match self.zone {
            GenerationZone::ToolCall => base.min(0.3),
            GenerationZone::StructuredOutput => base * 0.6,
            GenerationZone::Thinking => base,
            GenerationZone::FreeText => base,
        };

        // Apply entropy diversity boost (only for non-tool zones)
        let entropy_boost = self.entropy_diversity_boost();

        // Apply LZ compression ratio multiplier (only for non-tool zones)
        let lz_mult = self.lz_temperature_multiplier();

        (zone_temp + entropy_boost) * lz_mult
    }

    /// Greedy-threshold gate (arXiv:2510.05987).
    /// Returns true if the top-1 softmax probability exceeds the zone threshold,
    /// meaning argmax should be used regardless of temperature.
    pub fn should_use_greedy(&self, f32_logits: &[f32]) -> bool {
        if self.base_temperature == 0.0 {
            return true; // Already greedy
        }

        let threshold = match self.zone {
            GenerationZone::ToolCall => 0.8,
            GenerationZone::Thinking => 0.95,
            GenerationZone::StructuredOutput => 0.85,
            GenerationZone::FreeText => 0.9,
        };

        // Compute top-1 softmax probability efficiently
        let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return false;
        }
        let sum_exp: f32 = f32_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let top_prob = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

        top_prob >= threshold
    }

    /// Track per-token entropy for diversity monitoring.
    pub fn observe_entropy(&mut self, f32_logits: &[f32]) {
        let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return;
        }
        let sum_exp: f32 = f32_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        if sum_exp <= 0.0 {
            return;
        }
        let entropy: f32 = f32_logits
            .iter()
            .map(|&l| {
                let p = (l - max_logit).exp() / sum_exp;
                if p > 1e-10 { -p * p.ln() } else { 0.0 }
            })
            .sum();

        if self.entropy_window.len() >= 32 {
            self.entropy_window.pop_front();
        }
        self.entropy_window.push_back(entropy);

        if entropy < 0.5 {
            self.consecutive_low_entropy += 1;
        } else {
            self.consecutive_low_entropy = 0;
        }
    }

    /// Update LZ compression ratio every 16 tokens.
    pub fn update_lz_ratio(&mut self, output_tokens: &[u32]) {
        if output_tokens.len() < 32 || output_tokens.len() - self.lz_last_check < 16 {
            return;
        }
        self.lz_last_check = output_tokens.len();

        let window = &output_tokens[output_tokens.len().saturating_sub(128)..];
        let bytes: Vec<u8> = window.iter().flat_map(|&t| t.to_le_bytes()).collect();

        let mut seen = std::collections::HashSet::new();
        let mut total = 0usize;
        for n in 3..=6 {
            for w in bytes.windows(n) {
                seen.insert(w);
                total += 1;
            }
        }
        self.lz_ratio = if total > 0 {
            seen.len() as f32 / total as f32
        } else {
            1.0
        };
    }

    /// Temperature boost when entropy has been low for too long.
    fn entropy_diversity_boost(&self) -> f32 {
        if self.zone == GenerationZone::ToolCall {
            return 0.0;
        }
        match self.consecutive_low_entropy {
            0..=7 => 0.0,
            8..=15 => 0.1,
            16..=31 => 0.2,
            _ => 0.3,
        }
    }

    /// Temperature multiplier based on LZ compression ratio.
    fn lz_temperature_multiplier(&self) -> f32 {
        if self.zone == GenerationZone::ToolCall {
            return 1.0;
        }
        if self.lz_ratio < 0.15 {
            1.8
        } else if self.lz_ratio < 0.25 {
            1.4
        } else if self.lz_ratio < 0.35 {
            1.2
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(temp: f32, zone: GenerationZone) -> AdaptiveSamplingState {
        let mut s = AdaptiveSamplingState::new(temp);
        s.zone = zone;
        s
    }

    #[test]
    fn temperature_zero_stays_zero() {
        // Greedy request must not be overridden by adaptive logic.
        let s = st(0.0, GenerationZone::FreeText);
        assert_eq!(s.effective_temperature(), 0.0);
    }

    #[test]
    fn tool_call_zone_caps_temperature() {
        // ToolCall zone clamps base to <=0.3 even when caller wanted higher.
        let s = st(1.0, GenerationZone::ToolCall);
        assert!(s.effective_temperature() <= 0.3);
    }

    #[test]
    fn structured_zone_scales_down() {
        // StructuredOutput multiplies base by 0.6 (no entropy boost yet).
        let s = st(1.0, GenerationZone::StructuredOutput);
        let t = s.effective_temperature();
        assert!((t - 0.6).abs() < 1e-4, "expected ~0.6, got {t}");
    }

    #[test]
    fn freetext_zone_passes_through() {
        // FreeText with no entropy history is unchanged.
        let s = st(0.7, GenerationZone::FreeText);
        let t = s.effective_temperature();
        assert!((t - 0.7).abs() < 1e-4, "expected ~0.7, got {t}");
    }

    #[test]
    fn should_use_greedy_when_temp_zero() {
        let s = st(0.0, GenerationZone::FreeText);
        assert!(s.should_use_greedy(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn should_use_greedy_when_top_prob_high() {
        // Logits with one very dominant entry → top1 prob ≈ 1.0 > any threshold.
        let s = st(0.7, GenerationZone::FreeText);
        assert!(s.should_use_greedy(&[20.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn should_not_use_greedy_when_top_prob_low() {
        // Uniform logits → top1 prob = 1/N → well below threshold.
        let s = st(0.7, GenerationZone::FreeText);
        assert!(!s.should_use_greedy(&[1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn should_not_use_greedy_with_non_finite_logits() {
        // NaN-only / all-NEG_INF logits return false (safe default).
        let s = st(0.7, GenerationZone::FreeText);
        assert!(!s.should_use_greedy(&[f32::NEG_INFINITY, f32::NEG_INFINITY]));
    }

    #[test]
    fn entropy_boost_zero_under_threshold() {
        let s = st(0.7, GenerationZone::FreeText);
        // consecutive_low_entropy starts at 0 → no boost.
        assert_eq!(s.entropy_diversity_boost(), 0.0);
    }

    #[test]
    fn entropy_boost_tool_zone_disabled() {
        let mut s = st(0.7, GenerationZone::ToolCall);
        s.consecutive_low_entropy = 32; // would trigger 0.3 boost
        // ToolCall zone short-circuits to 0.
        assert_eq!(s.entropy_diversity_boost(), 0.0);
    }

    #[test]
    fn lz_multiplier_tool_zone_disabled() {
        let mut s = st(0.7, GenerationZone::ToolCall);
        s.lz_ratio = 0.05; // would trigger 1.8x boost
        // ToolCall zone short-circuits to 1.0.
        assert_eq!(s.lz_temperature_multiplier(), 1.0);
    }

    #[test]
    fn lz_multiplier_brackets() {
        let mut s = st(0.7, GenerationZone::FreeText);
        s.lz_ratio = 0.10;
        assert_eq!(s.lz_temperature_multiplier(), 1.8);
        s.lz_ratio = 0.20;
        assert_eq!(s.lz_temperature_multiplier(), 1.4);
        s.lz_ratio = 0.30;
        assert_eq!(s.lz_temperature_multiplier(), 1.2);
        s.lz_ratio = 0.50;
        assert_eq!(s.lz_temperature_multiplier(), 1.0);
    }
}
