// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from atlasfutures/semantic-router (https://github.com/atlasfutures/semantic-router, branch
// vsr-next, commit f6d0393), rayline-core/execution/capacity.go:14-68 and :161-206, published by Atlas Futures, Inc.
// under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0); the file carries no copyright line of
// its own. Translated from Go to Rust; changes: the per-request fact is split into a static window fact plus the
// prompt count, a pure `fits` verdict replaces the nullable fact, the shortfall relaxation works over a slice
// instead of the server's action map, and the provider-reported refusal is a provenance value. See
// RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: context-capacity facts. Whether a backend's window can hold this request's prompt plus its
//! output allowance and margin, in that backend's own tokenizer units, and which shortfalls stand when every
//! candidate falls short. A measured or bounded fit is a fit, an estimated fit is unknown, and any shortfall
//! excludes, as upstream's policy contract does.

/// How the prompt count behind a capacity verdict was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Provenance {
    /// Counted by the backend's own tokenizer (Atlas `/tokenize`).
    #[default]
    Measured,
    /// An upper bound (for example from the prepared request's bytes).
    Bounded,
    /// An estimate (for example an image charged a fixed number of tokens).
    Estimated,
    /// The runtime itself refused this exact request as over its context, before any output.
    ProviderReported,
}

/// What a backend's runtime does with a prompt over its window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Overflow {
    /// It truncates (or the behavior is unknown): an estimated shortfall stands.
    #[default]
    Truncates,
    /// It refuses with an error: an estimated or bounded shortfall may be left to the runtime to decide.
    Refuses,
}

/// A backend's window fact. `context_window` is `None` (or 0) when unknown, which never means "fits".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityFact {
    pub context_window: Option<u64>,
    /// The request's output allowance in tokens.
    pub output_limit: u64,
    /// The configured safety margin in tokens.
    pub margin_tokens: u64,
    pub provenance: Provenance,
    pub overflow: Overflow,
}

impl CapacityFact {
    /// Nothing is known about the window: every verdict is `Unknown`.
    pub const UNKNOWN: CapacityFact = CapacityFact {
        context_window: None,
        output_limit: 0,
        margin_tokens: 0,
        provenance: Provenance::Measured,
        overflow: Overflow::Truncates,
    };

    /// A window with a measured count.
    pub const fn measured(context_window: u64, output_limit: u64) -> Self {
        Self {
            context_window: Some(context_window),
            output_limit,
            margin_tokens: 0,
            provenance: Provenance::Measured,
            overflow: Overflow::Truncates,
        }
    }

    pub fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_overflow(mut self, overflow: Overflow) -> Self {
        self.overflow = overflow;
        self
    }

    pub fn with_margin(mut self, margin_tokens: u64) -> Self {
        self.margin_tokens = margin_tokens;
        self
    }
}

/// The verdict for one backend and one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    Fits,
    Exceeds,
    Unknown,
}

/// Tokens the request needs: prompt plus output allowance plus margin (saturating).
pub fn required_tokens(window: &CapacityFact, prompt_tokens: u64) -> u64 {
    prompt_tokens
        .saturating_add(window.output_limit)
        .saturating_add(window.margin_tokens)
}

/// Whether `prompt_tokens` (in the backend's own units) plus the output allowance and margin fit the window.
/// A provider-reported refusal always exceeds; an unknown window is unknown; an estimated fit is unknown
/// (only a measured or bounded count proves a fit); any shortfall exceeds.
pub fn fits(window: &CapacityFact, prompt_tokens: u64) -> Fit {
    if window.provenance == Provenance::ProviderReported {
        return Fit::Exceeds;
    }
    let Some(available) = window.context_window.filter(|w| *w > 0) else {
        return Fit::Unknown;
    };
    if required_tokens(window, prompt_tokens) > available {
        Fit::Exceeds
    } else if window.provenance == Provenance::Estimated {
        Fit::Unknown
    } else {
        Fit::Fits
    }
}

/// An estimated or bounded shortfall on a runtime that refuses an over-window prompt by itself: it never
/// settles that every backend falls short (upstream `yieldsToRuntime`).
pub fn yields_to_runtime(window: &CapacityFact, fit: Fit) -> bool {
    fit == Fit::Exceeds
        && matches!(
            window.provenance,
            Provenance::Estimated | Provenance::Bounded
        )
        && window.overflow == Overflow::Refuses
}

/// Settles the verdicts of every candidate of one request (upstream `relaxEstimatedShortfalls`). When every
/// candidate falls short, the shortfalls that yield to their runtime become `Unknown`, so the request is sent
/// there and the runtime's own refusal decides. While some candidate is not short, every shortfall stands.
/// `fits` and `windows` are parallel slices.
pub fn settle(fits: &mut [Fit], windows: &[&CapacityFact]) {
    debug_assert_eq!(fits.len(), windows.len());
    if fits.is_empty() || fits.iter().any(|f| *f != Fit::Exceeds) {
        return;
    }
    for (fit, window) in fits.iter_mut().zip(windows) {
        if yields_to_runtime(window, *fit) {
            *fit = Fit::Unknown;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_and_bounded_fits_count_estimated_is_unknown() {
        let w = CapacityFact::measured(1000, 100);
        assert_eq!(fits(&w, 900), Fit::Fits);
        assert_eq!(fits(&w, 901), Fit::Exceeds);
        assert_eq!(
            fits(&w.with_provenance(Provenance::Bounded), 900),
            Fit::Fits
        );
        assert_eq!(
            fits(&w.with_provenance(Provenance::Estimated), 900),
            Fit::Unknown
        );
        assert_eq!(
            fits(&w.with_provenance(Provenance::Estimated), 901),
            Fit::Exceeds
        );
    }

    #[test]
    fn margin_counts_and_unknown_window_is_unknown() {
        let w = CapacityFact::measured(1000, 100).with_margin(50);
        assert_eq!(fits(&w, 850), Fit::Fits);
        assert_eq!(fits(&w, 851), Fit::Exceeds);
        assert_eq!(fits(&CapacityFact::UNKNOWN, 1), Fit::Unknown);
        let zero = CapacityFact {
            context_window: Some(0),
            ..CapacityFact::UNKNOWN
        };
        assert_eq!(fits(&zero, 1), Fit::Unknown);
    }

    #[test]
    fn provider_refusal_always_exceeds_and_never_yields() {
        let w = CapacityFact::measured(1_000_000, 0).with_provenance(Provenance::ProviderReported);
        assert_eq!(fits(&w, 1), Fit::Exceeds);
        assert!(!yields_to_runtime(
            &w.with_overflow(Overflow::Refuses),
            Fit::Exceeds
        ));
    }

    #[test]
    fn saturating_requirement_never_wraps() {
        let w = CapacityFact::measured(u64::MAX, u64::MAX).with_margin(u64::MAX);
        assert_eq!(required_tokens(&w, u64::MAX), u64::MAX);
        assert_eq!(fits(&w, u64::MAX), Fit::Fits);
    }

    #[test]
    fn settle_relaxes_only_when_every_candidate_is_short() {
        let measured = CapacityFact::measured(100, 0);
        let est = CapacityFact::measured(100, 0)
            .with_provenance(Provenance::Estimated)
            .with_overflow(Overflow::Refuses);
        let est_trunc = CapacityFact::measured(100, 0).with_provenance(Provenance::Estimated);

        let mut all_short = [Fit::Exceeds, Fit::Exceeds, Fit::Exceeds];
        settle(&mut all_short, &[&measured, &est, &est_trunc]);
        assert_eq!(all_short, [Fit::Exceeds, Fit::Unknown, Fit::Exceeds]);

        let mut one_fits = [Fit::Fits, Fit::Exceeds];
        settle(&mut one_fits, &[&measured, &est]);
        assert_eq!(one_fits, [Fit::Fits, Fit::Exceeds]);

        let mut empty: [Fit; 0] = [];
        settle(&mut empty, &[]);
    }
}
