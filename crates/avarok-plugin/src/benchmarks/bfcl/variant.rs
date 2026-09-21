// SPDX-License-Identifier: AGPL-3.0-only

//! Per-variant constants for the BFCL family: which descriptor and plugin
//! metadata a variant carries, its draw percentages and floor, and the
//! sample count its draw is pinned to.
//!
//! Split out of `mod.rs` to keep that file under the repository's 500-LoC
//! cap. The `Variant` enum itself stays in `mod.rs`; only its inherent impl
//! moved, so nothing about the public surface changed.

use super::*;

impl Variant {
    pub(super) fn descriptor(self) -> &'static BenchmarkDescriptor {
        match self {
            Variant::Subset => &SUBSET_DESCRIPTOR,
            Variant::SubsetEcholp => &SUBSET_ECHOLP_DESCRIPTOR,
            Variant::Full => &FULL_DESCRIPTOR,
        }
    }
    pub(super) fn metadata(self) -> &'static PluginMetadata {
        match self {
            Variant::Subset => &SUBSET_METADATA,
            Variant::SubsetEcholp => &ECHOLP_METADATA,
            Variant::Full => &FULL_METADATA,
        }
    }
    pub(super) fn default_pct(self, category: &str) -> f64 {
        match (self, category) {
            (Variant::Full, _) => 100.0,
            (Variant::Subset, "non_live") => 62.0,
            (Variant::Subset, _) => 10.0,
            (Variant::SubsetEcholp, "non_live") => 46.0,
            (Variant::SubsetEcholp, "live") => 23.0,
            (Variant::SubsetEcholp, _) => 12.0,
        }
    }
    /// The subset floor this variant's draw is DEFINED with.
    ///
    /// ★ Read from the variant's own `DrawSpec`, never written out again here.
    /// `configure` rebuilds the whole spec from parameter defaults, so a floor
    /// spelled out a second time in this file is a second source of truth that
    /// silently wins. It already went wrong exactly that way: the echolp
    /// variant was added without extending an `if v == Variant::Subset { 25 }
    /// else { 0 }`, so its floor defaulted to 0. That takes `live_parallel`
    /// (16 rows) and `live_parallel_multiple` (24) by percentage instead of
    /// whole, and the draw silently became n=972 rather than the pinned 1004 --
    /// a plausible-looking score measured against a baseline for a different
    /// draw.
    pub(super) fn default_floor(self) -> usize {
        self.spec().subset_floor.unwrap_or(0)
    }

    /// The draw this variant is defined by. Single source of truth for both
    /// the constructor and the parameter defaults.
    pub(super) fn spec(self) -> DrawSpec {
        match self {
            Variant::Subset => DrawSpec::golden(),
            Variant::SubsetEcholp => DrawSpec::echolp(),
            Variant::Full => DrawSpec::full(),
        }
    }

    /// The sample count this draw must produce, if it is a pinned draw.
    ///
    /// A draw that silently drifts off its pinned n produces a score that looks
    /// fine and compares against nothing — the same failure mode as scoring one
    /// draw against another's threshold.
    pub(super) fn expected_samples(self) -> Option<usize> {
        match self {
            Variant::Subset => Some(995),
            Variant::SubsetEcholp => Some(1004),
            Variant::Full => None,
        }
    }
}
