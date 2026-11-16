// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the static data the routing policy reads: backends with their per-request prompt size, the
//! request class and the locality header, the static policy with its presets and validation, and the error type.
//! The per-request inputs live in `inputs.rs` and the answer in `decision.rs`; both are re-exported here so every
//! policy type is reachable as `types::<Name>`. No I/O: the caller builds every value.

use std::fmt;

use crate::capacity::CapacityFact;
use crate::swap::SwapPolicy;

pub use crate::decision::{Decision, Reason, Unavailable};
pub use crate::inputs::{
    Prediction, Predictions, Price, Prices, QueueEntry, Queues, Residency, ResidencyEntry,
    TokenUsage,
};

/// Value of a successful answer in US dollars for the Sage preset (`value_usd` 4.5; its `cost_weight` 0.2222 is
/// 1 / 4.5, so the two numbers state one trade-off).
pub const SAGE_VALUE_USD: f64 = 4.5;
/// Value for the Swift preset (`value_usd` 0.1; `cost_weight` 10.0 = 1 / 0.1).
pub const SWIFT_VALUE_USD: f64 = 0.1;
/// Default prefer-local acceptance threshold (`min_p_pass`).
pub const DEFAULT_MIN_P_PASS: f64 = 0.85;

/// A backend's identity inside one policy evaluation. The front assigns these from its config order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BackendId(pub u32);

impl fmt::Display for BackendId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One configured backend, as the front sees it for this request.
#[derive(Clone, Debug, PartialEq)]
pub struct Backend {
    pub id: BackendId,
    pub name: String,
    /// Runs on the user's or organization's own hardware. Only local backends survive local-only eligibility.
    pub local: bool,
    /// Reachable and serving. An unhealthy backend is never eligible.
    pub healthy: bool,
    /// The backend's context window fact (window, output allowance, margin, provenance of the count).
    pub window: CapacityFact,
    /// This request's prompt size in the backend's own tokenizer units, when the front counted it.
    /// Windows are never compared across backends.
    pub prompt_tokens: Option<u64>,
}

impl Backend {
    /// A healthy local backend with an unknown window and no prompt count.
    pub fn local(id: u32, name: impl Into<String>) -> Self {
        Self {
            id: BackendId(id),
            name: name.into(),
            local: true,
            healthy: true,
            window: CapacityFact::UNKNOWN,
            prompt_tokens: None,
        }
    }

    /// A healthy non-local (cloud) backend with an unknown window and no prompt count.
    pub fn cloud(id: u32, name: impl Into<String>) -> Self {
        Self {
            local: false,
            ..Self::local(id, name)
        }
    }

    pub fn with_window(mut self, window: CapacityFact) -> Self {
        self.window = window;
        self
    }

    pub fn with_prompt_tokens(mut self, tokens: u64) -> Self {
        self.prompt_tokens = Some(tokens);
        self
    }

    pub fn unhealthy(mut self) -> Self {
        self.healthy = false;
        self
    }
}

/// The request's class. `Privileged` (for example a privileged matter) is local-only whatever the policy says.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RequestClass {
    #[default]
    Standard,
    Privileged,
}

/// What the request's locality header asks for. A header can only tighten: `local` restricts to local
/// backends, `any` (or no header) leaves the policy as it is and never relaxes a local-only policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Locality {
    pub require_local: bool,
}

impl Locality {
    /// The request header the front reads.
    pub const HEADER: &'static str = "x-atlas-locality";

    pub const fn any() -> Self {
        Self {
            require_local: false,
        }
    }

    pub const fn local() -> Self {
        Self {
            require_local: true,
        }
    }

    /// Parses the header value. `local` tightens; `any`, `auto`, an empty value or no header change nothing.
    /// Any other value is an error (the front answers 400), so a typo can never relax a restriction.
    pub fn from_header(value: Option<&str>) -> Result<Self, PolicyError> {
        let Some(raw) = value else {
            return Ok(Self::any());
        };
        let v = raw.trim();
        if v.eq_ignore_ascii_case("local") {
            Ok(Self::local())
        } else if v.is_empty() || v.eq_ignore_ascii_case("any") || v.eq_ignore_ascii_case("auto") {
            Ok(Self::any())
        } else {
            Err(PolicyError::Locality(v.to_string()))
        }
    }

    /// The stricter of the two.
    pub fn tighten(self, other: Self) -> Self {
        Self {
            require_local: self.require_local || other.require_local,
        }
    }
}

/// How the target is picked among the eligible set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Preference {
    /// Expected utility `V * p - cost` over the whole eligible set.
    Utility,
    /// Use the resident local model when it fits and `p >= min_p_pass`; else utility over the eligible set.
    #[default]
    PreferLocal,
}

/// What a positive PII signal does. The signal can only add restrictions; a negative one never relaxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PiiRule {
    /// A detected PII signal restricts the eligible set to local backends.
    #[default]
    LocalOnly,
    /// The PII signal is ignored (the policy already restricts what it must).
    Ignore,
}

/// The static, configured policy (the front's TOML plus its flags).
#[derive(Clone, Debug, PartialEq)]
pub struct StaticPolicy {
    /// `--route-local-only`: only local backends are ever eligible.
    pub local_only: bool,
    /// An allow-list of backends; `None` allows every configured backend.
    pub allowed: Option<Vec<BackendId>>,
    pub preference: Preference,
    /// Prefer-local acceptance threshold in (0, 1]. 1.0 disables the local shortcut, so prefer-local then
    /// decides exactly as utility does.
    pub min_p_pass: f64,
    /// `V`: the dollar value of a successful answer.
    pub value_usd: f64,
    pub pii: PiiRule,
    pub swap: SwapPolicy,
}

impl Default for StaticPolicy {
    fn default() -> Self {
        Self {
            local_only: false,
            allowed: None,
            preference: Preference::PreferLocal,
            min_p_pass: DEFAULT_MIN_P_PASS,
            value_usd: SAGE_VALUE_USD,
            pii: PiiRule::LocalOnly,
            swap: SwapPolicy::default(),
        }
    }
}

impl StaticPolicy {
    /// The Sage preset (V = 4.5) with the defaults above.
    pub fn sage() -> Self {
        Self::default()
    }

    /// The Swift preset (V = 0.1) with the defaults above.
    pub fn swift() -> Self {
        Self {
            value_usd: SWIFT_VALUE_USD,
            ..Self::default()
        }
    }

    /// Startup validation for the front: every number in range, so no decision ever sees a NaN threshold.
    pub fn validate(&self) -> Result<(), PolicyError> {
        let threshold_ok = self.min_p_pass > 0.0 && self.min_p_pass <= 1.0;
        if !threshold_ok {
            return Err(PolicyError::MinPPass(self.min_p_pass));
        }
        let value_ok = self.value_usd.is_finite() && self.value_usd > 0.0;
        if !value_ok {
            return Err(PolicyError::ValueUsd(self.value_usd));
        }
        self.swap.validate()
    }
}

/// Startup validation of the backend list: ids are unique.
pub fn validate_backends(backends: &[Backend]) -> Result<(), PolicyError> {
    let mut ids: Vec<BackendId> = backends.iter().map(|b| b.id).collect();
    ids.sort_unstable();
    match ids.windows(2).find(|w| w[0] == w[1]) {
        Some(w) => Err(PolicyError::DuplicateBackend(w[0])),
        None => Ok(()),
    }
}

/// A configuration or header error, reported at startup or as a 400; never a panic.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicyError {
    Locality(String),
    MinPPass(f64),
    ValueUsd(f64),
    LambdaWait(f64),
    HourlyCap,
    DuplicateBackend(BackendId),
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyError::Locality(v) => {
                write!(
                    f,
                    "{} must be \"local\" or \"any\", got {v:?}",
                    Locality::HEADER
                )
            }
            PolicyError::MinPPass(v) => write!(f, "min_p_pass must be in (0, 1], got {v}"),
            PolicyError::ValueUsd(v) => write!(f, "value_usd must be a positive number, got {v}"),
            PolicyError::LambdaWait(v) => {
                write!(
                    f,
                    "swap lambda_wait_usd_per_s must be a non-negative number, got {v}"
                )
            }
            PolicyError::HourlyCap => write!(f, "swap hourly_cap must be at least 1"),
            PolicyError::DuplicateBackend(id) => write!(f, "backend id {id} is configured twice"),
        }
    }
}

impl std::error::Error for PolicyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locality_header_only_tightens() {
        assert_eq!(Locality::from_header(None), Ok(Locality::any()));
        assert_eq!(
            Locality::from_header(Some(" Local ")),
            Ok(Locality::local())
        );
        assert_eq!(Locality::from_header(Some("any")), Ok(Locality::any()));
        assert_eq!(Locality::from_header(Some("")), Ok(Locality::any()));
        assert!(Locality::from_header(Some("cloud")).is_err());
        assert_eq!(
            Locality::local().tighten(Locality::any()),
            Locality::local()
        );
        assert_eq!(Locality::any().tighten(Locality::any()), Locality::any());
    }

    #[test]
    fn policy_validation() {
        assert_eq!(StaticPolicy::sage().validate(), Ok(()));
        assert_eq!(StaticPolicy::swift().value_usd, SWIFT_VALUE_USD);
        let one = StaticPolicy {
            min_p_pass: 1.0,
            ..StaticPolicy::default()
        };
        assert_eq!(one.validate(), Ok(()));
        for bad in [0.0, -0.1, 1.01, f64::NAN] {
            let p = StaticPolicy {
                min_p_pass: bad,
                ..StaticPolicy::default()
            };
            assert!(p.validate().is_err(), "{bad}");
        }
        let v = StaticPolicy {
            value_usd: f64::INFINITY,
            ..StaticPolicy::default()
        };
        assert!(v.validate().is_err());
        assert!(!PolicyError::HourlyCap.to_string().is_empty());
    }

    #[test]
    fn duplicate_ids_are_reported() {
        let ok = [Backend::local(1, "a"), Backend::cloud(2, "b")];
        assert_eq!(validate_backends(&ok), Ok(()));
        let dup = [Backend::local(2, "a"), Backend::cloud(2, "b")];
        assert_eq!(
            validate_backends(&dup),
            Err(PolicyError::DuplicateBackend(BackendId(2)))
        );
    }
}
