// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the swap rule. A target that is not resident costs `swap_seconds` of dead time for every
//! queued request, so a swap is allowed only when the policy allows swaps at all (off by default), hysteresis
//! allows one now, and `expected_gain * V * n_hard_queued > swap_seconds * lambda_wait`. A forced swap (no
//! resident model is eligible) skips the gain test but still obeys the switch and hysteresis.

use std::time::{Duration, Instant};

use crate::hysteresis;
use crate::types::{BackendId, PolicyError, Residency};

/// Default dead-time price in US dollars per second (a tuning point for the front's config).
pub const DEFAULT_LAMBDA_WAIT_USD_PER_S: f64 = 0.01;
/// Default minimum time between two swaps.
pub const DEFAULT_DWELL: Duration = Duration::from_secs(600);
/// Default cap on swaps inside a rolling hour.
pub const DEFAULT_HOURLY_CAP: u32 = 4;

/// The swap part of the static policy (`--route-allow-swap` and its knobs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapPolicy {
    /// Off by default: a non-resident target is then served on a resident model instead.
    pub allow: bool,
    /// `lambda_wait`: what one second of dead time costs, in US dollars.
    pub lambda_wait_usd_per_s: f64,
    pub dwell: Duration,
    pub hourly_cap: u32,
}

impl Default for SwapPolicy {
    fn default() -> Self {
        Self {
            allow: false,
            lambda_wait_usd_per_s: DEFAULT_LAMBDA_WAIT_USD_PER_S,
            dwell: DEFAULT_DWELL,
            hourly_cap: DEFAULT_HOURLY_CAP,
        }
    }
}

impl SwapPolicy {
    pub fn validate(&self) -> Result<(), PolicyError> {
        let lambda_ok = self.lambda_wait_usd_per_s.is_finite() && self.lambda_wait_usd_per_s >= 0.0;
        if !lambda_ok {
            return Err(PolicyError::LambdaWait(self.lambda_wait_usd_per_s));
        }
        if self.hourly_cap == 0 {
            return Err(PolicyError::HourlyCap);
        }
        Ok(())
    }
}

/// Why a swap was not allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwapDenial {
    /// Swaps are off in the policy.
    Disabled,
    /// The last swap is younger than the dwell time.
    Dwell,
    /// The hourly cap is reached.
    HourlyCap,
    /// The expected gain does not pay for the dead time.
    Gain,
    /// The target has no swap-seconds estimate.
    UnknownSwapSeconds,
    /// The target or the resident alternative has no prediction, so the gain is unknown.
    UnknownGain,
}

impl SwapDenial {
    pub fn as_str(&self) -> &'static str {
        match self {
            SwapDenial::Disabled => "disabled",
            SwapDenial::Dwell => "dwell",
            SwapDenial::HourlyCap => "hourly_cap",
            SwapDenial::Gain => "gain",
            SwapDenial::UnknownSwapSeconds => "unknown_swap_seconds",
            SwapDenial::UnknownGain => "unknown_gain",
        }
    }
}

/// An allowed swap: the front evicts `from` (when one is resident), loads `to`, then records the swap
/// (`Residency::apply_swap`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapPlan {
    pub from: Option<BackendId>,
    pub to: BackendId,
    pub seconds: Option<f64>,
    /// No resident model was eligible, so the gain test was skipped.
    pub forced: bool,
}

/// The swap inequality: `expected_gain * value_usd * hard_queued > swap_seconds * lambda_wait`.
pub fn worth_it(
    expected_gain: f64,
    value_usd: f64,
    hard_queued: u32,
    swap_seconds: f64,
    lambda_wait: f64,
) -> bool {
    expected_gain * value_usd * f64::from(hard_queued) > swap_seconds * lambda_wait
}

/// The smallest expected gain in success probability that pays for a swap, or `None` when nothing can
/// (no queued hard request or no value).
pub fn break_even_gain(
    value_usd: f64,
    hard_queued: u32,
    swap_seconds: f64,
    lambda_wait: f64,
) -> Option<f64> {
    let scale = value_usd * f64::from(hard_queued);
    if scale > 0.0 && scale.is_finite() {
        Some(swap_seconds * lambda_wait / scale)
    } else {
        None
    }
}

/// Decides one swap to `target` at `now`. `gain` is the target's success probability minus the best resident
/// alternative's; `forced` means no resident model is eligible. Checks run in a fixed order: the policy switch,
/// hysteresis, then (unforced only) the estimate, the gain and the inequality.
pub(crate) fn evaluate(
    policy: &SwapPolicy,
    residency: &Residency,
    target: BackendId,
    gain: Option<f64>,
    value_usd: f64,
    hard_queued: u32,
    forced: bool,
    now: Instant,
) -> Result<SwapPlan, SwapDenial> {
    if !policy.allow {
        return Err(SwapDenial::Disabled);
    }
    hysteresis::check(&residency.history, policy, now)?;
    let seconds = residency.swap_seconds(target);
    if !forced {
        let s = seconds.ok_or(SwapDenial::UnknownSwapSeconds)?;
        let g = gain.ok_or(SwapDenial::UnknownGain)?;
        if !worth_it(g, value_usd, hard_queued, s, policy.lambda_wait_usd_per_s) {
            return Err(SwapDenial::Gain);
        }
    }
    Ok(SwapPlan {
        from: residency.evicted_by(target),
        to: target,
        seconds,
        forced,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hysteresis::SwapHistory;
    use crate::types::ResidencyEntry;

    fn residency(history: SwapHistory) -> Residency {
        Residency {
            entries: vec![
                ResidencyEntry {
                    id: BackendId(1),
                    resident: true,
                    swap_seconds: Some(5.0),
                },
                ResidencyEntry {
                    id: BackendId(2),
                    resident: false,
                    swap_seconds: Some(40.0),
                },
            ],
            history,
        }
    }

    fn allow() -> SwapPolicy {
        SwapPolicy {
            allow: true,
            ..SwapPolicy::default()
        }
    }

    #[test]
    fn inequality_and_break_even() {
        assert!(worth_it(0.15, 4.5, 1, 40.0, 0.01));
        assert!(!worth_it(0.15, 4.5, 0, 40.0, 0.01));
        assert!(!worth_it(0.05, 4.5, 1, 40.0, 0.01));
        let g = break_even_gain(4.5, 1, 40.0, 0.01).unwrap_or(f64::NAN);
        assert!((g - 0.4 / 4.5).abs() < 1e-12);
        assert_eq!(break_even_gain(4.5, 0, 40.0, 0.01), None);
    }

    #[test]
    fn evaluate_order_and_plan() {
        let now = Instant::now() + Duration::from_secs(7200);
        let r = residency(SwapHistory::new());
        let id = BackendId(2);
        assert_eq!(
            evaluate(
                &SwapPolicy::default(),
                &r,
                id,
                Some(0.5),
                4.5,
                1,
                false,
                now
            ),
            Err(SwapDenial::Disabled)
        );
        let plan = evaluate(&allow(), &r, id, Some(0.15), 4.5, 1, false, now);
        assert_eq!(
            plan,
            Ok(SwapPlan {
                from: Some(BackendId(1)),
                to: id,
                seconds: Some(40.0),
                forced: false
            })
        );
        assert_eq!(
            evaluate(&allow(), &r, id, None, 4.5, 1, false, now),
            Err(SwapDenial::UnknownGain)
        );
        assert_eq!(
            evaluate(&allow(), &r, id, Some(0.01), 4.5, 1, false, now),
            Err(SwapDenial::Gain)
        );
        let unknown = BackendId(9);
        assert_eq!(
            evaluate(&allow(), &r, unknown, Some(0.9), 4.5, 1, false, now),
            Err(SwapDenial::UnknownSwapSeconds)
        );
        let forced = evaluate(&allow(), &r, unknown, None, 4.5, 0, true, now);
        assert_eq!(
            forced,
            Ok(SwapPlan {
                from: Some(BackendId(1)),
                to: unknown,
                seconds: None,
                forced: true
            })
        );
        let recent = residency(SwapHistory::from_swaps([now - Duration::from_secs(60)]));
        assert_eq!(
            evaluate(&allow(), &recent, id, Some(0.9), 4.5, 1, true, now),
            Err(SwapDenial::Dwell)
        );
    }

    #[test]
    fn validate_rejects_bad_knobs() {
        assert_eq!(SwapPolicy::default().validate(), Ok(()));
        let bad = SwapPolicy {
            lambda_wait_usd_per_s: f64::NAN,
            ..SwapPolicy::default()
        };
        assert!(bad.validate().is_err());
        let zero = SwapPolicy {
            hourly_cap: 0,
            ..SwapPolicy::default()
        };
        assert_eq!(zero.validate(), Err(PolicyError::HourlyCap));
    }
}
