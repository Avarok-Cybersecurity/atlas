// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the prefer-local rule and the resident fallback. Prefer-local uses the loaded (resident),
//! eligible local model when its window provably fits and its predicted success `p >= min_p_pass`; otherwise
//! the decision falls through to utility over the whole eligible set. A threshold of 1.0 disables the shortcut,
//! so prefer-local then decides exactly as utility does. Without predictions the decision falls back to the
//! resident model, never to a backend outside E. Rule after the ARC technical report (section 2.2); the
//! `min_p_pass` name follows Raydock's hybrid profile; no code copied.

use std::cmp::{Ordering, Reverse};

use crate::capacity::Fit;
use crate::eligibility::Member;
use crate::types::{BackendId, Predictions, Queues, Residency};
use crate::utility::{self, Scored, UtilityCtx};

/// Whether `p` passes the prefer-local threshold. A threshold at or above 1.0 (or NaN) never passes.
pub fn passes(p: f64, min_p_pass: f64) -> bool {
    min_p_pass < 1.0 && p >= min_p_pass
}

/// Whether `m` may take the prefer-local shortcut at all: local, resident, and a proven fit.
pub fn shortcut_candidate(m: &Member, residency: &Residency) -> bool {
    m.local && m.fit == Fit::Fits && residency.is_resident(m.id)
}

/// The prefer-local pick among `members`: the passing candidate with the highest `p`, then the shorter queue,
/// then the lower id. `None` sends the decision to utility.
pub fn shortcut(
    members: &[&Member],
    preds: &Predictions,
    residency: &Residency,
    min_p_pass: f64,
    ctx: &UtilityCtx<'_>,
) -> Option<Scored> {
    members
        .iter()
        .filter(|m| shortcut_candidate(m, residency))
        .filter_map(|m| utility::score(m, preds, ctx))
        .filter(|s| passes(s.p, min_p_pass))
        .max_by(compare_local)
}

fn compare_local(a: &Scored, b: &Scored) -> Ordering {
    a.p.total_cmp(&b.p)
        .then_with(|| Reverse(a.depth).cmp(&Reverse(b.depth)))
        .then_with(|| Reverse(a.id).cmp(&Reverse(b.id)))
}

/// The target when nothing could be scored: `last` when it is a resident member, else the resident local member
/// with the shortest queue (then the lowest id), else the lowest resident member, else the lowest member (which
/// then needs a swap). `None` only for an empty slice.
pub fn resident_fallback(
    members: &[&Member],
    residency: &Residency,
    last: Option<BackendId>,
    queues: &Queues,
) -> Option<BackendId> {
    if let Some(l) =
        last.filter(|l| residency.is_resident(*l) && members.iter().any(|m| m.id == *l))
    {
        return Some(l);
    }
    let resident_local = members
        .iter()
        .filter(|m| m.local && residency.is_resident(m.id))
        .min_by_key(|m| (queues.depth(m.id), m.id))
        .map(|m| m.id);
    resident_local
        .or_else(|| {
            members
                .iter()
                .filter(|m| residency.is_resident(m.id))
                .map(|m| m.id)
                .min()
        })
        .or_else(|| members.iter().map(|m| m.id).min())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::Event;
    use crate::types::{Prediction, Prices, ResidencyEntry, TokenUsage};

    fn m(id: u32, local: bool, fit: Fit) -> Member {
        Member {
            id: BackendId(id),
            local,
            fit,
            prompt_tokens: None,
        }
    }

    fn residency() -> Residency {
        Residency {
            entries: vec![
                ResidencyEntry {
                    id: BackendId(1),
                    resident: true,
                    swap_seconds: None,
                },
                ResidencyEntry {
                    id: BackendId(2),
                    resident: false,
                    swap_seconds: Some(40.0),
                },
            ],
            ..Residency::default()
        }
    }

    #[test]
    fn threshold_rules() {
        assert!(passes(0.85, 0.85));
        assert!(!passes(0.84, 0.85));
        assert!(!passes(1.0, 1.0));
        assert!(!passes(1.0, f64::NAN));
    }

    #[test]
    fn shortcut_needs_resident_local_fit_and_pass() {
        let usage = TokenUsage::new(100.0, 10.0);
        let preds = Predictions::new()
            .with(BackendId(1), Prediction::new(0.9, usage))
            .with(BackendId(2), Prediction::new(0.99, usage))
            .with(BackendId(3), Prediction::new(0.99, usage))
            .with(BackendId(4), Prediction::new(0.99, usage));
        let prices = Prices::new();
        let queues = Queues::default();
        let ctx = UtilityCtx {
            value_usd: 4.5,
            event: Event::TaskStart,
            last: None,
            prices: &prices,
            queues: &queues,
        };
        let r = residency();
        let all = [
            m(1, true, Fit::Fits),
            m(2, true, Fit::Fits),
            m(3, false, Fit::Fits),
            m(4, true, Fit::Unknown),
        ];
        let refs: Vec<&Member> = all.iter().collect();
        assert_eq!(
            shortcut(&refs, &preds, &r, 0.85, &ctx).map(|s| s.id),
            Some(BackendId(1))
        );
        assert_eq!(shortcut(&refs, &preds, &r, 0.95, &ctx), None);
        assert_eq!(shortcut(&refs, &preds, &r, 1.0, &ctx), None);
    }

    #[test]
    fn fallback_order() {
        let r = residency();
        let q = Queues::default();
        let all = [
            m(1, true, Fit::Fits),
            m(2, true, Fit::Fits),
            m(3, false, Fit::Unknown),
        ];
        let refs: Vec<&Member> = all.iter().collect();
        assert_eq!(
            resident_fallback(&refs, &r, Some(BackendId(3)), &q),
            Some(BackendId(3))
        );
        assert_eq!(
            resident_fallback(&refs, &r, Some(BackendId(2)), &q),
            Some(BackendId(1))
        );
        assert_eq!(resident_fallback(&refs, &r, None, &q), Some(BackendId(1)));
        let no_local = [m(2, true, Fit::Fits), m(3, false, Fit::Unknown)];
        let refs: Vec<&Member> = no_local.iter().collect();
        assert_eq!(resident_fallback(&refs, &r, None, &q), Some(BackendId(3)));
        let only_swap = [m(2, true, Fit::Fits)];
        let refs: Vec<&Member> = only_swap.iter().collect();
        assert_eq!(resident_fallback(&refs, &r, None, &q), Some(BackendId(2)));
        assert_eq!(resident_fallback(&[], &r, None, &q), None);
    }
}
