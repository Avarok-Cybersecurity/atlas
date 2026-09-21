// SPDX-License-Identifier: AGPL-3.0-only

//! Gate A's PASS/FAIL decision and the two aggregates it divides.
//!
//! Split out of `score.rs` for the repo's 500-line ceiling. The seam is the
//! verdict itself: everything in `score.rs` MEASURES a tier, everything here
//! JUDGES one.
//!
//! Three bounds, and they are not the same kind of thing:
//!
//! - `webserver_ok` / `followed_directions` — CORRECTNESS, all-or-nothing.
//! - `s_per_turn` — SPEED, and only when the variant committed a bound (a 0.0
//!   budget means non-gating, not "everything fails"). It divides the AGENT's
//!   seconds, never the tier total, because the scorer's build is a
//!   per-ITERATION cost and a per-TURN ratio would let it shrink as turns grow.
//! - `Σwall` — BLOWUP. Σwall is a product of turns and per-turn cost, and turn
//!   count is drawn by the agent, not the engine, which is exactly why it is no
//!   longer the speed bound.

/// `wall` is the TOTAL tier wall (scorer included) that `wall_budget_s` bounds;
/// `agent_wall` is the agent's own, which is what the per-turn speed bound
/// divides. Separate arguments because they are separate measurements —
/// collapsing them is what let a per-ITERATION build cost leak into a per-TURN
/// ratio, where it shrinks as turns grow.
pub(super) fn verdict(
    rows: &[super::IterationRow],
    wall: f64,
    agent_wall: f64,
    wall_budget_s: f64,
    s_per_turn_budget: f64,
) -> crate::result::Verdict {
    use crate::result::Verdict;
    let n = rows.len();
    let ok = rows.iter().filter(|r| r.webserver_ok).count();
    let fd = rows.iter().filter(|r| r.directions.overall()).count();
    let turns = total_turns(rows);
    let mut failures = Vec::new();
    if ok < n {
        failures.push(format!("webserver_ok {ok}/{n}"));
    }
    if fd < n {
        failures.push(format!("followed_directions {fd}/{n}"));
    }
    // SPEED, and only when the variant committed a bound: 0.0 means NON-GATING,
    // not "everything fails". Three decimals so a marginal failure does not
    // print two identical-looking numbers either side of the `>`.
    let over_speed = seconds_per_turn(agent_wall, turns)
        .filter(|_| s_per_turn_budget > 0.0)
        .filter(|s| *s > s_per_turn_budget);
    if let Some(spt) = over_speed {
        failures.push(format!(
            "{spt:.3}s/turn > {s_per_turn_budget:.3}s/turn \
             ({agent_wall:.0}s agent / {turns} turns)"
        ));
    }
    // DEGENERACY, not speed: catches a tier that completes every task but
    // wanders through far more turns than the work needs. See the module doc.
    if wall > wall_budget_s {
        failures.push(format!("Σwall {wall:.0}s > {wall_budget_s:.0}s"));
    }
    if failures.is_empty() {
        let spt = match (seconds_per_turn(agent_wall, turns), s_per_turn_budget > 0.0) {
            (Some(s), true) => format!("{s:.3}s/turn ≤ {s_per_turn_budget:.3}"),
            (Some(s), false) => format!("{s:.3}s/turn (unbounded)"),
            (None, _) => "no turns".to_string(),
        };
        Verdict::pass(format!(
            "{ok}/{n} webserver_ok · {fd}/{n} followed_directions · \
             {spt} · Σwall {wall:.0}s ≤ {wall_budget_s:.0}s"
        ))
    } else {
        Verdict::fail(failures.join(" · "))
    }
}

/// Agent turns summed across the tier — the denominator the speed bound needs.
pub(super) fn total_turns(rows: &[super::IterationRow]) -> usize {
    rows.iter().map(|r| r.turns).sum()
}

/// Seconds of wall per agent turn, or `None` when the tier took no turns.
///
/// `None` rather than 0.0 or infinity: a zero-turn tier means the agent never
/// ran, which the correctness halves already fail on. Inventing a speed number
/// for it would either mask that (0.0 passes) or double-report it (inf fails).
pub(super) fn seconds_per_turn(wall: f64, turns: usize) -> Option<f64> {
    (turns > 0).then(|| wall / turns as f64)
}
