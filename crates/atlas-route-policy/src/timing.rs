// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: decision timing. The model may change only at a task start, a subagent start, a compaction or
//! a new user turn; between those (tool results, side calls, unknown turns) the previous target is kept, which
//! keeps the prefix cache and the recurrent state warm. A kept target must still be eligible and resident, so
//! timing can never route outside the eligible set or force a swap.

use crate::eligibility::EligibleSet;
use crate::events::Event;
use crate::types::{BackendId, Residency};

/// Whether `event` is a switching boundary.
pub fn may_switch(event: Event) -> bool {
    matches!(
        event,
        Event::TaskStart | Event::SubagentStart | Event::Compaction | Event::UserMessage
    )
}

/// The target to keep at a non-boundary event: `last`, when it is still eligible and resident.
pub fn hold(
    e: &EligibleSet,
    last: Option<BackendId>,
    event: Event,
    residency: &Residency,
) -> Option<BackendId> {
    if may_switch(event) {
        return None;
    }
    let last = last?;
    (e.contains(last) && residency.is_resident(last)).then_some(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundaries_are_exactly_the_four_switch_points() {
        let switching: Vec<Event> = Event::ALL.into_iter().filter(|e| may_switch(*e)).collect();
        assert_eq!(
            switching,
            vec![
                Event::TaskStart,
                Event::UserMessage,
                Event::Compaction,
                Event::SubagentStart
            ]
        );
        assert!(!may_switch(Event::Unknown));
        assert!(!may_switch(Event::ToolResult));
        assert!(!may_switch(Event::Sidecall));
    }
}
