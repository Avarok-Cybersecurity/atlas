// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: event inference on scripted request pairs: a conversation's consecutive requests (task
//! start, tool result, user turn, compaction, subagent, Anthropic-style tool result, explicit event, retry and a
//! resumed conversation) and the event each pair must yield, plus which events are switching boundaries.

use atlas_route_policy::{
    Event, MessageDigest, RequestDigest, Role, infer_event, infer_with_basis,
};

fn msg(role: Role, f: u64) -> MessageDigest {
    MessageDigest::new(role, f)
}

#[test]
fn event_inference_on_scripted_pairs() {
    let sys = msg(Role::System, 1);
    let u1 = msg(Role::User, 2);
    let a1 = msg(Role::Assistant, 3).with_tool_call();
    let t1 = msg(Role::Tool, 4);
    let a2 = msg(Role::Assistant, 5);
    let u2 = msg(Role::User, 6);
    let start = RequestDigest::main(vec![sys, u1]);
    let tool = RequestDigest::main(vec![sys, u1, a1, t1]);
    let turn = RequestDigest::main(vec![sys, u1, a1, t1, a2, u2]);
    let compacted = RequestDigest::main(vec![sys, msg(Role::User, 70), msg(Role::User, 8)]);
    let sub = RequestDigest::on_branch("explore-1", vec![msg(Role::User, 9)]);
    let anthropic_tool =
        RequestDigest::main(vec![sys, u1, a1, msg(Role::User, 10).with_tool_result()]);
    let explicit = RequestDigest::main(vec![sys, u1, a1, t1]).with_explicit(Event::Compaction);
    let script: [(&str, Option<&RequestDigest>, &RequestDigest, Event); 9] = [
        ("first request", None, &start, Event::TaskStart),
        ("tool result", Some(&start), &tool, Event::ToolResult),
        ("next user turn", Some(&tool), &turn, Event::UserMessage),
        ("compaction", Some(&turn), &compacted, Event::Compaction),
        ("subagent", Some(&turn), &sub, Event::SubagentStart),
        (
            "anthropic tool result in a user message",
            Some(&start),
            &anthropic_tool,
            Event::ToolResult,
        ),
        (
            "explicit event wins",
            Some(&turn),
            &explicit,
            Event::Compaction,
        ),
        (
            "retry of the same request",
            Some(&turn),
            &turn,
            Event::UserMessage,
        ),
        (
            "resumed conversation without state",
            None,
            &turn,
            Event::UserMessage,
        ),
    ];
    for (name, prev, cur, want) in script {
        assert_eq!(infer_event(prev, cur), want, "{name}");
    }
}

#[test]
fn boundaries_and_unknown_turns() {
    for e in Event::ALL {
        let boundary = matches!(
            e,
            Event::TaskStart | Event::SubagentStart | Event::Compaction | Event::UserMessage
        );
        assert_eq!(e.is_boundary(), boundary, "{}", e.as_str());
    }
    // An assistant-last transcript is an unknown turn, which is never a switching boundary.
    let odd = RequestDigest::main(vec![msg(Role::User, 1), msg(Role::Assistant, 2)]);
    let (event, _) = infer_with_basis(Some(&RequestDigest::main(vec![msg(Role::User, 1)])), &odd);
    assert_eq!(event, Event::Unknown);
    assert!(!event.is_boundary());
}
