// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from atlasfutures/semantic-router (https://github.com/atlasfutures/semantic-router, branch
// vsr-next, commit f6d0393), rayline-core/execution/events.go:12-26 and :130-162 and
// rayline-core/execution/deployment_wire.go:11-23, published by Atlas Futures, Inc. under the Apache License 2.0
// (https://www.apache.org/licenses/LICENSE-2.0); the files carry no copyright line of their own. Translated from Go
// to Rust; changes: the wire request is reduced to a per-message digest (role, tool-call and tool-result marks,
// wire group, content fingerprint) that the front computes, the "fresh branch" test reads the previous digest of
// the same branch, and a shrinking rewrite of the transcript is inferred as a compaction (upstream infers
// compaction only from an explicit event). See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: execution events. What just happened in an agent conversation (a task or subagent start, a
//! new user message, a tool result, a compaction, a side call), from the client's explicit event when it sends
//! one, else inferred from documented wire markers by comparing this request with the previous one on the same
//! branch. An unknown turn is never upgraded into a switching boundary.

/// The event kinds of an execution context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    TaskStart,
    UserMessage,
    ToolResult,
    Compaction,
    SubagentStart,
    Sidecall,
    Unknown,
}

impl Event {
    /// Every kind, for tests and label tables.
    pub const ALL: [Event; 7] = [
        Event::TaskStart,
        Event::UserMessage,
        Event::ToolResult,
        Event::Compaction,
        Event::SubagentStart,
        Event::Sidecall,
        Event::Unknown,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            Event::TaskStart => "task_start",
            Event::UserMessage => "user_message",
            Event::ToolResult => "tool_result",
            Event::Compaction => "compaction",
            Event::SubagentStart => "subagent_start",
            Event::Sidecall => "sidecall",
            Event::Unknown => "unknown",
        }
    }

    /// Parses an explicit client event kind. Only the known kinds parse; `unknown` and anything else is `None`
    /// (the front refuses the request, as upstream does for an unknown kind).
    pub fn parse(kind: &str) -> Option<Event> {
        Event::ALL
            .into_iter()
            .find(|e| *e != Event::Unknown && e.as_str() == kind)
    }

    /// Whether the model may change at this event (task start, subagent start, compaction, user turn).
    pub fn is_boundary(&self) -> bool {
        crate::timing::may_switch(*self)
    }
}

/// How the event was obtained.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Basis {
    ExplicitClient,
    MarkerInferred,
    Unknown,
}

/// A wire message role.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

/// What the event inference needs to know about one wire message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MessageDigest {
    pub role: Role,
    /// The message carries a tool call.
    pub tool_call: bool,
    /// The message carries a tool result (Anthropic puts tool results in user messages).
    pub tool_result: bool,
    /// Non-zero when the message belongs to a tool-result wire group (text beside Anthropic tool results).
    pub wire_group: u32,
    /// A content fingerprint independent of moving cache directives (`fingerprint_bytes` of the message with its
    /// `cache_control` fields removed).
    pub fingerprint: u64,
}

impl MessageDigest {
    pub const fn new(role: Role, fingerprint: u64) -> Self {
        Self {
            role,
            tool_call: false,
            tool_result: false,
            wire_group: 0,
            fingerprint,
        }
    }

    pub fn with_tool_call(mut self) -> Self {
        self.tool_call = true;
        self
    }

    pub fn with_tool_result(mut self) -> Self {
        self.tool_result = true;
        self
    }

    pub fn with_wire_group(mut self, group: u32) -> Self {
        self.wire_group = group;
        self
    }
}

/// FNV-1a 64 over bytes: a stable fingerprint the front can compute for each message.
pub fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The main conversation's branch name.
pub const MAIN_BRANCH: &str = "main";

/// One request as the event inference sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestDigest {
    /// The conversation branch (`main`, or a subagent's branch).
    pub branch: String,
    pub messages: Vec<MessageDigest>,
    /// The client's explicit event (an `x-atlas-event` header or body field), when it sent one.
    pub explicit: Option<Event>,
}

impl RequestDigest {
    pub fn main(messages: Vec<MessageDigest>) -> Self {
        Self {
            branch: MAIN_BRANCH.to_string(),
            messages,
            explicit: None,
        }
    }

    pub fn on_branch(branch: impl Into<String>, messages: Vec<MessageDigest>) -> Self {
        Self {
            branch: branch.into(),
            messages,
            explicit: None,
        }
    }

    pub fn with_explicit(mut self, event: Event) -> Self {
        self.explicit = Some(event);
        self
    }
}

/// Whether the request already carries execution history: an assistant or tool message, or a tool call or
/// tool result anywhere (upstream `hasExecutionHistory`).
pub fn has_execution_history(request: &RequestDigest) -> bool {
    request
        .messages
        .iter()
        .any(|m| matches!(m.role, Role::Assistant | Role::Tool) || m.tool_call || m.tool_result)
}

/// Leading messages two requests share, by fingerprint and role.
fn common_prefix(a: &[MessageDigest], b: &[MessageDigest]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(x, y)| x.fingerprint == y.fingerprint && x.role == y.role)
        .count()
}

/// The event for `cur`, given the previous request `prev` (any branch; only the same branch counts).
pub fn infer_event(prev: Option<&RequestDigest>, cur: &RequestDigest) -> Event {
    infer_with_basis(prev, cur).0
}

/// `infer_event` plus how the event was obtained.
pub fn infer_with_basis(prev: Option<&RequestDigest>, cur: &RequestDigest) -> (Event, Basis) {
    if let Some(event) = cur.explicit {
        return if event == Event::Unknown {
            (Event::Unknown, Basis::Unknown)
        } else {
            (event, Basis::ExplicitClient)
        };
    }
    let prev = prev.filter(|p| p.branch == cur.branch);
    if prev.is_none() && !has_execution_history(cur) {
        let kind = if cur.branch == MAIN_BRANCH {
            Event::TaskStart
        } else {
            Event::SubagentStart
        };
        return (kind, Basis::MarkerInferred);
    }
    if let Some(p) = prev {
        // The transcript no longer extends the previous one and is shorter: the history was replaced, as a
        // compaction does. A mere edit of an earlier message that keeps the length is not inferred as one.
        let shared = common_prefix(&p.messages, &cur.messages);
        let rewritten = shared < p.messages.len() && cur.messages.len() < p.messages.len();
        if rewritten && !cur.messages.is_empty() {
            return (Event::Compaction, Basis::MarkerInferred);
        }
    }
    let Some(last) = cur.messages.last() else {
        return (Event::Unknown, Basis::Unknown);
    };
    match last.role {
        Role::Tool => (Event::ToolResult, Basis::MarkerInferred),
        Role::User if last.wire_group != 0 || last.tool_result => {
            (Event::ToolResult, Basis::MarkerInferred)
        }
        Role::User => (Event::UserMessage, Basis::MarkerInferred),
        _ => (Event::Unknown, Basis::Unknown),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(role: Role, f: u64) -> MessageDigest {
        MessageDigest::new(role, f)
    }

    #[test]
    fn labels_round_trip() {
        for e in Event::ALL {
            if e == Event::Unknown {
                assert_eq!(Event::parse(e.as_str()), None);
            } else {
                assert_eq!(Event::parse(e.as_str()), Some(e));
            }
        }
        assert_eq!(Event::parse("subagent"), None);
    }

    #[test]
    fn fresh_branches_start_tasks_and_subagents() {
        let first = RequestDigest::main(vec![m(Role::System, 1), m(Role::User, 2)]);
        assert_eq!(
            infer_with_basis(None, &first),
            (Event::TaskStart, Basis::MarkerInferred)
        );
        let sub = RequestDigest::on_branch("sub-1", vec![m(Role::User, 3)]);
        assert_eq!(infer_event(Some(&first), &sub), Event::SubagentStart);
    }

    #[test]
    fn last_message_decides_turns() {
        let a = RequestDigest::main(vec![m(Role::User, 1)]);
        let b = RequestDigest::main(vec![
            m(Role::User, 1),
            m(Role::Assistant, 2),
            m(Role::User, 3),
        ]);
        assert_eq!(infer_event(Some(&a), &b), Event::UserMessage);
        let c = RequestDigest::main(vec![
            m(Role::User, 1),
            m(Role::Assistant, 2).with_tool_call(),
            m(Role::Tool, 4),
        ]);
        assert_eq!(infer_event(Some(&a), &c), Event::ToolResult);
        let d = RequestDigest::main(vec![
            m(Role::User, 1),
            m(Role::Assistant, 2),
            m(Role::User, 5).with_tool_result(),
        ]);
        assert_eq!(infer_event(Some(&a), &d), Event::ToolResult);
        let e = RequestDigest::main(vec![
            m(Role::User, 1),
            m(Role::Assistant, 2),
            m(Role::User, 6).with_wire_group(1),
        ]);
        assert_eq!(infer_event(Some(&a), &e), Event::ToolResult);
        let f = RequestDigest::main(vec![m(Role::User, 1), m(Role::Assistant, 2)]);
        assert_eq!(
            infer_with_basis(Some(&a), &f),
            (Event::Unknown, Basis::Unknown)
        );
    }

    #[test]
    fn shrinking_rewrite_is_a_compaction_and_explicit_wins() {
        let long = RequestDigest::main(vec![
            m(Role::System, 1),
            m(Role::User, 2),
            m(Role::Assistant, 3),
            m(Role::User, 4),
            m(Role::Assistant, 5),
            m(Role::User, 6),
        ]);
        let compacted = RequestDigest::main(vec![
            m(Role::System, 1),
            m(Role::User, 90),
            m(Role::User, 7),
        ]);
        assert_eq!(infer_event(Some(&long), &compacted), Event::Compaction);
        // Same length with an edited earlier message is a user turn, not a compaction.
        let mut edited = long.clone();
        edited.messages[1].fingerprint = 99;
        assert_eq!(infer_event(Some(&long), &edited), Event::UserMessage);
        let explicit = compacted.clone().with_explicit(Event::Sidecall);
        assert_eq!(
            infer_with_basis(Some(&long), &explicit),
            (Event::Sidecall, Basis::ExplicitClient)
        );
        let unknown = compacted.with_explicit(Event::Unknown);
        assert_eq!(
            infer_with_basis(Some(&long), &unknown),
            (Event::Unknown, Basis::Unknown)
        );
    }

    #[test]
    fn history_without_previous_request_reads_the_last_message() {
        let resumed = RequestDigest::main(vec![
            m(Role::User, 1),
            m(Role::Assistant, 2),
            m(Role::User, 3),
        ]);
        assert_eq!(infer_event(None, &resumed), Event::UserMessage);
        assert_eq!(
            infer_event(None, &RequestDigest::on_branch("s", vec![])),
            Event::SubagentStart
        );
        assert_ne!(fingerprint_bytes(b"a"), fingerprint_bytes(b"b"));
        assert_eq!(fingerprint_bytes(b""), 0xcbf2_9ce4_8422_2325);
    }
}
