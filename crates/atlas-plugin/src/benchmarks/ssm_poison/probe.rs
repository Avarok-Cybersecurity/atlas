// SPDX-License-Identifier: AGPL-3.0-only

//! The probe: a fixed conversation script that every round replays from
//! scratch against the same server.
//!
//! Nothing here may vary between rounds, between runs, or between boxes:
//! the invariant the gate asserts (identical input → identical output) is
//! only falsifiable when the input is pinned byte-for-byte. No dates, no
//! random seeds, no run ids in the text — `unique_prefix_tag` stays OUT of
//! this module on purpose, unlike the TTFT probes, because a unique prefix
//! would make every round cache-cold and the gate would never exercise the
//! prefix-cache restore path it exists to police.

use serde_json::{Value, json};

/// Fixed long prefix (~1.5K tokens) for turn 1. Sized so that, with the
/// flagship recipe's `enable_prefix_caching: true`, every later round's
/// prefill lands as a Marconi prefix-cache restore from round 0's state —
/// the exact path the 2026-08-11 batch4 regression poisoned (runs 8/9 of
/// the agentic gate restored a corrupted SSM snapshot and degenerated to
/// early-EOS). The content is an inert technical document: deterministic
/// to answer, and no part of it invites the model to vary its output.
pub const LONG_PREFIX: &str = "\
SYSTEM DOCUMENT 7741-C, revision 9, frozen text — quote from it, never extend it.

Section 1, Invariants. The ledger defines three invariants that every batch
must preserve end to end. First, monotonic sequence: each record carries a
sequence number exactly one greater than its predecessor, and gaps are
treated as corruption, never as reordering. Second, bounded drift: the
clock offset between any two nodes in the batch must stay under forty
milliseconds for the duration of the batch, and an offset beyond that bound
invalidates every record written after the breach. Third, closed membership:
a node joins the batch through a single signed admission record and leaves
through a single signed departure record, and no record may reference a node
outside its admission and departure interval.

Section 2, Batch lifecycle. A batch opens when the coordinator writes the
genesis record, assigning the batch id and the initial sequence window.
Nodes acknowledge the genesis record, after which the coordinator admits
writes. The batch runs until either the coordinator writes the seal record
or the drift invariant is breached. When a batch seals, every node flushes
its local ledger segment to the archive tier and reports the segment digest.
The digest is the hash of the concatenated record envelopes in sequence
order, excluding payloads. Two nodes that sealed the same batch must report
the same digest; a mismatch is escalated as a split-brain event and the
batch is quarantined.

Section 3, Recovery. Recovery replays the archived segments in sequence
order against an empty ledger until the digest matches the sealed value.
Replay is deterministic by construction: the envelope fixes the record
bytes, and the sequence order fixes their application order. Recovery never
consults a live node; a replay that cannot reproduce the sealed digest
marks the segment as untrusted and routes it to manual review. The recovery
window is bounded at ninety minutes per segment, and a segment that exceeds
the window is split at the nearest checkpoint boundary and retried in
halves.

Section 4, Checksum rule. The envelope checksum is computed over the header
fields in their serialized order: batch id, sequence number, node id,
timestamp, then payload length. The checksum excludes the payload itself,
which is carried separately and verified against the declared length only.
When a record is archived, the archive tier recomputes the checksum from the
stored header and refuses any record whose recomputed value differs from
the recorded one; such a record is quarantined with its recomputed value
attached. The checksum is sixteen bytes and is never truncated in transit.

Section 5, Escalation ladder. A drift breach escalates to the coordinator
within one heartbeat interval. A digest mismatch escalates immediately and
pauses admission. A split-brain event escalates to the on-call engineer and
freezes the archive tier until the quarantined batch is resolved. Escalation
records carry the breached invariant by name, the observed values, and the
bounds from this document, in that order.

Section 6, Admission and departure. Admission is a two-phase exchange. The
candidate node first presents a signed intent record naming the batch and its
own node id; the coordinator validates the signature against the roster key
and, if the batch is open, returns an admission record carrying the assigned
sequence window. The candidate becomes a member only after it acknowledges
the admission record. Departure is symmetric: a member submits a departure
intent, the coordinator drains the member's outstanding writes, then issues
the departure record. A node that leaves before its writes are drained is
marked delinquent, and its unacknowledged records are re-assigned to the
coordinator for replay. Membership changes are appended to the batch journal
in order and are themselves covered by the envelope checksum, so a roster
tamper is detected at the next archive recomputation.

Section 7, Archival layout. The archive tier stores segments in fixed-size
blocks aligned on checksum boundaries; a segment smaller than one block is
padded with a terminal envelope carrying a zero payload length. Block
headers repeat the batch id and the first and last sequence numbers of the
contained records, which lets a recovery scan locate a segment by sequence
number alone without reading payloads. The tier keeps two copies of every
block on distinct media and compares them on read; a copy disagreement
triggers a re-fetch from the member that sealed the batch, and if that
member is gone the block is marked degraded and excluded from digest
recomputation until restored. Degraded blocks are reported in the daily
integrity summary with their batch id, block offset, and the media that
disagreed.";

/// The user turns of the script, in order. Turn 1 is prefixed with
/// [`LONG_PREFIX`]; turns 2-4 ride on the accumulated conversation. Every
/// prompt is deterministic given the document: there is exactly one right
/// shape for each answer, and the gate does not care what that shape is —
/// only that it is the SAME shape every round.
pub const TURNS: [&str; 4] = [
    // Turn 1: acknowledge the document. The reply becomes part of the
    // conversation state every later turn depends on, so a poisoned prefix
    // restore corrupts turns 2-4 as well.
    "Reply with exactly one line: ACK 7741-C and the number of sections \
     listed in the document.",
    // Turn 2: reference the conversation state from turn 1.
    "List the three invariants from Section 1 of the document, numbered 1 \
     to 3, one per line, each in at most ten words.",
    // Turn 3: a transformation task whose output is fully determined by the
    // document — sensitive to any corruption of the prefilled context.
    "Rewrite Section 4 of the document as exactly two sentences, preserving \
     every rule it states.",
    // Turn 4: deep-context recall, longest expected reply of the script.
    "Answer from the document: (a) what does the envelope checksum cover, in \
     field order; (b) what is excluded from it; (c) what happens when the \
     archive tier recomputes a mismatching checksum. One short paragraph.",
];

/// The request every turn sends. Greedy, pinned seed, fixed output budget:
/// the transcript equality the gate asserts only holds when the sampler
/// cannot introduce variation of its own. `stream` is true because the
/// shared transport is a stream parser; the comparison consumes the
/// aggregated text, not the framing.
pub(super) fn request_body(model: &str, messages: &[Value], max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": max_tokens,
        "messages": messages,
    })
}

/// The full user message of turn 1.
pub(super) fn first_turn() -> String {
    format!("{LONG_PREFIX}\n\n{}", TURNS[0])
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod probe_tests;
