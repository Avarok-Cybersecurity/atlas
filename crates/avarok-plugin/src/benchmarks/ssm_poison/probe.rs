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

use crate::benchmarks::transcript::Transcript;

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
/// aggregated text, not the framing. `stream_options.include_usage` is the
/// OpenAI contract for usage on a stream: without it, correct
/// `completion_tokens` and the `cached_tokens` attestation the vacuity
/// check reads are an Atlas-specific courtesy, and running the gate against
/// a contract-faithful server would silently zero both.
///
/// Plain greedy body. The gate run pins TWO things via the serve config, both
/// required, established empirically 2026-08-15 (12/12 replays byte-identical
/// only with both):
///   1. the STREAMING content bug fixed in `api/strip.rs` — `strip_orphan_tool
///      _markup` ran `.trim_end()` per delta, deleting any delta that was
///      entirely trailing whitespace: Qwen's standalone space token before a
///      digit (" 7" -> "") and the `\n\n` between numbered lines. The model's
///      correct "ACK 7741-C … 7" streamed as "ACK7741-C … 7" and "1\n2\n3"
///      collapsed to one line — turns 1-2's anchors. (#473/680b3a568 regression,
///      exposed by #513's strict anchors.)
///   2. thinking OFF via `--serve-override disable_thinking=true`. Even with the
///      streaming bug fixed, thinking-ON burns the 768-token budget then
///      truncates turns 2-3 (finish_reason=length) and over-produces (turn 2
///      gave 5 numbered lines, not 3). The gate wants deterministic terse
///      replies, so it disables thinking through the serve-flag path BFCL uses
///      (NOT the per-request `chat_template_kwargs.enable_thinking:false`, which
///      degenerates to a 2-token stop on qwen3_6_moe). The shared bf16head
///      recipe carries `disable_thinking: false` in its defaults precisely so
///      this override is accepted; the other gates sharing it keep thinking on.
///
/// Nothing under test is lost by thinking-off: the gate polices the PREFILL
/// restore of the shared document prefix; thinking tokens are post-prefill
/// output the template strips from replayed history.
pub(super) fn request_body(model: &str, messages: &[Value], max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "stream_options": {"include_usage": true},
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

/// The number of sections in [`LONG_PREFIX`]. Turn 1 asks for this count, so
/// the anchor below must track the document; deriving it from the frozen text
/// keeps the two from drifting apart silently.
fn section_count() -> usize {
    (1..)
        .take_while(|n| LONG_PREFIX.contains(&format!("Section {n},")))
        .count()
}

fn contains_token(text: &str, expected: &str) -> bool {
    text.split(|c: char| !c.is_alphanumeric())
        .any(|token| token.eq_ignore_ascii_case(expected))
}

fn has_number_prefix(line: &str, number: usize) -> bool {
    line.strip_prefix(&number.to_string())
        .and_then(|rest| rest.chars().next())
        .is_some_and(|c| c.is_whitespace() || matches!(c, '.' | ')' | ':'))
}

fn contains_in_order(text: &str, phrases: &[&str]) -> bool {
    let text = text.to_lowercase();
    let mut cursor = 0;
    phrases.iter().all(|phrase| {
        let Some(found) = text[cursor..].find(phrase) else {
            return false;
        };
        cursor += found + phrase.len();
        true
    })
}

/// Semantic anchors on the REFERENCE round. Every violation found, or empty.
///
/// The comparison in `compare.rs` is purely relative: replays are held to
/// round 0, but nothing held round 0 to anything. Poisoning that is
/// deterministic from the first round — a corrupt prefill that produces the
/// same wrong bytes every time — therefore passed as Invariant. These anchors
/// pin the reference to the pinned script itself: turn 1 must acknowledge the
/// document and its section count, turn 2 must be the three numbered
/// invariants the prompt demands. They check the SCRIPT's contract, not any
/// model's phrasing, so the gate stays model-agnostic: a serving-grade model
/// that cannot follow "reply with exactly one line" or "numbered 1 to 3, one
/// per line" at temperature 0 cannot anchor this gate, and saying so loudly
/// beats certifying replays against garbage.
///
/// A turn that ran into the token budget (`finish_reason == "length"`) is a
/// violation for a second reason: a truncated reference caps every length
/// ratio at 1.0-ish and makes the runaway ceiling unreachable, so the budget
/// must be sized to let the reference finish on its own terms.
pub(super) fn validate_reference(reference: &[Transcript]) -> Vec<String> {
    let mut violations = Vec::new();
    if reference.len() != TURNS.len() {
        violations.push(format!(
            "reference has {} turn(s), script has {}",
            reference.len(),
            TURNS.len()
        ));
        return violations;
    }
    for (i, t) in reference.iter().enumerate() {
        match t.finish_reason.as_deref() {
            Some("stop") => {}
            Some("length") => violations.push(format!(
                "turn {}: reference hit the token budget (finish_reason=length) — a \
                 truncated reference cannot anchor the collapse ratios",
                i + 1
            )),
            other => violations.push(format!(
                "turn {}: reference did not finish normally (finish_reason={other:?})",
                i + 1
            )),
        }
    }
    let t1 = &reference[0].text;
    if !t1.contains("ACK 7741-C") {
        violations.push("turn 1: missing the demanded 'ACK 7741-C' acknowledgement".into());
    }
    // The section count must appear OUTSIDE the document id — "7741-C"
    // contains a 7 of its own, which would make the check vacuous. A model
    // may spell the number out; both forms anchor equally well.
    let sections = section_count();
    let words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let stripped = t1.replace("7741-C", "");
    let has_digit = contains_token(&stripped, &sections.to_string());
    let has_word = words
        .get(sections)
        .is_some_and(|word| contains_token(&stripped, word));
    if !has_digit && !has_word {
        violations.push(format!(
            "turn 1: does not state the document's section count ({sections})"
        ));
    }
    let lines: Vec<&str> = reference[1]
        .text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() != 3 {
        violations.push(format!(
            "turn 2: expected exactly 3 numbered lines, got {}",
            lines.len()
        ));
    } else {
        for (i, line) in lines.iter().enumerate() {
            if !has_number_prefix(line, i + 1) {
                violations.push(format!(
                    "turn 2: line {} does not start with its number: {:?}",
                    i + 1,
                    line
                ));
            }
        }
    }
    let t3 = reference[2].text.to_lowercase();
    let t3_sentences = t3
        .split_terminator(['.', '!', '?'])
        .filter(|sentence| !sentence.trim().is_empty())
        .count();
    // Each anchor named once, and used both to decide and to report. The
    // message used to lump all five under "does not preserve Section 4 in
    // exactly two sentences", so a field-order failure and a sentence-count
    // failure read identically.
    let t3_fields = contains_in_order(
        &t3,
        &[
            "batch id",
            "sequence number",
            "node id",
            "timestamp",
            "payload length",
        ],
    );
    // Stemmed, like `recompute` and `quarantin` beside it. The literal
    // "excludes payload" cannot match the document this reference is
    // rewriting: Section 4 says "The checksum excludes the payload itself",
    // and the definite article sits between the two words. A model quoting the
    // source faithfully — which at temperature 0 is exactly what it does —
    // failed an anchor that no wording in the document, and nothing the prompt
    // asked for, could satisfy.
    let t3_exclusion = t3.contains("exclud") && t3.contains("payload");
    let t3_recompute = t3.contains("recompute");
    let t3_quarantine = t3.contains("quarantin");
    if t3_sentences != 2 || !t3_fields || !t3_exclusion || !t3_recompute || !t3_quarantine {
        violations.push(format!(
            "turn 3: Section 4 rewrite failed an anchor (sentences={t3_sentences}, \
             fields_in_order={t3_fields}, exclusion={t3_exclusion}, \
             recompute={t3_recompute}, quarantine={t3_quarantine})"
        ));
    }
    let t4 = reference[3].text.to_lowercase();
    if !contains_in_order(
        &t4,
        &[
            "batch id",
            "sequence number",
            "node id",
            "timestamp",
            "payload length",
        ],
        // Stemmed for the same reason as turn 3: the document says "excludes",
        // and "excluded" only matches a passive rewording the prompt never asked
        // for. `payload` is still required independently on the next line.
    ) || !t4.contains("exclud")
        || !t4.contains("payload")
        || !t4.contains("recompute")
        || !t4.contains("quarantin")
    {
        violations.push("turn 4: does not answer every checksum question".into());
    }
    violations
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod probe_tests;
