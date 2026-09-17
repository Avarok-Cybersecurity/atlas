// SPDX-License-Identifier: AGPL-3.0-only

//! Every case below is a MUTATION this scorer must catch. The fixtures are
//! recorded outcomes, so none of it needs a GPU.

use super::super::transcript::{RequestOutcome, Transcript};
use super::{Legs, score, verdict};
use crate::result::VerdictKind;

fn t(text: &str, toks: usize) -> RequestOutcome {
    RequestOutcome::Ok(Box::new(Transcript {
        text: text.into(),
        finish_reason: Some("stop".into()),
        completion_tokens: toks,
        ..Default::default()
    }))
}
fn err() -> RequestOutcome {
    RequestOutcome::Error("connection reset".into())
}
fn canaries(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("CANARY-{i}-run")).collect()
}
fn legs<'a>(
    a: &'a [RequestOutcome],
    b: &'a [RequestOutcome],
    rungs: &'a [(String, Vec<RequestOutcome>)],
    post: &'a [RequestOutcome],
    cans: &'a [String],
) -> Legs<'a> {
    Legs {
        ref_a: a,
        ref_b: b,
        rungs,
        post,
        canaries: cans,
        min_completion_tokens: 16,
    }
}

#[test]
fn identical_streams_pass() {
    let c = canaries(1);
    let r = vec![t("hello world", 20)];
    let rungs = vec![("c2".to_string(), vec![t("hello world", 20)])];
    let post = vec![t("hello world", 20)];
    let s = score(&legs(&r, &r.clone(), &rungs, &post, &c));
    assert_eq!(s.identical, 2, "one rung + post");
    assert_eq!(s.diverged + s.contaminated + s.unmeasured, 0);
    assert!(
        matches!(verdict(&s).kind, VerdictKind::Pass),
        "{:?}",
        verdict(&s).reason
    );
    assert!(
        verdict(&s).reason.contains("tokens compared"),
        "a pass must state the volume it compared, or 3 identical tokens \
         masquerade as strong evidence"
    );
}

#[test]
fn divergence_at_k_is_localized_and_fails() {
    let c = canaries(1);
    let r = vec![t("abcdefgh", 20)];
    let rungs = vec![("c2".to_string(), vec![t("abcdXXXX", 20)])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(s.diverged, 1);
    // 5, not 4: `canonical()` opens with the (empty) reasoning field and its
    // \u{1} separator, so the common prefix is that separator plus "abcd".
    // Comparing the canonical form rather than `text` alone is what makes a
    // changed tool call or finish_reason count as divergence at all.
    assert_eq!(s.earliest_divergence, Some(5), "sep + 'abcd'");
    assert!(!matches!(verdict(&s).kind, VerdictKind::Pass));
}

#[test]
fn foreign_canary_is_contamination_not_divergence() {
    let c = canaries(2);
    let r = vec![
        t("mine CANARY-0-run ok", 20),
        t("theirs CANARY-1-run ok", 20),
    ];
    // prompt 0's reply now carries prompt 1's canary
    let rungs = vec![(
        "c2".to_string(),
        vec![
            t("mine CANARY-1-run ok", 20),
            t("theirs CANARY-1-run ok", 20),
        ],
    )];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(s.contaminated, 1, "leakage, not a generic diff");
    assert_eq!(s.diverged, 0, "must NOT be lumped in with divergence");
    assert_eq!(s.foreign_canaries, 1);
}

#[test]
fn solo_stream_or_usage_instability_disqualifies_attribution() {
    let c = canaries(1);
    let a = vec![t("alpha", 20)];
    let b = vec![t("beta", 20)]; // the two SOLO runs already disagree
    let rungs = vec![("c2".to_string(), vec![t("gamma", 20)])];
    let s = score(&legs(&a, &b, &rungs, &[], &c));
    assert_eq!(s.alone_unstable, 1);
    assert_eq!(
        s.diverged, 0,
        "must not blame concurrency for a #435 defect"
    );
    assert_eq!(
        s.compared, 0,
        "an unattributable prompt contributes no comparisons"
    );
    assert!(!matches!(verdict(&s).kind, VerdictKind::Pass));

    let a = vec![t("stable", 20)];
    let b = vec![t("stable", 21)];
    let rungs = vec![("c2".to_string(), vec![t("stable", 20)])];
    let s = score(&legs(&a, &b, &rungs, &[], &c));
    assert_eq!(s.alone_unstable, 1, "solo usage counts disagree");
    assert_eq!(
        s.compared, 0,
        "an unstable reference cannot attribute a rung"
    );
}

#[test]
fn one_crashed_request_cannot_pass() {
    let c = canaries(1);
    let r = vec![t("hello world", 20)];
    let rungs = vec![("c2".to_string(), vec![err()])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(s.unmeasured, 1);
    assert!(
        !matches!(verdict(&s).kind, VerdictKind::Pass),
        "an error is not vacuously identical"
    );
}

/// ★ THE HEADLINE TRAP: zero divergence over zero comparisons reading as PASS.
#[test]
fn all_crashed_is_fail_not_clean() {
    let c = canaries(1);
    let r = vec![err()];
    let rungs = vec![("c2".to_string(), vec![err()])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(s.compared, 0);
    assert_eq!(s.diverged, 0, "nothing diverged — because nothing ran");
    let v = verdict(&s);
    assert!(!matches!(v.kind, VerdictKind::Pass));
    assert!(v.reason.contains("nothing measured"), "{}", v.reason);
}

#[test]
fn short_reply_below_floor_is_unmeasured() {
    let c = canaries(1);
    let r = vec![t("hi", 3)];
    let rungs = vec![("c2".to_string(), vec![t("hi", 3)])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(s.unmeasured, 1, "3 tokens cannot witness contamination");
    assert_eq!(s.identical, 0, "equal-but-empty is not evidence");

    let reference = vec![t("enough", 16)];
    let rungs = vec![("c2".to_string(), vec![t("enough", 16)])];
    let s = score(&legs(&reference, &reference.clone(), &rungs, &[], &c));
    assert_eq!(s.identical, 1, "exactly at the floor is measurable");
    assert_eq!(s.unmeasured, 0);

    let short_reference = vec![t("same stream", 3)];
    let rungs = vec![("c2".to_string(), vec![t("same stream", 20)])];
    let s = score(&legs(
        &short_reference,
        &short_reference.clone(),
        &rungs,
        &[],
        &c,
    ));
    assert_eq!(s.unmeasured, 1, "the solo reference is below the floor");
    assert_eq!(s.compared, 0, "an empty reference cannot measure a rung");
}

#[test]
fn tool_call_argument_diff_is_divergence() {
    let c = canaries(1);
    let mk = |args: &str| {
        RequestOutcome::Ok(Box::new(Transcript {
            text: "same".into(),
            tool_calls: vec![("get_weather".into(), args.into())],
            finish_reason: Some("tool_calls".into()),
            completion_tokens: 20,
            ..Default::default()
        }))
    };
    let r = vec![mk(r#"{"city":"Paris"}"#)];
    let rungs = vec![("c2".to_string(), vec![mk(r#"{"city":"Berlin"}"#)])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(s.diverged, 1, "tool calls ride a separate field from text");
}

#[test]
fn finish_reason_diff_is_divergence() {
    let c = canaries(1);
    let mk = |fr: &str| {
        RequestOutcome::Ok(Box::new(Transcript {
            text: "same text".into(),
            finish_reason: Some(fr.into()),
            completion_tokens: 20,
            ..Default::default()
        }))
    };
    let r = vec![mk("stop")];
    let rungs = vec![("c2".to_string(), vec![mk("length")])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(s.diverged, 1);
}

#[test]
fn post_check_divergence_is_persistent() {
    let c = canaries(1);
    let r = vec![t("stable", 20)];
    let rungs = vec![("c2".to_string(), vec![t("stable", 20)])];
    let post = vec![t("corrupted", 20)];
    let s = score(&legs(&r, &r.clone(), &rungs, &post, &c));
    assert_eq!(s.persistent, 1, "state survived the concurrent episode");
    assert_eq!(s.diverged, 0, "persistent is its own, worse class");
    assert!(verdict(&s).reason.contains("PERSISTENT"));
}

#[test]
fn usage_count_mismatch_with_equal_text_is_divergence() {
    let c = canaries(1);
    let r = vec![t("identical text", 20)];
    let rungs = vec![("c2".to_string(), vec![t("identical text", 25)])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(
        s.diverged, 1,
        "the server accounted for the same text differently"
    );
}

#[test]
fn reasoning_is_compared_separately_from_text() {
    let c = canaries(1);
    let mk = |think: &str| {
        RequestOutcome::Ok(Box::new(Transcript {
            reasoning: think.into(),
            text: "same answer".into(),
            finish_reason: Some("stop".into()),
            completion_tokens: 20,
            ..Default::default()
        }))
    };
    let r = vec![mk("thought A")];
    let rungs = vec![("c2".to_string(), vec![mk("thought B")])];
    let s = score(&legs(&r, &r.clone(), &rungs, &[], &c));
    assert_eq!(
        s.diverged, 1,
        "a changed chain-of-thought behind an identical answer is still a change"
    );
}
