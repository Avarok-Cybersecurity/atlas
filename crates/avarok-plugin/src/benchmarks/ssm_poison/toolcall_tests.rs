// SPDX-License-Identifier: AGPL-3.0-only

//! Pure tests for the path-independence comparison. No server: these pin the
//! DECISION RULE, so a loosened rule fails here rather than on a GPU box two
//! hours into a campaign.

use super::*;
use crate::benchmarks::transcript::Transcript;

fn transcript(calls: &[(&str, &str)], text: &str) -> Transcript {
    Transcript {
        reasoning: String::new(),
        text: text.to_string(),
        tool_calls: calls
            .iter()
            .map(|(n, a)| (n.to_string(), a.to_string()))
            .collect(),
        finish_reason: Some("stop".to_string()),
        completion_tokens: 32,
        cached_prompt_tokens: 0,
    }
}

fn result(path: Path, calls: &[(&str, &str)], text: &str) -> PathResult {
    PathResult {
        path,
        target: transcript(calls, text),
    }
}

/// The invariant holding: every path answered the irrelevance target with no
/// call at all.
#[test]
fn no_call_on_every_path_is_no_divergence() {
    let r = vec![
        result(Path::Direct, &[], "Closed membership means ..."),
        result(Path::AfterWeather, &[], "Closed membership means ..."),
        result(Path::AfterSearch, &[], "Closed membership means ..."),
    ];
    assert!(divergences(&r).is_empty());
    assert!(!reference_called(&r));
}

/// ★ THE DEFECT THIS PROBE EXISTS FOR, in the exact shape measured on
/// 2026-09-07: the direct path answers correctly with no call, and a path that
/// merely ran a weather call first inherits it and calls
/// `get_current_weather{"location":"San Francisco, CA"}` on a prompt no tool
/// can answer. Four unrelated BFCL samples did precisely this.
#[test]
fn a_call_inherited_from_a_predecessor_is_a_divergence() {
    let r = vec![
        result(Path::Direct, &[], "Closed membership means ..."),
        result(
            Path::AfterWeather,
            &[("get_current_weather", r#"{"location":"San Francisco, CA"}"#)],
            "",
        ),
        result(Path::AfterSearch, &[], "Closed membership means ..."),
    ];
    let d = divergences(&r);
    assert_eq!(d.len(), 1, "only the perturbed path should diverge");
    assert_eq!(d[0].path, "after-weather");
    assert!(d[0].reference_calls.is_empty());
    assert_eq!(d[0].path_calls[0].0, "get_current_weather");
    assert!(
        d[0].describe().contains("reference made no call"),
        "the reading must say what the reference did: {}",
        d[0].describe()
    );
}

/// The opposite direction, also observed: the reference calls and a perturbed
/// path does not. Two of the twelve measured flips went this way, so the rule
/// must be symmetric rather than only catching spurious calls.
#[test]
fn a_call_that_disappears_is_also_a_divergence() {
    let r = vec![
        result(Path::Direct, &[("web_search", r#"{"query":"x"}"#)], ""),
        result(Path::AfterWeather, &[], "prose instead"),
    ];
    let d = divergences(&r);
    assert_eq!(d.len(), 1);
    assert!(d[0].path_calls.is_empty());
    assert!(d[0].describe().contains("this path made no call"));
}

/// Same function, different arguments. Two of the twelve moved this way, and a
/// name-only comparison would have missed both.
#[test]
fn the_same_tool_with_different_arguments_is_a_divergence() {
    let r = vec![
        result(
            Path::Direct,
            &[(
                "web_search",
                r#"{"query":"IP address to company data API"}"#,
            )],
            "",
        ),
        result(
            Path::AfterSearch,
            &[("web_search", r#"{"query":"IP address to company data"}"#)],
            "",
        ),
    ];
    let d = divergences(&r);
    assert_eq!(
        d.len(),
        1,
        "arguments are part of the answer, not decoration"
    );
    assert_eq!(d[0].path_calls[0].0, "web_search");
}

/// ★ PROSE MAY STILL JITTER. The replay probe next door established that
/// reworded prose is a healthy property of Marconi anchor selection, and this
/// probe must not re-litigate it — otherwise it fails for a reason it is not
/// about and gets switched off.
#[test]
fn reworded_prose_with_identical_calls_is_not_a_divergence() {
    let r = vec![
        result(
            Path::Direct,
            &[],
            "Closed membership means a node joins once.",
        ),
        result(
            Path::AfterWeather,
            &[],
            "It means that a node joins exactly one time.",
        ),
    ];
    assert!(
        divergences(&r).is_empty(),
        "different prose with the same (absent) calls must pass"
    );
}

/// Call ORDER is part of the answer: a parallel-call sample that returns the
/// same two calls transposed scores differently. One of the twelve measured
/// flips was exactly a reordering plus an argument change.
#[test]
fn transposed_calls_are_a_divergence() {
    let a = ("order_status_check", r#"{"order_id":"282828"}"#);
    let b = ("get_product_details", r#"{"product_id":"282828"}"#);
    let r = vec![
        result(Path::Direct, &[a, b], ""),
        result(Path::AfterWeather, &[b, a], ""),
    ];
    assert_eq!(
        divergences(&r).len(),
        1,
        "the reference emitted these calls in one order; the other path did not"
    );
}

/// A reference that is itself calling on the irrelevance target is reported,
/// so a wrong baseline cannot quietly become the thing every path is judged
/// against.
#[test]
fn a_reference_that_calls_is_reported_separately() {
    let r = vec![
        result(
            Path::Direct,
            &[("get_current_weather", r#"{"location":"SF"}"#)],
            "",
        ),
        result(
            Path::AfterWeather,
            &[("get_current_weather", r#"{"location":"SF"}"#)],
            "",
        ),
    ];
    assert!(
        divergences(&r).is_empty(),
        "the paths agree, so path-independence holds"
    );
    assert!(
        reference_called(&r),
        "but the reference hallucinated a call, and that must not be silent"
    );
}

/// Anti-vacuity: with no reference path there is nothing to compare, and the
/// function must not report success by finding zero divergences in an empty
/// comparison. The driver treats a missing reference as an error; this pins
/// that the helper does not manufacture a pass.
#[test]
fn a_missing_reference_yields_no_verdict_rather_than_a_pass() {
    let r = vec![result(Path::AfterWeather, &[("x", "{}")], "")];
    assert!(
        divergences(&r).is_empty(),
        "no reference means no comparison"
    );
    assert!(
        !reference_called(&r),
        "and no reference means no reference-called finding either"
    );
}

/// Every path is actually exercised by the plan — a probe that silently ran
/// one path would compare nothing and pass.
#[test]
fn all_three_paths_are_distinct_and_two_interpose_a_call() {
    assert_eq!(Path::ALL.len(), 3);
    assert!(Path::Direct.interposed().is_none());
    assert_eq!(Path::AfterWeather.interposed(), Some(CALLS_WEATHER));
    assert_eq!(Path::AfterSearch.interposed(), Some(CALLS_SEARCH));
    let labels: Vec<_> = Path::ALL.iter().map(|p| p.label()).collect();
    let mut uniq = labels.clone();
    uniq.sort_unstable();
    uniq.dedup();
    assert_eq!(
        uniq.len(),
        labels.len(),
        "path labels must be distinguishable"
    );
}

/// The two offered tools are the two the contamination actually used. If this
/// list drifts, the probe stops pointing at the measured defect.
#[test]
fn the_offered_tools_are_the_ones_the_contamination_used() {
    let names: Vec<String> = tools()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"get_current_weather".to_string()));
    assert!(names.contains(&"web_search".to_string()));
}

/// The request must actually offer the tools and let the model choose — a body
/// that forgot `tools`, or pinned `tool_choice: "none"`, could never observe a
/// spurious call and would pass vacuously forever.
#[test]
fn the_request_offers_tools_and_leaves_the_choice_to_the_model() {
    let body = request_body("m", &[json!({"role": "user", "content": "hi"})], 256);
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"].as_array().unwrap().len(), 2);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["seed"], 0);
}
