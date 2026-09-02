// SPDX-License-Identifier: AGPL-3.0-only
//! Card rendering. The interesting cases are the ones that produce a WRONG card
//! quietly: a metric key that does not exist, a slot with no data, a model id
//! with an `&` in it.
use super::card::{Fmt, parse_args, render, spec_for};
use super::record::GateRecord;
use super::tests::{SHA, hw, run_record};
use crate::result::Verdict;

/// A record for `id` carrying `metrics`, built the way the gate builds one.
fn rec(id: &str, metrics: &[(&str, f64)]) -> GateRecord {
    let mut r = run_record(
        metrics
            .iter()
            .map(|(k, v)| ((*k).to_string(), *v))
            .collect(),
        Verdict::pass("ok"),
    );
    r.benchmark_id = id.to_string();
    GateRecord::from_run(
        &r,
        hw(),
        SHA.to_string(),
        Vec::new(),
        None,
        Default::default(),
    )
    .expect("record")
}

fn template() -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    std::fs::read_to_string(root.join("assets/cards/result-card.svg"))
        .expect("the card template is committed")
}

/// Every metric key a card spec names must exist on a real committed record.
///
/// ★ This is the test that matters. A typo'd key renders an EMPTY box — the card
/// still looks fine, and nobody notices until it is on social media. So the
/// specs are checked against the records the repo actually has rather than
/// against my memory of them.
#[test]
fn every_card_key_exists_on_a_real_record() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    let benchmarks = root.join(".benchmarks");
    if !benchmarks.exists() {
        return;
    }
    let mut missing = Vec::new();
    for entry in std::fs::read_dir(&benchmarks).unwrap().flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        // Newest record for this gate.
        let mut files: Vec<_> = std::fs::read_dir(entry.path())
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|e| e == "json")
                    && p.file_name().is_some_and(|n| n != "BASELINE.json")
            })
            .collect();
        files.sort();
        let Some(newest) = files.last() else { continue };
        let Ok(rec) = super::record::read_record(newest) else {
            continue;
        };
        let spec = spec_for(&id);
        if !spec.hero_key.is_empty() && !rec.metrics.contains_key(spec.hero_key) {
            missing.push(format!("{id}: hero key `{}` absent", spec.hero_key));
        }
        for s in spec.slots.iter().flatten() {
            if !rec.metrics.contains_key(s.key) {
                missing.push(format!("{id}: slot `{}` key `{}` absent", s.label, s.key));
            }
        }
    }
    assert!(
        missing.is_empty(),
        "card specs name metrics no record carries — these render as blank boxes:\n  {}",
        missing.join("\n  ")
    );
}

#[test]
fn a_missing_hero_metric_renders_an_em_dash_not_a_zero() {
    // A failed run has no metrics. The template ships the placeholder `000.0`,
    // and leaving it would publish a card claiming a measurement of zero.
    let r = rec("decode-floor", &[]);
    let svg = render(&template(), &r, &Default::default());
    assert!(svg.contains(">—<"), "expected an em dash for the hero");
    assert!(!svg.contains(">000.0<"), "the placeholder survived");
}

/// The group carrying an unused slot must be HIDDEN, not merely emptied.
///
/// Blanking the text leaves an empty bordered rectangle, which reads as a metric
/// that failed to render rather than as a benchmark that has three numbers and
/// not four. `display="none"` on the group takes the box, its accent bar and its
/// label with it.
#[test]
fn an_unused_slot_hides_its_whole_box() {
    let r = rec(
        "decode-floor",
        &[
            ("server_decode_tok_s", 22.7),
            ("accept_len_mean", 2.68),
            ("output_tokens", 795.0),
            ("runs", 3.0),
        ],
    );
    let svg = render(&template(), &r, &Default::default());
    assert!(
        hidden(&svg, "field-m4"),
        "slot 4 has no data, box still visible"
    );
    assert!(
        !hidden(&svg, "field-m1"),
        "slot 1 HAS data, must not be hidden"
    );
}

/// Is `display="none"` set on the element carrying this id?
fn hidden(svg: &str, id: &str) -> bool {
    let Some(at) = svg.find(&format!("id=\"{id}\"")) else {
        return false;
    };
    let open = svg[..at].rfind('<').unwrap_or(0);
    let close = svg[at..].find('>').map(|i| at + i).unwrap_or(svg.len());
    svg[open..close].contains("display=\"none\"")
}

/// The chip's own `fill`, not "does this colour appear anywhere" — brand green
/// is also the hero and the section heading, so a substring check would pass on
/// a card with no chip at all.
fn attr_of(svg: &str, id: &str, attr: &str) -> String {
    let Some(at) = svg.find(&format!("id=\"{id}\"")) else {
        return String::new();
    };
    let open = svg[..at].rfind('<').unwrap_or(0);
    let close = svg[at..].find('>').map(|i| at + i).unwrap_or(svg.len());
    let tag = &svg[open..close];
    let key = format!("{attr}=\"");
    match tag.find(&key) {
        Some(k) => {
            let v = &tag[k + key.len()..];
            v[..v.find('"').unwrap_or(0)].to_string()
        }
        None => String::new(),
    }
}

#[test]
fn a_passing_run_gets_a_green_chip_and_a_failing_one_gets_gold() {
    // `verdict_passes()` matches "PASS" exactly, so the fixtures use the real
    // spelling rather than a lowercase stand-in that would silently take the
    // failing branch and make this test agree with itself.
    let mut pass = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    pass.verdict = Some("PASS".to_string());
    let svg = render(&template(), &pass, &Default::default());
    assert!(svg.contains(">PASS<"), "no PASS text on a passing record");
    assert_eq!(attr_of(&svg, "verdict-chip", "fill"), "#12B981");

    let mut fail = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    fail.verdict = Some("FAIL".to_string());
    let svg = render(&template(), &fail, &Default::default());
    assert!(svg.contains(">FAIL<"), "no FAIL text on a failing record");
    // Gold, not red — the brand palette has no red.
    assert_eq!(attr_of(&svg, "verdict-chip", "fill"), "#EFB338");
    // And the hero stops being a large green dash.
    assert_eq!(attr_of(&svg, "value-toks", "fill"), "#82868F");
}

/// An ungated run has no verdict, and a placeholder pill invites the reader to
/// wonder what it means.
#[test]
fn a_record_with_no_verdict_hides_the_chip_entirely() {
    let mut r = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    r.verdict = None;
    let svg = render(&template(), &r, &Default::default());
    assert!(hidden(&svg, "field-verdict"), "the chip is still showing");
}

#[test]
fn absent_attribution_hides_its_box() {
    let r = rec("decode-floor", &[("server_decode_tok_s", 22.7)]);
    let svg = render(&template(), &r, &parse_args("author=Ada").unwrap());
    assert!(!hidden(&svg, "field-author"), "author was given");
    assert!(hidden(&svg, "field-handle"), "handle was not given");
    assert!(hidden(&svg, "field-site"), "website was not given");
}

#[test]
fn an_ampersand_in_a_model_id_does_not_break_the_svg() {
    let mut r = rec("decode-floor", &[("server_decode_tok_s", 1.0)]);
    r.target_model = "vendor/model&<thing>".to_string();
    let svg = render(&template(), &r, &Default::default());
    assert!(
        svg.contains("vendor/model&amp;&lt;thing&gt;"),
        "not escaped"
    );
}

#[test]
fn args_parse_and_a_missing_equals_is_refused() {
    let ok = parse_args("author=Ada Lovelace, handle=@ada ,website=ada.dev,").unwrap();
    assert_eq!(ok.get("author").map(String::as_str), Some("Ada Lovelace"));
    assert_eq!(ok.get("handle").map(String::as_str), Some("@ada"));
    assert_eq!(ok.get("website").map(String::as_str), Some("ada.dev"));
    // Silently dropping this would produce a card with no author and no complaint.
    assert!(parse_args("authorAda").is_err());
    assert!(parse_args("=nokey").is_err());
}

#[test]
fn formats_round_the_way_a_reader_expects() {
    assert_eq!(Fmt::Int.apply(114.6), "115");
    assert_eq!(Fmt::One.apply(22.68), "22.7");
    assert_eq!(Fmt::Two.apply(84.219), "84.22");
    assert_eq!(Fmt::Ms.apply(8288.4), "8288 ms");
}
