// SPDX-License-Identifier: AGPL-3.0-only

//! The shape rules in `.github/pr-taxonomy.json`'s own `_doc` are worthless
//! unless something enforces them. These do.
//!
//! The load-bearing one is `benches_may_only_add`: it is the property that
//! makes it acceptable to let a language model influence a merge gate at all.

use super::*;

fn tree(json: &str) -> Vec<Node> {
    let v: serde_json::Value = serde_json::from_str(json).expect("fixture parses");
    parse_children(&v).expect("fixture builds")
}

fn tree_err(json: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(json).expect("fixture parses");
    parse_children(&v).unwrap_err().to_string()
}

/// ★ `_benches` as a BARE STRING used to parse as empty here while the ci.yml
/// jq read it fine — two implementations of one function disagreeing, with the
/// Rust half failing in the REMOVING direction. A typo must be loud.
#[test]
fn a_bare_string_benches_is_rejected_not_silently_dropped() {
    let err = tree_err(r#"{ "a": { "_benches": "bfcl-subset" }, "b": {} }"#);
    assert_eq!(
        err,
        "a: _benches must be an ARRAY of benchmark ids, got \"bfcl-subset\". A bare string \
         parses as empty here while jq reads it, so the two halves would disagree — in the \
         removing direction."
    );
}

#[test]
fn a_non_string_benches_entry_is_rejected() {
    let err = tree_err(r#"{ "a": { "_benches": [1, "bfcl-subset"] }, "b": {} }"#);
    assert_eq!(
        err,
        "a: _benches contains a non-string entry (1). A silently-dropped entry removes a benchmark."
    );
    let err = tree_err(r#"{ "a": { "_benches": [["bfcl-subset"]] }, "b": {} }"#);
    assert_eq!(
        err,
        "a: _benches contains a non-string entry ([\"bfcl-subset\"]). A silently-dropped entry removes a benchmark."
    );
}

// ── The real file ──────────────────────────────────────────────────────────

/// The shipped taxonomy loads and satisfies every rule it documents.
#[test]
fn the_real_taxonomy_is_valid() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let roots = load(&root).expect("`.github/pr-taxonomy.json` loads and validates");
    fn project(nodes: &[Node], parent: &str, out: &mut Vec<String>) {
        for node in nodes {
            let path = if parent.is_empty() {
                node.name.clone()
            } else {
                format!("{parent}/{}", node.name)
            };
            out.push(format!("{path}={}", node.benches.join(",")));
            project(&node.children, &path, out);
        }
    }
    let mut projection = Vec::new();
    project(&roots, "", &mut projection);
    assert_eq!(
        projection,
        [
            "correctness=bfcl-subset",
            "correctness/numerics=bfcl-subset-echolp",
            "correctness/loader=bfcl-subset-echolp,ttft-cold-gate",
            "correctness/tool-calling=bfcl-subset-echolp",
            "correctness/ssm-state=ttft-warm-gate,ssm-state-poisoning-gate",
            "correctness/kv-cache=ttft-warm-gate,ttft-cold-gate,ssm-state-poisoning-gate",
            "correctness/sampling=bfcl-subset-echolp,agentic-webserver",
            "performance=agentic-webserver",
            "performance/decode=bfcl-subset,ttft-warm-gate,decode-floor",
            "performance/prefill=ttft-cold-gate,ttft-warm-gate",
            "performance/kernel-dispatch=bfcl-subset,decode-floor",
            "performance/memory-traffic=bfcl-subset,ttft-warm-gate,decode-floor",
            "performance/scheduling=ttft-warm-gate,concurrency-sweep",
            "performance/speculation=bfcl-subset,decode-floor",
            "capability=bfcl-subset,agentic-webserver",
            "capability/new-model=ttft-cold-gate",
            "capability/new-hardware=ttft-cold-gate,ttft-warm-gate",
            "capability/quantization=bfcl-subset-echolp",
            "capability/adapters=bfcl-subset-echolp,ttft-cold-gate",
            "capability/serving-api=concurrency-sweep,ttft-warm-gate",
            "infrastructure=",
            "infrastructure/ci=",
            "infrastructure/benchmark-gate=",
            "infrastructure/release=",
            "infrastructure/observability=",
            "infrastructure/build-system=",
            "documentation=",
            "documentation/reference=",
            "documentation/design-record=",
            "unknown=",
        ]
    );
}

/// The 2026-08-16 fill: `correctness/ssm-state` names the gate built for the
/// Marconi SSM-snapshot poisoning class. Before the fill the ssm-state intent
/// implied only a warm-TTFT leg — the one gate that polices restored state
/// directly was unreachable through intent.
#[test]
fn ssm_state_intent_implies_the_poison_gate() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let roots = load(&root).unwrap();
    for path in [
        vec!["correctness".to_string(), "ssm-state".into()],
        vec!["correctness".to_string(), "kv-cache".into()],
    ] {
        let got = benches_for(&roots, &path);
        let expected: std::collections::BTreeSet<String> = match path[1].as_str() {
            "ssm-state" => &["bfcl-subset", "ssm-state-poisoning-gate", "ttft-warm-gate"][..],
            "kv-cache" => &[
                "bfcl-subset",
                "ssm-state-poisoning-gate",
                "ttft-cold-gate",
                "ttft-warm-gate",
            ][..],
            _ => unreachable!(),
        }
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        assert_eq!(got, expected, "{path:?}");
    }
}

// ── ★ The safety property ──────────────────────────────────────────────────

/// **The whole design rests on this.** Descending FURTHER can only grow the
/// implied set; it can never shrink it.
///
/// If a deeper path could drop an ancestor's benchmark, then writing a
/// misleading PR title would skip tests — and the diff would not even have to
/// lie, only the prose. Union semantics is what makes a misclassification cost
/// GPU minutes instead of coverage.
#[test]
fn benches_may_only_add() {
    let roots = tree(
        r#"{
            "correctness": {
              "_benches": ["bfcl-subset"],
              "kv-cache":  { "_benches": ["ttft-warm-gate"] },
              "sampling":  {}
            },
            "unknown": {}
        }"#,
    );
    let parent = benches_for(&roots, &["correctness".into()]);
    for child in ["kv-cache", "sampling"] {
        let deeper = benches_for(&roots, &["correctness".into(), child.into()]);
        assert!(
            parent.is_subset(&deeper),
            "descending into {child} DROPPED {:?} — a deeper path must never \
             imply fewer benchmarks than its ancestor",
            parent.difference(&deeper).collect::<Vec<_>>()
        );
    }
    // And it genuinely adds where declared.
    assert!(
        benches_for(&roots, &["correctness".into(), "kv-cache".into()]).contains("ttft-warm-gate")
    );
}

/// A path that has gone stale — a category renamed out from under an old label
/// — must degrade to FEWER extra benches, never take the view down.
#[test]
fn an_unknown_segment_degrades_instead_of_failing() {
    let roots = tree(
        r#"{ "performance": { "_benches": ["agentic-webserver"], "decode": {}, "prefill": {} },
             "unknown": {} }"#,
    );
    let got = benches_for(
        &roots,
        &["performance".into(), "a-category-that-was-renamed".into()],
    );
    assert_eq!(
        got.iter().cloned().collect::<Vec<_>>(),
        vec!["agentic-webserver".to_string()],
        "should keep what matched and stop, not panic and not return nothing"
    );
}

// ── The descent ────────────────────────────────────────────────────────────

/// The classifier is offered exactly the children of where it stands.
#[test]
fn options_are_the_current_nodes_children() {
    let roots = tree(r#"{ "a": { "x": {}, "y": {} }, "b": {}, "unknown": {} }"#);
    assert_eq!(
        options_at(&roots, &[]).unwrap(),
        vec!["a".to_string(), "b".into(), "unknown".into()]
    );
    assert_eq!(
        options_at(&roots, &["a".into()]).unwrap(),
        vec!["x".to_string(), "y".into()]
    );
}

/// A leaf offers nothing — the descent is over, and asking again would invent
/// a level that does not exist.
#[test]
fn a_leaf_offers_no_options() {
    let roots = tree(r#"{ "a": { "x": {}, "y": {} }, "b": {}, "unknown": {} }"#);
    assert!(options_at(&roots, &["a".into(), "x".into()]).is_none());
    assert!(options_at(&roots, &["b".into()]).is_none());
    assert!(is_complete(&roots, &["b".into()]));
    assert!(!is_complete(&roots, &["a".into()]));
}

/// A set of one is not a choice. `options_at` must not offer it, and `resolve`
/// follows it automatically — otherwise every such node burns an API call to
/// produce an answer that was already determined.
#[test]
fn a_lone_child_is_followed_not_asked() {
    // Built directly: `validate` rejects this shape, and that rejection is
    // itself tested below. This pins the behaviour if the rule is relaxed.
    let roots = tree(r#"{ "a": { "only": { "p": {}, "q": {} } }, "b": {} }"#);
    assert!(
        options_at(&roots, &["a".into()]).is_none(),
        "a single child must not be offered as a choice"
    );
    assert_eq!(
        resolve(&roots, &["a".into()]),
        vec!["a".to_string(), "only".into()],
        "resolve should have walked through the forced step"
    );
}

// ── Validation ─────────────────────────────────────────────────────────────

#[test]
fn a_single_child_node_is_rejected() {
    let roots = tree(r#"{ "a": { "only": {} }, "b": {} }"#);
    let err = validate(&roots).unwrap_err().to_string();
    assert_eq!(
        err,
        "a has exactly one child (only). Either give it a sibling or make a a leaf."
    );
}

#[test]
fn a_non_kebab_key_is_rejected() {
    let roots = tree(r#"{ "Perf Stuff": {}, "b": {} }"#);
    let err = validate(&roots).unwrap_err().to_string();
    assert_eq!(
        err,
        "Perf Stuff: keys must be lowercase kebab-case so a path is a safe label"
    );
}

#[test]
fn a_bench_that_is_not_a_required_gate_is_rejected() {
    let roots = tree(r#"{ "a": { "_benches": ["no-such-bench"] }, "b": {} }"#);
    let err = validate(&roots).unwrap_err().to_string();
    assert_eq!(
        err,
        "a: _benches names \"no-such-bench\", which is not a required benchmark. A path that \
         selects a benchmark nobody runs is a silent no-op."
    );
}

#[test]
fn a_one_root_tree_is_rejected() {
    let roots = tree(r#"{ "only": {} }"#);
    let err = validate(&roots).unwrap_err().to_string();
    assert_eq!(
        err,
        "the taxonomy needs at least two roots; one root is not a choice"
    );
}

/// `_doc` and `_benches` are metadata, not categories. If they leaked into the
/// children they would be offered to the classifier as things to pick.
#[test]
fn reserved_keys_are_not_categories() {
    let roots = tree(r#"{ "_doc": ["notes"], "a": { "_benches": ["bfcl-subset"] }, "b": {} }"#);
    let names: Vec<&str> = roots.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(names, vec!["a", "b"]);
    assert_eq!(roots[0].benches, ["bfcl-subset"]);
    assert!(roots[0].is_leaf(), "_benches must not count as a child");
}

/// The ledger is grow-only on disk and DEDUPLICATED ON READ — `Event::identity`
/// excludes `at`, so a re-run collapses instead of accumulating. Three appends,
/// one of them a replay with a different timestamp, must read back as two.
#[test]
fn a_replayed_ledger_event_collapses_on_read() {
    let dir = super::super::tests::tempdir::Dir::new();
    let root = dir.path();
    let path = atlas_governance::ledger::path_for(root, 433);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let ev = |run: &str, at: u64, value: &str| atlas_governance::event::Event {
        pr: 433,
        head_sha: "abc1234567".into(),
        run_id: run.into(),
        attempt: 1,
        at,
        kind: atlas_governance::event::EventKind::Category {
            value: value.into(),
            status: "ok".into(),
        },
    };
    for e in [
        ev("31300000", 1_786_280_000, "infrastructure/benchmark-gate"),
        ev("31300001", 1_786_280_100, "tooling"),
        // Same identity as the first, different clock: a re-run of one job.
        ev("31300000", 1_786_288_888, "infrastructure/benchmark-gate"),
    ] {
        atlas_governance::ledger::append(&path, &e).unwrap();
    }
    let raw = atlas_governance::ledger::read_all(&path).unwrap();
    let deduped = raw.deduplicated();
    assert_eq!(
        deduped
            .events
            .iter()
            .map(|event| {
                let atlas_governance::event::EventKind::Category { value, status } = &event.kind
                else {
                    panic!("fixture contains only category events")
                };
                (
                    event.run_id.as_str(),
                    event.at,
                    value.as_str(),
                    status.as_str(),
                )
            })
            .collect::<Vec<_>>(),
        [
            (
                "31300000",
                1_786_280_000,
                "infrastructure/benchmark-gate",
                "ok",
            ),
            ("31300001", 1_786_280_100, "tooling", "ok"),
        ],
        "a replayed event must collapse; identity excludes `at` precisely so that \
         re-running a job does not inflate the journey, and the first observation survives"
    );
}

// ── matched depth: the stale-segment signal ────────────────────────────────

/// `performance` and `performance/decodes` imply the SAME set — the degrade
/// rule is right for a gate and invisible to a human. The matched depth is what
/// tells them apart, so a renamed category is reported rather than silently
/// costing coverage forever.
#[test]
fn matched_depth_distinguishes_a_stale_segment_from_a_real_leaf() {
    let roots = tree(
        r#"{ "performance": { "_benches": ["agentic-webserver"],
                              "decode": { "_benches": ["bfcl-subset"] },
                              "prefill": {} },
             "unknown": {} }"#,
    );

    let full = benches_for_matched(&roots, &["performance".into(), "decode".into()]);
    assert_eq!(full.1, 2, "both segments matched");

    let stale = benches_for_matched(&roots, &["performance".into(), "decodes".into()]);
    assert_eq!(stale.1, 1, "the typo must be reported as a partial match");
    assert_eq!(
        stale.0,
        benches_for(&roots, &["performance".into()]),
        "a stale tail still degrades to the matched prefix's union"
    );

    // A real leaf is NOT stale: `prefill` matches fully and simply declares
    // nothing. Conflating the two is what the jq walk did.
    let leaf = benches_for_matched(&roots, &["performance".into(), "prefill".into()]);
    assert_eq!(
        leaf.1, 2,
        "a benchless leaf is a full match, not a stale one"
    );

    // And the empty path matches nothing while being perfectly valid.
    assert_eq!(benches_for_matched(&roots, &[]).1, 0);
}

/// `benches_for` must stay exactly the first half of `benches_for_matched`, or
/// the codebase has two walks again — the bug this whole file exists to prevent.
#[test]
fn benches_for_is_the_first_half_of_benches_for_matched() {
    let roots = tree(
        r#"{ "a": { "_benches": ["bfcl-subset"], "x": { "_benches": ["ttft-warm-gate"] }, "y": {} },
             "b": {} }"#,
    );
    for path in [
        vec![],
        vec!["a".to_string()],
        vec!["a".to_string(), "x".into()],
        vec!["a".to_string(), "nope".into()],
        vec!["zzz".to_string()],
    ] {
        assert_eq!(
            benches_for(&roots, &path),
            benches_for_matched(&roots, &path).0,
            "the two walks disagreed on {path:?}"
        );
    }
}
