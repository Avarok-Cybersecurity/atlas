// SPDX-License-Identifier: AGPL-3.0-only

use super::disclosed_from;
use crate::recipe::Recipe;
use std::collections::BTreeMap;

fn recipe(defaults: &str) -> Recipe {
    let text = format!(
        "recipe_version: \"2\"\nmodel: org/model\ncontainer: avarok\nruntime: atlas\n\
         metadata:\n  updated: \"2026-08-28\"\ndefaults:\n{defaults}"
    );
    Recipe::parse("fam/stem", &text).expect("the fixture recipe parses")
}

fn disclosed(defaults: &str, overrides: &[(&str, &str)]) -> Vec<(String, String)> {
    let overrides: BTreeMap<String, String> = overrides
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let args = recipe(defaults)
        .serve_args(&overrides)
        .unwrap_or_else(|e| panic!("the fixture renders: {e:#}"));
    disclosed_from(&args).into_iter().collect()
}

fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

/// The record reads what the RENDERED command resolved — the recipe as
/// pinned, the operator's override on top, and `--hermetic` closing the gate
/// whatever the recipe said.
#[test]
fn the_disclosure_is_read_off_the_rendered_serve() {
    // `atlas-recipes#16`: the pin the record must now be able to prove.
    assert_eq!(
        disclosed("  speculative: \"true\"\n  mtp_gate: force\n", &[]),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // The pre-#16 recipe: speculation on, nothing pinned — and the record
    // says so by carrying no `mtp_gate` at all, not by inventing `auto`.
    assert_eq!(
        disclosed("  speculative: \"true\"\n", &[]),
        pairs(&[("speculative", "true")])
    );
    // An explicit `auto` is a resolution and is recorded as one.
    assert_eq!(
        disclosed("  speculative: \"true\"\n  mtp_gate: auto\n", &[]),
        pairs(&[("mtp_gate", "auto"), ("speculative", "true")])
    );
    // A `--serve-override mtp_gate=force` on an `auto` recipe wins, as it
    // does on the command line.
    assert_eq!(
        disclosed(
            "  speculative: \"true\"\n  mtp_gate: auto\n",
            &[("mtp_gate", "force")]
        ),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // `--hermetic` forces the gate where the recipe pinned nothing (beside a
    // contradicting `auto` the validator refuses the serve outright), and
    // the disclosure follows the server's resolution, not the raw flag.
    assert_eq!(
        disclosed("  speculative: \"true\"\n", &[("hermetic", "true")]),
        pairs(&[("mtp_gate", "force"), ("speculative", "true")])
    );
    // No speculation: `mtp_gate` is moot and the record shows why.
    assert_eq!(
        disclosed("  max_batch_size: \"8\"\n", &[]),
        pairs(&[("speculative", "false")])
    );
}

/// G22: `--prefill-codispatch` is a flag, so the record reads it off the
/// rendered serve — the concurrency ladder's `prefill_codispatch = "true"`
/// pin must disclose `true`, where `perf_env` used to say `0` for it.
#[test]
fn the_codispatch_flag_is_disclosed_off_the_rendered_serve() {
    assert_eq!(
        disclosed(
            "  max_batch_size: \"8\"\n",
            &[("prefill_codispatch", "true")]
        ),
        pairs(&[("prefill_codispatch", "true"), ("speculative", "false")])
    );
    assert_eq!(
        disclosed("  prefill_codispatch: \"false\"\n", &[]),
        pairs(&[("prefill_codispatch", "false"), ("speculative", "false")])
    );
}
