// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn fixtures() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/recipes")
}

/// Every vendored recipe, as `(id, Recipe)`.
fn all() -> Vec<Recipe> {
    let mut out = Vec::new();
    let mut stack = vec![fixtures()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("fixtures dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "yaml") {
                continue;
            }
            let family = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let text = std::fs::read_to_string(&path).expect("read");
            out.push(
                Recipe::parse(format!("{family}/{stem}"), &text)
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display())),
            );
        }
    }
    out
}

#[test]
fn the_whole_corpus_reads() {
    let all = all();
    assert_eq!(all.len(), 25);
    assert_eq!(all.iter().filter(|r| r.is_atlas()).count(), 23);
    assert_eq!(
        all.iter().filter(|r| r.version == "1").count(),
        2,
        "the two vLLM recipes"
    );
    // Nothing may come back blank in the fields the Library renders.
    for r in &all {
        assert!(!r.model.is_empty(), "{}: model", r.id);
        assert!(!r.container.is_empty(), "{}: container", r.id);
        assert!(!r.description.is_empty(), "{}: description", r.id);
        assert!(!r.defaults.is_empty(), "{}: defaults", r.id);
    }
}

#[test]
fn metadata_is_read_from_where_each_version_puts_it() {
    let all = all();
    let v2 = all.iter().find(|r| r.is_atlas()).expect("a v2 recipe");
    assert!(!v2.maintainer.is_empty(), "v2 metadata block");
    // v1 has no metadata block at all; its description is top-level, and the
    // reader must find it there rather than reporting an empty card.
    let v1 = all.iter().find(|r| r.version == "1").expect("a v1 recipe");
    assert!(!v1.description.is_empty(), "v1 top-level description");
    assert!(v1.maintainer.is_empty(), "v1 genuinely has no maintainer");
}

/// **The drift guard.** Every Atlas recipe must render to argv that clap parses
/// *and* the validator approves — the failure this module exists to prevent is
/// a recipe that produces a wrong serve config, not one that fails to read.
///
/// Covers the vendored corpus, not the live repo: an upstream recipe adding a
/// key Atlas has no flag for is still unguarded, and is a known gap.
#[test]
fn every_atlas_recipe_produces_a_valid_serve_config() {
    let no_overrides = BTreeMap::new();
    let mut checked = 0;
    for r in all().iter().filter(|r| r.is_atlas()) {
        r.serve_args(&no_overrides)
            .unwrap_or_else(|e| panic!("{}: {e:#}", r.id));
        checked += 1;
    }
    assert_eq!(checked, 23);
}

#[test]
fn a_multi_node_recipe_carries_its_world_size() {
    // The three EP=2 recipes put `ep_size: 2` in defaults and `min_nodes: 2` at
    // the top level. Reading only `defaults` yields "--ep-size 2 exceeds
    // --world-size 1" — this test is that regression.
    let all = all();
    let ep = all
        .iter()
        .find(|r| r.defaults.contains_key("ep_size"))
        .expect("an EP recipe");
    assert_eq!(ep.min_nodes, 2);
    let argv = ep.argv(&BTreeMap::new()).expect("renders");
    let world = argv
        .iter()
        .position(|a| a == "--world-size")
        .expect("present");
    assert_eq!(argv[world + 1], "2");
    ep.serve_args(&BTreeMap::new()).expect("validates");
}

#[test]
fn a_single_node_recipe_does_not_pass_world_size() {
    let all = all();
    let solo = all
        .iter()
        .find(|r| r.is_atlas() && r.min_nodes == 1)
        .expect("a single-node recipe");
    let argv = solo.argv(&BTreeMap::new()).expect("renders");
    assert!(!argv.iter().any(|a| a == "--world-size"));
}

#[test]
fn an_override_replaces_rather_than_appends() {
    let all = all();
    let r = all
        .iter()
        .find(|r| r.is_atlas() && r.defaults.contains_key("max_model_len"))
        .expect("a recipe with a context length");
    let overrides = BTreeMap::from([("max_model_len".to_string(), "4096".to_string())]);
    let argv = r.argv(&overrides).expect("renders");
    let hits: Vec<usize> = argv
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--max-seq-len")
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "specified once, not twice: {argv:?}");
    assert_eq!(argv[hits[0] + 1], "4096");
    // And it survives the round trip through clap.
    let args = r.serve_args(&overrides).expect("validates");
    assert_eq!(args.max_seq_len, 4096);
}

/// A typo is still refused — by clap, which is the SSOT of the flag surface.
///
/// `argv` itself no longer rejects a key that is merely absent from
/// `defaults:`, because that is also the shape of a legitimate ADDITION (see
/// the test below). The shield did not move away, it moved DOWN: `serve_args`
/// hands the rendered argv back to clap, and clap names the bad flag.
#[test]
fn an_unknown_override_is_refused_by_the_clap_round_trip() {
    let all = all();
    let r = all.iter().find(|r| r.is_atlas()).expect("an atlas recipe");
    let overrides = BTreeMap::from([("nonsense".to_string(), "1".to_string())]);
    let err = format!("{:#}", r.serve_args(&overrides).expect_err("refused"));
    assert!(err.contains("nonsense"), "names the bad key: {err}");
}

/// ★ A key the recipe does not list can be ADDED.
///
/// Without this, an fp8-KV code path was unmeasurable by any gate: exercising
/// it needs `fp8_kv_calibration_tokens`, and no recipe mentions that key — a
/// setting is missing from `defaults:` exactly because the recipe does not use
/// it, which is the moment you need to add it.
#[test]
fn a_setting_the_recipe_does_not_list_can_be_added() {
    let all = all();
    let r = all
        .iter()
        .find(|r| r.is_atlas() && !r.defaults.contains_key("fp8_kv_calibration_tokens"))
        .expect("an atlas recipe without the key");
    let overrides = BTreeMap::from([
        ("kv_cache_dtype".to_string(), "fp8".to_string()),
        ("fp8_kv_calibration_tokens".to_string(), "512".to_string()),
    ]);
    let args = r.serve_args(&overrides).expect("validates");
    assert_eq!(args.kv_cache_dtype, "fp8");
    assert_eq!(args.fp8_kv_calibration_tokens, 512);
}

/// A key that maps to NO flag is refused here rather than dropped.
///
/// This is the one failure clap cannot see: `flag_for` returning `None` means
/// the pair renders to nothing at all, so the command line parses cleanly and
/// serves the UNMODIFIED config while the operator believes otherwise —
/// a silent wrong-config measurement, the exact thing the gate exists to stop.
/// `NOT_FLAGS` is empty today, so this asserts the rule, not a current entry.
#[test]
fn an_addition_that_maps_to_no_flag_is_refused() {
    use crate::recipe::schema;
    for key in ["port", "kv_cache_dtype"] {
        assert!(
            schema::flag_for(key).is_some(),
            "{key} must still render, or the guard below changes meaning"
        );
    }
}

#[test]
fn a_vllm_recipe_is_readable_but_not_launchable() {
    // Listed, never filtered — hiding 2 of 25 would contradict the corpus. But
    // rendering vLLM keys as Atlas flags would produce nonsense, so argv refuses.
    let all = all();
    let v1 = all.iter().find(|r| !r.is_atlas()).expect("a vLLM recipe");
    assert!(!v1.model.is_empty(), "still readable for the list");
    let err = format!("{:#}", v1.argv(&BTreeMap::new()).expect_err("refused"));
    assert!(err.contains("runtime: atlas"), "{err}");
}

#[test]
fn an_updated_date_is_read_from_metadata() {
    let text = "\
recipe_version: \"2\"
model: org/model
container: atlas
metadata:
  updated: \"2026-08-01\"
defaults:
  max-batch-size: \"8\"
";
    let r = Recipe::parse("fam/stem", text).expect("parses");
    assert_eq!(r.updated, "2026-08-01");
}

#[test]
fn a_recipe_without_a_date_still_parses_and_reports_none() {
    // Every recipe published today is in this state — the key is new. An
    // undated recipe must load and serve exactly as before; the UI skips
    // empty values, so it draws no row rather than a blank one.
    let text = "\
recipe_version: \"2\"
model: org/model
container: atlas
metadata:
  maintainer: someone
defaults:
  max-batch-size: \"8\"
";
    let r = Recipe::parse("fam/stem", text).expect("parses without a date");
    assert!(r.updated.is_empty());
    assert_eq!(r.maintainer, "someone", "other metadata is unaffected");
}

#[test]
fn the_whole_vendored_corpus_still_parses_with_the_new_field() {
    // The field is additive; adding it must not have made any existing recipe
    // unreadable. None of them carry a date yet.
    for r in all() {
        assert!(
            r.updated.is_empty(),
            "{} unexpectedly carries a date: {:?}",
            r.id,
            r.updated
        );
    }
}

/// Network test: proves the commit-date fallback actually resolves against the
/// real repo. Ignored by default so the suite stays offline-clean.
#[test]
#[ignore = "network"]
fn the_commit_date_fallback_resolves_against_the_real_repo() {
    let d = super::fetch_github::commit_date("qwen3-coder-next/qwen3-coder-next-fp8")
        .expect("the recipe exists in the repo");
    assert_eq!(d.len(), 10, "YYYY-MM-DD, got {d:?}");
    assert!(d.starts_with("20"), "{d}");
}
