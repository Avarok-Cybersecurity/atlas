// SPDX-License-Identifier: AGPL-3.0-only

//! The snapshot has one job: let a downstream renderer build a command clap
//! will accept. Each test below is a way it could fail to do that while still
//! looking like a valid document.

use super::*;

fn flag<'a>(m: &'a Manifest, key: &str) -> &'a Flag {
    m.flags
        .iter()
        .find(|f| f.key == key)
        .unwrap_or_else(|| panic!("{key} is not in the manifest"))
}

/// The whole reason this exists. `--gdn-fused-norm` takes no value and
/// `--ssm-h-dtype` does; a renderer that guesses produces a command clap
/// refuses, on the serving machine, after the operator has reviewed it.
#[test]
fn presence_only_is_reported_per_flag_not_guessed() {
    let m = build();
    // A bare toggle: clap parses it as SetTrue.
    assert!(
        flag(&m, "video_allow_ffmpeg").presence_only,
        "video_allow_ffmpeg is a bare flag"
    );
    // A value flag that happens to be a bool: `--flag true` is required.
    assert!(
        !flag(&m, "gdn_fused_norm").presence_only,
        "gdn_fused_norm takes an explicit value"
    );
    // A value flag that is not a bool at all.
    assert!(!flag(&m, "ssm_h_dtype").presence_only);
    assert!(!flag(&m, "ssm_h_dtype").is_bool);
}

/// A closed value set must come from the module the validator enforces, or a
/// picker will offer something the launch refuses. This is the `fcfs` bug in
/// test form: atlas-recipes offered a policy the engine never accepted.
#[test]
fn enumerated_values_come_from_the_validator_not_from_prose() {
    let m = build();
    assert_eq!(
        flag(&m, "scheduling_policy").options,
        vec!["fifo".to_owned(), "slai".to_owned()]
    );
    assert!(
        flag(&m, "kv_cache_dtype").options.len() >= 16,
        "the KV catalog has 16 variants, the manifest offered {}",
        flag(&m, "kv_cache_dtype").options.len()
    );
    assert!(
        flag(&m, "lm_head_dtype")
            .options
            .contains(&"nvfp4".to_owned()),
        "lm_head_dtype must carry its closed set"
    );
}

/// The key is the recipe `defaults:` spelling, because that is what a
/// downstream table is indexed by.
/// The key is derived from the flag, so a recipe that uses a different
/// spelling needs the alias carried explicitly. `max_model_len` is vLLM's
/// spelling kept in shipping recipes; the flag is `--max-seq-len`. A consumer
/// cannot work that out, and dropping it means a generated table silently
/// stops claiming a key recipes actually set.
#[test]
fn recipe_aliases_travel_with_the_flag_that_owns_them() {
    let m = build();
    let seq = flag(&m, "max_seq_len");
    assert_eq!(seq.flag, "max-seq-len");
    assert!(
        seq.recipe_aliases.contains(&"max_model_len".to_owned()),
        "max_model_len must map to --max-seq-len, got {:?}",
        seq.recipe_aliases
    );

    let tp = flag(&m, "tp_size");
    assert!(tp.recipe_aliases.contains(&"tensor_parallel".to_owned()));

    // And a flag with no alias carries none, rather than an empty guess.
    assert!(flag(&m, "port").recipe_aliases.is_empty());
}

#[test]
fn flags_are_the_cli_spelling_and_keys_the_underscored_one() {
    let m = build();
    for f in &m.flags {
        assert!(!f.flag.contains('_'), "the flag is hyphenated: {}", f.flag);
        assert!(!f.flag.starts_with('-'), "no leading dashes: {}", f.flag);
        assert_eq!(f.key, f.flag.replace('-', "_"));
    }
}

/// clap's own `--help`/`--version` describe nothing a recipe can set, and a
/// positional has no long name to key on.
#[test]
fn clap_own_arguments_are_not_flags_a_recipe_can_set() {
    let m = build();
    for k in ["help", "version"] {
        assert!(
            !m.flags.iter().any(|f| f.key == k),
            "{k} must not appear as a settable flag"
        );
    }
}

/// A consumer pins the shape and refuses a document it cannot read. If this
/// number moves, every downstream generator has to be looked at.
#[test]
fn the_document_declares_its_shape_and_its_engine() {
    let m = build();
    assert_eq!(m.schema_version, SCHEMA_VERSION);
    assert!(!m.spark_version.is_empty());
    assert!(
        m.flags.len() > 90,
        "ServeArgs has ~100 settable flags; the manifest found {}",
        m.flags.len()
    );
}

/// Two flags sharing a key would make a downstream map silently lose one.
#[test]
fn every_key_is_unique() {
    let m = build();
    let mut keys: Vec<&str> = m.flags.iter().map(|f| f.key.as_str()).collect();
    keys.sort_unstable();
    let before = keys.len();
    keys.dedup();
    assert_eq!(before, keys.len(), "duplicate keys in the manifest");
}
