// SPDX-License-Identifier: AGPL-3.0-only

//! The lever contract behind #1242: what is a lever, what a declaration
//! means, and what the harness may inherit (nothing it did not declare).

use super::*;

fn map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn a_lever_is_any_avarok_name_that_is_not_a_harness_variable() {
    assert!(is_lever("AVAROK_FP8_ROWWISE"));
    assert!(is_lever("AVAROK_MTP_K_LADDER"));
    assert!(is_lever("AVAROK_PREFILL_CODISPATCH"));
    // A read the server grows tomorrow is a lever by default.
    assert!(is_lever("AVAROK_SOMETHING_NEW"));
    for harness in HARNESS_VARS {
        assert!(!is_lever(harness), "{harness} is the box's, not a lever");
    }
    // Sibling namespaces and the legacy prefix are not levers HERE: the legacy
    // name is mirrored onto AVAROK_* at process start and counted there.
    assert!(!is_lever("ATLASCTL_CONFIG_DIR"));
    assert!(!is_lever("ATLAS_FP8_ROWWISE"));
    assert!(!is_lever("PATH"));
    assert!(!is_lever("CUDARC_CUDA_VERSION"));
}

#[test]
fn levers_filters_an_environment_and_keeps_only_what_the_server_reads() {
    let env = [
        ("PATH", "/usr/bin"),
        ("AVAROK_HOME", "/workspace/.atlas"),
        ("AVAROK_FP8_ROWWISE", "1"),
        ("CUDARC_CUDA_VERSION", "13000"),
        ("AVAROK_MTP_DCUT_RATIO", "1.0"),
    ];
    assert_eq!(
        levers(env),
        map(&[
            ("AVAROK_FP8_ROWWISE", "1"),
            ("AVAROK_MTP_DCUT_RATIO", "1.0")
        ])
    );
}

/// The fingerprint is a function of the SET: order-free, value-sensitive,
/// and unable to confuse `A=1 B=2` with any other spelling.
#[test]
fn the_fingerprint_separates_sets_that_differ_and_nothing_else() {
    let dense = map(&[
        ("AVAROK_FP8_ROWWISE", "1"),
        ("AVAROK_MTP_DCUT_RATIO", "1.0"),
        ("AVAROK_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
    ]);
    let same = map(&[
        ("AVAROK_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
        ("AVAROK_MTP_DCUT_RATIO", "1.0"),
        ("AVAROK_FP8_ROWWISE", "1"),
    ]);
    assert_eq!(fingerprint(&dense), fingerprint(&same));
    assert_eq!(fingerprint(&dense).len(), 64);
    let mut off = dense.clone();
    off.insert("AVAROK_FP8_ROWWISE".into(), "0".into());
    assert_ne!(fingerprint(&dense), fingerprint(&off), "a value moved");
    let mut fewer = dense.clone();
    fewer.remove("AVAROK_MTP_K_LADDER");
    assert_ne!(fingerprint(&dense), fingerprint(&fewer), "a lever dropped");
    assert_ne!(
        fingerprint(&BTreeMap::new()),
        fingerprint(&map(&[("AVAROK_X", "")])),
        "an empty value is not an absent lever"
    );
    assert_ne!(
        fingerprint(&map(&[("AVAROK_A", "1B=2")])),
        fingerprint(&map(&[("AVAROK_A", "1"), ("B", "2")])),
    );
}

#[test]
fn a_declaration_is_validated_and_a_legacy_spelling_is_renamed() {
    let ok = declared(
        "recipe q/x",
        &map(&[("AVAROK_FP8_ROWWISE", "1"), ("ATLAS_MTP_DCUT_RATIO", "1.0")]),
    )
    .unwrap();
    assert_eq!(
        ok,
        map(&[
            ("AVAROK_FP8_ROWWISE", "1"),
            ("AVAROK_MTP_DCUT_RATIO", "1.0")
        ]),
        "the legacy name means its current one, which is what the server reads"
    );
    // Both spellings at ONE value are one lever; at two values they are a
    // contradiction and refused by name.
    assert_eq!(
        declared(
            "recipe q/x",
            &map(&[("AVAROK_FP8_ROWWISE", "1"), ("ATLAS_FP8_ROWWISE", "1")])
        )
        .unwrap(),
        map(&[("AVAROK_FP8_ROWWISE", "1")])
    );
    let e = declared(
        "recipe q/x",
        &map(&[("AVAROK_FP8_ROWWISE", "1"), ("ATLAS_FP8_ROWWISE", "0")]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("AVAROK_FP8_ROWWISE twice"), "{e}");
    assert!(e.contains("recipe q/x"), "{e}");

    for (bad, why) in [
        (("CUDA_VISIBLE_DEVICES", "0"), "not an AVAROK_* serve lever"),
        (("AVAROK_HOME", "/tmp/x"), "harness variable"),
        (("ATLAS_HOME", "/tmp/x"), "harness variable"),
        (("AVAROK_FP8_ROWWISE", "  "), "empty value"),
    ] {
        let e = declared("recipe q/x", &map(&[bad]))
            .unwrap_err()
            .to_string();
        assert!(e.contains(why), "{bad:?}: {e}");
        assert!(e.contains(bad.0), "names the key: {e}");
    }
}

#[test]
fn the_gate_pin_wins_over_the_recipe_on_a_clash_and_adds_otherwise() {
    let merged = merge_declared(
        map(&[("AVAROK_A", "recipe"), ("AVAROK_B", "recipe")]),
        map(&[("AVAROK_B", "pin"), ("AVAROK_C", "pin")]),
    );
    assert_eq!(
        merged,
        map(&[
            ("AVAROK_A", "recipe"),
            ("AVAROK_B", "pin"),
            ("AVAROK_C", "pin")
        ])
    );
}

/// The #1242 case, verbatim: a node's bench.yaml exported the dense
/// concurrency levers, and `bfcl-subset`'s recipe declares none of them.
#[test]
fn an_undeclared_lever_in_the_harness_is_refused_by_name() {
    let node = map(&[
        ("AVAROK_FP8_ROWWISE", "1"),
        ("AVAROK_MTP_DCUT_RATIO", "1.0"),
        ("AVAROK_MTP_K_LADDER", "1:3,2:1,4:2,8:2,16:1"),
    ]);
    let e = reconcile(
        "recipe qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl",
        &BTreeMap::new(),
        &node,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("3 serve lever(s)"), "{e}");
    assert!(e.contains("AVAROK_FP8_ROWWISE=1"), "{e}");
    assert!(
        e.contains("AVAROK_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1"),
        "{e}"
    );
    assert!(
        e.contains("recipe qwen3.8/qwen3.8-27b-nvfp4-unsloth-bfcl"),
        "names the recipe: {e}"
    );
    assert!(e.contains("bench.yaml"), "names the remedy: {e}");
    // One stray lever beside a fully declared set is still refused.
    let declared = map(&[("AVAROK_FP8_ROWWISE", "1")]);
    let mut present = declared.clone();
    present.insert("AVAROK_PREFILL_CODISPATCH".into(), "1".into());
    let e = reconcile("recipe r", &declared, &present)
        .unwrap_err()
        .to_string();
    assert!(e.contains("1 serve lever(s)"), "{e}");
    assert!(e.contains("AVAROK_PREFILL_CODISPATCH=1"), "{e}");
    assert!(
        !e.contains("AVAROK_FP8_ROWWISE"),
        "the declared one is not blamed: {e}"
    );
}

/// A declared lever is APPLIED: absent from the harness it is what the
/// child must be given; present at the declared value there is nothing to
/// give; present at another value it is a contradiction, not a silent win.
#[test]
fn a_declared_lever_is_applied_and_a_contradicted_one_is_refused() {
    let declared = map(&[
        ("AVAROK_FP8_ROWWISE", "1"),
        ("AVAROK_MTP_DCUT_RATIO", "1.0"),
    ]);
    let fresh = reconcile("recipe r", &declared, &BTreeMap::new()).unwrap();
    assert_eq!(
        fresh.env, declared,
        "the server runs under the whole declaration"
    );
    assert_eq!(fresh.missing, declared, "and none of it was already there");

    let half = reconcile("recipe r", &declared, &map(&[("AVAROK_FP8_ROWWISE", "1")])).unwrap();
    assert_eq!(half.env, declared);
    assert_eq!(half.missing, map(&[("AVAROK_MTP_DCUT_RATIO", "1.0")]));

    let all = reconcile("recipe r", &declared, &declared).unwrap();
    assert!(all.missing.is_empty(), "nothing to apply, nothing refused");

    let e = reconcile(
        "recipe r",
        &declared,
        &map(&[
            ("AVAROK_FP8_ROWWISE", "0"),
            ("AVAROK_MTP_DCUT_RATIO", "1.0"),
        ]),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("contradicts recipe r"), "{e}");
    assert!(
        e.contains("AVAROK_FP8_ROWWISE: harness \"0\", recipe r declares \"1\""),
        "{e}"
    );
    assert!(
        !e.contains("AVAROK_MTP_DCUT_RATIO"),
        "the agreeing one is not blamed: {e}"
    );
}

/// A harness with no levers and a recipe declaring none is the common case
/// and must cost nothing: no refusal, nothing to apply, the empty digest.
#[test]
fn no_levers_anywhere_reconciles_to_nothing() {
    let r = reconcile("recipe r", &BTreeMap::new(), &BTreeMap::new()).unwrap();
    assert!(r.env.is_empty() && r.missing.is_empty());
    assert_eq!(fingerprint(&r.env), fingerprint(&BTreeMap::new()));
}
