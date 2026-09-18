// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `--control-vector*` resolution.

use super::*;

fn p(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect()
}

#[test]
fn an_unknown_model_requires_the_layer_range() {
    // Regression pin. This used to default to 1..n_layer-1, i.e. EVERY layer,
    // which is not the configuration any shipped vector was validated at —
    // `--control-vector refusal=f.gguf` alone silently ran 1..47 while the
    // docs described 4..44. With no curated range for the model, refusing is
    // the fail-closed choice and matches how this function already treats an
    // undeclared-vector modifier.
    let v = p(&[("refusal", "/m/v.gguf")]);
    let e = resolve(&v, &[], &[], &[], 48, "unknown")
        .unwrap_err()
        .to_string();
    assert!(e.contains("--control-vector-layers"), "{e}");
    assert!(e.contains("refusal"), "{e}");
}

#[test]
fn a_known_model_gets_its_curated_range() {
    // qwen4_exp is 4..44 because all three known vectors for it use that range,
    // which makes it a property of the model rather than of any one vector.
    let v = p(&[("refusal", "/m/v.gguf")]);
    let out = resolve(&v, &[], &[], &[], 48, "qwen4_exp").unwrap();
    assert_eq!((out[0].1.layer_start, out[0].1.layer_end), (4, 44));
}

#[test]
fn an_explicit_range_overrides_the_curated_one() {
    // The curated value is a convenience, not a policy: a future model, or a
    // vector characterised elsewhere, must still be able to say otherwise.
    let v = p(&[("refusal", "/m/v.gguf")]);
    let out = resolve(&v, &p(&[("refusal", "2-40")]), &[], &[], 48, "qwen4_exp").unwrap();
    assert_eq!((out[0].1.layer_start, out[0].1.layer_end), (2, 40));
}

#[test]
fn a_curated_range_past_the_end_of_a_smaller_checkpoint_is_refused() {
    // Same model_type, fewer layers than the table assumes. Silently clamping
    // would serve a different configuration than the one the table names, so
    // fall through to requiring the flag.
    let v = p(&[("refusal", "/m/v.gguf")]);
    let e = resolve(&v, &[], &[], &[], 20, "qwen4_exp")
        .unwrap_err()
        .to_string();
    assert!(e.contains("--control-vector-layers"), "{e}");
}

#[test]
fn scale_and_mode_default_when_the_range_is_given() {
    let v = p(&[("refusal", "/m/v.gguf")]);
    let out = resolve(&v, &p(&[("refusal", "4-44")]), &[], &[], 48, "unknown").unwrap();
    assert_eq!(out.len(), 1);
    let (name, spec) = &out[0];
    assert_eq!(name, "refusal");
    assert_eq!((spec.layer_start, spec.layer_end), (4, 44));
    assert_eq!(spec.scale, 1.0);
    assert_eq!(spec.mode, CvecMode::Project);
}

#[test]
fn modifiers_apply_to_the_named_vector() {
    let v = p(&[("a", "/m/a.gguf"), ("b", "/m/b.gguf")]);
    let out = resolve(
        &v,
        &p(&[("a", "4-44"), ("b", "1-47")]),
        &p(&[("b", "0.1")]),
        &p(&[("b", "add")]),
        48,
        "unknown",
    )
    .unwrap();
    let a = &out.iter().find(|(n, _)| n == "a").unwrap().1;
    let b = &out.iter().find(|(n, _)| n == "b").unwrap().1;
    assert_eq!((a.layer_start, a.layer_end), (4, 44));
    assert_eq!(a.scale, 1.0);
    assert_eq!(a.mode, CvecMode::Project);
    assert_eq!((b.layer_start, b.layer_end), (1, 47));
    assert_eq!(b.scale, 0.1);
    assert_eq!(b.mode, CvecMode::Add);
}

#[test]
fn a_modifier_naming_an_undeclared_vector_is_rejected() {
    // Silently ignoring it would serve the DEFAULT while the operator reads
    // their flag back off the command line and believes otherwise.
    let v = p(&[("refusal", "/m/v.gguf")]);
    let e = resolve(&v, &p(&[("refusl", "4-44")]), &[], &[], 48, "unknown")
        .unwrap_err()
        .to_string();
    assert!(e.contains("refusl"), "{e}");
    assert!(e.contains("refusal"), "{e}");
}

#[test]
fn duplicate_names_are_rejected() {
    let v = p(&[("a", "/m/a.gguf"), ("a", "/m/b.gguf")]);
    // The range is supplied so this test fails on the DUPLICATE and not on the
    // missing-range check, which would pass for the wrong reason.
    let e = resolve(&v, &p(&[("a", "4-44")]), &[], &[], 48, "unknown")
        .unwrap_err()
        .to_string();
    assert!(e.contains("twice"), "{e}");
}

#[test]
fn no_vectors_resolves_empty() {
    assert!(
        resolve(&[], &[], &[], &[], 48, "unknown")
            .unwrap()
            .is_empty()
    );
}

#[test]
fn range_parser_rejects_an_inverted_or_malformed_range() {
    assert!(parse_control_vector_layers("a=44-4").is_err());
    assert!(parse_control_vector_layers("a=4..44").is_err());
    assert!(parse_control_vector_layers("a=x-44").is_err());
    assert!(parse_control_vector_layers("a=4-44").is_ok());
}

#[test]
fn mode_parser_rejects_anything_but_project_or_add() {
    assert!(parse_control_vector_mode("a=project").is_ok());
    assert!(parse_control_vector_mode("a=add").is_ok());
    let e = parse_control_vector_mode("a=ablate").unwrap_err();
    assert!(e.contains("project"), "{e}");
}

#[test]
fn scale_parser_rejects_non_numbers_and_non_finite() {
    assert!(parse_control_vector_scale("a=1.0").is_ok());
    assert!(parse_control_vector_scale("a=-0.5").is_ok());
    assert!(parse_control_vector_scale("a=lots").is_err());
    assert!(parse_control_vector_scale("a=inf").is_err());
    assert!(parse_control_vector_scale("a=NaN").is_err());
}

#[test]
fn a_path_containing_a_colon_or_comma_survives() {
    // The reason the modifiers are separate flags rather than a packed
    // PATH:SCALE:A-B:MODE spec string.
    let v = p(&[("a", "/m/odd:name,v1.gguf")]);
    let out = resolve(&v, &p(&[("a", "4-44")]), &[], &[], 48, "unknown").unwrap();
    assert_eq!(
        out[0].1.path.to_string_lossy(),
        "/m/odd:name,v1.gguf",
        "path was mangled"
    );
}
