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
fn defaults_cover_every_layer_at_scale_one_in_project_mode() {
    let v = p(&[("refusal", "/m/v.gguf")]);
    let out = resolve(&v, &[], &[], &[], 48).unwrap();
    assert_eq!(out.len(), 1);
    let (name, spec) = &out[0];
    assert_eq!(name, "refusal");
    // Layer 0 never carries a direction, so the default range starts at 1.
    assert_eq!((spec.layer_start, spec.layer_end), (1, 47));
    assert_eq!(spec.scale, 1.0);
    assert_eq!(spec.mode, CvecMode::Project);
}

#[test]
fn modifiers_apply_to_the_named_vector() {
    let v = p(&[("a", "/m/a.gguf"), ("b", "/m/b.gguf")]);
    let out = resolve(
        &v,
        &p(&[("a", "4-44")]),
        &p(&[("b", "0.1")]),
        &p(&[("b", "add")]),
        48,
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
    let e = resolve(&v, &p(&[("refusl", "4-44")]), &[], &[], 48)
        .unwrap_err()
        .to_string();
    assert!(e.contains("refusl"), "{e}");
    assert!(e.contains("refusal"), "{e}");
}

#[test]
fn duplicate_names_are_rejected() {
    let v = p(&[("a", "/m/a.gguf"), ("a", "/m/b.gguf")]);
    let e = resolve(&v, &[], &[], &[], 48).unwrap_err().to_string();
    assert!(e.contains("twice"), "{e}");
}

#[test]
fn no_vectors_resolves_empty() {
    assert!(resolve(&[], &[], &[], &[], 48).unwrap().is_empty());
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
    let out = resolve(&v, &[], &[], &[], 48).unwrap();
    assert_eq!(
        out[0].1.path.to_string_lossy(),
        "/m/odd:name,v1.gguf",
        "path was mangled"
    );
}
