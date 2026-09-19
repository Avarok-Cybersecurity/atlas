// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn int_domain_is_enforced_with_the_bound_in_the_message() {
    let kind = ParamKind::Int { min: 1, max: 64 };
    assert_eq!(kind.parse("16").unwrap(), ParamValue::Int(16));
    let err = kind.parse("0").unwrap_err().to_string();
    assert!(err.contains("between 1 and 64"), "{err}");
    assert!(kind.parse("65").is_err(), "the upper bound is inclusive");
    assert!(kind.parse("nope").is_err());
}

#[test]
fn int_list_round_trips_and_rejects_empty() {
    let kind = ParamKind::IntList { min: 1, max: 128 };
    let v = kind.parse(" 1, 2,4 , 8,16 ").unwrap();
    assert_eq!(v, ParamValue::IntList(vec![1, 2, 4, 8, 16]));
    assert_eq!(v.to_edit_string(), "1, 2, 4, 8, 16");
    assert_eq!(kind.parse(&v.to_edit_string()).unwrap(), v);
    assert!(kind.parse("  ").is_err());
    assert!(kind.parse("1,999").is_err());
    assert!(
        kind.parse("0,1").is_err(),
        "every entry respects the minimum"
    );
}

#[test]
fn bool_aliases_canonicalize_and_invalid_text_is_rejected() {
    let kind = ParamKind::Bool;
    assert_eq!(kind.parse(" YES ").unwrap(), ParamValue::Bool(true));
    assert_eq!(kind.parse("0").unwrap(), ParamValue::Bool(false));
    assert_eq!(
        kind.parse("maybe").unwrap_err().to_string(),
        "expected true or false, got \"maybe\""
    );
}

#[test]
fn float_domain_is_finite_and_inclusive() {
    let kind = ParamKind::Float { min: 0.0, max: 1.0 };
    assert_eq!(kind.parse("0").unwrap(), ParamValue::Float(0.0));
    assert_eq!(kind.parse("1").unwrap(), ParamValue::Float(1.0));
    assert!(kind.parse("-0.01").is_err());
    assert!(kind.parse("1.01").is_err());
    for non_finite in ["NaN", "inf", "-inf"] {
        assert!(kind.parse(non_finite).is_err(), "accepted {non_finite}");
    }
}

#[test]
fn free_text_is_trimmed_and_cannot_be_empty() {
    let kind = ParamKind::Text;
    assert_eq!(
        kind.parse("  model/name  ").unwrap(),
        ParamValue::Text("model/name".into())
    );
    assert_eq!(
        kind.parse(" \t ").unwrap_err().to_string(),
        "must not be empty"
    );
}

#[test]
fn choice_is_case_insensitive_and_lists_the_options_on_failure() {
    let kind = ParamKind::Choice(&["hello", "count"]);
    assert_eq!(
        kind.parse("COUNT").unwrap(),
        ParamValue::Text("count".into())
    );
    let err = kind.parse("other").unwrap_err().to_string();
    assert!(err.contains("hello, count"), "{err}");
}

#[test]
fn defaults_come_from_the_schema_only() {
    let specs = vec![
        ParamSpec::new(
            "conc",
            "Concurrency",
            "levels to sweep",
            ParamKind::IntList { min: 1, max: 64 },
            ParamValue::IntList(vec![1, 2, 4, 8, 16]),
        ),
        ParamSpec::new(
            "osl",
            "Output tokens",
            "max tokens per request",
            ParamKind::Int { min: 1, max: 8192 },
            ParamValue::Int(128),
        ),
    ];
    let v = ParamValues::defaults(&specs);
    assert_eq!(v.int_list("conc").unwrap(), &[1, 2, 4, 8, 16]);
    assert_eq!(v.usize("osl").unwrap(), 128);
    v.validate_against(&specs).unwrap();
}

#[test]
fn missing_key_is_an_error_not_a_silent_default() {
    let v = ParamValues::default();
    let err = v.int("osl").unwrap_err().to_string();
    assert!(err.contains("never set"), "{err}");
}

#[test]
fn mistyped_key_names_both_types() {
    let mut v = ParamValues::default();
    v.set("osl", ParamValue::Text("128".into()));
    let err = v.int("osl").unwrap_err().to_string();
    assert!(
        err.contains("is text") && err.contains("expected int"),
        "{err}"
    );
}

#[test]
fn validate_against_reports_the_field_label() {
    let specs = vec![ParamSpec::new(
        "osl",
        "Output tokens",
        "max tokens per request",
        ParamKind::Int { min: 1, max: 8192 },
        ParamValue::Int(128),
    )];
    let mut v = ParamValues::defaults(&specs);
    v.set("osl", ParamValue::Int(99_999));
    let err = v.validate_against(&specs).unwrap_err().to_string();
    assert!(err.starts_with("Output tokens:"), "{err}");
}

#[test]
fn validate_against_rejects_values_outside_the_schema() {
    let specs = vec![ParamSpec::new(
        "osl",
        "Output tokens",
        "max tokens per request",
        ParamKind::Int { min: 1, max: 8192 },
        ParamValue::Int(128),
    )];
    let mut v = ParamValues::defaults(&specs);
    v.set("os1", ParamValue::Int(8));
    assert_eq!(
        v.validate_against(&specs).unwrap_err().to_string(),
        "unknown parameter \"os1\" — this benchmark takes: osl"
    );
}
