// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: Python `str.startswith` / `str.endswith` as chat-template
//! string methods.
//!
//! minijinja has no string methods of its own, only filters and tests, so a
//! template written for Python Jinja2 that calls `content.startswith('<think>')`
//! fails with `UnknownMethod` the first time the call is evaluated. The Sharp
//! chat template (`assets/chat_templates/qwen-sharp-v22.5.0.jinja`) calls both
//! methods on every assistant turn in history, so any multi-turn or tool-loop
//! request needs them. They are reached from the unknown-method callback in
//! `jinja_helpers.rs`, which only fires after minijinja has found no method of
//! that name: a template that already works is never routed through here.
//!
//! Python semantics kept: one argument that is either a string or a sequence
//! (a tuple in Python) of strings, in which case the result is true when any
//! member matches; the empty string always matches; anything else is a type
//! error, as in Python. The optional `start`/`end` arguments are not supported
//! and fail with an argument-count error rather than being ignored.

use minijinja::value::{ValueKind, from_args};
use minijinja::{Error, ErrorKind, Value};

/// Evaluate `subject.startswith(affix)` or `subject.endswith(affix)`.
///
/// `method` is `"startswith"` or `"endswith"`; the caller has already checked
/// that `subject` is a string.
pub(super) fn affix_method(subject: &Value, method: &str, args: &[Value]) -> Result<Value, Error> {
    let (affix,): (Value,) = from_args(args)?;
    let text = subject.as_str().unwrap_or_default();
    let at_start = method == "startswith";
    let test = |needle: &str| {
        if at_start {
            text.starts_with(needle)
        } else {
            text.ends_with(needle)
        }
    };
    if let Some(needle) = affix.as_str() {
        return Ok(Value::from(test(needle)));
    }
    if affix.kind() == ValueKind::Seq {
        let members = affix.try_iter()?;
        let mut matched = false;
        for member in members {
            let Some(needle) = member.as_str() else {
                return Err(type_error(method, "a tuple member"));
            };
            matched |= test(needle);
        }
        return Ok(Value::from(matched));
    }
    Err(type_error(method, "the argument"))
}

fn type_error(method: &str, what: &str) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("str.{method}: {what} must be a string or a sequence of strings"),
    )
}

#[cfg(test)]
mod tests {
    use super::super::{ToolJsonStyle, build_jinja_env_with};

    fn render(template: &str, subject: &str) -> Result<String, minijinja::Error> {
        let env = build_jinja_env_with(template, ToolJsonStyle::Compact).expect("compiles");
        env.get_template("chat")
            .expect("template registered")
            .render(minijinja::context! { x => subject })
    }

    #[test]
    fn startswith_and_endswith_follow_python() {
        let t = "{{ x.startswith('<think>') }} {{ x.endswith('</think>') }}";
        assert_eq!(render(t, "<think>a</think>").unwrap(), "true true");
        assert_eq!(render(t, "a</think>").unwrap(), "false true");
        assert_eq!(render(t, "<think>a").unwrap(), "true false");
        assert_eq!(render(t, "").unwrap(), "false false");
    }

    #[test]
    fn empty_affix_always_matches() {
        let t = "{{ x.startswith('') }} {{ x.endswith('') }}";
        assert_eq!(render(t, "abc").unwrap(), "true true");
        assert_eq!(render("{{ x.startswith('') }}", "").unwrap(), "true");
    }

    #[test]
    fn affix_longer_than_subject_is_false() {
        assert_eq!(
            render("{{ x.startswith('abcd') }}", "abc").unwrap(),
            "false"
        );
        assert_eq!(render("{{ x.endswith('abcd') }}", "abc").unwrap(), "false");
    }

    #[test]
    fn multibyte_text_is_compared_by_bytes_of_valid_utf8() {
        let t = "{{ x.startswith('⚠️') }} {{ x.endswith('é') }}";
        assert_eq!(render(t, "⚠️ warning é").unwrap(), "true true");
        assert_eq!(render(t, "warning").unwrap(), "false false");
    }

    #[test]
    fn a_tuple_of_prefixes_matches_when_any_member_does() {
        let t = "{{ x.startswith(('exception:', 'failed to ')) }}";
        assert_eq!(render(t, "failed to open").unwrap(), "true");
        assert_eq!(render(t, "exception: boom").unwrap(), "true");
        assert_eq!(render(t, "ok").unwrap(), "false");
        assert_eq!(
            render("{{ x.endswith(('.rs', '.py')) }}", "main.py").unwrap(),
            "true"
        );
    }

    #[test]
    fn works_in_conditions_and_in_boolean_chains() {
        let t = "{% if x.startswith('a') and x.endswith('c') %}yes{% else %}no{% endif %}";
        assert_eq!(render(t, "abc").unwrap(), "yes");
        assert_eq!(render(t, "abd").unwrap(), "no");
    }

    #[test]
    fn a_non_string_argument_is_a_type_error() {
        assert!(render("{{ x.startswith(1) }}", "abc").is_err());
        assert!(render("{{ x.endswith(['a', 2]) }}", "abc").is_err());
    }

    #[test]
    fn wrong_argument_counts_are_errors() {
        assert!(render("{{ x.startswith() }}", "abc").is_err());
        assert!(render("{{ x.startswith('a', 1) }}", "abc").is_err());
    }

    #[test]
    fn split_result_is_a_list_that_indexes_from_both_ends() {
        // Regression: the split filter yields a lazy iterable, on which
        // `[-1]` is undefined, so `content.split(end)[-1]` rendered empty and
        // dropped every assistant turn that carried an inline think block.
        let t = "{{ x.split('</think>')[0] }}|{{ x.split('</think>')[-1] }}|{{ x.split('</think>') | length }}";
        assert_eq!(render(t, "a</think>b").unwrap(), "a|b|2");
        assert_eq!(render(t, "ab").unwrap(), "ab|ab|1");
        let var = "{% set sep = '\\n</think>' %}{{ x.split(sep)[-1] | trim }}";
        assert_eq!(render(var, "r\n</think>\n\nans").unwrap(), "ans");
    }

    #[test]
    fn unknown_methods_still_fail_as_before() {
        assert!(render("{{ x.casefold() }}", "abc").is_err());
    }

    #[test]
    fn rewritten_literal_spellings_still_use_the_test_form() {
        // The text rewrite in `convert_python_jinja_to_minijinja` turns the
        // literal `.startswith('<tool_response>')` into the minijinja test
        // `is startingwith`; both spellings must agree.
        let converted = super::super::convert_python_jinja_to_minijinja(
            "{{ x.startswith('<tool_response>') and x.endswith('</tool_response>') }}",
        );
        assert!(converted.contains("is startingwith"));
        let probe = "<tool_response>r</tool_response>";
        assert_eq!(render(&converted, probe).unwrap(), "true");
        assert_eq!(
            render("{{ x.startswith('<tool_response>') }}", probe).unwrap(),
            "true"
        );
    }
}
