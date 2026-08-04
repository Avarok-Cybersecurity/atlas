// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn most_keys_are_the_flag_with_dashes() {
    assert_eq!(
        flag_for("gpu_memory_utilization").as_deref(),
        Some("gpu-memory-utilization")
    );
    assert_eq!(flag_for("port").as_deref(), Some("port"));
}

#[test]
fn the_three_renames_are_applied() {
    // Each of these silently serves the wrong thing if dropped: `max_model_len`
    // and `tensor_parallel` are not fields at all, and the listen address is
    // `--bind`, so a pass-through would be rejected or ignored.
    assert_eq!(flag_for("max_model_len").as_deref(), Some("max-seq-len"));
    assert_eq!(flag_for("tensor_parallel").as_deref(), Some("tp-size"));
    assert_eq!(flag_for("host").as_deref(), Some("bind"));
}

#[test]
fn a_true_boolean_is_a_bare_flag_and_a_false_one_is_omitted() {
    assert_eq!(
        argv_for("enable_prefix_caching", "true"),
        Some(vec!["--enable-prefix-caching".to_string()])
    );
    // `--enable-prefix-caching false` is not accepted by a SetTrue flag, so a
    // recipe restating the default must not break the parse.
    assert_eq!(argv_for("enable_prefix_caching", "false"), None);
}

#[test]
fn a_flag_that_takes_an_explicit_bool_keeps_its_value() {
    // `--disable-tool-grammar` is Option<bool>; dropping `false` would change
    // behaviour rather than restate a default.
    assert_eq!(
        argv_for("disable_tool_grammar", "false"),
        Some(vec!["--disable-tool-grammar".into(), "false".into()])
    );
}

#[test]
fn values_pass_through_verbatim() {
    assert_eq!(
        argv_for("max_model_len", "65536"),
        Some(vec!["--max-seq-len".into(), "65536".into()])
    );
}
