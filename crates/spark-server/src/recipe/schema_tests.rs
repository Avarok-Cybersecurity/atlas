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
fn a_presence_only_boolean_is_a_bare_flag_and_a_false_one_is_omitted() {
    // `--speculative` is a clap SetTrue flag: it takes no value, so `false`
    // can only be expressed by not passing it — which the shipped
    // qwen3.5-0.8b recipe (`speculative: false`) depends on.
    assert_eq!(
        argv_for("speculative", "true"),
        Some(vec!["--speculative".to_string()])
    );
    assert_eq!(argv_for("speculative", "false"), None);
    // `enable_prefix_caching` is `bool` (not `Option<bool>`), so the derive
    // gives it SetTrue despite its `num_args = 0..=1` — clap, not the field
    // syntax, is the authority the presence-only set is read from.
    assert_eq!(argv_for("enable_prefix_caching", "false"), None);
    // `video_allow_ffmpeg` is the key the video-fidelity gate recipes pin
    // (2026-08-15: the certified self-start 400'd every MP4 leg without it).
    // It must render as the bare flag, or the recipe's serve fails to parse
    // and the gate never comes up with the subprocess decoder enabled.
    assert_eq!(
        argv_for("video_allow_ffmpeg", "true"),
        Some(vec!["--video-allow-ffmpeg".to_string()])
    );
}

#[test]
fn a_flag_that_takes_an_explicit_bool_keeps_its_value() {
    // `--disable-tool-grammar` is Option<bool>; dropping `false` would change
    // behaviour rather than restate a default.
    assert_eq!(
        argv_for("disable_tool_grammar", "false"),
        Some(vec!["--disable-tool-grammar".into(), "false".into()])
    );
    // The Option<bool> levers are the case the old hand-kept exception list
    // missed: absent leaves the legacy env fallback live, explicit false
    // seals it, so rendering `false` as NOTHING changed the config.
    assert_eq!(
        argv_for("gdn_fused_norm", "false"),
        Some(vec!["--gdn-fused-norm".into(), "false".into()])
    );
    // `--ssm-tail-midchunk` REQUIRES a value (ArgAction::Set, one arg), so
    // `true` must be passed through too — a bare flag fails its parse.
    assert_eq!(
        argv_for("ssm_tail_midchunk", "true"),
        Some(vec!["--ssm-tail-midchunk".into(), "true".into()])
    );
}

#[test]
fn values_pass_through_verbatim() {
    assert_eq!(
        argv_for("max_model_len", "65536"),
        Some(vec!["--max-seq-len".into(), "65536".into()])
    );
}
