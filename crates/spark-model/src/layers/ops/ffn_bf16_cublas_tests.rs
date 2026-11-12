// SPDX-License-Identifier: AGPL-3.0-only

//! The `ffn_bf16_cublas_prefill` row, resolved through the whole target table
//! (`target_defaults::resolve`) so the wiring is graded, not only the helper.

use super::*;
use crate::layers::ops::target_defaults::{TargetLevers, format_levers, resolve as resolve_all};
use avarok_kernels::TargetDefaults;

/// `kernels/ada/HARDWARE.toml` `[defaults]`: the baseline plus the one row.
const ADA: TargetDefaults = TargetDefaults {
    hw: "ada",
    lm_head_batchm_max: 8,
    ssm_batched_recurrent: false,
    gdn_prefill_tc: false,
    ssm_ba_gates_hopper: false,
    fp8_act_quant_hopper: false,
    decode_split_silu: true,
    attn_decode_splitk: "legacy",
    ffn_m16_tc: false,
    attn_m16_tc: false,
    lm_head_m16_tc: false,
    attn_ncol_gemv: false,
    ffn_gateup_fused: false,
    dense_fp8_from_bf16: false,
    ffn_bf16_cublas_prefill: true,
    gdn_batched_gemm: false,
    w8a8_prefill_max_m_widening: u32::MAX,
    w8a8_prefill_max_m_narrowing: u32::MAX,
};

/// Any target that does not declare the row (gb10, hopper, b200, b300).
const OTHER: TargetDefaults = TargetDefaults {
    hw: "gb10",
    ffn_bf16_cublas_prefill: false,
    ..ADA
};

fn levers(defaults: &TargetDefaults, env: &[(&str, &str)]) -> TargetLevers {
    resolve_all(defaults, |name| {
        env.iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| (*v).to_owned())
    })
}

fn row_of(defaults: &TargetDefaults, env: &[(&str, &str)]) -> (bool, Source) {
    let r = levers(defaults, env).ffn_bf16_cublas_prefill;
    (r.value, r.source)
}

#[test]
fn ada_takes_cublas_with_an_empty_environment() {
    assert_eq!(row_of(&ADA, &[]), (true, Source::Target));
    // And the layer's decision: no scope armed, the row decides.
    assert!(takes_cublas(false, row_of(&ADA, &[]).0));
}

#[test]
fn a_target_without_the_row_keeps_dense_gemm_tc() {
    assert_eq!(row_of(&OTHER, &[]), (false, Source::Target));
    assert!(!takes_cublas(false, row_of(&OTHER, &[]).0));
}

/// `AVAROK_CUBLAS_GEMM=off` (or `0`, `false`, empty) is a SET variable, so it
/// decides: the row is off on ada, and the row's own variable cannot re-arm it.
#[test]
fn a_set_cublas_gemm_of_off_wins_over_the_ada_declaration() {
    for off in ["off", "0", "false", ""] {
        assert_eq!(
            row_of(&ADA, &[("AVAROK_CUBLAS_GEMM", off)]),
            (false, Source::Env),
            "AVAROK_CUBLAS_GEMM={off:?}"
        );
        assert_eq!(
            row_of(&ADA, &[("AVAROK_CUBLAS_GEMM", off), (ENV, "1")]),
            (false, Source::Env),
            "AVAROK_CUBLAS_GEMM={off:?} must outrank {ENV}"
        );
    }
    // A scope that names other families only is also "set, without ffn".
    assert_eq!(
        row_of(&ADA, &[("AVAROK_CUBLAS_GEMM", "attn,ssm")]),
        (false, Source::Env)
    );
}

/// `AVAROK_CUBLAS_GEMM=ffn` (or `all`, `1`) decides ON everywhere, as before.
#[test]
fn a_set_cublas_gemm_of_ffn_wins_on_every_target() {
    for t in [&ADA, &OTHER] {
        for on in ["ffn", "all", "1", "ffn,attn"] {
            assert_eq!(
                row_of(t, &[("AVAROK_CUBLAS_GEMM", on), (ENV, "0")]),
                (true, Source::Env),
                "{}: AVAROK_CUBLAS_GEMM={on:?}",
                t.hw
            );
        }
    }
}

/// With `AVAROK_CUBLAS_GEMM` absent the row's own variable is the A/B, in both
/// directions, under the 2026-09-11 grammar.
#[test]
fn the_row_variable_overrides_the_declaration_in_both_directions() {
    for off in ["0", "false", "off", "no"] {
        assert_eq!(row_of(&ADA, &[(ENV, off)]), (false, Source::Env), "{off}");
    }
    assert_eq!(row_of(&OTHER, &[(ENV, "1")]), (true, Source::Env));
}

/// When `AVAROK_CUBLAS_GEMM` is set the row equals the scope's `ffn` bit, so
/// `takes_cublas` is exactly the pre-row decision for every spelling.
#[test]
fn a_set_cublas_gemm_reproduces_the_pre_row_decision_exactly() {
    for raw in ["off", "ffn", "attn", "all", "junk", "ffn,junk"] {
        let scope_ffn = parse_cublas_scope(Some(raw)).0.ffn;
        for t in [&ADA, &OTHER] {
            let row = row_of(t, &[("AVAROK_CUBLAS_GEMM", raw)]).0;
            assert_eq!(takes_cublas(scope_ffn, row), scope_ffn, "{raw:?}");
        }
    }
}

#[test]
fn the_pure_resolver_matches_the_table() {
    assert!(resolve(true, None, None).value);
    assert!(!resolve(false, None, None).value);
    assert!(!resolve(true, Some("off"), None).value);
    assert!(resolve(false, Some("ffn"), None).value);
}

#[test]
fn the_serve_line_names_the_row_and_marks_an_override() {
    let line = format_levers(&levers(&ADA, &[]));
    assert!(line.contains("ffn_bf16_cublas_prefill=on"), "{line}");
    assert!(!line.contains("ffn_bf16_cublas_prefill=on (env)"), "{line}");
    let line = format_levers(&levers(&ADA, &[("AVAROK_CUBLAS_GEMM", "off")]));
    assert!(line.contains("ffn_bf16_cublas_prefill=off (env)"), "{line}");
    let line = format_levers(&levers(&OTHER, &[]));
    assert!(line.contains("ffn_bf16_cublas_prefill=off"), "{line}");
}
