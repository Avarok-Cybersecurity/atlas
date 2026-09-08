// SPDX-License-Identifier: AGPL-3.0-only

//! Lever resolution tests: the polarity of every switch, the presence-vs-truth
//! wart the graph kill switch inherits, and the guard that keeps raw
//! environment reads off the per-decode-step path.

use super::*;
use std::collections::HashMap;

/// Resolve against a fixed map instead of the process environment.
///
/// `set_var` is unsafe and process-global, so a test that mutated the
/// environment would race every other test in this binary. Driving
/// `from_values` directly exercises the PRODUCTION resolution, not a copy of
/// it — `from_env` is nothing but this with two closures over `std::env`.
fn resolve(vars: &[(&str, &str)]) -> DFlashLevers {
    let map: HashMap<&str, &str> = vars.iter().copied().collect();
    from_values(
        |var| map.get(var).map(|v| (*v).to_string()),
        |var| map.contains_key(var),
    )
}

#[test]
fn nothing_set_resolves_to_defaults() {
    assert_eq!(resolve(&[]), DFlashLevers::defaults());
}

#[test]
fn defaults_are_spelled_out_not_derived() {
    let d = DFlashLevers::defaults();
    // The two fields whose shipped value is not `T::default()`. If either
    // regresses to the derived zero, graph capture warms up zero times and
    // row 0 loses its anchor bias — both silent.
    assert_eq!(d.propose_warmup_n, 2);
    assert!(d.dspark_anchor_bias);
    assert!(!d.any_diagnostic_armed);
}

#[test]
fn every_opt_in_is_off_until_it_is_exactly_one() {
    // `=1` arms; any other spelling does not. Pinned per field because a
    // single shared helper would not catch one field wired to the wrong
    // variable name — which is the failure this table exists to find.
    let cases: [(&str, fn(&DFlashLevers) -> bool); 8] = [
        ("ATLAS_DFLASH_DEBUG_DUMP", |l| l.debug_dump),
        ("ATLAS_DFLASH_DEBUG_DUMP_FULL", |l| l.debug_dump_full),
        ("ATLAS_DFLASH_LOG_DRAFTS", |l| l.log_drafts),
        ("ATLAS_DFLASH_BLOCK_DUMP", |l| l.block_dump),
        ("ATLAS_DFLASH_OPTION_B_DIAG", |l| l.option_b_diag),
        ("ATLAS_DFLASH_DEBUG_FORCE_PATTERN", |l| l.force_pattern),
        ("ATLAS_DFLASH_PRECOMPUTE", |l| l.precompute),
        ("ATLAS_DSPARK_CONF_TRACE", |l| l.dspark_conf_trace),
    ];
    for (var, read) in cases {
        assert!(!read(&resolve(&[])), "{var} armed with nothing set");
        assert!(read(&resolve(&[(var, "1")])), "{var} did not arm at =1");
        assert!(!read(&resolve(&[(var, "0")])), "{var} armed at =0");
        assert!(
            !read(&resolve(&[(var, "true")])),
            "{var} armed at =true; these levers are strict `1`"
        );
    }
}

#[test]
fn the_anchor_bias_is_the_one_opt_out() {
    assert!(resolve(&[]).dspark_anchor_bias);
    assert!(resolve(&[("ATLAS_DSPARK_ANCHOR_BIAS", "1")]).dspark_anchor_bias);
    assert!(!resolve(&[("ATLAS_DSPARK_ANCHOR_BIAS", "0")]).dspark_anchor_bias);
}

#[test]
fn dspark_shift_defers_to_the_checkpoint_unless_spelled() {
    assert_eq!(resolve(&[]).dspark_shift, None);
    assert_eq!(
        resolve(&[("ATLAS_DSPARK_SHIFT", "1")]).dspark_shift,
        Some(true)
    );
    assert_eq!(
        resolve(&[("ATLAS_DSPARK_SHIFT", "0")]).dspark_shift,
        Some(false)
    );
    // Anything else is not an override — the drafter config still decides.
    assert_eq!(resolve(&[("ATLAS_DSPARK_SHIFT", "yes")]).dspark_shift, None);
}

#[test]
fn numeric_levers_fall_back_when_unparseable() {
    assert_eq!(
        resolve(&[("ATLAS_DFLASH_PROPOSE_WARMUP_N", "5")]).propose_warmup_n,
        5
    );
    assert_eq!(
        resolve(&[("ATLAS_DFLASH_PROPOSE_WARMUP_N", "x")]).propose_warmup_n,
        2
    );
    assert_eq!(
        resolve(&[("ATLAS_DFLASH_BLOCK_DUMP_AT_POS", "64")]).block_dump_at_pos,
        64
    );
    assert_eq!(
        resolve(&[("ATLAS_DFLASH_DEBUG_CTX_USED", "7")]).force_ctx_used,
        Some(7)
    );
    assert_eq!(
        resolve(&[("ATLAS_DFLASH_DEBUG_CTX_USED", "-1")]).force_ctx_used,
        None
    );
}

/// ★ The wart, pinned deliberately.
///
/// The chain this replaced tested `std::env::var(..).is_err()`, so a
/// diagnostic variable set to ANY value — including `0` — suppresses CUDA
/// graph capture even though it enables no diagnostic. Preserved because a
/// graph capture that appears only when a flag is spelled a particular way is
/// a worse surprise for an operator than an over-eager kill switch, and
/// because changing it would change a measured perf path under cover of a
/// refactor.
#[test]
fn a_diagnostic_set_to_zero_still_suppresses_graphs() {
    for var in GRAPH_SUPPRESSING_DIAGNOSTICS {
        assert!(
            resolve(&[(var, "0")]).any_diagnostic_armed,
            "{var}=0 must still force the eager path"
        );
        assert!(
            resolve(&[(var, "")]).any_diagnostic_armed,
            "{var}= (empty) must still force the eager path"
        );
    }
    // …and nothing else does. An unrelated DFlash variable must not cost the
    // graphs: `ATLAS_DFLASH2=0` and `ATLAS_DFLASH_OPTION_B=0` are path
    // selectors, not diagnostics.
    assert!(!resolve(&[("ATLAS_DFLASH2", "0")]).any_diagnostic_armed);
    assert!(!resolve(&[("ATLAS_DFLASH_OPTION_B", "0")]).any_diagnostic_armed);
    assert!(!resolve(&[("ATLAS_DFLASH_PROPOSE_WARMUP_N", "4")]).any_diagnostic_armed);
}

#[test]
fn the_block_dump_arms_only_at_or_past_its_position() {
    let armed = resolve(&[
        ("ATLAS_DFLASH_BLOCK_DUMP", "1"),
        ("ATLAS_DFLASH_BLOCK_DUMP_AT_POS", "64"),
    ]);
    assert!(!armed.block_dump_armed_at(63));
    assert!(armed.block_dump_armed_at(64));
    assert!(armed.block_dump_armed_at(65));
    // Position alone never arms it.
    let off = resolve(&[("ATLAS_DFLASH_BLOCK_DUMP_AT_POS", "0")]);
    assert!(!off.block_dump_armed_at(1_000_000));
}

/// ★ THE ENVIRONMENT IS READ ONCE PER HEAD. This test is the enforcement.
///
/// `forward_block` runs once per decode step and its layer helpers run
/// `num_layers` times inside that, so a raw `std::env::var` there is an
/// allocation plus the process-wide environment lock on the hottest path the
/// drafter has — and the lock cost GROWS with concurrency (0.57 us at one
/// thread, 4.00 us at eight), which hides it from every single-stream
/// benchmark. The chain this module replaced read eleven variables to answer
/// one question.
///
/// A source-level check because the property is "who may read the
/// environment", which no runtime assertion can observe.
#[test]
fn dflash_levers_are_resolved_once() {
    // The per-decode-step path. `from_weights.rs` is deliberately absent: it
    // builds the head, so it is where the reads belong.
    const HOT: [&str; 3] = [
        "forward_block.rs",
        "forward_block_layer.rs",
        "forward_block_layer_paged.rs",
    ];
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/layers/dflash_head");
    let mut offenders = Vec::new();
    for file in HOT {
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} is the path this guard protects: {e}", path.display()));
        // Comments are stripped first: these files DOCUMENT the variables
        // they no longer read, and a guard that cannot tell a call from a
        // sentence about a call would be satisfied by rewording the comment.
        // Count, not just detect: the message should say how far it slipped.
        let n = text
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .map(|code| code.matches("std::env::var").count())
            .sum::<usize>();
        if n > 0 {
            offenders.push(format!("{file} ({n})"));
        }
    }
    assert!(
        offenders.is_empty(),
        "raw environment reads on the per-decode-step DFlash path — every one \
         allocates and takes the process-wide env lock. Use the resolved \
         `self.levers` (`DFlashLevers`) instead: {offenders:?}"
    );
}
