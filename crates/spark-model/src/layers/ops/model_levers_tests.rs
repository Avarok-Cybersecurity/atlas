// SPDX-License-Identifier: AGPL-3.0-only

//! Lever resolution tests: polarity of every switch, and the guard that
//! keeps `from_env` off hot paths.
//!
//! Split out of `model_levers.rs` to keep it under the repository's
//! 500-LoC cap, the same pattern `mtp_carry_tests.rs` uses.

use super::*;
use std::collections::HashMap;

/// ★ THE ENVIRONMENT IS READ ONCE. This test is the enforcement; the
/// module doc is only the explanation.
///
/// `from_env()` reads ~30 variables, each allocating a `String` and each
/// taking the process-wide environment lock — 0.57 us per resolve
/// single-threaded, 4.00 us at 8 threads, because the lock serialises. It
/// was called 32,513 times in one `concurrency-sweep` from a
/// per-layer-per-prefill site while its doc claimed "called once".
///
/// Only two callers are legitimate: `ModelLevers::get`, which caches it in
/// a `OnceLock`, and the model build, which needs an owned mutable copy to
/// overwrite `max_decode_seqs`. Everything else must use `get()` or take
/// `levers` from the context it already has.
///
/// A source-level check because the property is "who may call this", which
/// no runtime assertion can observe.
#[test]
fn from_env_is_called_only_where_it_is_allowed() {
    const ALLOWED: [&str; 3] = [
        // caches the result in a OnceLock — this IS the once.
        "crates/spark-model/src/layers/ops/model_levers.rs",
        // needs an owned mutable copy; takes it from `*get()`.
        "crates/spark-model/src/model/impl_a1.rs",
        // This file. The guard names what it forbids, so it matches itself —
        // it flagged its own new home the instant these tests were split out
        // of `model_levers.rs`, which is the guard working, not a false
        // positive.
        "crates/spark-model/src/layers/ops/model_levers_tests.rs",
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let mut offenders = Vec::new();
    let mut stack = vec![root.join("crates")];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|n| n == "target") {
                    continue;
                }
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                if !text.contains("ModelLevers::from_env()") {
                    continue;
                }
                let rel = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .replace('\\', "/");
                if !ALLOWED.contains(&rel.as_str()) {
                    offenders.push(rel);
                }
            }
        }
    }
    offenders.sort();
    assert!(
        offenders.is_empty(),
        "ModelLevers::from_env() re-reads ~30 env vars under a global lock. \
         These call it instead of the once-resolved ModelLevers::get(): {offenders:?}"
    );
}

fn resolve(values: &[(&str, &str)]) -> ModelLevers {
    let values: HashMap<_, _> = values.iter().copied().collect();
    from_values(
        |name| values.get(name).map(|value| (*value).to_owned()),
        |name| values.contains_key(name),
        0,
        crate::model::drafter_context::DrafterContext::BOTH,
    )
}

#[test]
fn the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off() {
    let d = ModelLevers::defaults();
    assert_eq!(
        resolve(&[]),
        d,
        "absent environment uses the public default"
    );
    assert_eq!(
        d,
        ModelLevers {
            gdn_regresident: true,
            gdn_wy17: true,
            gdn_wyn: true,
            gemv_sw: true,
            ffn_small_m: true,
            // The SSM batch-4 GEMV tier is the sixth opt-out lever. This
            // literal is spelled out rather than derived so that adding a
            // lever forces an author to state its polarity HERE, in the
            // test, instead of inheriting whatever `Default` gives.
            ssm_gemv_batch4: true,
            max_decode_seqs: 1,
            drafter: crate::model::drafter_context::DrafterContext::BOTH,
            ..ModelLevers::default()
        }
    );
}

#[test]
fn exact_one_opt_ins_map_to_their_own_fields() {
    let cases = [
        ("ATLAS_KV_POISON", [true, false, false, false, false, false]),
        (
            "ATLAS_GDN_BATCHED_FLA",
            [false, true, false, false, false, false],
        ),
        (
            "ATLAS_DECODE_FFN_VIA_GEMM",
            [false, false, true, false, false, false],
        ),
        (
            "ATLAS_MOE_UNION_STATS",
            [false, false, false, true, false, false],
        ),
        (
            "ATLAS_DFLASH_CONTIG_ATTN",
            [false, false, false, false, true, false],
        ),
        ("ATLAS_K4_DIAG", [false, false, false, false, false, true]),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "1")]);
        assert_eq!(
            [
                d.kv_poison,
                d.gdn_batched_fla,
                d.decode_ffn_via_gemm,
                d.moe_union_stats,
                d.dflash_contig_attn,
                d.k4_diag
            ],
            expected,
            "{name}"
        );
    }
    assert!(!resolve(&[("ATLAS_K4_DIAG", "true")]).k4_diag);
}

#[test]
fn truthy_opt_ins_map_independently_and_presence_is_distinct() {
    let cases = [
        (
            "ATLAS_HOLO_MOE_DOWN_FP4",
            [true, false, false, false, false],
        ),
        (
            "ATLAS_HOLO_MOE_GATEUP_FP4",
            [false, true, false, false, false],
        ),
        ("ATLAS_LORA_EAGER", [false, false, true, false, false]),
        ("ATLAS_LORA_ROTATE", [false, false, false, true, false]),
        ("ATLAS_DIAG_GEMMA4", [false, false, false, false, true]),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "TrUe")]);
        assert_eq!(
            [
                d.holo_moe_down_fp4,
                d.holo_moe_gateup_fp4,
                d.lora_eager,
                d.lora_rotate,
                d.gemma4_diag
            ],
            expected,
            "{name}"
        );
    }
    assert!(resolve(&[("ATLAS_BF16_TC_PROJ", "0")]).bf16_tc_proj);
    // `TQ_PLUS_WEIGHT_ROTATION` is VALUE-gated, not presence-gated — the
    // opposite of the line above. All five former implementations agreed
    // on `=1`-or-`true`, and this pins that the consolidation kept it.
    assert!(!resolve(&[]).weight_pre_rotated);
    assert!(resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "1")]).weight_pre_rotated);
    assert!(resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "TRUE")]).weight_pre_rotated);
    assert!(!resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "0")]).weight_pre_rotated);

    // ★ THE SSM DECODE FIVE, AND THEIR POLARITIES DIFFER. Three are opt-in
    // diagnostics, one ships ON and opts out with `=0`, and the fifth
    // stores the POSITIVE of a variable whose call site reads the negative.
    // Getting any of these backwards silently changes which kernel runs on
    // the decode path, so the defaults are pinned explicitly.
    let d = resolve(&[]);
    assert!(!d.ssm_ms_profile, "profiling is off unless asked for");
    assert!(!d.ssm_detail_profile);
    assert!(!d.gdn_fused_conv);
    assert!(!d.moe_legacy_pertoken_decode, "default is token-major MoE");
    assert!(d.ssm_gemv_batch4, "batch-4 GEMV ships ON");

    assert!(resolve(&[("ATLAS_SSM_MS_PROFILE", "1")]).ssm_ms_profile);
    assert!(resolve(&[("ATLAS_SSM_DETAIL_PROFILE", "1")]).ssm_detail_profile);
    assert!(resolve(&[("ATLAS_GDN_FUSED_CONV", "1")]).gdn_fused_conv);
    assert!(resolve(&[("ATLAS_MOE_LEGACY_PERTOKEN_DECODE", "1")]).moe_legacy_pertoken_decode);
    assert!(!resolve(&[("ATLAS_SSM_GEMV_BATCH4", "0")]).ssm_gemv_batch4);
    // `=0` on an opt-in is NOT enabling — the trap `ATLAS_BF16_TC_PROJ`
    // falls into by being presence-gated.
    assert!(!resolve(&[("ATLAS_GDN_FUSED_CONV", "0")]).gdn_fused_conv);
}

#[test]
fn kill_switches_and_zero_opt_outs_keep_their_distinct_polarities() {
    let d = resolve(&[
        ("ATLAS_NO_GDN_REGRESIDENT", "1"),
        ("ATLAS_NO_GEMV_SW", "1"),
        ("ATLAS_GDN_WY17", "0"),
        ("ATLAS_GDN_WYN", "0"),
        ("ATLAS_FFN_SMALLM", "0"),
    ]);
    assert!(!d.gdn_regresident);
    assert!(!d.gemv_sw);
    assert!(!d.gdn_wy17);
    assert!(!d.gdn_wyn);
    assert!(!d.ffn_small_m);
    assert!(resolve(&[("ATLAS_NO_GDN_REGRESIDENT", "0")]).gdn_regresident);
    assert!(resolve(&[("ATLAS_GDN_WY17", "1")]).gdn_wy17);
}

#[test]
fn externally_resolved_shadow_and_drafter_values_are_carried() {
    let d = from_values(
        |_| None,
        |_| false,
        7,
        crate::model::drafter_context::DrafterContext::OFF,
    );
    assert_eq!(d.shadow_topk, 7);
    assert_eq!(
        d.drafter,
        crate::model::drafter_context::DrafterContext::OFF
    );
}
