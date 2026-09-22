// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

/// ★ Every committed `[benchmarks.param_overrides]` pin in the REAL tree must
/// hold against its gate's actual schema: name a registered benchmark, name a
/// parameter that exists, parse through that parameter's own kind, and never
/// name a `threshold_params`-coupled key (whose value comes from the paired
/// metric's bound). A pin that fails any of these is discovered here in
/// milliseconds instead of at serve time on a gate run.
#[test]
fn every_committed_param_override_parses_against_its_gates_schema() {
    let root = repo_root();
    let mut observed = Vec::new();
    for (target, entry) in load_all(&root).expect("tree loads") {
        if entry.param_overrides.is_empty() {
            continue;
        }
        let descriptor = crate::registry::find(&entry.gate).unwrap_or_else(|| {
            panic!(
                "{}/{}: param_overrides on unregistered benchmark {:?}",
                target.hardware, target.model, entry.gate
            )
        });
        let specs = descriptor.build().parameters();
        for (key, raw) in &entry.param_overrides {
            observed.push((
                target.hardware.clone(),
                target.model.clone(),
                entry.gate.clone(),
                key.clone(),
                raw.clone(),
            ));
            assert!(
                !descriptor.threshold_params.iter().any(|(p, _)| p == key),
                "{}/{}/{}: pin {key:?} names a threshold-coupled param",
                target.hardware,
                target.model,
                entry.gate
            );
            let spec = specs
                .iter()
                .find(|s| s.key == key.as_str())
                .unwrap_or_else(|| {
                    panic!(
                        "{}/{}/{}: pin {key:?} names no schema parameter",
                        target.hardware, target.model, entry.gate
                    )
                });
            spec.kind.parse(raw).unwrap_or_else(|e| {
                panic!(
                    "{}/{}/{}: pin {key}={raw} does not parse: {e:#}",
                    target.hardware, target.model, entry.gate
                )
            });
        }
    }
    // `load_all` walks `kernels/<hw>/<model>` in sorted order, so the MoE's
    // pins (qwen3.6-35b-a3b) precede the dense 27B's (qwen3.8-27b).
    let moe = |key: &str, value: &str| {
        (
            "gb10".to_string(),
            "qwen3.6-35b-a3b".to_string(),
            "concurrency-sweep-moe".to_string(),
            key.to_string(),
            value.to_string(),
        )
    };
    assert_eq!(
        observed,
        vec![
            // The MoE ladder is pinned to the PUBLISHED instrument (bench/
            // baselines/qwen36-35b-a3b/published.json: ISL 128 / OSL 1024,
            // the harness's essay request, C=1..16) rather than the dense
            // gate's ISL 512 / OSL 320 natural fixture. It has no history on
            // any instrument, so nothing is lost by choosing the one the only
            // vLLM MoE one-shot was measured on; the two instruments' tok/s
            // are ~4x apart and must never share an axis.
            moe("concurrencies", "1,2,4,8,16"),
            moe("isls", "128"),
            moe("osl", "1024"),
            moe("prompt_mode", "essay"),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep".into(),
                "concurrencies".into(),
                "1,2,4,8,16,32,64,128".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep".into(),
                "isls".into(),
                "512".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep".into(),
                "osl".into(),
                "320".into(),
            ),
            // The DFlash2 gate's instrument is the plain one's TRUNCATED at
            // the widest rung its serve can admit — same isl, same osl, so
            // the rungs they share stay directly comparable. It stops at 16
            // because a DFlash2 serve refuses to start above a narrow batch
            // (measured; see the BENCH.toml note), and a rung above the batch
            // cap would measure the cap rather than the engine.
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep-dflash2".into(),
                "concurrencies".into(),
                "1,2,4,8,16".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep-dflash2".into(),
                "isls".into(),
                "512".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "concurrency-sweep-dflash2".into(),
                "osl".into(),
                // 200, not the plain gate's 320: at 320 this gate's C=1 cell
                // is deterministically vacuity-flagged (completion ~229 =
                // 71.5% of budget against an 80% floor), so no threshold makes
                // it certifiable. Lowering the budget below the natural stop
                // makes every finish a "length" finish — the comparability
                // property the vacuity rule protects.
                "200".into(),
            ),
            // kat-equality-gate, added with the 2026-09-10 promotion. All
            // three pins are load-bearing for a bound: `orders` and
            // `sample_cap` are what the `orders`/`samples` metric pins are
            // statements ABOUT, and `max_new_tokens` is BFCL's own budget —
            // this gate replays BFCL's request body, and shipped once at 512
            // against BFCL's 1024.
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "kat-equality-gate".into(),
                "max_new_tokens".into(),
                "1024".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "kat-equality-gate".into(),
                "orders".into(),
                "2".into(),
            ),
            (
                "gb10".into(),
                "qwen3.8-27b".into(),
                "kat-equality-gate".into(),
                "sample_cap".into(),
                // 257 is the end of live_parallel_multiple in the golden
                // draw's sorted concatenation, i.e. the smallest prefix
                // covering every subset in which order-dependence has been
                // observed. `truncate` selects a prefix, not a sample, so this
                // number decides WHICH subsets are compared.
                "257".into(),
            ),
        ],
        "the committed override validation must not pass vacuously or skip a pin"
    );
}

/// ★ The MoE concurrency entry, pinned BY VALUE. It is the only entry for its
/// gate id, it is that gate's declared subject, and it is UNMEASURED: no
/// `[benchmarks.metrics]` table, so `baseline_for` drops it and a
/// `--pull-request-gate` run refuses rather than passing a run nobody can
/// judge. The first measured floors (three fresh reps on this instrument,
/// mean - max(3*sigma, 5%) per rung, see the BENCH.toml note) flip `status`
/// and MUST replace the `metrics.is_none()` assertion below with the per-rung
/// values — that edit is the point: a floor that lands without touching a
/// test is a floor nobody reviewed.
///
/// The serve pins are the vLLM one-shot's parity profile as the manifest
/// records it (ctx 2048, batch cap 128, util 0.85, bf16 KV, MTP K=4, thinking
/// off, fifo scheduling), so a record from this entry DECLARES every axis
/// `site/src/lib/ladder-baselines.js` fingerprints and can be judged against
/// that one-shot axis by axis instead of being refused as "undeclared".
#[test]
fn the_moe_concurrency_entry_is_the_published_instrument_and_still_unmeasured() {
    use std::collections::BTreeMap;
    let root = repo_root();
    let all = load_all(&root).expect("tree loads");
    let moe: Vec<_> = all
        .iter()
        .filter(|(target, entry)| {
            target.hardware == "gb10"
                && target.model == "qwen3.6-35b-a3b"
                && entry.checkpoint == "Qwen/Qwen3.6-35B-A3B-FP8"
                && entry.gate.starts_with("concurrency-sweep")
        })
        .collect();
    assert_eq!(
        moe.iter().map(|(_, e)| e.gate.as_str()).collect::<Vec<_>>(),
        ["concurrency-sweep-moe"],
        "the MoE ladder has exactly one gate id — a second entry under \
         `concurrency-sweep` would split its records across two directories \
         and the site would file them under two subjects"
    );
    let (_, entry) = moe[0];
    assert!(
        entry.default,
        "the only checkpoint on this gate must declare itself its subject"
    );
    assert_eq!(entry.status, "unmeasured");
    assert!(
        entry.metrics.is_none(),
        "the MoE ladder has no committed measurement; a threshold here would be a guess. \
         When the first floors land, replace this assertion with their values."
    );
    assert_eq!(
        entry.recipe.as_deref(),
        Some("qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head")
    );
    let pins = |kv: &[(&str, &str)]| {
        kv.iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(
        entry.param_overrides,
        pins(&[
            ("concurrencies", "1,2,4,8,16"),
            ("isls", "128"),
            ("osl", "1024"),
            ("prompt_mode", "essay"),
        ])
    );
    assert_eq!(
        entry.serve_overrides,
        pins(&[
            ("disable_thinking", "true"),
            ("gpu_memory_utilization", "0.85"),
            ("kv_cache_dtype", "bf16"),
            ("max_batch_size", "128"),
            ("max_model_len", "2048"),
            ("num_drafts", "1"),
            ("scheduling_policy", "fifo"),
            ("ssm_cache_slots", "32"),
        ])
    );
    assert!(
        !entry.serve_overrides.contains_key("lm_head_dtype"),
        "the bf16 head is the recipe's correctness pin, not a throughput knob"
    );
}
