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
    assert_eq!(
        observed,
        vec![
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
                "320".into(),
            ),
        ],
        "the committed override validation must not pass vacuously or skip a pin"
    );
}
