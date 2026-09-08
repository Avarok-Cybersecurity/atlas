// SPDX-License-Identifier: AGPL-3.0-only

//! ★ WHERE THE ENVIRONMENT MAY BE READ. One table, one enforcement.
//!
//! `std::env::var` allocates a `String` and takes the PROCESS-WIDE
//! environment lock; `var_os` skips the allocation but still takes the lock.
//! So concurrent decode threads SERIALISE against each other on every read,
//! and the cost grows with concurrency — MEASURED on GB10, one resolve of ~30
//! variables costs 0.57 us single-threaded, **4.00 us at 8 threads and
//! 5.76 us at 16**. That growth is precisely why no single-stream benchmark
//! ever showed it, and why `ModelLevers::from_env` was called **32,513 times
//! in one `concurrency-sweep`** while its own doc claimed "called once".
//!
//! These are source-level checks because the property is "who may read the
//! environment", which no runtime assertion can observe.
//!
//! To add a file here, do NOT add its hot function to the allow list — move
//! the variable into [`super::ModelLevers`] (or the module's own levers
//! struct) and read the resolved field instead.

/// Files on a per-token, per-layer or per-forward path, each with the
/// functions in it that are still ALLOWED to read the environment.
///
/// An empty allow list means the file must not read the environment at all.
/// Paths are relative to `crates/spark-model/src`.
const GUARDED: [(&str, &[&str]); 10] = [
    // ── Dense FFN ──
    (
        "layers/dense_ffn.rs",
        &[
            // `OnceLock`ed kernel-choice helpers — resolved at first touch
            // and stable thereafter, which graph capture requires.
            "mmq_small_tile_enabled",
            "mmq_tile64_enabled",
            // Run once per weight, at load.
            "finalize_q4k_load",
            "finalize_nvfp4_mmq_load",
        ],
    ),
    // ── MoE routed prefill: once per layer per prefill chunk ──
    ("layers/moe/forward_prefill_routed.rs", &[]),
    // ── The decode step itself. `ATLAS_SSM_SAVE_DUMP` was asked THREE times
    //    per token here, each read only to decide whether to do nothing. ──
    ("model/trait_impl/decode_a.rs", &[]),
    // ── DFlash drafter: once per DECODE STEP, and the layer helpers run
    //    `num_layers` times inside that. `dflash_head/from_weights.rs` is
    //    deliberately absent — it builds the head, so it is where the reads
    //    belong, and `dflash_head.rs` keeps two cold readers of its own
    //    (`fp8_rt_enabled` behind a `OnceLock` so the kernel choice cannot
    //    change across graph capture, and `dflash_ctx_cap` called at model
    //    build to size the capture buffer).
    ("layers/dflash_head/forward_block.rs", &[]),
    ("layers/dflash_head/forward_block_layer.rs", &[]),
    ("layers/dflash_head/forward_block_layer_paged.rs", &[]),
    ("layers/dflash_head/propose.rs", &[]),
    ("layers/dflash_head/markov.rs", &[]),
    ("layers/dflash_head/dflash2.rs", &[]),
    ("layers/dflash_head/precompute_ctx_kv.rs", &[]),
];

fn src_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Every environment read in `text`, as `(line number, enclosing fn name)`.
///
/// Comments are stripped first: these files DOCUMENT the variables they no
/// longer read, and a guard that could not tell a call from a sentence about
/// a call would be satisfied by rewording the comment.
///
/// Attribution is to the innermost item-level `fn`. Closures do not declare
/// `fn`, so a read inside one is correctly charged to its enclosing function.
fn env_reads(text: &str) -> Vec<(usize, String)> {
    let mut current = "<file scope>".to_string();
    let mut found = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim_start();
        for prefix in ["pub(crate) fn ", "pub(super) fn ", "pub fn ", "fn "] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                current = rest
                    .split(['(', '<'])
                    .next()
                    .unwrap_or("?")
                    .trim()
                    .to_string();
                break;
            }
        }
        let code = line.split("//").next().unwrap_or("");
        if code.contains("std::env::var") {
            found.push((i + 1, current.clone()));
        }
    }
    found
}

/// ★ THE ENFORCEMENT. A read on any guarded path, in any function not named
/// in its allow list, fails the build.
#[test]
fn the_environment_is_read_only_where_it_is_allowed() {
    let root = src_root();
    let mut offenders = Vec::new();
    for (rel, allowed) in GUARDED {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} is a guarded path: {e}", path.display()));
        for (line, func) in env_reads(&text) {
            if !allowed.contains(&func.as_str()) {
                offenders.push(format!("{rel}:{line} in `{func}`"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "environment reads on a per-token / per-layer / per-forward path. Each \
         takes the process-wide env lock, which serialises concurrent decode \
         threads. Resolve the variable ONCE into a levers struct and read the \
         field instead — do not extend the allow list: {offenders:?}"
    );
}

/// The scanner must be able to tell a call from a comment about a call, and
/// an allowed function from a forbidden one. Without this the guard above
/// could be green because it measures nothing.
#[test]
fn the_scanner_discriminates() {
    let sample = "\
fn cold() {
    let a = std::env::var(\"X\").ok();
}
pub fn hot() {
    // this used to call std::env::var(\"Y\") and no longer does
    let b = 1;
}
pub(super) fn hotter(&self) {
    let c = std::env::var_os(\"Z\").is_some();
}
";
    let reads = env_reads(sample);
    assert_eq!(
        reads,
        vec![(2, "cold".to_string()), (9, "hotter".to_string())],
        "the comment inside `hot` must not count, and `var_os` inside \
         `hotter` must"
    );
}

/// Only two callers of `ModelLevers::from_env` are legitimate: `get()`, which
/// caches it in a `OnceLock`, and the model build, which needs an owned
/// mutable copy to overwrite `max_decode_seqs`. Everything else must use
/// `get()` or take `levers` from the context it already has.
///
/// Crate-wide rather than table-driven, because this one is about a single
/// function name and the answer is the same everywhere.
#[test]
fn from_env_is_called_only_where_it_is_allowed() {
    const ALLOWED: [&str; 3] = [
        // caches the result in a OnceLock — this IS the once.
        "crates/spark-model/src/layers/ops/model_levers.rs",
        // needs an owned mutable copy; takes it from `*get()`.
        "crates/spark-model/src/model/impl_a1.rs",
        // This file. The guard names what it forbids, so it matches itself.
        "crates/spark-model/src/layers/ops/hot_path_env_guards.rs",
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
        "ModelLevers::from_env() re-reads ~40 env vars under a global lock. \
         These call it instead of the once-resolved ModelLevers::get(): {offenders:?}"
    );
}
