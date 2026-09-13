// SPDX-License-Identifier: AGPL-3.0-only

//! On-disk persistence of the cross-grammar token-mask cache (#918).
//!
//! WHAT IS PERSISTED, AND WHY IT IS NOT KEYED BY SCHEMA
//! ----------------------------------------------------
//! #918 describes the cold cost as "per new tool schema". The CPU
//! measurement behind this module says otherwise — it is per *process*.
//! M5 Max, release, Qwen3 ByteLevel-BPE tokenizer (151,669 tokens), the
//! coherency gate's `get_weather` schema through
//! `compile_qwen3_coder_tool_grammar`, single-shot:
//!
//! ```text
//! schema                                   construct   mask prewarm
//! get_weather(city, days)                     5.5 ms       621.2 ms
//! get_weather(city, days)  — again            0.1 ms         0.0 ms
//! search_docs(query, limit) — NEW schema      4.9 ms        16.0 ms
//! run_cmd(command, timeout) — NEW schema      4.9 ms        14.2 ms
//! ```
//!
//! A second, entirely different schema costs ~2.5% of the first because
//! xgrammar's Tier-2 `RuleLevelCache` keys masks *structurally*: every
//! JSON tool schema reuses the same string / number / whitespace /
//! punctuation sub-rules. So what has to survive a restart is the
//! RULE-level cache, not a compiled grammar keyed by schema text — a
//! schema-keyed file would only ever help an exact repeat and would
//! miss the 97% that already transfers. The snapshot is therefore keyed
//! by the TOKENIZER (fingerprint + vocab size); see
//! `xgrammar::compiler::mask_snapshot`.
//!
//! WHEN IT IS WRITTEN
//! ------------------
//! From the background prewarm thread, after it has warmed a grammar —
//! never from the request thread. The first grammar-bearing request of
//! the first process still pays today's cost (minus whatever the
//! prefill overlap hides); every process after that starts warm, for
//! schemas it has never seen. Same harness, n=3: the whole cold path
//! (schema in hand to first constrained mask filled) goes 624.8 ms ->
//! 5.2-5.3 ms for the same schema and 5.0-19.1 ms for one this build
//! has never compiled.
//!
//! Controls: `ATLAS_GRAMMAR_CACHE=0` disables persistence entirely;
//! `ATLAS_GRAMMAR_CACHE_DIR` relocates the file (for a read-only model
//! directory, or to share one warm cache across several model copies of
//! the same tokenizer).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use xgrammar::compiler::{RuleLevelCache, SnapshotIdentity, mask_snapshot};

use super::engine::GrammarEngine;

/// Directory name created under the model directory.
const CACHE_DIR_NAME: &str = ".atlas-grammar-cache";

/// Upper bound on masks written to disk, most-recently-used first.
///
/// The in-memory rule cache is bounded by ~1/3 GiB, which is far more
/// than belongs in a file next to a checkpoint. Measured mask footprint
/// is ~19 KB at a 151,669-token vocabulary (2.36 MB for the 123 masks
/// of one tool grammar), so 512 caps the snapshot around 10 MB there
/// and ~16 MB at Qwen3.6's 248,320 — several distinct tool grammars'
/// worth, which is all the reuse the structural key can deliver anyway.
const MAX_PERSISTED_MASKS: usize = 512;

/// Everything the background saver needs, with no reference to the
/// (`!Sync`) compiler.
pub(super) struct MaskSnapshot {
    path: PathBuf,
    identity: SnapshotIdentity,
    cache: RuleLevelCache,
    /// Entry count at the last successful write — the save is skipped
    /// unless the cache has grown since.
    saved_entries: Arc<AtomicUsize>,
    /// One writer at a time; a second concurrent prewarm just skips.
    writing: Arc<AtomicBool>,
}

/// A callback the background prewarm invokes once a grammar's masks are
/// warm. Boxed so `state.rs` needs no knowledge of persistence.
pub type PrewarmHook = Arc<dyn Fn(usize) + Send + Sync>;

/// `ATLAS_GRAMMAR_CACHE=0` (or `false`/`off`/`no`) disables the on-disk
/// mask cache. Anything else — including unset — enables it.
pub(super) fn cache_enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    )
}

/// Where the snapshot for `fingerprint` lives.
///
/// `ATLAS_GRAMMAR_CACHE_DIR` wins when set — a model directory pulled
/// from a read-only mount cannot host the file, and one warm cache can
/// legitimately serve several copies of the same checkpoint.
pub(super) fn snapshot_path(
    model_dir: &Path,
    dir_override: Option<&str>,
    fingerprint: u64,
) -> PathBuf {
    let dir = match dir_override.map(str::trim).filter(|s| !s.is_empty()) {
        Some(d) => PathBuf::from(d),
        None => model_dir.join(CACHE_DIR_NAME),
    };
    dir.join(format!("masks-{fingerprint:016x}.bin"))
}

impl GrammarEngine {
    /// Seed the cross-grammar mask cache from a snapshot written by an
    /// earlier process, and arm the background saver (#918).
    ///
    /// Returns the number of masks imported; `0` is a miss (no file, a
    /// different tokenizer, a different build, or a corrupt file) and is
    /// not an error — the engine just compiles as it did before.
    /// Called once, at server startup, off any request path.
    pub fn attach_mask_cache(&mut self, model_dir: &Path) {
        if !cache_enabled_from(std::env::var("ATLAS_GRAMMAR_CACHE").ok().as_deref()) {
            tracing::info!("Grammar: on-disk mask cache disabled (ATLAS_GRAMMAR_CACHE)");
            return;
        }
        let Some(cache) = self.compiler.rule_cache_handle() else {
            return; // compiler built with caching off — nothing to persist
        };
        let identity = self.compiler.snapshot_identity();
        let path = snapshot_path(
            model_dir,
            std::env::var("ATLAS_GRAMMAR_CACHE_DIR").ok().as_deref(),
            identity.tokenizer_fingerprint,
        );
        let imported = match self.compiler.load_mask_snapshot(&path) {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(
                    "Grammar: mask snapshot unreadable at {}: {e}",
                    path.display()
                );
                0
            }
        };
        if imported > 0 {
            tracing::info!(
                "Grammar: warmed {imported} token masks from {} — the first tool-call \
                 request skips the cold mask compile (#918)",
                path.display(),
            );
        } else {
            tracing::info!(
                "Grammar: no usable mask snapshot at {}; the first grammar of this \
                 process will compute and persist one (#918)",
                path.display(),
            );
        }
        self.snapshot = Some(MaskSnapshot {
            path,
            identity,
            cache,
            saved_entries: Arc::new(AtomicUsize::new(imported)),
            writing: Arc::new(AtomicBool::new(false)),
        });
    }

    /// The callback handed to a request's background prewarm, which
    /// persists the grown mask cache once that prewarm finishes.
    /// `None` when persistence is off or the engine was never attached.
    pub(crate) fn mask_snapshot_hook(&self) -> Option<PrewarmHook> {
        let snap = self.snapshot.as_ref()?;
        let (path, identity) = (snap.path.clone(), snap.identity);
        let cache = snap.cache.clone();
        let saved = Arc::clone(&snap.saved_entries);
        let writing = Arc::clone(&snap.writing);
        Some(Arc::new(move |_warmed: usize| {
            if cache.len() <= saved.load(Ordering::Relaxed) {
                return; // nothing new to persist
            }
            if writing.swap(true, Ordering::AcqRel) {
                return; // another prewarm is already writing
            }
            // The encode + fsync (~27 ms for 2.4 MB at a 151,669-token
            // vocabulary) runs on a THIRD thread, not this one: the
            // request's first constrained fill joins the prewarm
            // thread, and it must wait for masks, never for I/O.
            let (path, cache, saved) = (path.clone(), cache.clone(), Arc::clone(&saved));
            let done = Arc::clone(&writing);
            let spawned = std::thread::Builder::new()
                .name("grammar-mask-save".to_string())
                .spawn(move || {
                    let written =
                        mask_snapshot::save_to_file(&cache, identity, &path, MAX_PERSISTED_MASKS);
                    match written {
                        Ok(n) => {
                            saved.store(n, Ordering::Relaxed);
                            tracing::debug!(
                                "Grammar: persisted {n} token masks to {}",
                                path.display()
                            );
                        }
                        // A read-only model directory is a normal
                        // deployment, not a fault: the server keeps
                        // working, just cold at boot.
                        Err(e) => tracing::debug!(
                            "Grammar: could not persist masks to {}: {e}",
                            path.display()
                        ),
                    }
                    done.store(false, Ordering::Release);
                });
            if let Err(e) = spawned {
                tracing::debug!("Grammar: mask-snapshot writer not spawned: {e}");
                writing.store(false, Ordering::Release);
            }
        }))
    }
}
