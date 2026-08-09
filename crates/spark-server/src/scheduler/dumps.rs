// SPDX-License-Identifier: AGPL-3.0-only

//! [`RunDumps`] — the diagnostic file sinks a run writes to.
//!
//! Two `OnceLock<Option<Mutex<..>>>` file appenders, each opened lazily from an
//! `ATLAS_*` path on its first write. Both are sinks for one run's decode
//! path, and holding them in process globals had two consequences:
//!
//! * The handle outlived the run that opened it, so a second model appended
//!   into the same file with nothing marking the boundary — and the analysis
//!   these files exist for is per-model.
//! * `OnceLock` made the *decision* permanent too. A run started with the env
//!   unset poisoned the slot with `None`, so no later run in that process could
//!   dump at all, however the environment changed.
//!
//! Opened when the run starts rather than on first write. The env gate is
//! unchanged, so a serve with neither variable set opens nothing; what changes
//! is that a bad path fails at run start instead of silently at the first
//! record.

use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::sync::Mutex;

/// Diagnostic file sinks for one scheduler run.
#[derive(Debug, Default)]
pub struct RunDumps {
    /// `ATLAS_LOGIT_DUMP=path` — per-position logits.
    pub logits: Option<Mutex<BufWriter<File>>>,
    /// `ATLAS_ADADEC_DIAGNOSTIC=dir` — the adaptive-decode entropy trace.
    pub adadec: Option<Mutex<File>>,
}

impl RunDumps {
    /// Open whichever sinks this run's environment asks for.
    pub fn from_env() -> Self {
        Self {
            logits: Self::open_append("ATLAS_LOGIT_DUMP", |p| p.to_path_buf())
                .map(|f| Mutex::new(BufWriter::new(f))),
            adadec: Self::open_append("ATLAS_ADADEC_DIAGNOSTIC", |p| {
                p.join("adadec_entropy.jsonl")
            })
            .map(Mutex::new),
        }
    }

    /// Resolve `var` to a path, run `to_file` over it, and open for append.
    /// `None` when the variable is unset or empty; an open failure is logged
    /// and treated as unset, so a bad diagnostic path never fails a serve.
    fn open_append(
        var: &str,
        to_file: impl FnOnce(&std::path::Path) -> std::path::PathBuf,
    ) -> Option<File> {
        let raw = std::env::var(var).ok()?;
        if raw.is_empty() {
            return None;
        }
        let base = std::path::Path::new(&raw);
        let path = to_file(base);
        if let Some(dir) = path.parent()
            && dir != std::path::Path::new("")
            && raw.as_str() == base.to_string_lossy()
            && path != base
        {
            let _ = std::fs::create_dir_all(dir);
        }
        match OpenOptions::new().create(true).append(true).open(&path) {
            Ok(f) => Some(f),
            Err(e) => {
                tracing::error!("{var}: cannot open {}: {e}", path.display());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_environment_opens_nothing() {
        // Also the shape that made the `OnceLock` wrong: this answer used to be
        // cached for the life of the process, so a run started without the
        // variable set locked every later run out of dumping.
        let d = RunDumps::default();
        assert!(d.logits.is_none() && d.adadec.is_none());
    }

    #[test]
    fn two_runs_hold_independent_sinks() {
        let a = RunDumps::default();
        let b = RunDumps::default();
        assert!(a.logits.is_none());
        assert!(b.logits.is_none());
    }
}
