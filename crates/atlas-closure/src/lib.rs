// SPDX-License-Identifier: AGPL-3.0-only

//! What a kernel target is actually compiled from, as one hash.
//!
//! # Why a file *set* is not enough
//!
//! `atlas-kernels/build.rs` resolves a target's sources by merging
//! `kernels/<hw>/common/*.cu` with `kernels/<hw>/<model>/<quant>/*.cu`, keyed by
//! file stem, model winning. It is tempting to hash that resolved set and call a
//! benchmark record still valid when the set is unchanged. That is wrong twice,
//! and both were found by reading the tree rather than reasoning about it:
//!
//! 1. **A shadow file may `#include` the very file it shadows.**
//!    `kernels/gb10/qwen3.6-27b/nvfp4/inferspark_prefill_paged_indirect.cu`
//!    contains `#include "../../common/inferspark_prefill_paged_indirect.cu"`,
//!    and it is not alone. A set hash reports "this model shadows that stem, so
//!    a change to the common copy cannot reach it" — while the edited bytes are
//!    compiled straight into the model's kernel. Silent, and fails OPEN.
//!
//! 2. **Headers are in no set at all.** The resolver matches `*.cu`
//!    non-recursively, so `common/*.cuh` — including the one defining `BR64` —
//!    is invisible. Editing a header would invalidate nothing.
//!
//! Following includes fixes both, because an included file's bytes are inside
//! the hash no matter which directory it lives in.
//!
//! # What the hash covers
//!
//! The transitive quoted-`#include` closure of the resolved sources, plus the
//! things that change the emitted code without appearing in any source: nvcc
//! flags, `MODEL.toml`, `KERNEL.toml`, `HARDWARE.toml`, the arch string, and the
//! compiler version.
//!
//! # What it does NOT cover — read this before trusting it
//!
//! A hash that under-specifies its inputs is a cache that returns stale results
//! and reports success, so the omissions are listed rather than implied:
//!
//! - **Angle-bracket includes** (`#include <cuda_fp16.h>`). Toolchain headers,
//!   covered coarsely by the recorded compiler version.
//! - **Include search paths.** Only quoted, path-relative includes are
//!   resolved. A `-I`-found or generated header is NOT followed — and an
//!   include that cannot be resolved makes the whole computation fail rather
//!   than silently omit a file. See [`ClosureError::UnresolvedInclude`].
//! - **Host code.** Anything under `crates/` is outside this hash entirely and
//!   keeps invalidating every gate through the existing path boundary.
//! - **Out-of-repo inputs.** Checkpoint revision, recipe content, serve
//!   environment, driver, container, box state. Recorded as provenance
//!   elsewhere; no tree hash can see them.
//!
//! Equal hash therefore proves *the same device code was compiled from the same
//! sources and config* — not that two runs will produce the same numbers under
//! load. It is sound for "this record still describes this binary" and says
//! nothing about concurrency-dependent behaviour.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Version of the hash *definition*.
///
/// Bumping it invalidates every stored hash on purpose. Any change to what is
/// fed into the digest — a new input, a different ordering, a changed
/// separator — must bump this, or old and new records compare equal while
/// meaning different things.
pub const CLOSURE_SCHEMA: u32 = 1;

#[derive(Debug)]
pub enum ClosureError {
    /// A quoted include did not resolve to a file on disk.
    ///
    /// Deliberately fatal. Skipping it would drop real compiled bytes out of
    /// the hash, which is the one failure mode this module exists to prevent —
    /// so an unresolvable include invalidates everything instead.
    UnresolvedInclude { from: PathBuf, include: String },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl std::fmt::Display for ClosureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnresolvedInclude { from, include } => write!(
                f,
                "{}: #include \"{include}\" does not resolve to a file. The closure hash \
                 refuses to omit compiled bytes, so this invalidates every gate rather than \
                 hashing an incomplete source set.",
                from.display()
            ),
            Self::Io { path, source } => write!(f, "reading {}: {source}", path.display()),
        }
    }
}

impl std::error::Error for ClosureError {}

type Result<T> = std::result::Result<T, ClosureError>;

/// Everything that decides a target's compiled device code.
#[derive(Debug, Clone)]
pub struct ClosureInputs {
    /// Resolved sources, post-shadowing, in any order — the hash sorts them.
    pub sources: Vec<PathBuf>,
    /// Config files whose contents are compiled in or steer the compile.
    pub configs: Vec<PathBuf>,
    /// Merged nvcc flags, in the order they are passed.
    pub flags: Vec<String>,
    /// e.g. `"sm_121a"`.
    pub arch: String,
    /// Compiler identification, e.g. the first line of `nvcc --version`.
    pub compiler: String,
}

/// Hash a target's full closure.
///
/// Deterministic across machines and runs: paths enter the digest as
/// repo-relative strings sorted lexicographically, never as absolute paths,
/// which would otherwise make two checkouts of the same commit disagree.
pub fn hash(root: &Path, inputs: &ClosureInputs) -> Result<String> {
    let mut closure: BTreeSet<PathBuf> = BTreeSet::new();
    for src in &inputs.sources {
        expand(src, &mut closure)?;
    }
    for cfg in &inputs.configs {
        closure.insert(cfg.clone());
    }

    let mut digest = Sha256::new();
    // Domain separation: the schema and field labels go in first, so a future
    // input added without bumping CLOSURE_SCHEMA cannot silently produce the
    // same digest as an older, smaller input set.
    digest.update(b"atlas-closure\x00");
    digest.update(CLOSURE_SCHEMA.to_le_bytes());
    digest.update(b"\x00arch\x00");
    digest.update(inputs.arch.as_bytes());
    digest.update(b"\x00compiler\x00");
    digest.update(inputs.compiler.as_bytes());
    digest.update(b"\x00flags\x00");
    for flag in &inputs.flags {
        digest.update(flag.as_bytes());
        digest.update(b"\x1f");
    }

    digest.update(b"\x00files\x00");
    for path in &closure {
        let rel = path.strip_prefix(root).unwrap_or(path);
        // The NAME is hashed as well as the bytes: moving identical content to
        // a different stem changes which kernel it shadows, so it is a
        // different compile even though the bytes match.
        digest.update(rel.to_string_lossy().as_bytes());
        digest.update(b"\x1f");
        let bytes = std::fs::read(path).map_err(|source| ClosureError::Io {
            path: path.clone(),
            source,
        })?;
        digest.update(bytes.len().to_le_bytes());
        digest.update(&bytes);
        digest.update(b"\x1e");
    }

    Ok(format!("{:x}", digest.finalize()))
}

/// Add `file` and everything it quoted-includes, transitively.
///
/// The `BTreeSet` doubles as the cycle guard: a file already inserted is not
/// walked again, so mutually-including headers terminate instead of recursing
/// forever. Include guards make that pattern normal in this tree.
fn expand(file: &Path, out: &mut BTreeSet<PathBuf>) -> Result<()> {
    let canonical = file.canonicalize().unwrap_or_else(|_| file.to_path_buf());
    if !out.insert(canonical.clone()) {
        return Ok(());
    }
    let text = std::fs::read_to_string(&canonical).map_err(|source| ClosureError::Io {
        path: canonical.clone(),
        source,
    })?;
    let dir = canonical.parent().unwrap_or(Path::new("."));
    for include in quoted_includes(&text) {
        let target = dir.join(&include);
        if !target.exists() {
            return Err(ClosureError::UnresolvedInclude {
                from: canonical.clone(),
                include,
            });
        }
        expand(&target, out)?;
    }
    Ok(())
}

/// Quoted include paths, in source order.
///
/// Angle-bracket includes are intentionally skipped — they are toolchain
/// headers, covered by the compiler version rather than by content.
///
/// Commented-out includes are skipped too. A `//`-prefixed include is not
/// compiled, and hashing it would make a comment edit look like a source
/// change; the point of this hash is to be precise about what the compiler
/// saw. Block comments are not tracked: a `#include` inside `/* … */` is rare
/// enough that over-including it costs a re-run, which is the safe direction.
fn quoted_includes(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        let Some(rest) = trimmed.strip_prefix("#include") else {
            continue;
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('"') else {
            continue;
        };
        if let Some(end) = rest.find('"') {
            found.push(rest[..end].to_string());
        }
    }
    found
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod lib_tests;
