// SPDX-License-Identifier: AGPL-3.0-only
//! What makes one `spark serve` "the same server" as another: the bytes of
//! the binary and the arguments it was started with.
//!
//! A benchmark that reuses a server somebody else started has to know it is
//! measuring what it would have started itself — the same build, the same
//! recipe rendering, the same overrides, the same ENGINE ENVIRONMENT — or its
//! record names a config it never ran. The server publishes them as digests
//! (`GET /serve-config`), never as the arguments themselves: an argv can carry
//! `--auth-token`. The digests are what both sides compute, one spelling, here.
//!
//! ## Why the environment is part of the identity (owner, 2026-09-22)
//!
//! "If the server config has to change to be accurate, we should NOT reuse the
//! server." The recipe half of that was already enforced — a recipe renders to
//! flags, so a different recipe is a different `argv_sha256`. The environment
//! half was NOT: `AVAROK_PREFILL_CODISPATCH=1` and `=0` produce byte-identical
//! argv, so a leased server carrying the wrong lever passed the reuse check and
//! the run measured a config it did not declare. That lever is worth +4.78% on
//! warm TTFT, about 107x the control spread, so this is not a hypothetical.
//!
//! The digest covers EVERY `AVAROK_*` variable rather than a curated list.
//! `gate::record::PERF_CONTROLS` would have been the tempting SSOT, but it
//! holds only the three codispatch keys — fingerprinting it would silently miss
//! `AVAROK_FP8_ROWWISE`, `AVAROK_MTP_DCUT_RATIO` and `AVAROK_MTP_K_LADDER`,
//! which the published recipe also sets, and a safety check that misses three
//! of four levers is worse than none because it gets trusted. The failure
//! directions are not symmetric: refusing a reusable server costs one server
//! start, while reusing a wrong one silently corrupts a record, so this errs
//! toward refusing.

use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// The digest of a server's arguments — everything after the program name,
/// joined by NUL so `["a b"]` and `["a", "b"]` differ.
#[must_use]
pub fn argv_fingerprint(args_after_program: &[String]) -> String {
    let mut h = Sha256::new();
    for a in args_after_program {
        h.update(a.as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// The SHA-256 of a file's bytes; a binary's identity, since `spark` embeds
/// no version of its own source.
pub fn file_sha256(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).with_context(|| format!("reading {}", path.display()))?;
    Ok(format!("{:x}", h.finalize()))
}

/// The two digests of THIS process, computed once: its own executable and
/// its own arguments after the program name.
pub fn this_process() -> &'static ServeIdentity {
    static ID: std::sync::OnceLock<ServeIdentity> = std::sync::OnceLock::new();
    ID.get_or_init(|| {
        let args: Vec<String> = std::env::args().skip(1).collect();
        ServeIdentity {
            argv_sha256: argv_fingerprint(&args),
            binary_sha256: std::env::current_exe()
                .context("current_exe")
                .and_then(|p| file_sha256(&p))
                .unwrap_or_else(|e| format!("unavailable: {e:#}")),
            engine_env_sha256: engine_env_fingerprint(std::env::vars()),
            pid: std::process::id(),
        }
    })
}

/// What `GET /serve-config` answers.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServeIdentity {
    pub argv_sha256: String,
    pub binary_sha256: String,
    /// Digest of every `AVAROK_*` variable this server was started with.
    ///
    /// `serde(default)` so a server built before this field existed still
    /// PARSES — it then reports the empty string, which
    /// [`crate::serve_identity::engine_env_is_unknown`] treats as UNKNOWN and
    /// the reuse check refuses on. An old server is exactly the case where the
    /// environment cannot be verified, so "cannot tell" must not read as
    /// "matches".
    #[serde(default)]
    pub engine_env_sha256: String,
    pub pid: u32,
}

/// The digest of the engine-relevant environment: every `AVAROK_*` variable,
/// sorted by name, NUL-separated so `A=1,B=` and `A=1B,=` cannot collide.
///
/// Pure over the iterator so it is testable without mutating the process
/// environment — `set_var` is unsafe and process-global.
#[must_use]
pub fn engine_env_fingerprint(vars: impl Iterator<Item = (String, String)>) -> String {
    let mut kept: Vec<(String, String)> = vars.filter(|(k, _)| k.starts_with("AVAROK_")).collect();
    kept.sort();
    let mut h = Sha256::new();
    for (k, v) in kept {
        h.update(k.as_bytes());
        h.update([0u8]);
        h.update(v.as_bytes());
        h.update([0u8]);
    }
    format!("{:x}", h.finalize())
}

/// Whether a reported digest carries no information — an empty string, which is
/// what a server predating [`ServeIdentity::engine_env_sha256`] reports.
#[must_use]
pub fn engine_env_is_unknown(reported: &str) -> bool {
    reported.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fingerprint_separates_arguments_and_orders_them() {
        let a = argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "1".into()]);
        let b = argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "2".into()]);
        let c = argv_fingerprint(&["serve m".into(), "--port".into(), "1".into()]);
        assert_ne!(a, b, "a different port is a different server");
        assert_ne!(a, c, "joined arguments are not the same argv");
        assert_eq!(
            a,
            argv_fingerprint(&["serve".into(), "m".into(), "--port".into(), "1".into()])
        );
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn a_file_digest_is_its_bytes() {
        let dir = std::env::temp_dir().join(format!("serve-identity-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bin");
        std::fs::write(&p, b"hello").unwrap();
        assert_eq!(
            file_sha256(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        std::fs::write(&p, b"hellO").unwrap();
        assert_ne!(
            file_sha256(&p).unwrap(),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
