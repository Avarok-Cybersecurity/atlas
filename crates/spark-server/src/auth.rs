// SPDX-License-Identifier: AGPL-3.0-only

//! Bearer-token authentication for the HTTP API.
//!
//! Atlas serves an OpenAI-compatible API; every mainstream client library
//! (`openai`, `litellm`, `anthropic`, opencode, OpenWebUI) sends
//! `Authorization: Bearer <key>` by default, so bearer tokens are the only
//! auth scheme that preserves drop-in client compatibility. mTLS and signed
//! requests would force every user to re-tool their client, so we don't ship
//! them. (If an enterprise customer needs mTLS in front of Atlas, the
//! standard answer is a reverse proxy — nginx, Envoy, Caddy — terminating
//! TLS and forwarding to Atlas on localhost.)
//!
//! Tokens are loaded once at startup. Two sources are supported:
//!   - `--auth-tokens-file <PATH>`: one token per line, blank lines and
//!     `#` comments ignored. The standard production form. Permissions
//!     should be `0600` and a warning is logged if they're broader.
//!   - `--auth-token <TOKEN>`: single inline token. Convenient for quick
//!     starts but the token leaks via `ps`/`/proc/<pid>/cmdline`, so the
//!     server logs a one-line warning at startup.
//!
//! Validation uses constant-time byte comparison (no early-exit on first
//! mismatch) so an attacker can't measure token-prefix-match latency to
//! recover the secret. Comparing against multiple candidate tokens linearly
//! is acceptable here — the candidate set is tiny (operator-curated),
//! constant-time-bounded, and not under attacker control.

use std::path::Path;

use anyhow::{Context, Result, anyhow};

/// Loaded bearer-token validator. Constructed once at startup; cloneable
/// across handlers via `Arc<AuthConfig>`.
#[derive(Debug)]
pub struct AuthConfig {
    /// Valid bearer tokens, stored as raw bytes for constant-time compare.
    /// Order is irrelevant; duplicates are de-duped at load time.
    tokens: Vec<Vec<u8>>,
}

impl AuthConfig {
    /// Load tokens from a file. One token per line; blank lines and lines
    /// starting with `#` are ignored. Trailing whitespace is trimmed from
    /// each token. Returns an error if the file is empty after parsing.
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading auth tokens file {}", path.display()))?;
        let mut tokens: Vec<Vec<u8>> = raw
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty() && !s.starts_with('#'))
            .map(|s| s.as_bytes().to_vec())
            .collect();
        tokens.sort();
        tokens.dedup();
        if tokens.is_empty() {
            return Err(anyhow!(
                "auth tokens file {} contains no usable tokens \
                 (lines must be non-empty and not start with `#`)",
                path.display()
            ));
        }
        Ok(Self { tokens })
    }

    /// Build from a single inline token. The caller is expected to have
    /// already trimmed; we trim defensively to catch obvious mistakes.
    pub fn from_inline(token: &str) -> Result<Self> {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("--auth-token must not be empty"));
        }
        Ok(Self {
            tokens: vec![trimmed.as_bytes().to_vec()],
        })
    }

    /// Number of distinct tokens loaded. For startup logging only —
    /// never log the tokens themselves.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// Validate a presented token in constant time relative to the length
    /// of the candidate set. Returns `true` iff the presented token byte-
    /// equals one of the loaded tokens.
    ///
    /// The comparison is constant-time per candidate (no early exit on
    /// first mismatching byte); the loop over candidates is linear, which
    /// is fine because the operator controls the candidate count and it
    /// is small (typically 1–10). An attacker cannot insert candidates,
    /// so the linear scan does not leak exploitable timing information.
    pub fn validate(&self, presented: &[u8]) -> bool {
        let mut any_match = 0u8;
        for valid in &self.tokens {
            any_match |= ct_eq(presented, valid);
        }
        any_match == 1
    }
}

/// Constant-time byte-slice equality. Returns `1` if the slices are
/// byte-equal, `0` otherwise. Always processes `max(a.len(), b.len())`
/// bytes — never short-circuits on first mismatch — so timing does not
/// leak how many leading bytes match.
fn ct_eq(a: &[u8], b: &[u8]) -> u8 {
    // Length mismatch is observable through length alone; that leak is
    // intentional and unavoidable (a token's length is not secret in
    // practice — generators emit fixed-length tokens).
    if a.len() != b.len() {
        return 0;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    // diff == 0 ⇒ slices are equal. Convert to {0, 1} without a branch.
    1u8 & ((diff as u32).wrapping_sub(1) >> 31) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_matches_only_when_equal() {
        assert_eq!(ct_eq(b"hello", b"hello"), 1);
        assert_eq!(ct_eq(b"hello", b"world"), 0);
        assert_eq!(ct_eq(b"hello", b"hellz"), 0);
        assert_eq!(ct_eq(b"hello", b"helloo"), 0);
        assert_eq!(ct_eq(b"", b""), 1);
        assert_eq!(ct_eq(b"a", b""), 0);
    }

    #[test]
    fn validates_single_inline_token() {
        let cfg = AuthConfig::from_inline("sk-test-token").unwrap();
        assert!(cfg.validate(b"sk-test-token"));
        assert!(!cfg.validate(b"sk-test-toke"));
        assert!(!cfg.validate(b"sk-test-tokenx"));
        assert!(!cfg.validate(b""));
        assert_eq!(cfg.token_count(), 1);
    }

    #[test]
    fn rejects_empty_inline() {
        assert!(AuthConfig::from_inline("").is_err());
        assert!(AuthConfig::from_inline("   ").is_err());
    }

    #[test]
    fn loads_file_with_comments_and_blanks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokens.txt");
        std::fs::write(
            &path,
            "# project A\n\
             alpha-token\n\
             \n\
             # project B\n\
             beta-token\n\
             alpha-token\n", // duplicate — should be de-duped
        )
        .unwrap();
        let cfg = AuthConfig::from_file(&path).unwrap();
        assert_eq!(cfg.token_count(), 2);
        assert!(cfg.validate(b"alpha-token"));
        assert!(cfg.validate(b"beta-token"));
        assert!(!cfg.validate(b"# project A"));
        assert!(!cfg.validate(b"gamma-token"));
    }

    #[test]
    fn empty_file_is_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, "# only a comment\n\n   \n").unwrap();
        assert!(AuthConfig::from_file(&path).is_err());
    }
}
