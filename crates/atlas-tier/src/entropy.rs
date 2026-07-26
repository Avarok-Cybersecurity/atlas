// SPDX-License-Identifier: AGPL-3.0-only

//! OS entropy for **per-process key salts** (the SSM decode-tier client salt).
//!
//! Deliberately NOT in [`crate::hash`]: that module is the durable-determinism
//! seam (same input ⇒ same `u64` forever, a fleet-wide on-disk contract); this
//! one is its exact opposite — a value that must be FRESH per process so two
//! same-model clients sharing one paging peer can never derive the same
//! slot-coordinate wire keys.
//!
//! Failure is a HARD error, never a silent zero/degraded salt: a degraded
//! shared salt would quietly reintroduce the exact cross-client key sharing
//! the salt exists to prevent (PCND — no silent fallthrough).

use anyhow::Result;

/// 8 bytes of OS entropy as a `u64`.
///
/// Linux: `libc::getrandom` (flags = 0: the `/dev/urandom` pool, blocking only
/// pre-seed at early boot), looping on `EINTR` and partial reads. Other unix
/// (macOS): `/dev/urandom` via `read_exact`. Windows: `BCryptGenRandom`.
pub fn random_u64() -> Result<u64> {
    let mut buf = [0u8; 8];
    fill(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(target_os = "linux")]
fn fill(buf: &mut [u8]) -> Result<()> {
    let mut got = 0usize;
    while got < buf.len() {
        let r = unsafe { libc::getrandom(buf[got..].as_mut_ptr().cast(), buf.len() - got, 0) };
        if r < 0 {
            let e = std::io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            anyhow::bail!("getrandom failed (refusing a degraded zero-entropy salt): {e}");
        }
        got += r as usize;
    }
    Ok(())
}

#[cfg(all(unix, not(target_os = "linux")))]
fn fill(buf: &mut [u8]) -> Result<()> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| anyhow::anyhow!("open /dev/urandom failed (refusing a degraded salt): {e}"))?;
    f.read_exact(buf)
        .map_err(|e| anyhow::anyhow!("read /dev/urandom failed (refusing a degraded salt): {e}"))
}

/// Windows: `BCryptGenRandom`, reached through the `getrandom` crate that is
/// already in this workspace's dependency graph (0.3.4, pulled transitively) —
/// so this arm adds no new supply-chain surface. Same PCND stance as the unix
/// arms above: a failure is a hard error, never a degraded zero-entropy salt.
#[cfg(windows)]
fn fill(buf: &mut [u8]) -> Result<()> {
    getrandom::fill(buf)
        .map_err(|e| anyhow::anyhow!("BCryptGenRandom failed (refusing a degraded salt): {e}"))
}

#[cfg(test)]
#[path = "entropy_tests.rs"]
mod tests;
