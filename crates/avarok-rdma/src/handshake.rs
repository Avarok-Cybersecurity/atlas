// SPDX-License-Identifier: AGPL-3.0-only
//
// The client side of the rail handshake as PURE byte functions, generic over
// `Read`/`Write` and taking QP identities as plain `(qpn, psn, gid)` tuples —
// no `Verbs` (and thus no ibverbs) required. `railset::RailSet` delegates to
// these in production; tests drive them against a scripted fake stream to pin
// the client's complete emitted byte sequence (`tests/transcript_golden.rs`),
// which is the only test class that catches a write REORDER.
//
// Wire order per dialect (all little-endian, a frozen external contract):
//   client: [u8 n_rails]                                   (`write_n_rails`)
//   server: RO  → [u8 n]{VerbsServerParams}×n              (`wire::read_server_rails`)
//           RW  → [u8 n echo]{CacheServerParams}×n         (`read_rw_server_params`)
//   client: [u8 n_rails] {VerbsClientParams}×n             (`write_client_params`)
//   server: [u8 ack] == STATUS_OK                          (`read_ack`)
// The n_rails byte is deliberately sent TWICE by the client — that is the wire
// format, not a redundancy to deduplicate.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};

use crate::wire::{CacheServerParams, STATUS_OK, VerbsClientParams};

/// Step 1: tell the peer how many rails we want to stripe across.
pub fn write_n_rails<W: Write>(w: &mut W, n: usize) -> Result<()> {
    w.write_all(&[n as u8]).context("send n_rails")?;
    Ok(())
}

/// RW-blade dialect (KV overflow / snapshots): read the peer's `[u8 n]` echo
/// (must equal the negotiated `want`; deliberately NOT bounded to 8 — the RO
/// dialect's 1..=8 bound is its own, and unifying validation would change
/// accepted wire inputs) followed by `n` `CacheServerParams`.
pub fn read_rw_server_params<R: Read>(
    r: &mut R,
    want: usize,
    peer: &str,
) -> Result<Vec<CacheServerParams>> {
    let mut b1 = [0u8; 1];
    if let Err(e) = r.read_exact(&mut b1) {
        // A clean EOF here is the peer REJECTING us, not a transport fault: it
        // hangs up after logging its own reason (e.g. "paging blade cap") and
        // the v2 wire has no error frame to carry that reason back. Bare
        // `read_exact` context yields "failed to fill whole buffer", which
        // sends operators hunting a network problem that does not exist.
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            bail!(
                "{peer} closed the connection during the rail handshake without sending rail \
                 params. The peer REJECTED this client — read ITS log for the reason. Most \
                 common: the arena this client requested exceeds the peer's --max-blade-gb \
                 cap (peer logs \"paging blade cap\"); also possible: a blob_bytes/kind \
                 mismatch against an arena a prior client already fixed."
            );
        }
        return Err(e).context("read peer n_rails");
    }
    if b1[0] as usize != want {
        bail!("{peer} granted {} rails, wanted {want}", b1[0]);
    }
    let mut server = Vec::with_capacity(want);
    for _ in 0..want {
        server
            .push(CacheServerParams::read_from(r).with_context(|| format!("read {peer} params"))?);
    }
    Ok(server)
}

/// Reply with our rail count (again — wire format) + each rail's QP identity.
pub fn write_client_params<W: Write>(w: &mut W, ids: &[(u32, u32, [u8; 16])]) -> Result<()> {
    w.write_all(&[ids.len() as u8])
        .context("send client n_rails")?;
    for &(qpn, psn, gid) in ids {
        VerbsClientParams { qpn, psn, gid }
            .write_to(w)
            .context("send verbs client params")?;
    }
    Ok(())
}

/// Final step: the peer's one-byte ready ack, which must be `STATUS_OK`.
pub fn read_ack<R: Read>(r: &mut R, peer: &str) -> Result<()> {
    let mut ack = [0u8; 1];
    r.read_exact(&mut ack).context("read verbs ready ack")?;
    if ack[0] != STATUS_OK {
        bail!("{peer} refused connection (ack {})", ack[0]);
    }
    Ok(())
}
