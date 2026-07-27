// SPDX-License-Identifier: AGPL-3.0-only
//
// RailSet — the ONE client-side rail bring-up, extracted from the five
// hand-rolled copies (rdma_kv_backend / expert_tier_rdma / weight_tier_rdma /
// weight_lora_rdma / rdma_snapshot). Compositional by design, byte-identical
// by construction:
//
//   * The stream must be ALREADY PREAMBLED — the five preambles (expert
//     manifest + MODE_VERBS, weight/LoRA model request + manifest +
//     MODE_VERBS, KV `[u64 total_bytes]`, snapshot paging magic) differ and
//     stay in the clients. RailSet never owns the `TcpStream` (the snapshot
//     tier keeps it as a live paging control channel after connect).
//   * MR registration stays AT THE CALL SITE: there is no registration method
//     here, so the access flag (`reg_mr(.., false)` for every client landing
//     buffer, `true` / `reg_mr_rw` on the untouched server side) can never be
//     defaulted or homogenized. Callers do `rail.verbs.reg_mr(..)` directly —
//     `verbs` is deliberately `pub` (the KV zero-copy hot path and the
//     snapshot staged path also post/poll on it raw).
//   * PSN is CALLER-SUPPLIED via `RailSpec` (clients keep their
//     `rand::random::<u32>() & 0xff_ffff` per rail; `rand` stays out of this
//     crate and fixed PSNs make the handshake transcript-testable).
//
// Handshake order (the wire steps are frozen; local steps are free):
//   begin: write [u8 n_rails] → Verbs::create per rail
//   caller: reg_mr its landing buffers on each rail.verbs
//   read_server_ro / read_server_rw: the dialect's server params
//   caller (RO tiers): validate the table against its manifest — a failed
//     validation must bail BEFORE client params are written, exactly as today
//   complete: write [u8 n_rails] + client params → connect each rail
//     INIT→RTR→RTS → read [u8 ack] == STATUS_OK
//   into_verbs: decompose — clients re-own each `Verbs` in their own rail
//     structs, so drop order (MR dereg before buffer free) is unchanged.

use anyhow::{Context, Result, bail};
use std::io::{Read, Write};

use crate::handshake;
use crate::verbs::Verbs;
use crate::wire::{CacheServerParams, RemoteQp, VerbsServerParams, read_server_rails};

/// Guard for `complete`'s rail/param pairing. A bare `zip` would SILENTLY
/// truncate: a server slice shorter than the rail count leaves the trailing
/// rails stuck in INIT (never RTS) while `complete` still returns `Ok`, so the
/// caller believes it is connected and fails later, obscurely, on its first
/// RDMA op. Pure so it is unit-testable without a NIC.
pub(crate) fn check_rail_count(server_len: usize, rails_len: usize, peer: &str) -> Result<()> {
    if server_len != rails_len {
        bail!(
            "{peer}: server returned {server_len} rail params for {rails_len} client rails — \
             refusing to leave rails unconnected (a zip would silently truncate)"
        );
    }
    Ok(())
}

/// One rail's bring-up parameters: device name, RoCEv2 GID index, and the
/// caller-chosen 24-bit send PSN.
#[derive(Clone, Debug)]
pub struct RailSpec {
    pub dev: String,
    pub gid_idx: u32,
    pub psn: u32,
}

impl RailSpec {
    /// A spec with a fresh random 24-bit PSN — what every client tier does.
    pub fn new(dev: String, gid_idx: u32, psn: u32) -> Self {
        Self { dev, gid_idx, psn }
    }
}

/// One connected (or connecting) rail. `verbs` is public: registration and
/// the data plane (post_read/post_write/poll) belong to the caller.
pub struct Rail {
    pub verbs: Verbs,
}

/// All rails of one client connection, mid-handshake.
pub struct RailSet {
    pub rails: Vec<Rail>,
}

impl RailSet {
    /// Wire step 1 + local QP creation: write `[u8 n_rails]` to the
    /// already-preambled stream, then `Verbs::create` one RC QP per spec.
    pub fn begin<W: Write>(stream: &mut W, specs: &[RailSpec]) -> Result<Self> {
        handshake::write_n_rails(stream, specs.len())?;
        let rails = specs
            .iter()
            .map(|s| Verbs::create(&s.dev, s.gid_idx, s.psn).map(|verbs| Rail { verbs }))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { rails })
    }

    pub fn n_rails(&self) -> usize {
        self.rails.len()
    }

    /// RO dialect (expert/weight/LoRA): `[u8 n]{VerbsServerParams}×n`, with
    /// the framed count validated == ours and 1..=8. Returned un-consumed so
    /// the caller can validate the per-layer/per-shard tables against its
    /// manifest BEFORE `complete` writes anything back.
    pub fn read_server_ro<R: Read>(&self, stream: &mut R) -> Result<Vec<VerbsServerParams>> {
        read_server_rails(stream, self.rails.len())
    }

    /// RW-blade dialect (KV/snapshot): `[u8 n echo]{CacheServerParams}×n`.
    pub fn read_server_rw<R: Read>(
        &self,
        stream: &mut R,
        peer: &str,
    ) -> Result<Vec<CacheServerParams>> {
        handshake::read_rw_server_params(stream, self.rails.len(), peer)
    }

    /// The common handshake tail: write `[u8 n_rails]` (again — wire format)
    /// plus each rail's client QP params, connect each rail to its peer rail
    /// (INIT→RTR→RTS), then read the one-byte ready ack.
    pub fn complete<S: Read + Write, P: RemoteQp>(
        &mut self,
        stream: &mut S,
        server: &[P],
        peer: &str,
    ) -> Result<()> {
        check_rail_count(server.len(), self.rails.len(), peer)?;
        let ids: Vec<(u32, u32, [u8; 16])> = self
            .rails
            .iter()
            .map(|r| (r.verbs.qpn(), r.verbs.psn(), r.verbs.gid()))
            .collect();
        handshake::write_client_params(stream, &ids)?;
        for (rail, sp) in self.rails.iter_mut().zip(server) {
            let (qpn, psn, gid) = sp.qp_identity();
            rail.verbs
                .connect(qpn, psn, &gid)
                .with_context(|| format!("connect {peer} rail"))?;
        }
        handshake::read_ack(stream, peer)
    }

    /// `read_server_rw` + `complete` in one call — the RW clients have no
    /// mid-handshake validation. Returns the params for rkey/base capture
    /// (each caller keeps its own policy; KV/snapshot use the LAST base).
    pub fn finish_rw<S: Read + Write>(
        &mut self,
        stream: &mut S,
        peer: &str,
    ) -> Result<Vec<CacheServerParams>> {
        let server = self.read_server_rw(stream, peer)?;
        self.complete(stream, &server, peer)?;
        Ok(server)
    }

    /// Decompose after `complete`: hand each `Verbs` back to the caller's own
    /// rail struct so ownership (and thus drop order vs the registered
    /// buffers) is exactly what it was before the extraction.
    pub fn into_verbs(self) -> Vec<Verbs> {
        self.rails.into_iter().map(|r| r.verbs).collect()
    }
}

#[cfg(test)]
#[path = "railset_tests.rs"]
mod tests;
