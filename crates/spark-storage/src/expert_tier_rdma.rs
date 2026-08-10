// SPDX-License-Identifier: AGPL-3.0-only
//
// RdmaTier — the peer weight-fetch tier (Stage 4).
//
// Fetches expert records from an `atlas-expert-peer` straight into the pinned
// arena, then returns residency addresses pointing INTO that arena — exactly
// like `UmaArenaTier`, only the source is a peer instead of local NVMe. Two
// transports share the tier and the arena machinery:
//
//   * `Transport::Tcp`   — Phase A: two-sided record streaming. The peer `pread`s
//     each record and writes it back over the socket; simple, bit-identical, but
//     the peer CPU is busy and single-stream bandwidth is ~5 GB/s.
//   * `Transport::Verbs` — Phase B: one-sided `IBV_WR_RDMA_READ`. The client
//     pulls each record directly out of the peer's registered store MR into the
//     arena slot with zero peer-CPU involvement (~14 GB/s, measured). This is a
//     pure transport swap — the bytes still land in the same pinned LPDDR that
//     the GPU reads at the same VA, and `residency_from` + the record header's
//     identity check catch any misplacement, so it cannot change a GEMM byte.
//
// The transport is chosen by the `--expert-backend` value: `rdma` = TCP,
// `rdma-verbs` = one-sided verbs. Device/GID for verbs come from
// `$ATLAS_EXPERT_RDMA_DEV` (default `roceP2p1s0f1`) / `$ATLAS_EXPERT_RDMA_GID`
// (default 3, the RoCEv2 IPv4 GID on GB10/CX7).

use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{Context, Result, bail};

use crate::expert::{ExpertKey, ExpertLayout, ExpertRecordSpec};
use crate::expert_arena::ExpertArena;
use crate::expert_peer::{MODE_TCP, STATUS_OK, encode_request, read_manifest};
use crate::expert_tier::{ArenaSlot, ExpertResidency, ExpertTier, TierKind, residency_from};

/// The active peer transport. `Verbs` only exists where the shim is compiled.
/// The verbs transport is N-rail: one `Rail` per CX7 adapter, and a fetch is
/// striped to `rail = expert % n_rails` (single-rail = the unchanged path).
enum Transport {
    Tcp,
    #[cfg(atlas_rdma_verbs)]
    Verbs(Vec<Rail>),
}

/// One-sided verbs state for a single rail: the QP, this rail's arena MR lkey,
/// and the per-layer remote MR `{base, rkey}` table the peer published for it.
/// The base VA is shared across rails (the peer mmaps each layer once); only the
/// rkey (and QP/NIC) differ per rail.
#[cfg(atlas_rdma_verbs)]
struct Rail {
    verbs: atlas_rdma::verbs::Verbs,
    arena_lkey: u32,
    /// `(remote_base_addr, rkey)` per MoE layer, layer-indexed.
    layers: Vec<(u64, u32)>,
}

pub struct RdmaTier {
    stream: TcpStream,
    // `transport` is declared BEFORE `arena` so it drops first: the verbs rails
    // hold MRs registered over the arena's pinned pages, so their `ibv_dereg_mr`
    // must run before the arena frees those pages. (Struct fields drop in
    // declaration order.) With N rails this is load-bearing — reverting the order
    // would dereg N MRs over freed memory.
    transport: Transport,
    arena: ExpertArena,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    healthy: bool,
}

impl RdmaTier {
    /// Connect to a peer at `addr`, receive its manifest, allocate the arena, and
    /// bring up the chosen transport. The peer's `ExpertIndex` geometry defines
    /// the record stride, so the arena matches the remote store exactly.
    pub fn connect(
        addr: &str,
        num_slabs: u32,
        slots_per_slab: u32,
        use_verbs: bool,
    ) -> Result<Self> {
        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect expert peer {addr}"))?;
        stream.set_nodelay(true).ok();
        let index = read_manifest(&mut stream)?;
        let spec = index.spec();
        let layout = index.layout();
        let arena = ExpertArena::new(num_slabs, slots_per_slab, layout.record_stride as usize)?;

        let transport = if use_verbs {
            #[cfg(atlas_rdma_verbs)]
            {
                connect_verbs(&mut stream, &arena, index.num_moe_layers)?
            }
            // Built without rdma-core (no C shim) — verbs is unavailable; the
            // TCP `rdma` backend still works. Keeps the crate compiling under
            // ATLAS_SKIP_BUILD / hosts without libibverbs.
            #[cfg(not(atlas_rdma_verbs))]
            {
                let _ = &arena;
                bail!(
                    "--expert-backend rdma-verbs needs a build with rdma-core \
                     (atlas_rdma_verbs cfg); use --expert-backend rdma (TCP) instead"
                );
            }
        } else {
            stream
                .write_all(&[MODE_TCP])
                .context("send TCP transport mode")?;
            Transport::Tcp
        };

        let label = match &transport {
            Transport::Tcp => "TCP".to_string(),
            #[cfg(atlas_rdma_verbs)]
            Transport::Verbs(rails) => {
                format!("verbs (one-sided RDMA READ, {} rail(s))", rails.len())
            }
        };
        tracing::info!(
            "RdmaTier[{label}] connected to {addr}: {} layers, {} experts, stride {}",
            index.num_moe_layers,
            index.num_experts,
            layout.record_stride
        );
        Ok(Self {
            stream,
            arena,
            spec,
            layout,
            transport,
            healthy: true,
        })
    }

    pub fn arena(&self) -> &ExpertArena {
        &self.arena
    }

    /// Two-sided TCP fetch: request the record, read `[status][stride bytes]`
    /// straight into the pinned slot.
    fn fetch_tcp(&mut self, key: ExpertKey, host: *mut u8, stride: usize) -> Result<()> {
        if let Err(e) = self
            .stream
            .write_all(&encode_request(key.layer, key.expert))
        {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer request {:?}", key));
        }
        let mut status = [0u8; 1];
        if let Err(e) = self.stream.read_exact(&mut status) {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer status {:?}", key));
        }
        if status[0] != STATUS_OK {
            bail!("peer returned error status {} for {:?}", status[0], key);
        }
        // Land the record bytes DIRECTLY into the pinned, GPU-addressable slot.
        // SAFETY: `host` points at a `stride`-byte slot inside the pinned arena.
        let dst = unsafe { std::slice::from_raw_parts_mut(host, stride) };
        if let Err(e) = self.stream.read_exact(dst) {
            self.healthy = false;
            return Err(e).with_context(|| format!("peer payload {:?}", key));
        }
        Ok(())
    }
}

/// Bring up the one-sided verbs transport via [`atlas_rdma::railset::RailSet`]:
/// create N rails, register the arena on each, exchange per-rail QP params over
/// the TCP control channel, connect INIT->RTR->RTS, await the ack. Dual-rail is
/// env-driven (ATLAS_EXPERT_DUAL_RAIL=1): rail 0 = ATLAS_EXPERT_RDMA_DEV/GID
/// (the existing single-rail defaults), rail 1 = ATLAS_EXPERT_RAIL2_DEV/GID
/// (default rocep1s0f1 / 3). Single-rail is the default and is byte-for-byte
/// the previous path.
#[cfg(atlas_rdma_verbs)]
fn connect_verbs(
    stream: &mut TcpStream,
    arena: &ExpertArena,
    num_layers: u32,
) -> Result<Transport> {
    use crate::expert_peer::MODE_VERBS;
    use atlas_rdma::env::{first_set, first_set_u32};
    use atlas_rdma::railset::{RailSet, RailSpec};

    stream
        .write_all(&[MODE_VERBS])
        .context("send verbs transport mode")?;

    // Rail 0 from the expert env (the cabled CX7 link); rail 1 from the expert
    // rail-2 env. Dual-rail only when ATLAS_EXPERT_DUAL_RAIL=1. PSN = fresh
    // random 24-bit per rail (caller-supplied by RailSet design).
    let spec = |dev: String, gid: u32| RailSpec::new(dev, gid, rand::random::<u32>() & 0xff_ffff);
    let rail0 = spec(
        first_set(&["ATLAS_EXPERT_RDMA_DEV"], "roceP2p1s0f1"),
        first_set_u32(&["ATLAS_EXPERT_RDMA_GID"], 3),
    );
    let dual = std::env::var("ATLAS_EXPERT_DUAL_RAIL").ok().as_deref() == Some("1");
    let specs: Vec<RailSpec> = if dual {
        let rail1 = spec(
            first_set(&["ATLAS_EXPERT_RAIL2_DEV"], "rocep1s0f1"),
            first_set_u32(&["ATLAS_EXPERT_RAIL2_GID"], 3),
        );
        vec![rail0, rail1]
    } else {
        vec![rail0]
    };

    // [u8 n_rails] + one QP per rail, then register the WHOLE arena as each
    // rail's READ landing MR (LOCAL_WRITE only — `remote_read == false` is the
    // access-flag invariant for every client landing buffer). The N MRs pin the
    // SAME arena pages (one lkey per rail).
    let mut rs = RailSet::begin(stream, &specs)?;
    let mut arena_lkeys: Vec<u32> = Vec::with_capacity(rs.n_rails());
    for rail in &mut rs.rails {
        // SAFETY: the arena's pinned buffer lives as long as the tier (and thus
        // every MR); base_ptr()/total_bytes() describe exactly that allocation.
        let keys = unsafe {
            rail.verbs
                .reg_mr(arena.base_ptr(), arena.total_bytes(), false)?
        };
        arena_lkeys.push(keys.lkey);
    }

    // Peer publishes N per-rail QP + per-layer MR tables; validate each rail's
    // layer count against the manifest BEFORE replying (a mismatch must bail
    // with no client params written — the pre-RailSet behavior).
    let server = rs
        .read_server_ro(stream)
        .context("read verbs server params")?;
    for sp in &server {
        if sp.layers.len() != num_layers as usize {
            bail!(
                "verbs peer published {} layer MRs but manifest has {num_layers} MoE layers",
                sp.layers.len()
            );
        }
    }

    // Reply with each rail's client QP, connect each rail, await the ack.
    rs.complete(stream, &server, "verbs peer")?;
    let rails: Vec<Rail> = rs
        .into_verbs()
        .into_iter()
        .zip(arena_lkeys)
        .zip(server)
        .map(|((verbs, arena_lkey), sp)| Rail {
            verbs,
            arena_lkey,
            layers: sp.layers,
        })
        .collect();
    Ok(Transport::Verbs(rails))
}

impl ExpertTier for RdmaTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, _stream: u64) -> Result<ExpertResidency> {
        let stride = self.layout.record_stride as usize;
        let host = self.arena.slot_host_ptr(slot.slab, slot.slot)?;
        let dev_va = self.arena.slot_dev_va(slot.slab, slot.slot)?;
        let spec = self.spec; // Copy — release the field borrow before matching.

        match &mut self.transport {
            Transport::Tcp => {
                self.fetch_tcp(key, host, stride)?;
            }
            #[cfg(atlas_rdma_verbs)]
            Transport::Verbs(rails) => {
                // Stripe the fetch onto rail = expert % n_rails. Single-rail
                // (n == 1) => always rail 0, the unchanged path.
                let ri = (key.expert as usize) % rails.len();
                let rail = &mut rails[ri];
                let (base, rkey) = *rail.layers.get(key.layer as usize).with_context(|| {
                    format!("verbs: no layer MR for layer {} ({:?})", key.layer, key)
                })?;
                let remote_addr = base + (key.expert as u64) * (stride as u64);
                let wr_id = ((key.layer as u64) << 32) | (key.expert as u64);
                // SAFETY: `host` is a `stride`-byte slot inside this rail's arena
                // MR (arena_lkey); remote_addr/rkey address the peer's layer MR on
                // the SAME rail.
                let post = unsafe {
                    rail.verbs.post_read(
                        host as *mut std::ffi::c_void,
                        rail.arena_lkey,
                        remote_addr,
                        rkey,
                        stride as u32,
                        wr_id,
                    )
                };
                if let Err(e) = post {
                    self.healthy = false;
                    return Err(e).with_context(|| format!("verbs post_read {:?}", key));
                }
                match rail.verbs.poll() {
                    Ok(got) if got == wr_id => {}
                    Ok(got) => {
                        self.healthy = false;
                        bail!("verbs completion wr_id {got:#x} != expected {wr_id:#x} ({key:?})");
                    }
                    Err(e) => {
                        self.healthy = false;
                        return Err(e).with_context(|| format!("verbs poll {:?}", key));
                    }
                }
            }
        }

        // SAFETY: the slot now holds `stride` valid bytes (landed by TCP or RDMA).
        let record = unsafe { std::slice::from_raw_parts(host, stride) };
        residency_from(&spec, record, dev_va, key)
    }

    fn kind(&self) -> TierKind {
        TierKind::Rdma
    }

    /// Link health: false after any transport error so the streamer can fall
    /// back to the local NVMe UMA tier (graceful degradation on CX7 flap).
    fn healthy(&self) -> bool {
        self.healthy
    }
}
