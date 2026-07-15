// SPDX-License-Identifier: AGPL-3.0-only

//! `KvPagingBackend` — the KV overflow tier as a PAGING client (flag-ON arm
//! of `ATLAS_KV_PAGING`), plus [`connect_kv_peer_backend`], the single
//! selection seam `HighSpeedSwap` calls (flag OFF ⇒ the raw one-sided
//! `RdmaKvBackend`, identical data plane; its handshake is the
//! v2 header with `blob_bytes == 0`).
//!
//! Data plane (strictly synchronous, ONE control op in flight — the peer's
//! read-pin lifecycle releases a GET pin on the connection's NEXT op, which
//! is only sound because this client never pipelines; pipelining needs N
//! control connections or a PIN/UNPIN protocol extension and is a measured
//! follow-up, NOT this MVP):
//!   PUT  = ALLOC(RTT, offset) → RDMA-WRITE block → poll → COMMIT(RTT)
//!   GET  = GET(RTT, offset)   → RDMA-READ block  → poll (BEFORE the next
//!          control op, so the pinned slot can never be evicted mid-read)
//!
//! Zero-copy landing (`ATLAS_KV_ZERO_COPY=1`) is preserved: the whole UMA
//! scratch pool registers as ONE landing MR per rail
//! (`register_landing_region`, `remote_read == false`) and a GET's RDMA READ
//! lands directly in the destination slot — only the raddr provenance changed
//! (GET-reply offset instead of `base + group_id·stride`). The leading WAR
//! `stream_sync` is kept verbatim from `RdmaKvBackend::read_zero_copy` (the
//! NIC write is off-stream). The MR-dereg-before-pool-free drop order is
//! inherited: `HighSpeedSwap` declares `backend` before `pool`, and this
//! backend's rails (whose `Verbs` own the MRs) drop with it.
//!
//! MISS = HARD ERROR ([`super::kv_miss_error`]): deployment requirement is a
//! miss-proof peer (`--swap-cap-gb-kv 0`).

use std::ffi::c_void;
use std::io::Write;
use std::net::TcpStream;
use std::num::NonZeroU64;

use anyhow::{Context, Result, bail};

use super::ns;
use crate::backend::{BlockReadRequest, ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout, KvKind};
use crate::snapshot_swap::{
    PagingKind, client_alloc, client_bye, client_commit, client_get, encode_paging_v2_header,
};
use atlas_rdma::verbs::Verbs;

/// Fully-resolved connect parameters (env resolution lives in
/// [`connect_kv_peer_backend`]; the smoke example constructs this directly
/// with exact byte sizes / explicit salts).
#[derive(Clone, Copy, Debug)]
pub struct KvPagingConnect {
    /// Peer warm-arena bytes: non-zero multiple of `layout.block_bytes()`.
    pub arena_bytes: u64,
    /// The KV namespace (see [`ns::derive_kv_ns`]) — model fp + layout
    /// identity + per-client salt already folded.
    pub ns: NonZeroU64,
}

/// One QP on one CX7 adapter with a single registered block-sized bounce.
struct PagingRail {
    verbs: Verbs,
    bounce: PinnedBuffer,
    bounce_lkey: u32,
    remote_rkey: u32,
    /// Pre-registered whole UMA landing region `(base, len, lkey)` for
    /// zero-copy restore (per-slot sub-registration fails on GB10).
    region: Option<(u64, u64, u32)>,
}

pub struct KvPagingBackend {
    rails: Vec<PagingRail>,
    layout: GroupLayout,
    remote_base: u64,
    ns: NonZeroU64,
    zero_copy: bool,
    rr: usize,
    next_wr: u64,
    /// The live paging control channel (ALLOC/COMMIT/GET ride here; data
    /// moves one-sided over the rails).
    ctrl: TcpStream,
}

// SAFETY: mirrors `RdmaKvBackend` — both `StorageBackend` methods take
// `&mut self` and no `&self` method touches a QP/bounce, so `Sync` is sound
// (the swap orchestrator owns it single-threaded regardless).
unsafe impl Sync for KvPagingBackend {}

impl KvPagingBackend {
    /// Connect to a paging peer at `addr` with the v2 handshake
    /// (`[PAGING_MAGIC_V2][kind=KV][arena_bytes][blob_bytes=block_bytes]`),
    /// bring up the rails (same env triple as the legacy KV backend:
    /// `ATLAS_EXPERT_RDMA_DEV`/`GID` = rail 0, `ATLAS_KV_DUAL_RAIL=1` +
    /// `ATLAS_KV_RAIL2_DEV`/`GID` = rail 1), and register one block-sized
    /// bounce per rail (`remote_read == false`, invariant).
    pub fn connect(addr: &str, layout: GroupLayout, cfg: KvPagingConnect) -> Result<Self> {
        use atlas_rdma::env::{first_set, first_set_u32};
        use atlas_rdma::railset::{RailSet, RailSpec};

        let block_bytes = layout.block_bytes();
        if cfg.arena_bytes == 0 || !cfg.arena_bytes.is_multiple_of(block_bytes) {
            bail!(
                "kv-paging: arena_bytes {} must be a non-zero multiple of block_bytes {block_bytes}",
                cfg.arena_bytes
            );
        }
        let spec =
            |dev: String, gid: u32| RailSpec::new(dev, gid, rand::random::<u32>() & 0xff_ffff);
        let rail0 = spec(
            first_set(&["ATLAS_EXPERT_RDMA_DEV"], "roceP2p1s0f1"),
            first_set_u32(&["ATLAS_EXPERT_RDMA_GID"], 3),
        );
        let dual = std::env::var("ATLAS_KV_DUAL_RAIL").ok().as_deref() == Some("1");
        let specs: Vec<RailSpec> = if dual {
            vec![
                rail0,
                spec(
                    first_set(&["ATLAS_KV_RAIL2_DEV"], "rocep1s0f1"),
                    first_set_u32(&["ATLAS_KV_RAIL2_GID"], 3),
                ),
            ]
        } else {
            vec![rail0]
        };

        let mut stream =
            TcpStream::connect(addr).with_context(|| format!("connect kv paging peer {addr}"))?;
        stream.set_nodelay(true).ok();
        stream
            .write_all(&encode_paging_v2_header(
                PagingKind::KV,
                cfg.arena_bytes,
                block_bytes,
            ))
            .context("send kv paging v2 header")?;

        let bb = block_bytes as usize;
        // `parts` (the pinned bounce buffers) MUST be declared BEFORE `rs` (the
        // Verbs that hold the MRs registered against those buffers). reg_mr's
        // contract: the addr must outlive the Verbs (the MR is dereg'd on
        // `Verbs::drop` → rs_destroy). On the SUCCESS path both are consumed
        // into `PagingRail { verbs, bounce }`, whose field order dereg's before
        // free. But on ANY early return between reg_mr and building `rails`
        // (e.g. `finish_rw` failing when the peer rejects), the two are dropped
        // as locals in REVERSE declaration order — so `parts` last means the
        // pinned memory is freed AFTER `rs` deregisters the MRs, not before.
        let mut parts: Vec<(PinnedBuffer, u32)> = Vec::new();
        let mut rs = RailSet::begin(&mut stream, &specs)?;
        parts.reserve(rs.n_rails());
        for rail in &mut rs.rails {
            let bounce = PinnedBuffer::new(bb).context("alloc pinned kv paging bounce")?;
            // SAFETY: bounce lives as long as the rail (and thus the MR) — on
            // both the success path (co-owned in PagingRail) and the error path
            // (parts outlives rs by declaration order above).
            let keys = unsafe { rail.verbs.reg_mr(bounce.ptr, bb, false)? };
            parts.push((bounce, keys.lkey));
        }
        let server = rs.finish_rw(&mut stream, "kv paging peer")?;
        let base = server.last().map(|sp| sp.base_addr).unwrap_or(0);
        let rails: Vec<PagingRail> = rs
            .into_verbs()
            .into_iter()
            .zip(parts)
            .zip(&server)
            .map(|((verbs, (bounce, bounce_lkey)), sp)| PagingRail {
                verbs,
                bounce,
                bounce_lkey,
                remote_rkey: sp.rkey,
                region: None,
            })
            .collect();
        let zero_copy = std::env::var("ATLAS_KV_ZERO_COPY").ok().as_deref() == Some("1");
        tracing::info!(
            "KvPagingBackend connected to {addr}: kind=KV, blob {block_bytes} B, arena {:.3} GiB, \
             ns {:#018x}, {} rail(s), zero_copy={zero_copy} (strictly synchronous MVP: 1 control \
             op in flight)",
            cfg.arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            cfg.ns.get(),
            rails.len(),
        );
        Ok(Self {
            rails,
            layout,
            remote_base: base,
            ns: cfg.ns,
            zero_copy,
            rr: 0,
            next_wr: 1, // 0 == "no completion yet" sentinel convention
            ctrl: stream,
        })
    }

    /// The wire key of one KV block (base dense group id folded with the ns).
    fn block_key(&self, layer: u32, block: u32) -> u64 {
        let base = self
            .layout
            .group_id(GroupKey::new(layer, block, 0, KvKind::K))
            .0;
        ns::wire_key(self.ns, base)
    }

    /// Control GET: the peer offset of `(layer, block)`. A miss is a HARD
    /// error — see the module doc / `kv_miss_error`.
    fn get_block_offset(&mut self, layer: u32, block: u32) -> Result<u64> {
        let key = self.block_key(layer, block);
        match client_get(&mut self.ctrl, key)? {
            Some(off) => Ok(off),
            None => Err(super::kv_miss_error(layer, block)),
        }
    }

    fn pick_rail(&mut self) -> (usize, u64) {
        let ri = self.rr % self.rails.len();
        self.rr = self.rr.wrapping_add(1);
        let wr = self.next_wr;
        self.next_wr = self.next_wr.wrapping_add(1).max(1);
        (ri, wr)
    }

    /// RDMA-READ `len` bytes from the peer into the rail bounce (drained),
    /// then `copy_h2d` to `dst` and sync `stream` — the sync is the bounce
    /// reuse guard (each rail has ONE bounce; the next op may refill it).
    fn rdma_read_bounce(&mut self, raddr: u64, len: usize, dst: u64, stream: u64) -> Result<()> {
        let (ri, wr) = self.pick_rail();
        let rail = &mut self.rails[ri];
        // SAFETY: bounce is a live registered MR of >= len; raddr/rkey are the
        // peer arena (a GET-pinned slot, released on our next control op).
        unsafe {
            rail.verbs.post_read(
                rail.bounce.ptr,
                rail.bounce_lkey,
                raddr,
                rail.remote_rkey,
                len as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        copy_h_to_d_async(dst, rail.bounce.ptr as *const c_void, len, stream)?;
        stream_sync(stream)
    }

    /// Zero-copy RDMA-READ straight into the (UMA) destination via the
    /// pre-registered landing region (caller checked coverage + did the
    /// leading WAR sync). On poll the bytes are GPU-visible at `dst`.
    fn rdma_read_direct(&mut self, raddr: u64, len: usize, dst: u64) -> Result<()> {
        let (ri, wr) = self.pick_rail();
        let rail = &mut self.rails[ri];
        let (_, _, lkey) = rail.region.expect("caller verified region coverage");
        // SAFETY: dst is inside the live UMA landing MR (lkey); raddr/rkey
        // address the peer arena. The NIC DMAs straight into dst.
        unsafe {
            rail.verbs.post_read(
                dst as *mut c_void,
                lkey,
                raddr,
                rail.remote_rkey,
                len as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        Ok(())
    }

    /// Whether every rail's landing region covers `[dst, dst+len)`.
    fn all_regions_cover(&self, dst: u64, len: usize) -> bool {
        self.rails.iter().all(|r| match r.region {
            Some((base, rlen, _)) => dst >= base && dst + len as u64 <= base + rlen,
            None => false,
        })
    }
}

impl StorageBackend for KvPagingBackend {
    /// Per-head restore: GET the block (pinning it), then RDMA-READ the one
    /// `group_stride` stripe at `offset + (kind·nkv + kv_head)·gs`. Reachable
    /// only via un-coalesced callers (construction requires coalescing, so
    /// this is a correctness backstop, not a hot path).
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let gs = self.layout.group_stride;
        let nkv = self.layout.num_kv_heads as u64;
        for req in requests {
            let g = req.group;
            let off = self.get_block_offset(g.layer, g.block)?;
            let idx = (g.kv_kind as u64) * nkv + g.kv_head as u64;
            let raddr = self.remote_base + off + idx * gs;
            self.rdma_read_bounce(raddr, gs as usize, req.dst_dev_ptr, stream)?;
        }
        Ok(())
    }

    // read_async: default = the synchronous `read` (correct, just not async).
    // The sync control RTT at the prefetch boundary is the documented MVP
    // cost; the parent measures it on hardware before any default flip.

    /// Block-granular restore — ONE control GET + ONE RDMA READ per block
    /// (the whole point of block-sized paging records).
    fn read_blocks(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let bb = self.layout.block_bytes() as usize;
        let zc = self.zero_copy
            && requests
                .iter()
                .all(|r| self.all_regions_cover(r.dst_dev_ptr, bb));
        if zc {
            // WAR barrier (verbatim from RdmaKvBackend::read_zero_copy): the
            // NIC is about to DMA into UMA slots a previous tile's attention
            // kernel may still be reading on `stream`; the NIC write is
            // off-stream, so drain in-flight consumers first.
            stream_sync(stream)?;
        }
        for req in requests {
            let off = self.get_block_offset(req.base_key.layer, req.base_key.block)?;
            let raddr = self.remote_base + off;
            if zc {
                self.rdma_read_direct(raddr, bb, req.dst_dev_ptr)?;
            } else {
                self.rdma_read_bounce(raddr, bb, req.dst_dev_ptr, stream)?;
            }
        }
        Ok(())
    }

    /// Per-head writes cannot be served by a peer-owned BLOCK arena: ALLOC
    /// hands out a whole-block slot, so committing after one `group_stride`
    /// stripe would mark `2·nkv − 1` garbage stripes resident. Construction
    /// requires block coalescing precisely so this is unreachable.
    fn write_from_host(&mut self, key: GroupKey, _src: &[u8]) -> Result<()> {
        bail!(
            "kv-paging: per-head write_from_host (layer {}, block {}) is unsupported — the \
             paging record is one whole KV block; run with ATLAS_HSS_COALESCE_BLOCKS on \
             (default) so offload uses write_block_from_host",
            key.layer,
            key.block
        )
    }

    /// Offload one whole block: ALLOC (never full — the peer spills its
    /// coldest to NVMe) → RDMA-WRITE via the rail bounce → poll → COMMIT.
    /// Commit-before-get visibility replaces the legacy drain-before-read.
    fn write_block_from_host(&mut self, base_key: GroupKey, src: &[u8]) -> Result<()> {
        let bb = self.layout.block_bytes() as usize;
        if src.len() != bb {
            bail!(
                "kv-paging write_block_from_host: src len {} != block bytes {bb}",
                src.len()
            );
        }
        let key = self.block_key(base_key.layer, base_key.block);
        let off = client_alloc(&mut self.ctrl, key).with_context(|| {
            format!(
                "kv-paging ALLOC (layer {}, block {}) refused — peer arena exhausted by \
                 reservations/read-pins? Grow ATLAS_KV_PAGING_ARENA_GB (and the peer's \
                 --max-blade-gb)",
                base_key.layer, base_key.block
            )
        })?;
        let raddr = self.remote_base + off;
        let (ri, wr) = self.pick_rail();
        let rail = &mut self.rails[ri];
        // SAFETY: bounce is a live MR of block_bytes; copy the block image in,
        // RDMA-WRITE it to the ALLOC-reserved (peer-pinned) slot, drain.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), rail.bounce.ptr as *mut u8, bb);
            rail.verbs.post_write(
                rail.bounce.ptr,
                rail.bounce_lkey,
                raddr,
                rail.remote_rkey,
                bb as u32,
                wr,
            )?;
        }
        while rail.verbs.poll()? != wr {}
        client_commit(&mut self.ctrl, key)
    }

    // write_blocks_run: default fans out to write_block_from_host — correct;
    // peer-assigned slots are non-contiguous in disk-id space, so wide-write
    // coalescing is layout-incompatible here (see below).

    /// Peer-assigned paging slots are NOT contiguous in disk-id space —
    /// claiming run coalescing would corrupt the caller's layout assumptions.
    fn supports_write_run_coalescing(&self) -> bool {
        false
    }

    fn group_layout(&self) -> GroupLayout {
        self.layout
    }

    /// Register the whole (UMA) scratch pool as ONE landing MR per rail so
    /// zero-copy restore reuses that lkey for every slot (per-slot
    /// registration fails on GB10). `remote_read == false` — invariant.
    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        for rail in &mut self.rails {
            // SAFETY: base/len describe the pool's live UMA allocation, which
            // outlives every rail (backend declared before pool ⇒ MRs dereg
            // before cuMemFreeHost).
            let keys = unsafe { rail.verbs.reg_mr(base as *mut c_void, len, false) }
                .context("kv-paging: register UMA landing region")?;
            rail.region = Some((base, len as u64, keys.lkey));
        }
        tracing::info!(
            "KvPagingBackend: registered UMA landing region {:.1} MiB on {} rail(s) — \
             zero-copy restore live",
            len as f64 / (1024.0 * 1024.0),
            self.rails.len(),
        );
        Ok(())
    }
}

impl Drop for KvPagingBackend {
    fn drop(&mut self) {
        // Strictly synchronous: no in-flight RDMA ops to drain. Tell the peer
        // to close the paging loop (releases any lingering read-pin).
        let _ = client_bye(&mut self.ctrl);
    }
}
