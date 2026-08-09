// SPDX-License-Identifier: AGPL-3.0-only

//! Offset-addressed RDMA arena for the SSM-snapshot spill tier.
//!
//! A minimal, **synchronous** transport over the CX7 verbs + RW-blade paging
//! peer protocol the KV overflow tier uses, but addressed by a flat byte
//! **offset** (snapshots are keyed by an opaque id → arena slot) rather than the
//! KV `GroupKey`/`group_stride` layout — reusing that layout would corrupt live
//! KV (its `write_from_host` asserts `src.len()==group_bytes`).
//!
//! The paging peer is layout-agnostic (client sends `total_bytes`, the peer
//! registers ONE RW MR and serves `base+offset`), so a second peer instance on
//! its own port serves the snapshot arena with zero peer-side change. Each op is
//! drained to completion before returning (one blob ~64 MB, ~5–7 ms — the
//! spill/fault path is latency-, not throughput-critical), so the caller's
//! `SnapshotBlobStore::{put,get}` contract (durable on return) holds.
//!
//! Gathering the scattered per-layer SSM state into the contiguous blob and all
//! device-stream ordering already happen in `SsmSnapshotPool::{spill_slot,
//! fault_in_slot}`; this transport only moves host bytes.

// The real transport needs the CUDA pinned bounce + the verbs FFI; when either
// is absent, a stub whose `connect` always errors lets dependents reference the
// type unconditionally (the tier selector then falls back to host-RAM).
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use imp::RdmaSnapshotArena;
#[cfg(not(all(feature = "cuda", atlas_rdma_verbs)))]
pub use stub::RdmaSnapshotArena;

#[cfg(not(all(feature = "cuda", atlas_rdma_verbs)))]
mod stub {
    use anyhow::{Result, bail};
    /// Placeholder when RDMA verbs / CUDA aren't built. `connect` always errors,
    /// so [`crate`] dependents degrade to the host-RAM tier; `write`/`read` are
    /// unreachable (a stub arena is never successfully constructed).
    pub struct RdmaSnapshotArena;
    impl RdmaSnapshotArena {
        pub fn connect(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
            bail!("RDMA snapshot tier not built (needs feature `cuda` + atlas_rdma_verbs)")
        }
        pub fn connect_paging(_addr: &str, _arena_bytes: u64, _blob_bytes: usize) -> Result<Self> {
            bail!("RDMA snapshot tier not built (needs feature `cuda` + atlas_rdma_verbs)")
        }
        pub fn write(&self, _offset: u64, _bytes: &[u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn read(&self, _offset: u64, _out: &mut [u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_put(&self, _key: u64, _bytes: &[u8]) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_get(&self, _key: u64, _out: &mut [u8]) -> Result<bool> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
        pub fn paging_remove(&self, _key: u64) -> Result<()> {
            unreachable!("stub RdmaSnapshotArena is never constructed")
        }
    }
}

#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
mod imp {
    use std::io::Write;
    use std::net::TcpStream;
    use std::sync::Mutex;

    use anyhow::{Result, bail};

    use crate::cuda_min::PinnedBuffer;
    use atlas_rdma::env::{first_set, first_set_u32};
    use atlas_rdma::railset::{RailSet, RailSpec};
    use atlas_rdma::verbs::Verbs;

    /// One rail: its QP + a single persistent registered bounce (`blob_bytes`).
    struct SnapRail {
        verbs: Verbs,
        bounce: PinnedBuffer,
        lkey: u32,
        remote_rkey: u32,
        /// lkey for the SHARED striped-staging buffer registered on THIS rail (0 if
        /// staging disabled). Every rail registers the SAME contiguous staging
        /// buffer, so whichever rail fetches a chunk it lands at its true offset.
        staging_lkey: u32,
    }

    /// Mutable rail state, serialized under one lock (the trait exposes `&self`).
    struct ArenaInner {
        rails: Vec<SnapRail>,
        /// ONE contiguous `blob_bytes` staging buffer for the striped/pipelined
        /// dual-rail path (ATLAS_SSM_STAGING); None = single-WR bounce fallback.
        staging: Option<PinnedBuffer>,
        rr: usize,
        next_wr: u64,
        /// In raw (dumb one-sided) mode: kept alive for the QP's lifetime, else idle.
        /// In paging mode: the live control channel — alloc/commit/get/remove
        /// requests ride this stream, interleaved with the RDMA data plane below.
        stream: TcpStream,
    }

    /// Offset-addressed RDMA snapshot arena. Connect to the paging peer sized for
    /// `arena_slots × blob_bytes`; `write`/`read` move one `blob_bytes` blob to/from
    /// `base + offset`.
    pub struct RdmaSnapshotArena {
        inner: Mutex<ArenaInner>,
        remote_base: u64,
        blob_bytes: usize,
    }

    // SAFETY: every access to the raw verbs/bounce state goes through `inner`'s
    // Mutex, so there is no unsynchronized sharing; mirrors the KV backend's
    // single-owner contract. `Verbs` is `Send`; `PinnedBuffer` is `Send + Sync`.
    unsafe impl Send for RdmaSnapshotArena {}
    unsafe impl Sync for RdmaSnapshotArena {}

    impl RdmaSnapshotArena {
        /// Handshake with the snapshot peer at `addr` and register `blob_bytes`
        /// bounces. Rail devices/GIDs reuse the KV env (`ATLAS_EXPERT_RDMA_DEV`/`GID`
        /// = rail 0, `ATLAS_KV_RAIL2_DEV`/`GID` = rail 1, dual only when
        /// `ATLAS_KV_DUAL_RAIL=1`). `arena_bytes` = `arena_slots × blob_bytes`.
        pub fn connect(addr: &str, arena_bytes: u64, blob_bytes: usize) -> Result<Self> {
            Self::connect_inner(addr, arena_bytes, blob_bytes, false)
        }

        /// Paging-mode connect: the peer arena becomes a page-cache over an
        /// NVMe swap file and OWNS residency; this client uses the control channel
        /// (`paging_put`/`paging_get`/`paging_remove`) instead of a client-side
        /// allocator. Requires the peer be started with `--swap-dir`.
        pub fn connect_paging(addr: &str, arena_bytes: u64, blob_bytes: usize) -> Result<Self> {
            Self::connect_inner(addr, arena_bytes, blob_bytes, true)
        }

        fn connect_inner(
            addr: &str,
            arena_bytes: u64,
            blob_bytes: usize,
            paging: bool,
        ) -> Result<Self> {
            // Rail env: the KV triple verbatim (shared CX7 fabric config). Fresh
            // random 24-bit PSN per rail.
            let spec =
                |dev: String, gid: u32| RailSpec::new(dev, gid, rand::random::<u32>() & 0xff_ffff);
            let rail0 = spec(
                first_set(&["ATLAS_EXPERT_RDMA_DEV"], "roceP2p1s0f1"),
                first_set_u32(&["ATLAS_EXPERT_RDMA_GID"], 3),
            );
            let dual = std::env::var("ATLAS_KV_DUAL_RAIL").ok().as_deref() == Some("1");
            let specs: Vec<RailSpec> = if dual {
                let rail1 = spec(
                    first_set(&["ATLAS_KV_RAIL2_DEV"], "rocep1s0f1"),
                    first_set_u32(&["ATLAS_KV_RAIL2_GID"], 3),
                );
                vec![rail0, rail1]
            } else {
                vec![rail0]
            };
            let n_rails = specs.len();

            let mut stream = TcpStream::connect(addr)
                .map_err(|e| anyhow::anyhow!("connect snapshot peer {addr}: {e}"))?;
            stream.set_nodelay(true).ok();
            // v2 handshake: both modes send the same 25-byte header.
            // Paging carries the real blob size; the raw (dumb one-sided) mode
            // signals itself with blob_bytes == 0 — the peer then hands this
            // connection a private arena with a client-owned allocator.
            stream.write_all(&crate::snapshot_swap::encode_paging_v2_header(
                crate::snapshot_swap::PagingKind::SSM,
                arena_bytes,
                if paging { blob_bytes as u64 } else { 0 },
            ))?;
            // [u8 n_rails] + one QP per rail.
            let mut rs = RailSet::begin(&mut stream, &specs)?;

            // ONE contiguous staging buffer (ATLAS_SSM_STAGING) registered on EVERY
            // rail → each rail gets its own lkey over the SAME memory, so a chunk
            // striped to any rail lands at its true offset (the inc-6 reassembly fix).
            let staging_on = std::env::var("ATLAS_SSM_STAGING").ok().as_deref() == Some("1");
            let staging = if staging_on {
                Some(PinnedBuffer::new(blob_bytes)?)
            } else {
                None
            };

            // Per-rail bounce + shared staging MRs, both LOCAL_WRITE only
            // (`remote_read == false`, invariant — we WRITE from and READ into them).
            let mut parts: Vec<(PinnedBuffer, u32, u32)> = Vec::with_capacity(n_rails);
            for rail in &mut rs.rails {
                let bounce = PinnedBuffer::new(blob_bytes)?;
                // SAFETY: bounce lives as long as the rail (and thus the MR).
                let keys = unsafe { rail.verbs.reg_mr(bounce.ptr, blob_bytes, false)? };
                // SAFETY: staging outlives the rail (both live in ArenaInner,
                // dropped together).
                let staging_lkey = match &staging {
                    Some(s) => unsafe { rail.verbs.reg_mr(s.ptr, blob_bytes, false)?.lkey },
                    None => 0,
                };
                parts.push((bounce, keys.lkey, staging_lkey));
            }

            // Peer's per-rail QP + shared arena base/rkey, client params, connect,
            // ack. Shared base: keep the LAST (pre-RailSet loop-overwrite behavior).
            let server = rs.finish_rw(&mut stream, "snapshot peer")?;
            let base = server.last().map(|sp| sp.base_addr).unwrap_or(0);
            let rails: Vec<SnapRail> = rs
                .into_verbs()
                .into_iter()
                .zip(parts)
                .zip(&server)
                .map(|((verbs, (bounce, lkey, staging_lkey)), sp)| SnapRail {
                    verbs,
                    bounce,
                    lkey,
                    remote_rkey: sp.rkey,
                    staging_lkey,
                })
                .collect();
            tracing::info!(
                "RdmaSnapshotArena connected to {addr}: {:.1} GiB arena, {n_rails} rail(s), blob {blob_bytes} B",
                arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            Ok(Self {
                inner: Mutex::new(ArenaInner {
                    rails,
                    staging,
                    rr: 0,
                    next_wr: 1, // 0 == "no completion yet" sentinel in the poll loop
                    stream,
                }),
                remote_base: base,
                blob_bytes,
            })
        }

        #[inline]
        pub fn blob_bytes(&self) -> usize {
            self.blob_bytes
        }

        /// RDMA-WRITE one `blob_bytes` blob to `base + offset`, drained to completion.
        pub fn write(&self, offset: u64, bytes: &[u8]) -> Result<()> {
            if bytes.len() != self.blob_bytes {
                bail!(
                    "snapshot write: {} != blob_bytes {}",
                    bytes.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            self.rdma_write_locked(&mut g, self.remote_base + offset, bytes)
        }

        /// RDMA-READ one `blob_bytes` blob from `base + offset` into `out`, drained.
        pub fn read(&self, offset: u64, out: &mut [u8]) -> Result<()> {
            if out.len() != self.blob_bytes {
                bail!(
                    "snapshot read: {} != blob_bytes {}",
                    out.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            self.rdma_read_locked(&mut g, self.remote_base + offset, out)
        }

        /// Pick a rail (round-robin) and a fresh wr id.
        fn rail_and_wr(g: &mut ArenaInner) -> (usize, u64) {
            let n = g.rails.len();
            let ri = g.rr % n;
            g.rr = g.rr.wrapping_add(1);
            let wr = g.next_wr;
            g.next_wr = g.next_wr.wrapping_add(1).max(1);
            (ri, wr)
        }

        fn rdma_write_locked(&self, g: &mut ArenaInner, raddr: u64, bytes: &[u8]) -> Result<()> {
            if g.staging.is_some() {
                return self.rdma_staged(g, raddr, Some(bytes), None);
            }
            let (ri, wr) = Self::rail_and_wr(g);
            let rail = &mut g.rails[ri];
            // SAFETY: bounce is a live registered MR of blob_bytes; copy the blob in,
            // RDMA-WRITE it to the peer arena, drain the single completion.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    rail.bounce.ptr as *mut u8,
                    self.blob_bytes,
                );
                rail.verbs.post_write(
                    rail.bounce.ptr,
                    rail.lkey,
                    raddr,
                    rail.remote_rkey,
                    self.blob_bytes as u32,
                    wr,
                )?;
            }
            while rail.verbs.poll()? != wr {}
            Ok(())
        }

        fn rdma_read_locked(&self, g: &mut ArenaInner, raddr: u64, out: &mut [u8]) -> Result<()> {
            if g.staging.is_some() {
                return self.rdma_staged(g, raddr, None, Some(out));
            }
            let (ri, wr) = Self::rail_and_wr(g);
            let rail = &mut g.rails[ri];
            // SAFETY: read into the live bounce MR, drain, then copy host-side to out.
            unsafe {
                rail.verbs.post_read(
                    rail.bounce.ptr,
                    rail.lkey,
                    raddr,
                    rail.remote_rkey,
                    self.blob_bytes as u32,
                    wr,
                )?;
            }
            while rail.verbs.poll()? != wr {}
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rail.bounce.ptr as *const u8,
                    out.as_mut_ptr(),
                    self.blob_bytes,
                );
            }
            Ok(())
        }

        /// Striped, pipelined, dual-rail transfer of ONE blob through the shared
        /// contiguous staging buffer (ATLAS_SSM_STAGING). `write_src` set = WRITE
        /// (memcpy in, stripe out); `read_dst` set = READ (stripe in, memcpy out).
        /// Chunks round-robin across rails, ≤ depth in-flight per rail (bounds the
        /// send queue). Because every rail registers the SAME staging buffer, chunk
        /// j lands at its true offset regardless of rail → one memcpy reassembles.
        fn rdma_staged(
            &self,
            g: &mut ArenaInner,
            raddr: u64,
            write_src: Option<&[u8]>,
            read_dst: Option<&mut [u8]>,
        ) -> Result<()> {
            let staging = g.staging.as_ref().expect("staging present").ptr as *mut u8;
            let is_read = read_dst.is_some();
            if let Some(src) = write_src {
                // WRITE: assemble the whole blob into staging first.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), staging, self.blob_bytes) };
            }
            let n = g.rails.len();
            let chunk = crate::snapshot_swap::staging_chunk_bytes();
            let depth = crate::snapshot_swap::staging_depth();
            let plan = crate::snapshot_swap::stripe_plan(self.blob_bytes, chunk, n);
            let total: usize = plan.iter().map(|w| w.len()).sum();
            let mut posted = vec![0usize; n];
            let mut reaped = vec![0usize; n];
            let mut done = 0usize;
            while done < total {
                // Post up to `depth` in-flight per rail (bounds each SQ).
                for ri in 0..n {
                    while posted[ri] < plan[ri].len() && (posted[ri] - reaped[ri]) < depth {
                        let (off, len) = plan[ri][posted[ri]];
                        let wr = g.next_wr;
                        g.next_wr = g.next_wr.wrapping_add(1).max(1);
                        let rail = &mut g.rails[ri];
                        let local = unsafe { staging.add(off) } as *mut _;
                        let raddr_chunk = raddr + off as u64;
                        unsafe {
                            if is_read {
                                rail.verbs.post_read(
                                    local,
                                    rail.staging_lkey,
                                    raddr_chunk,
                                    rail.remote_rkey,
                                    len as u32,
                                    wr,
                                )?;
                            } else {
                                rail.verbs.post_write(
                                    local,
                                    rail.staging_lkey,
                                    raddr_chunk,
                                    rail.remote_rkey,
                                    len as u32,
                                    wr,
                                )?;
                            }
                        }
                        posted[ri] += 1;
                    }
                }
                // Reap one completion from each rail with work outstanding.
                for ri in 0..n {
                    if posted[ri] > reaped[ri] {
                        g.rails[ri].verbs.poll()?;
                        reaped[ri] += 1;
                        done += 1;
                    }
                }
            }
            if let Some(dst) = read_dst {
                // READ: one memcpy reassembles (every chunk is at its true offset).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        staging as *const u8,
                        dst.as_mut_ptr(),
                        self.blob_bytes,
                    )
                };
            }
            Ok(())
        }

        // ─────────────────────────── paging data path ───────────────────────
        // The peer owns residency; we ALLOC a slot (control), RDMA-WRITE the blob,
        // then COMMIT — all under one lock so the peer's single-threaded per-conn
        // request order is respected. GET faults from the peer's NVMe swap if needed.

        /// PUT `key`'s blob into the tier. Never "full" — the peer spills to NVMe.
        pub fn paging_put(&self, key: u64, bytes: &[u8]) -> Result<()> {
            if bytes.len() != self.blob_bytes {
                bail!(
                    "paging_put: {} != blob_bytes {}",
                    bytes.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            let off = crate::snapshot_swap::client_alloc(&mut g.stream, key)?;
            self.rdma_write_locked(&mut g, self.remote_base + off, bytes)?;
            crate::snapshot_swap::client_commit(&mut g.stream, key)
        }

        /// GET `key`'s blob into `out`. `Ok(false)` = the peer has no such key.
        pub fn paging_get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
            if out.len() != self.blob_bytes {
                bail!(
                    "paging_get: {} != blob_bytes {}",
                    out.len(),
                    self.blob_bytes
                );
            }
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            match crate::snapshot_swap::client_get(&mut g.stream, key)? {
                Some(off) => {
                    self.rdma_read_locked(&mut g, self.remote_base + off, out)?;
                    Ok(true)
                }
                None => Ok(false),
            }
        }

        /// Drop `key` from the peer cache.
        pub fn paging_remove(&self, key: u64) -> Result<()> {
            let mut g = self.inner.lock().expect("snapshot arena mutex");
            crate::snapshot_swap::client_remove(&mut g.stream, key)
        }
    }
}
