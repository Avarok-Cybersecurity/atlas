// SPDX-License-Identifier: AGPL-3.0-only
//
// Per-rail state for the RDMA KV backend: the registered pinned bounce ring,
// the in-flight work-request map, and the rail's post/poll/drain loop. Split
// out of rdma_kv_backend.rs (was 715 LoC) to satisfy the CI 500-LoC cap — pure
// code motion, no behavior change. `InFlight` stays private here; only
// `Bounce` and `Rail` are needed by the parent module.

use std::collections::HashMap;
use std::ffi::c_void;

use anyhow::{Context, Result};

use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_async};
use atlas_rdma::verbs::Verbs;

/// One registered pinned bounce in a rail's pipeline ring.
pub(super) struct Bounce {
    pub(super) buf: PinnedBuffer,
    pub(super) lkey: u32,
    /// #11-refinement: async prefetch reuse guard. `read_async` records a CUDA
    /// event on the copy stream AFTER `copy_h_to_d_async` drains this bounce;
    /// the next reuse of this bounce `sync`s it first (via `wait_bounce_free`),
    /// so the host cannot re-fill the bounce before the device copy has read it —
    /// the reuse hazard the deleted terminal host stream_sync used to prevent.
    /// `None` on the synchronous path (that path keeps its terminal stream_sync),
    /// so the guard is a pure no-op there → byte-identical for prefetch-OFF.
    pub(super) copy_done: Option<CudaEvent>,
}

/// An in-flight RDMA op, keyed by its `wr_id`, so a completion can be dispatched.
pub(super) enum InFlight {
    /// A restore: after the READ lands, copy the bounce to this HBM dest.
    Read { bounce: usize, dst: u64 },
    /// An offload: the WRITE from this bounce; just free it on completion.
    Write { bounce: usize },
}

/// One QP on one CX7 adapter, with its own bounce ring + completion tracking.
pub(super) struct Rail {
    pub(super) verbs: Verbs,
    pub(super) remote_rkey: u32,
    pub(super) bounces: Vec<Bounce>,
    pub(super) free: std::collections::VecDeque<usize>,
    pub(super) inflight: HashMap<u64, InFlight>,
    pub(super) next_wr: u64,
    /// Zero-copy restore: lkeys of destination MRs registered on demand (a UMA
    /// dst is GPU-addressable, so RDMA lands there directly — no bounce, no
    /// copy_h2d). Cached by dst address; KV scratch slots are reused.
    pub(super) dst_lkeys: HashMap<u64, u32>,
    /// Pre-registered whole landing region `(base, len, lkey)`: one MR covering
    /// the entire (UMA) scratch pool. Any dst inside it reuses this lkey, so we
    /// never register per-slot sub-regions (which fail on GB10).
    pub(super) region: Option<(u64, u64, u32)>,
    /// In-flight direct (zero-copy) reads on this rail — no bounce to free.
    pub(super) direct_inflight: usize,
}

impl Rail {
    #[inline]
    pub(super) fn fresh_wr(&mut self) -> u64 {
        let w = self.next_wr;
        self.next_wr = self.next_wr.wrapping_add(1);
        w
    }

    /// Register (once, cached) a `bytes`-sized destination MR at `addr` for a
    /// zero-copy RDMA READ landing. On GB10 UMA the dst is GPU-addressable pinned
    /// host memory, so `ibv_reg_mr` on its VA succeeds and the GPU reads the
    /// landed bytes at the same address — no `copy_h2d`.
    /// Register `[base, base+len)` as ONE landing MR on this rail (the whole UMA
    /// scratch pool). Called once, before any restore.
    pub(super) fn register_region(&mut self, base: u64, len: usize) -> Result<()> {
        // SAFETY: base/len describe the pool's live UMA (pinned) allocation,
        // which outlives every rail (deregistered on drop before the pool frees).
        let keys = unsafe { self.verbs.reg_mr(base as *mut c_void, len, false) }
            .context("register UMA landing region")?;
        self.region = Some((base, len as u64, keys.lkey));
        Ok(())
    }

    pub(super) fn reg_dst(&mut self, addr: u64, bytes: usize) -> Result<u32> {
        // Whole-region fast path: any dst inside the pre-registered pool reuses
        // its single lkey (no per-slot registration — that fails on GB10).
        if let Some((base, len, lkey)) = self.region
            && addr >= base
            && addr + bytes as u64 <= base + len
        {
            return Ok(lkey);
        }
        if let Some(&lk) = self.dst_lkeys.get(&addr) {
            return Ok(lk);
        }
        // SAFETY: caller guarantees zero-copy mode => addr is a live UMA buffer
        // of at least `bytes` (else reg_mr fails, surfacing a clear error).
        let keys = unsafe { self.verbs.reg_mr(addr as *mut c_void, bytes, false) }
            .context("zero-copy restore needs a UMA (GPU-addressable) dst; reg_mr failed")?;
        self.dst_lkeys.insert(addr, keys.lkey);
        Ok(keys.lkey)
    }

    /// Reap exactly one completion on this rail, freeing its bounce. For a READ,
    /// first `copy_h2d` the landed bytes to its HBM dest on `stream`. When
    /// `track` is set (async prefetch), record a CUDA event on `stream` after
    /// that copy so a later reuse of this bounce can `sync` on the copy draining
    /// the bounce (replacing the deleted terminal host stream_sync). `track` is
    /// only ever true on `read_async`'s READ reaps; the sync path and all write
    /// drains pass false, leaving `copy_done = None` → byte-identical.
    pub(super) fn reap_one(&mut self, group_bytes: usize, stream: u64, track: bool) -> Result<()> {
        let wr = self.verbs.poll()?;
        let op = self
            .inflight
            .remove(&wr)
            .with_context(|| format!("kv: completion for unknown wr_id {wr:#x}"))?;
        let bounce = match op {
            InFlight::Read { bounce, dst } => {
                copy_h_to_d_async(
                    dst,
                    self.bounces[bounce].buf.ptr as *const _,
                    group_bytes,
                    stream,
                )?;
                if track {
                    let ev = CudaEvent::new()?;
                    ev.record(stream)?;
                    self.bounces[bounce].copy_done = Some(ev);
                }
                bounce
            }
            InFlight::Write { bounce } => bounce,
        };
        self.free.push_back(bounce);
        Ok(())
    }

    /// #11-refinement: before reusing bounce `b`, wait for any still-in-flight
    /// async copy_h2d that is draining it (recorded by a prior `read_async`
    /// reap). No-op on the sync path (`copy_done` is always `None`), so it never
    /// perturbs prefetch-OFF. Off the decode run-ahead loop it only ever fires
    /// under genuine copy-engine backpressure (correct async pushback).
    pub(super) fn wait_bounce_free(&mut self, b: usize) -> Result<()> {
        if let Some(ev) = self.bounces[b].copy_done.take() {
            ev.sync()?;
        }
        Ok(())
    }

    pub(super) fn drain(&mut self, group_bytes: usize, stream: u64) -> Result<()> {
        while !self.inflight.is_empty() {
            // Drained ops here are writes (before a read) or teardown reaps — no
            // new mirror-RAW consumer needs the copy event, so track=false.
            self.reap_one(group_bytes, stream, false)?;
        }
        Ok(())
    }

    /// # Safety: bounce/len/remote must describe a live MR and the peer arena.
    pub(super) unsafe fn post_read(
        &mut self,
        bounce: usize,
        raddr: u64,
        bytes: usize,
        dst: u64,
    ) -> Result<()> {
        let wr = self.fresh_wr();
        unsafe {
            self.verbs.post_read(
                self.bounces[bounce].buf.ptr,
                self.bounces[bounce].lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        self.inflight.insert(wr, InFlight::Read { bounce, dst });
        Ok(())
    }

    /// # Safety: as `post_read`; `src` bytes already copied into the bounce.
    pub(super) unsafe fn post_write(
        &mut self,
        bounce: usize,
        raddr: u64,
        bytes: usize,
    ) -> Result<()> {
        let wr = self.fresh_wr();
        unsafe {
            self.verbs.post_write(
                self.bounces[bounce].buf.ptr,
                self.bounces[bounce].lkey,
                raddr,
                self.remote_rkey,
                bytes as u32,
                wr,
            )?;
        }
        self.inflight.insert(wr, InFlight::Write { bounce });
        Ok(())
    }
}
