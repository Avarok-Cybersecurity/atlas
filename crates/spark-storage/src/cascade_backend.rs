// SPDX-License-Identifier: AGPL-3.0-only
//
// CascadeBackend — a T1 local pinned-LPDDR write-back cache in front of any
// `StorageBackend` backing tier (the RDMA peer, or the SSD io_uring backend).
//
// The KV cache overflow already spills to `backing`; this inserts the handoff's
// missing middle tier: hot groups live in a bounded pinned-host cache (fast,
// GPU-addressable, no RDMA), and only evicted groups flush DOWN to `backing`.
// A restore hits T1 (a local copy_h2d) or falls through to `backing`. Purely a
// placement layer — no tier transforms bytes, so the composite is bit-identical
// to the backing alone (the group-id -> address bijection is the same on every
// tier). Enabled by $ATLAS_KV_LOCAL_GB; 0 (default) leaves the path untouched.

use std::ffi::c_void;

use anyhow::{Context, Result};

use crate::backend::{ReadRequest, StorageBackend};
use crate::cascade_policy::SlotCache;
use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};

/// The T1 byte store: one pinned buffer of `cap_slots * group_bytes`, slot `i`
/// at `ptr + i*group_bytes`. Pinned so the flush-read is a plain host slice and
/// the restore copy_h2d is fast (same LPDDR is GPU-addressable on GB10).
struct PinnedStore {
    buf: PinnedBuffer,
    group_bytes: usize,
}

impl PinnedStore {
    fn new(cap_slots: u32, group_bytes: usize) -> Result<Self> {
        let bytes = (cap_slots as usize)
            .checked_mul(group_bytes)
            .context("CascadeBackend T1 size overflow")?;
        Ok(Self {
            buf: PinnedBuffer::new(bytes).context("alloc T1 pinned store")?,
            group_bytes,
        })
    }
    #[inline]
    fn slot_host_ptr(&self, slot: u32) -> *const c_void {
        // SAFETY: slot < cap_slots (SlotCache invariant); offset within the buf.
        unsafe {
            (self.buf.ptr as *const u8).add(slot as usize * self.group_bytes) as *const c_void
        }
    }
    /// Copy the group bytes for `slot` out into a fresh Vec (releases the borrow
    /// so the flush can call `&mut backing`).
    fn slot_bytes(&self, slot: u32) -> Vec<u8> {
        // SAFETY: as above; the slot holds `group_bytes` valid bytes.
        unsafe {
            std::slice::from_raw_parts(
                (self.buf.ptr as *const u8).add(slot as usize * self.group_bytes),
                self.group_bytes,
            )
            .to_vec()
        }
    }
    fn write_slot(&mut self, slot: u32, src: &[u8]) {
        debug_assert_eq!(src.len(), self.group_bytes);
        // SAFETY: slot in range; src is exactly group_bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                (self.buf.ptr as *mut u8).add(slot as usize * self.group_bytes),
                self.group_bytes,
            );
        }
    }
}

pub struct CascadeBackend {
    hot: SlotCache,
    store: PinnedStore,
    backing: Box<dyn StorageBackend>,
    group_bytes: usize,
    /// #11-refinement: last async T1 hit-copy event. `read_async` records it
    /// after its hit `copy_h2d`s (which read the pinned slots); a subsequent
    /// eviction `write_slot` over one of those slots `sync`s it first, so the
    /// host cannot overwrite a slot a still-in-flight hit-copy is reading.
    /// `None` on the sync path → byte-identical for prefetch-OFF.
    last_read_event: Option<CudaEvent>,
}

// Single-owner rationale identical to RdmaKvBackend: both trait methods take
// `&mut self`, no `&self` method touches shared state, and HighSpeedSwap owns it
// single-threaded. The pinned store's raw ptr is only used under `&mut self`.
unsafe impl Sync for CascadeBackend {}

impl CascadeBackend {
    pub fn new(
        backing: Box<dyn StorageBackend>,
        layout: GroupLayout,
        cap_slots: u32,
    ) -> Result<Self> {
        let group_bytes = layout.group_bytes() as usize;
        tracing::info!(
            "high-speed-swap: T1 cascade cache = {cap_slots} slots × {group_bytes} B = {:.1} GiB local pinned RAM, backing below",
            (cap_slots as f64 * group_bytes as f64) / (1024.0 * 1024.0 * 1024.0),
        );
        Ok(Self {
            hot: SlotCache::new(cap_slots),
            store: PinnedStore::new(cap_slots, group_bytes)?,
            backing,
            group_bytes,
            last_read_event: None,
        })
    }

    /// Flush every resident T1 group down to backing (durability on teardown).
    fn flush_all(&mut self) -> Result<()> {
        for (key, slot) in self.hot.residents() {
            let bytes = self.store.slot_bytes(slot);
            self.backing.write_from_host(key, &bytes)?;
        }
        Ok(())
    }

    /// Body shared by `read` (sync) and `read_async` (#11-refinement). The T1
    /// hit copies pipeline identically; only the tail differs:
    ///   * sync  (`false`): forward misses via `backing.read`, terminal
    ///     `stream_sync`. Byte-identical to the pre-refinement `read`.
    ///   * async (`true`): forward misses via `backing.read_async` (no terminal
    ///     host sync there either), record `last_read_event` after the hit
    ///     copies so a later eviction `write_slot` can guard against them, and
    ///     OMIT the terminal `stream_sync`. Mirror-RAW for the hits is closed by
    ///     the HSS `kv_prefetch_done` (the hit `copy_h2d`s are enqueued on
    ///     `prefetch_stream` before HSS records that event).
    fn read_common(&mut self, requests: &[ReadRequest], stream: u64, is_async: bool) -> Result<()> {
        let keys: Vec<GroupKey> = requests.iter().map(|r| r.group).collect();
        let (hits, misses) = self.hot.plan_read(&keys);
        // T1 hits: local copy_h2d straight from the pinned slot into HBM.
        for (i, slot) in &hits {
            let src = self.store.slot_host_ptr(*slot);
            copy_h_to_d_async(requests[*i].dst_dev_ptr, src, self.group_bytes, stream)?;
            self.hot.touch(*slot);
        }
        // Async: record an event AFTER the hit copies read the pinned slots, so a
        // subsequent eviction write_slot over one of those slots waits it out.
        if is_async && !hits.is_empty() {
            let ev = CudaEvent::new()?;
            ev.record(stream)?;
            self.last_read_event = Some(ev);
        }
        // Misses fall through to backing (peer RDMA or SSD). Non-promoting: a
        // miss is NOT pulled up into T1 (smaller correctness surface; write-back
        // populates T1). backing.read syncs the stream for its own dsts; the
        // trailing stream_sync also covers the hit copy_h2d above.
        if !misses.is_empty() {
            let miss_reqs: Vec<ReadRequest> = misses.iter().map(|&i| requests[i]).collect();
            if is_async {
                self.backing.read_async(&miss_reqs, stream)?;
            } else {
                self.backing.read(&miss_reqs, stream)?;
            }
        }
        if !is_async {
            stream_sync(stream)?;
        }
        Ok(())
    }
}

impl StorageBackend for CascadeBackend {
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let plan = self.hot.plan_write(key);
        // Evicted a resident group → flush its (still-in-slot) bytes DOWN first.
        if let Some((victim_key, victim_slot)) = plan.flush_victim {
            let victim_bytes = self.store.slot_bytes(victim_slot);
            self.backing
                .write_from_host(victim_key, &victim_bytes)
                .context("cascade: flush T1 victim to backing")?;
        }
        // #11-refinement: an eviction is about to overwrite this slot in place.
        // If a prior async hit-copy is still reading it (recorded in read_async),
        // wait it out first — else the host `write_slot` below corrupts the bytes
        // the copy engine is mid-read. No-op on the sync path (`None`) →
        // byte-identical; when it fires it is on the offload/eviction path, never
        // the decode run-ahead loop, so it never stalls decode.
        if let Some(ev) = self.last_read_event.take() {
            ev.sync()?;
        }
        // Then cache the new group in T1 (write-back — lives here until evicted).
        self.store.write_slot(plan.slot, src);
        Ok(())
    }

    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, false)
    }

    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read_common(requests, stream, true)
    }

    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        // Forward to backing so RDMA zero-copy restore of MISSES still lands
        // directly into the UMA pool. (T1 hits copy_h2d locally regardless.)
        self.backing.register_landing_region(base, len)
    }

    fn group_layout(&self) -> GroupLayout {
        // Geometry is the backing's (the group-id ↔ address bijection is the
        // same on every tier). Cascade inherits the DEFAULT block read/write
        // (per-head fan-out through its own read/write_from_host = T1 caching) —
        // correct, just un-coalesced. Native T1 block coalescing is a follow-up.
        self.backing.group_layout()
    }
}

impl Drop for CascadeBackend {
    fn drop(&mut self) {
        // Durability: push any T1-resident groups down to backing before the
        // pinned store frees. Best-effort on teardown.
        let _ = self.flush_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::PosixBackend;
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h_async, stream_sync};
    use crate::group::KvKind;
    use crate::layout::Layout;

    fn tempdir(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("atlas-cascade-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// #11-refinement staging-reuse gate for the cascade T1 tier: an eviction
    /// `write_slot` must not overwrite a pinned slot that a still-in-flight async
    /// hit-copy (`read_async`) is reading — `last_read_event.sync` guards it.
    /// We size T1 to 2 slots over a 4-group set (forcing eviction), do a
    /// `read_async` that hits both resident slots, then `write_from_host` a new
    /// group whose eviction victim is one of those just-read slots, and finally
    /// verify EVERY group reads back correctly (through T1 hits, evicted-then-
    /// flushed backing reads, and a sync/async parity leg). A missing guard
    /// would corrupt the victim's flushed-to-backing bytes silently.
    #[test]
    #[ignore = "requires GPU"]
    fn read_async_eviction_no_corruption() {
        let _ctx = CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("evict");
        // 4 blocks, T1 caps at 2 slots → writing block 2/3 evicts 0/1.
        let spec = GroupLayout::new(1, 4, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let bytes = spec.group_bytes() as usize;
        let backing = Box::new(PosixBackend::new(layout).unwrap());
        let mut cascade = CascadeBackend::new(backing, spec, 2).unwrap();

        let keys: Vec<GroupKey> = (0..4u32)
            .map(|b| GroupKey::new(0, b, 0, KvKind::K))
            .collect();
        let pat = |b: usize| -> Vec<u8> {
            (0..bytes)
                .map(|i| ((i * 7 + b * 11) & 0xFF) as u8)
                .collect()
        };

        // Cache g0,g1 in T1 (fits in the 2 slots).
        cascade.write_from_host(keys[0], &pat(0)).unwrap();
        cascade.write_from_host(keys[1], &pat(1)).unwrap();

        // Async restore g0,g1 → both T1 hits; records last_read_event over the
        // slots holding g0,g1.
        let d0 = DeviceBuffer::new(bytes).unwrap();
        let d1 = DeviceBuffer::new(bytes).unwrap();
        cascade
            .read_async(
                &[
                    ReadRequest {
                        group: keys[0],
                        dst_dev_ptr: d0.ptr,
                    },
                    ReadRequest {
                        group: keys[1],
                        dst_dev_ptr: d1.ptr,
                    },
                ],
                _ctx.stream,
            )
            .unwrap();
        // Evict: caching g2,g3 flushes g0,g1's slots DOWN to backing. Each
        // write_from_host must sync the in-flight hit-copy before write_slot.
        cascade.write_from_host(keys[2], &pat(2)).unwrap();
        cascade.write_from_host(keys[3], &pat(3)).unwrap();
        // The async reads' consumer would device-wait kv_prefetch_done; here we
        // host-sync to read device memory back.
        stream_sync(_ctx.stream).unwrap();
        let readback = |d: &DeviceBuffer, want: &[u8]| {
            let mut got = vec![0u8; bytes];
            copy_d_to_h_async(got.as_mut_ptr() as *mut c_void, d.ptr, bytes, _ctx.stream).unwrap();
            stream_sync(_ctx.stream).unwrap();
            assert_eq!(&got, want, "cascade async restore corrupted");
        };
        readback(&d0, &pat(0));
        readback(&d1, &pat(1));

        // Now g0,g1 live in backing (evicted); g2,g3 in T1. A mixed sync read
        // (hits g2/g3, misses g0/g1 → backing) must return every original byte.
        let devs: Vec<DeviceBuffer> = (0..4).map(|_| DeviceBuffer::new(bytes).unwrap()).collect();
        let reqs: Vec<ReadRequest> = keys
            .iter()
            .zip(&devs)
            .map(|(k, d)| ReadRequest {
                group: *k,
                dst_dev_ptr: d.ptr,
            })
            .collect();
        cascade.read(&reqs, _ctx.stream).unwrap();
        for (b, d) in devs.iter().enumerate() {
            readback(d, &pat(b));
        }
        drop(cascade);
        std::fs::remove_dir_all(&dir).ok();
    }
}
