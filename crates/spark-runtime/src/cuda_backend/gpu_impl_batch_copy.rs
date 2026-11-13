// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache csrc/cuda/mp_mem_kernels.cu:358-849 (commit 3aefae1, Apache-2.0): one cuMemcpyBatchAsync per transfer, the CUDA 12.8 versus 13.x signature split and the support probe; no code copied.
// Design after vLLM csrc/cache_kernels.cu:110-190 (https://github.com/vllm-project/vllm, commit 8a36f08, Apache-2.0): the entry point resolved through cuGetProcAddress, the legacy-stream rejection and the per-copy fallback; no code copied.

//! Responsibility: batched async host/device copies (`cuMemcpyBatchAsync`, with a
//! per-copy fallback), non-blocking event polling, timing events (L06's device phase
//! timer) and host-memory registration for [`AvarokCudaBackend`] (port sprint,
//! contract 3.7.1). Inherent `_cu` bodies; the `GpuBackend` impl in `gpu_impl.rs`
//! delegates one line to each.
//!
//! **When the batch path runs.** Only when the entry point resolved (`port_ffi`), the
//! one-time probe passed (a scrambled round trip compared byte for byte), the stream
//! is a real stream (the driver rejects the legacy NULL stream) and the stream is not
//! capturing (inside a capture the per-copy form records ordinary memcpy nodes).
//! The first failing batch call latches the path off for the process and that call's
//! copies are re-issued one by one: re-issuing a copy writes the same bytes, so a
//! batch the driver half-enqueued before failing is harmless.
//!
//! **The per-copy path** issues `cuMemcpyDtoHAsync_v2` / `cuMemcpyHtoDAsync_v2` per
//! entry on the same stream and adds no synchronization: a batch's host ranges are
//! caller-kept by contract, exactly like `copy_h2d_async_retained`.
//!
//! The safety contract at the top of `gpu_impl.rs` applies to every driver call here.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use avarok_core::registry::cuda_error_text;

use super::port_ffi::{self, MemcpyAttributes, MemcpyBatch};
use super::{
    AvarokCudaBackend, cuEventCreate, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoHAsync_v2,
    cuMemcpyHtoDAsync_v2, cuStreamSynchronize,
};
use crate::gpu::{CopyEntry, DevicePtr, GpuBackend, port_ext};
use crate::pinned_hosts;

/// Set by the first failing `cuMemcpyBatchAsync`: every later batch takes the per-copy path.
static BATCH_OFF: AtomicBool = AtomicBool::new(false);
/// Set once the "registered memory takes the per-copy path" line was logged.
static REGISTERED_NOTED: AtomicBool = AtomicBool::new(false);
/// The probe's verdict, computed once on first use.
static BATCH_PROBE: OnceLock<bool> = OnceLock::new();

/// `CUDA_ERROR_NOT_READY`: `cuEventQuery`'s "still running".
const CUDA_ERROR_NOT_READY: i32 = 600;
/// `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`: page-locked for every
/// context, and mapped so `host_ptr_to_device` works on the region (zero-copy readers).
const HOST_REGISTER_FLAGS: u32 = 0x01 | 0x02;
/// Bytes per half of the probe's round trip.
const PROBE_HALF: usize = 2048;

/// Direction of one batch.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Dir {
    ToHost,
    ToDevice,
}

impl Dir {
    fn name(self) -> &'static str {
        match self {
            Dir::ToHost => "D2H",
            Dir::ToDevice => "H2D",
        }
    }
}

impl AvarokCudaBackend {
    /// `supports_copy_batch`: resolved, not latched off, and the probe passed.
    pub(super) fn supports_copy_batch_cu(&self) -> bool {
        if BATCH_OFF.load(Ordering::Relaxed) || port_ffi::driver().memcpy_batch.is_none() {
            return false;
        }
        *BATCH_PROBE.get_or_init(|| self.probe_copy_batch())
    }

    /// One batch of copies on `stream`; see the module doc for when the batch path runs.
    pub(super) fn copy_batch_cu(
        &self,
        entries: &[CopyEntry],
        stream: u64,
        dir: Dir,
        any_order: bool,
    ) -> Result<()> {
        port_ext::validate_entries(entries)?;
        if entries.iter().all(|e| e.bytes == 0) {
            return Ok(());
        }
        // 0 is the legacy NULL stream, 1 is `CU_STREAM_LEGACY`: both refused by the batch API.
        let real_stream = stream > 1;
        // `cuMemcpyBatchAsync` into or out of memory locked with `cuMemHostRegister` (the host
        // tier's pinned pool) raises CUDA_ERROR_ILLEGAL_ADDRESS asynchronously on WSL2
        // (driver 13030), whatever the register flags and copy attributes, in both directions
        // (`examples/batch_host_kind_probe.rs`; `cuMemAllocHost` memory and the per-copy path
        // are fine). The fault is sticky, so it cannot be probed in process: such batches
        // always take the per-copy path.
        let registered_host =
            pinned_hosts::any_tier_owned(entries.iter().map(|e| (e.host, e.bytes)));
        if registered_host && !REGISTERED_NOTED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                "batched copies touching cuMemHostRegister memory (the host KV tier) use one \
                 async copy per entry: cuMemcpyBatchAsync faults on that memory kind"
            );
        }
        if real_stream
            && !registered_host
            && self.supports_copy_batch_cu()
            && !GpuBackend::stream_is_capturing(self, stream)
        {
            let status = self.batch_submit(entries, stream, dir, any_order);
            if status == 0 {
                return Ok(());
            }
            latch_batch_off(status);
        }
        copy_each(entries, stream, dir)
    }

    /// Submit every non-empty entry in one driver call; returns the driver status.
    fn batch_submit(&self, entries: &[CopyEntry], stream: u64, dir: Dir, any_order: bool) -> i32 {
        let Some(batch) = port_ffi::driver().memcpy_batch else {
            return -1;
        };
        let live = entries.iter().filter(|e| e.bytes > 0);
        let n = live.clone().count();
        let mut dsts: Vec<u64> = Vec::with_capacity(n);
        let mut srcs: Vec<u64> = Vec::with_capacity(n);
        let mut sizes: Vec<usize> = Vec::with_capacity(n);
        for e in live {
            // A host address is a valid operand under unified addressing (every 64-bit
            // target Atlas runs on), which is how both upstreams pass it.
            let (dst, src) = match dir {
                Dir::ToHost => (e.host as u64, e.dev.0),
                Dir::ToDevice => (e.dev.0, e.host as u64),
            };
            dsts.push(dst);
            srcs.push(src);
            sizes.push(e.bytes);
        }
        let mut attrs = MemcpyAttributes {
            src_access_order: if any_order && dir == Dir::ToDevice {
                port_ffi::SRC_ACCESS_ORDER_ANY
            } else {
                port_ffi::SRC_ACCESS_ORDER_STREAM
            },
            ..Default::default()
        };
        // One attribute set, applying from copy 0 to the end.
        let mut attr_idx = [0usize];
        // SAFETY: the three arrays hold `n` entries each; `attrs`/`attr_idx` hold one;
        // every pointer was validated by `validate_entries` and is caller-kept until the
        // stream syncs (the batch contract).
        match batch {
            MemcpyBatch::V13(f) => unsafe {
                f(
                    dsts.as_mut_ptr(),
                    srcs.as_mut_ptr(),
                    sizes.as_mut_ptr(),
                    n,
                    &mut attrs,
                    attr_idx.as_mut_ptr(),
                    1,
                    stream,
                )
            },
            MemcpyBatch::V128(f) => {
                let mut fail_idx = 0usize;
                unsafe {
                    f(
                        dsts.as_mut_ptr(),
                        srcs.as_mut_ptr(),
                        sizes.as_mut_ptr(),
                        n,
                        &mut attrs,
                        attr_idx.as_mut_ptr(),
                        1,
                        &mut fail_idx,
                        stream,
                    )
                }
            }
        }
    }

    /// The probe: two scrambled halves up, the whole buffer down, on a private stream,
    /// compared byte for byte. Any driver error or mismatch reports the path unusable.
    fn probe_copy_batch(&self) -> bool {
        let verdict = self.probe_round_trip();
        let version = port_ffi::driver().driver_version;
        match &verdict {
            Ok(()) => tracing::info!(
                "cuMemcpyBatchAsync: probe passed (driver {version}); batched copies use it"
            ),
            Err(e) => tracing::info!(
                "cuMemcpyBatchAsync unusable (driver {version}): {e:#}; batched copies run \
                 one async copy per entry"
            ),
        }
        verdict.is_ok()
    }

    fn probe_round_trip(&self) -> Result<()> {
        let bytes = 2 * PROBE_HALF;
        let stream = self.create_stream_cu()?;
        let host = match self.alloc_host_pinned_cu(2 * bytes) {
            Ok(h) => h,
            Err(e) => {
                destroy_stream(stream);
                return Err(e);
            }
        };
        let mut dev: u64 = 0;
        let alloc_status = unsafe { cuMemAlloc_v2(&mut dev, bytes) };
        let result = if alloc_status != 0 {
            Err(anyhow::anyhow!(
                "cuMemAlloc_v2 failed: status {alloc_status}"
            ))
        } else {
            self.probe_on(host, DevicePtr(dev), stream)
        };
        if alloc_status == 0 {
            unsafe { cuMemFree_v2(dev) };
        }
        let freed = self.free_host_pinned_cu(host, 2 * bytes);
        destroy_stream(stream);
        result.and(freed)
    }

    fn probe_on(&self, host: *mut u8, dev: DevicePtr, stream: u64) -> Result<()> {
        let half = PROBE_HALF;
        let bytes = 2 * half;
        // SAFETY: `host` is a fresh page-locked region of `2 * bytes` bytes owned here.
        let (src, dst) = unsafe {
            let all = std::slice::from_raw_parts_mut(host, 2 * bytes);
            all.split_at_mut(bytes)
        };
        for (i, b) in src.iter_mut().enumerate() {
            *b = (i.wrapping_mul(131) ^ (i >> 7)) as u8 | 1;
        }
        let up = [
            CopyEntry {
                dev: dev.offset(half),
                host: src.as_ptr() as usize,
                bytes: half,
            },
            CopyEntry {
                dev,
                host: src[half..].as_ptr() as usize,
                bytes: half,
            },
        ];
        let down = [CopyEntry {
            dev,
            host: dst.as_mut_ptr() as usize,
            bytes,
        }];
        let s = self.batch_submit(&up, stream, Dir::ToDevice, false);
        if s != 0 {
            bail!("H2D batch: {}", cuda_error_text(s));
        }
        let s = self.batch_submit(&down, stream, Dir::ToHost, false);
        if s != 0 {
            bail!("D2H batch: {}", cuda_error_text(s));
        }
        let s = unsafe { cuStreamSynchronize(stream) };
        if s != 0 {
            bail!("probe sync: {}", cuda_error_text(s));
        }
        // The halves went up swapped, so the device holds [src[half..], src[..half]].
        if dst[..half] != src[half..] || dst[half..] != src[..half] {
            bail!("round trip returned different bytes");
        }
        Ok(())
    }

    /// `event_query`: `cuEventQuery`, or (entry point unavailable) a wait plus `true`.
    pub(super) fn event_query_cu(&self, event: u64) -> Result<bool> {
        let Some(query) = port_ffi::driver().event_query else {
            self.event_synchronize_cu(event)?;
            return Ok(true);
        };
        match unsafe { query(event) } {
            0 => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            s => bail!("cuEventQuery failed: {}", cuda_error_text(s)),
        }
    }

    /// `create_timing_event`: an event WITH timing (`cuEventCreate` flags 0). The
    /// `create_event` flavor passes `CU_EVENT_DISABLE_TIMING`, which
    /// `cuEventElapsedTime` rejects.
    pub(super) fn create_timing_event_cu(&self) -> Result<u64> {
        let mut event: u64 = 0;
        let status = unsafe { cuEventCreate(&mut event, 0x00) };
        if status != 0 {
            bail!("cuEventCreate (timing) failed: {}", cuda_error_text(status));
        }
        Ok(event)
    }

    /// `event_elapsed_ms`: never blocks. `Ok(None)` while either event is incomplete.
    pub(super) fn event_elapsed_ms_cu(&self, start: u64, end: u64) -> Result<Option<f32>> {
        let Some(elapsed) = port_ffi::driver().event_elapsed else {
            bail!("event_elapsed_ms: cuEventElapsedTime is not available from this driver");
        };
        let mut ms: f32 = 0.0;
        match unsafe { elapsed(&mut ms, start, end) } {
            0 => Ok(Some(ms)),
            CUDA_ERROR_NOT_READY => Ok(None),
            s => bail!("cuEventElapsedTime failed: {}", cuda_error_text(s)),
        }
    }

    /// `host_register`: `cuMemHostRegister`, then a tier-owned entry in `pinned_hosts`.
    pub(super) fn host_register_cu(&self, ptr: usize, bytes: usize) -> Result<()> {
        port_ext::check_register_range(ptr, bytes)?;
        let Some(register) = port_ffi::driver().host_register else {
            bail!("host_register: cuMemHostRegister is not available from this driver");
        };
        let status = unsafe { register(ptr as *mut c_void, bytes, HOST_REGISTER_FLAGS) };
        if status != 0 {
            bail!(
                "cuMemHostRegister of {bytes} bytes at {ptr:#x} failed: {}",
                cuda_error_text(status)
            );
        }
        // After the lock succeeded, so the registry never claims pages that are not locked.
        crate::pinned_hosts::register_tier(ptr as *const u8, bytes);
        Ok(())
    }

    /// `host_unregister`: forget the region, then `cuMemHostUnregister`.
    pub(super) fn host_unregister_cu(&self, ptr: usize) -> Result<()> {
        if ptr == 0 {
            bail!("host_unregister: null address");
        }
        let Some(unregister) = port_ffi::driver().host_unregister else {
            bail!("host_unregister: cuMemHostUnregister is not available from this driver");
        };
        // Before the unlock, as `free_host_pinned_cu` does, so a reused address is never
        // reported as still page-locked.
        crate::pinned_hosts::unregister(ptr as *const u8);
        let status = unsafe { unregister(ptr as *mut c_void) };
        // As for `cuMemFreeHost`: at exit the driver may have torn the context down first,
        // and the pages were unlocked with it.
        if status != 0 && !avarok_core::registry::is_teardown_noop(status) {
            bail!(
                "cuMemHostUnregister at {ptr:#x} failed: {}",
                cuda_error_text(status)
            );
        }
        Ok(())
    }
}

/// The per-copy path: one async copy per non-empty entry, in entry order, no sync.
fn copy_each(entries: &[CopyEntry], stream: u64, dir: Dir) -> Result<()> {
    for (i, e) in entries.iter().enumerate().filter(|(_, e)| e.bytes > 0) {
        let status = match dir {
            Dir::ToHost => unsafe {
                cuMemcpyDtoHAsync_v2(e.host as *mut c_void, e.dev.0, e.bytes, stream)
            },
            Dir::ToDevice => unsafe {
                cuMemcpyHtoDAsync_v2(e.dev.0, e.host as *const c_void, e.bytes, stream)
            },
        };
        if status != 0 {
            bail!(
                "batched {} copy, entry {i} ({} bytes): {}",
                dir.name(),
                e.bytes,
                cuda_error_text(status)
            );
        }
    }
    Ok(())
}

fn latch_batch_off(status: i32) {
    if !BATCH_OFF.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            "cuMemcpyBatchAsync failed ({}); batched copies use one async copy per entry \
             from now on",
            cuda_error_text(status)
        );
    }
}

/// Destroy a stream this module created, when the driver offers `cuStreamDestroy`
/// (otherwise the one probe stream lives as long as the context, like every stream
/// `create_stream` hands out).
fn destroy_stream(stream: u64) {
    if let Some(destroy) = port_ffi::driver().stream_destroy {
        unsafe { destroy(stream) };
    }
}
