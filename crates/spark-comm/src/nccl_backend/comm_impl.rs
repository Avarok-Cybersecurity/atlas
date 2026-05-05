// SPDX-License-Identifier: AGPL-3.0-only

//! `impl CommBackend for NcclBackend` — collective operations.
//!
//! See `crate::nccl_backend` module-level docs for the global SAFETY
//! contract that applies to every `unsafe { nccl*(...) }` call here:
//! comm/buffers/sizes/streams come from valid prior allocations on this
//! rank's device, and the `extern "C"` ABI matches NCCL 2.28+.

use anyhow::Result;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::{COLLECTIVE_TIMEOUT_SECS, NcclBackend};
use crate::CommBackend;
use crate::nccl::{self, NcclDataType, NcclRedOp};

impl CommBackend for NcclBackend {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()> {
        if self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0 {
            return self.all_reduce_2rank(ptr, bytes, self.legacy_stream);
        }
        let count = bytes / 2;
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        if self.world_size == 2 && self.add_kernel.load(Ordering::Relaxed) != 0 {
            // Use event-based async with 2-rank send/recv path.
            nccl::record_event(self.compute_done_event, compute_stream)?;
            nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

            self.all_reduce_2rank(ptr, bytes, self.comm_stream)?;

            nccl::record_event(self.comm_done_event, self.comm_stream)?;
            nccl::stream_wait_event(compute_stream, self.comm_done_event)?;
            return Ok(());
        }

        let count = bytes / 2;
        let comm = *self.comm.lock();

        // 1. Record "MoE compute done" on compute stream
        nccl::record_event(self.compute_done_event, compute_stream)?;

        // 2. Comm stream waits for compute to finish
        nccl::stream_wait_event(self.comm_stream, self.compute_done_event)?;

        // 3. Launch all_reduce on dedicated comm stream
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr as *const _,
                ptr as *mut _,
                count,
                NcclDataType::Bfloat16,
                NcclRedOp::Sum,
                comm,
                self.comm_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllReduce (async)")?;

        // 4. Record "all_reduce done" on comm stream
        nccl::record_event(self.comm_done_event, self.comm_stream)?;

        // 5. Compute stream waits for all_reduce before residual_add
        nccl::stream_wait_event(compute_stream, self.comm_done_event)?;

        // Check for async errors (non-blocking, just flags unhealthy).
        self.check_async_error(comm);

        Ok(())
    }

    fn register_buffer(&self, ptr: u64, bytes: usize) -> Result<u64> {
        let mut handle: *mut c_void = ptr::null_mut();
        let comm = *self.comm.lock();
        let result =
            unsafe { nccl::ncclCommRegister(comm, ptr as *mut c_void, bytes, &mut handle) };
        nccl::check_nccl(result, "ncclCommRegister")?;
        self.registered_handles.lock().push(handle);
        Ok(handle as u64)
    }

    fn deregister_buffer(&self, handle: u64) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe { nccl::ncclCommDeregister(comm, handle as *mut c_void) };
        nccl::check_nccl(result, "ncclCommDeregister")
    }

    fn symmetric_alloc(&self, bytes: usize) -> Result<u64> {
        // Safety: matched by the caller's symmetric_free; both go through
        // ncclMemAlloc/ncclMemFree on the live communicator's allocator.
        let ptr = unsafe { nccl::nccl_mem_alloc(bytes) }?;
        Ok(ptr as u64)
    }

    fn symmetric_free(&self, ptr: u64) -> Result<()> {
        // Safety: caller guarantees `ptr` was returned by `symmetric_alloc`.
        unsafe { nccl::nccl_mem_free(ptr as *mut c_void) }
    }

    fn set_add_kernel(&self, handle: u64) {
        self.add_kernel.store(handle, Ordering::Relaxed);
        tracing::info!(
            "NCCL backend: bf16_add_inplace kernel set \
             (2-rank send/recv enabled)"
        );
    }

    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllGather(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclAllGather")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclReduceScatter(
                send_ptr as *const c_void,
                recv_ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclReduceScatter")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()> {
        let start = Instant::now();
        let comm = *self.comm.lock();

        // Broadcast raw bytes as Uint8
        let result = unsafe {
            nccl::ncclBroadcast(
                ptr as *const _,
                ptr as *mut _,
                bytes,
                NcclDataType::Uint8,
                root as i32,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "ncclBroadcast")?;

        // Synchronize to measure wall-clock time for timeout detection.
        nccl::sync_stream(self.legacy_stream)?;

        let elapsed = start.elapsed();
        if elapsed.as_secs() >= COLLECTIVE_TIMEOUT_SECS {
            tracing::error!(
                "NCCL broadcast took {:.1}s (threshold: {}s) \
                 — marking communicator unhealthy",
                elapsed.as_secs_f64(),
                COLLECTIVE_TIMEOUT_SECS,
            );
            self.unhealthy.store(true, Ordering::Release);
        }

        // Also check for async errors.
        self.check_async_error(comm);

        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclAllReduce(
                ptr::null(),
                ptr::null_mut(),
                0,
                NcclDataType::Float32,
                NcclRedOp::Sum,
                comm,
                self.legacy_stream,
            )
        };
        nccl::check_nccl(result, "barrier (ncclAllReduce count=0)")?;
        self.check_async_error(comm);
        Ok(())
    }

    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                bytes,
                NcclDataType::Uint8,
                dest_rank as i32,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend (send_to)")
    }

    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()> {
        let comm = *self.comm.lock();
        let result = unsafe {
            nccl::ncclRecv(
                ptr as *mut c_void,
                bytes,
                NcclDataType::Uint8,
                src_rank as i32,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv (recv_from)")
    }

    fn group_start(&self) -> Result<()> {
        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")
    }

    fn group_end(&self) -> Result<()> {
        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")
    }

    fn is_healthy(&self) -> bool {
        if self.unhealthy.load(Ordering::Acquire) {
            return false;
        }
        // Actively probe the communicator for async errors.
        let comm = *self.comm.lock();
        self.check_async_error(comm)
    }

    fn attempt_reconnect(&self) -> Result<()> {
        self.reconnect_inner()
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_collective_timeout_constant() {
        // Sanity: timeout should be a reasonable value (not 0, not too large).
        assert!(COLLECTIVE_TIMEOUT_SECS >= 10);
        assert!(COLLECTIVE_TIMEOUT_SECS <= 300);
    }

    #[test]
    fn test_recv_buffer_size_sufficient() {
        use crate::nccl_backend::RECV_BUFFER_SIZE;
        // Verify buffer covers all use cases including large models.
        let single_decode = 6144 * 2; // 12 KB (122B hidden)
        let k3_verify = 6144 * 3 * 2; // 36 KB
        let prefill_chunk = 6144 * 4096 * 2; // 48 MB (122B, 4096 chunk)
        assert!(RECV_BUFFER_SIZE >= single_decode);
        assert!(RECV_BUFFER_SIZE >= k3_verify);
        assert!(RECV_BUFFER_SIZE >= prefill_chunk);
    }
}
