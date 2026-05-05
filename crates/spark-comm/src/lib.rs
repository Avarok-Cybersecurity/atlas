// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

//! Communication backend abstraction (SBIO IORouter for collective ops).
//!
//! All distributed communication flows through [`CommBackend`]. Business
//! logic never calls NCCL or MPI directly.
//!
//! - [`SingleGpuBackend`] — all ops are no-ops (single GPU).
//! - [`NcclBackend`] — real multi-GPU via NCCL (expert parallelism).

use anyhow::Result;

pub mod nccl;
pub mod nccl_backend;
pub use nccl_backend::NcclBackend;

/// Communication backend trait for distributed operations.
///
/// All collective operations take raw device pointer + byte count.
/// The pointer type is u64 (matching CUDA CUdeviceptr) to avoid
/// coupling this crate to spark-runtime's DevicePtr.
pub trait CommBackend: Send + Sync {
    /// All-reduce: sum across all ranks, result on all ranks.
    fn all_reduce(&self, ptr: u64, bytes: usize) -> Result<()>;

    /// All-gather: each rank contributes a chunk, all ranks get full buffer.
    fn all_gather(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// Reduce-scatter: reduce + scatter (inverse of all-gather).
    fn reduce_scatter(&self, send_ptr: u64, recv_ptr: u64, bytes: usize) -> Result<()>;

    /// Broadcast from root rank to all ranks.
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> Result<()>;

    /// Barrier: block until all ranks reach this point.
    fn barrier(&self) -> Result<()>;

    /// Async all-reduce using GPU-side event synchronization.
    ///
    /// Replaces `gpu.synchronize(stream) + all_reduce(ptr, bytes)`.
    /// Uses a dedicated comm stream + CUDA events so the CPU never blocks.
    /// `compute_stream` is where MoE kernels ran and where residual_add will run.
    fn all_reduce_async(&self, ptr: u64, bytes: usize, compute_stream: u64) -> Result<()> {
        let _ = compute_stream;
        self.all_reduce(ptr, bytes)
    }

    /// Pre-register a GPU buffer with the communication backend.
    ///
    /// For NCCL over IB/RoCE, this caches the IB memory registration
    /// (`ibv_reg_mr`), avoiding per-call overhead in all_reduce.
    /// Returns an opaque handle for deregistration.
    fn register_buffer(&self, _ptr: u64, _bytes: usize) -> Result<u64> {
        Ok(0)
    }

    /// Deregister a previously registered buffer.
    fn deregister_buffer(&self, _handle: u64) -> Result<()> {
        Ok(())
    }

    /// Allocate a GPU buffer in NCCL's symmetric-memory window
    /// (NCCL ≥ 2.28 / `ncclMemAlloc`). Returns the device pointer as `u64`.
    ///
    /// Symmetric-memory allocations are the substrate for:
    ///   1. Copy-engine offload of NVLink collectives (frees SMs for compute).
    ///   2. Device-side communication API (kernels invoke collectives in-kernel),
    ///      which TokenWeave-style fused AR+RMSNorm+Residual builds on.
    ///
    /// On Atlas's 2-rank Spark over RoCE, the copy-engine offload itself does
    /// not apply (RoCE is not NVLink), but the symmetric windows are still
    /// required for future device-API fusions and to reduce per-call setup.
    /// Returns an error if the linked NCCL is < 2.28; backends that don't
    /// support symmetric memory return `Err` and callers must fall back.
    fn symmetric_alloc(&self, _bytes: usize) -> Result<u64> {
        anyhow::bail!("symmetric_alloc not supported by this CommBackend");
    }

    /// Free a buffer previously returned by `symmetric_alloc`.
    fn symmetric_free(&self, _ptr: u64) -> Result<()> {
        anyhow::bail!("symmetric_free not supported by this CommBackend");
    }

    /// Provide a kernel handle for the BF16 in-place addition kernel.
    ///
    /// Used by the 2-rank send/recv all-reduce path. The kernel is loaded
    /// by the model layer (which has access to AtlasRegistry) and passed
    /// to the comm backend at init time.
    fn set_add_kernel(&self, _handle: u64) {
        // Default: no-op (single GPU or backends that don't need it)
    }

    /// Send tokens to a specific rank (for EP token dispatch).
    ///
    /// Sends `bytes` from `ptr` on this rank to `dest_rank`.
    /// Must be paired with a matching `recv_from` on the destination rank.
    /// `stream` is the CUDA stream on which the operation is enqueued.
    fn send_to(&self, ptr: u64, bytes: usize, dest_rank: usize, stream: u64) -> Result<()>;

    /// Receive tokens from a specific rank (for EP token combine).
    ///
    /// Receives `bytes` into `ptr` on this rank from `src_rank`.
    /// Must be paired with a matching `send_to` on the source rank.
    /// `stream` is the CUDA stream on which the operation is enqueued.
    fn recv_from(&self, ptr: u64, bytes: usize, src_rank: usize, stream: u64) -> Result<()>;

    /// Begin a group of point-to-point operations (send_to/recv_from).
    ///
    /// All send_to/recv_from calls between group_start and group_end are
    /// batched into a single NCCL launch for efficiency.
    fn group_start(&self) -> Result<()> {
        Ok(())
    }

    /// End a group of point-to-point operations.
    fn group_end(&self) -> Result<()> {
        Ok(())
    }

    /// Check if the communicator is healthy (no async errors, no timeouts).
    ///
    /// Returns `true` if the communicator is operational. Implementations
    /// may actively probe the underlying transport (e.g., `ncclCommGetAsyncError`).
    fn is_healthy(&self) -> bool {
        true
    }

    /// Attempt to recover a degraded communicator.
    ///
    /// For NCCL, this aborts the dead communicator and re-initializes
    /// via TCP bootstrap. Both ranks must call this concurrently.
    /// Returns `Ok(())` on successful recovery, `Err` if recovery failed.
    fn attempt_reconnect(&self) -> Result<()> {
        Ok(())
    }

    /// This rank's index (0-based).
    fn rank(&self) -> usize;

    /// Total number of ranks.
    fn world_size(&self) -> usize;
}

/// Single-GPU backend: all collective ops are no-ops.
///
/// Used in Phase 1 where the entire model fits on one GPU.
pub struct SingleGpuBackend;

impl CommBackend for SingleGpuBackend {
    fn all_reduce(&self, _ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn all_gather(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn reduce_scatter(&self, _send_ptr: u64, _recv_ptr: u64, _bytes: usize) -> Result<()> {
        Ok(())
    }

    fn broadcast(&self, _ptr: u64, _bytes: usize, _root: usize) -> Result<()> {
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }

    fn send_to(&self, _ptr: u64, _bytes: usize, _dest_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn recv_from(&self, _ptr: u64, _bytes: usize, _src_rank: usize, _stream: u64) -> Result<()> {
        Ok(())
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_gpu_noop() {
        let comm = SingleGpuBackend;
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
        comm.all_reduce(0x1000, 1024).unwrap();
        comm.all_gather(0x1000, 0x2000, 512).unwrap();
        comm.reduce_scatter(0x1000, 0x2000, 512).unwrap();
        comm.broadcast(0x1000, 256, 0).unwrap();
        comm.barrier().unwrap();
        comm.send_to(0x1000, 256, 0, 0).unwrap();
        comm.recv_from(0x2000, 256, 0, 0).unwrap();
        comm.group_start().unwrap();
        comm.group_end().unwrap();
        assert!(comm.is_healthy());
        comm.attempt_reconnect().unwrap();
    }

    #[test]
    fn test_single_gpu_symmetric_alloc_unsupported() {
        // SingleGpuBackend doesn't override symmetric_alloc/free — it must
        // fall back to the trait default which returns an error. This is
        // the contract callers depend on for fallback paths.
        let comm = SingleGpuBackend;
        assert!(comm.symmetric_alloc(1024).is_err());
        assert!(comm.symmetric_free(0x1000).is_err());
    }
}
