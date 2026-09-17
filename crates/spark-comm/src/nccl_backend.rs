// SPDX-License-Identifier: AGPL-3.0-only

//! NCCL-based communication backend for expert parallelism.
//!
//! Uses TCP bootstrap: rank 0 generates a unique ID and sends it to
//! all other ranks via a TCP listener. Then all ranks call
//! `ncclCommInitRank` with the shared ID.
//!
//! Optimizations for 2-rank EP with small messages (4 KB):
//! - Pre-registers buffers with NCCL (`ncclCommRegister`) to cache IB memory registration
//! - Uses paired `ncclSend`/`ncclRecv` + local BF16 add instead of `ncclAllReduce`
//!
//! Health monitoring and recovery:
//! - Checks `ncclCommGetAsyncError` after each collective
//! - Detects broadcast timeouts (>30s) via stream sync + wall-clock check
//! - Aborts dead communicators via `ncclCommAbort` and reconnects
//!
//! ## Safety contract for the `unsafe { ... }` calls below
//!
//! All unsafe blocks in this file wrap a single FFI call into either
//! NCCL (`nccl*`) or the CUDA Driver API (`cu*`). The invariants are
//! uniform:
//!
//! - **NCCL handles**: `NcclComm` instances are constructed via
//!   `nccl::comm_init_rank` after a successful TCP bootstrap and are
//!   `Drop`-cleaned via `ncclCommDestroy`. They are never aliased
//!   across threads without a `Mutex` guarding the comm.
//! - **CUDA buffers** passed to NCCL come from a prior `cuMemAlloc_v2`
//!   on the same device that owns the comm; size in bytes matches the
//!   allocation.
//! - **Streams** referenced via `u64` are owned by the caller and
//!   outlive the in-flight collective.
//! - **`extern "C"` ABI**: matches the NCCL 2.20+ headers and the
//!   `cuMemAlloc_v2`/`cuLaunchKernel`/etc. shapes declared just below.
//!
//! Per-site `// SAFETY:` comments are omitted because the contract is
//! identical for every call. Deviations get a per-site comment.

use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::ffi::c_void;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::nccl::{self, NcclComm, NcclDataType, NcclResult, NcclUniqueId};

// CUDA driver API for recv buffer allocation and kernel launch.
unsafe extern "C" {
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuLaunchKernel(
        f: u64,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: u64,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
}

mod recv_buffer;
use recv_buffer::ensure_payload_fits;
pub use recv_buffer::{ALL_REDUCE_DTYPE_BYTES, required_recv_bytes};

/// Timeout threshold for a single synchronous collective operation.
/// If a broadcast + stream sync takes longer than this, mark the communicator unhealthy.
pub(super) const COLLECTIVE_TIMEOUT_SECS: u64 = 30;

/// NCCL communication backend for multi-GPU / multi-node EP.
pub struct NcclBackend {
    /// Protected for abort-and-reconnect. All NCCL calls acquire this lock.
    comm: Mutex<NcclComm>,
    rank: usize,
    world_size: usize,
    /// Dedicated stream for NCCL collectives (separate from compute).
    comm_stream: u64,
    /// Event: signals "MoE compute done" on the compute stream.
    compute_done_event: u64,
    /// Event: signals "all_reduce done" on the comm stream.
    comm_done_event: u64,
    /// Legacy compute stream for synchronous barrier/broadcast.
    legacy_stream: u64,
    /// Persistent receive buffer for 2-rank send/recv all-reduce.
    recv_buffer: u64,
    /// Allocated capacity of `recv_buffer`, in bytes (0 when `world_size != 2`).
    ///
    /// Every 2-rank all-reduce payload is checked against this before any NCCL
    /// call or kernel launch. Derived from the configured maximum transfer at
    /// construction — see [`required_recv_bytes`].
    recv_capacity: usize,
    /// Handles from ncclCommRegister (deregistered in Drop).
    registered_handles: Mutex<Vec<*mut c_void>>,
    /// Kernel handle for bf16_add_inplace (set via set_add_kernel).
    add_kernel: AtomicU64,
    /// Whether the communicator is in a degraded/unhealthy state.
    unhealthy: AtomicBool,
    /// Number of successful reconnections (for diagnostics).
    reconnect_count: AtomicU64,
    /// Bootstrap parameters stored for reconnection.
    master_addr: String,
    master_port: u16,
}

// SAFETY: `NcclComm` is an opaque NCCL handle. NCCL guarantees the handle
// is thread-safe for non-overlapping operations on the same communicator;
// Atlas serializes access through the inner `Mutex<NcclComm>` (one in-flight
// collective per rank at a time) and the host-side completions are bound
// to the user-supplied CUDA stream. The raw pointer never escapes the
// backend, so any aliasing is bounded by the Mutex guard's lifetime.
unsafe impl Send for NcclBackend {}
unsafe impl Sync for NcclBackend {}

impl NcclBackend {
    /// Initialize NCCL with TCP bootstrap.
    ///
    /// Rank 0 listens on `master_addr:master_port`, generates a unique ID,
    /// and sends it to all connecting ranks. All ranks then call
    /// `ncclCommInitRank` which internally synchronizes.
    /// `recv_capacity` is the largest all-reduce payload this backend will ever
    /// be asked to carry, in bytes — compute it with [`required_recv_bytes`]
    /// from the serve configuration. It is only consulted when
    /// `world_size == 2` (the send/recv fast path); other world sizes reduce
    /// in-place via `ncclAllReduce` and allocate no receive buffer.
    pub fn new(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
        stream: u64,
        recv_capacity: usize,
    ) -> Result<Self> {
        Self::log_nccl_env_vars();

        let unique_id = if rank == 0 {
            let id = Self::generate_unique_id()?;
            Self::distribute_id(&id, master_addr, master_port, world_size)?;
            id
        } else {
            Self::receive_id(master_addr, master_port)?
        };

        let mut comm: NcclComm = ptr::null_mut();
        let result =
            unsafe { nccl::ncclCommInitRank(&mut comm, world_size as i32, unique_id, rank as i32) };
        nccl::check_nccl(result, "ncclCommInitRank")?;

        let comm_stream = nccl::create_stream()?;
        let compute_done_event = nccl::create_event()?;
        let comm_done_event = nccl::create_event()?;

        // Allocate persistent recv buffer for 2-rank send/recv all-reduce,
        // sized to the caller's configured maximum transfer.
        let mut recv_buffer: u64 = 0;
        if world_size == 2 {
            if recv_capacity == 0 {
                anyhow::bail!(
                    "world_size == 2 requires a non-zero receive-buffer capacity; \
                     compute it with required_recv_bytes(max_batch_tokens, hidden_size, \
                     ALL_REDUCE_DTYPE_BYTES)"
                );
            }
            let status = unsafe { cuMemAlloc_v2(&mut recv_buffer, recv_capacity) };
            if status != 0 {
                anyhow::bail!(
                    "cuMemAlloc_v2 for recv_buffer ({recv_capacity} bytes) failed: status {status}"
                );
            }
            // Register recv buffer with NCCL for IB memory caching.
            let mut handle: *mut c_void = ptr::null_mut();
            let result = unsafe {
                nccl::ncclCommRegister(comm, recv_buffer as *mut c_void, recv_capacity, &mut handle)
            };
            if result != NcclResult::Success {
                tracing::warn!("ncclCommRegister for recv_buffer failed (non-fatal): {result:?}");
            } else {
                tracing::info!(
                    "Registered recv_buffer ({} KB) with NCCL",
                    recv_capacity / 1024
                );
            }
        }

        Ok(Self {
            comm: Mutex::new(comm),
            rank,
            world_size,
            comm_stream,
            compute_done_event,
            comm_done_event,
            legacy_stream: stream,
            recv_buffer,
            recv_capacity: if world_size == 2 { recv_capacity } else { 0 },
            registered_handles: Mutex::new(Vec::new()),
            add_kernel: AtomicU64::new(0),
            unhealthy: AtomicBool::new(false),
            reconnect_count: AtomicU64::new(0),
            master_addr: master_addr.to_owned(),
            master_port,
        })
    }

    /// Log NCCL-related environment variables at init time for diagnostics.
    fn log_nccl_env_vars() {
        let vars = [
            "NCCL_TIMEOUT",
            "NCCL_WATCHDOG_TIMEOUT",
            "NCCL_IB_TIMEOUT",
            "NCCL_IB_RETRY_CNT",
            "NCCL_SOCKET_IFNAME",
            "NCCL_DEBUG",
        ];
        for var in &vars {
            match std::env::var(var) {
                Ok(val) => tracing::info!("NCCL env: {var}={val}"),
                Err(_) => tracing::debug!("NCCL env: {var} not set"),
            }
        }
    }

    /// Check the NCCL communicator for asynchronous errors.
    ///
    /// Returns `true` if the communicator is healthy (no async errors).
    /// If an async error is detected, sets the unhealthy flag and returns `false`.
    fn check_async_error(&self, comm: NcclComm) -> bool {
        let mut async_err = NcclResult::Success;
        let result = unsafe { nccl::ncclCommGetAsyncError(comm, &mut async_err) };
        if result != NcclResult::Success {
            tracing::error!(
                "ncclCommGetAsyncError call itself failed: {result:?} \
                 — marking unhealthy"
            );
            self.unhealthy.store(true, Ordering::Release);
            return false;
        }
        if async_err != NcclResult::Success {
            tracing::error!("NCCL async error detected: {async_err:?} — marking unhealthy");
            self.unhealthy.store(true, Ordering::Release);
            return false;
        }
        true
    }

    /// Abort the current communicator and re-initialize via TCP bootstrap.
    ///
    /// Both ranks must call this concurrently (the reconnect protocol
    /// mirrors the initial bootstrap). After reconnect, the recv_buffer
    /// is re-registered with the new communicator.
    fn reconnect_inner(&self) -> Result<()> {
        let mut comm_guard = self.comm.lock();

        // Double-check: another thread may have already reconnected.
        if !self.unhealthy.load(Ordering::Acquire) {
            tracing::info!("NCCL communicator already recovered by another thread");
            return Ok(());
        }

        let old_comm = *comm_guard;
        let attempt = self.reconnect_count.load(Ordering::Relaxed) + 1;
        tracing::warn!(
            "NCCL reconnect: aborting old communicator \
             (rank={}, reconnect #{})",
            self.rank,
            attempt,
        );

        // Abort the dead communicator (non-blocking cleanup).
        if !old_comm.is_null() {
            let result = unsafe { nccl::ncclCommAbort(old_comm) };
            if result != NcclResult::Success {
                tracing::warn!("ncclCommAbort returned {result:?} (proceeding anyway)");
            }
        }

        // Re-bootstrap: rank 0 distributes a new unique ID.
        // Use master_port + 1 to avoid bind conflicts with a lingering listener.
        let reconnect_port = self.master_port.wrapping_add(1);
        let unique_id = if self.rank == 0 {
            let id = Self::generate_unique_id()?;
            Self::distribute_id(&id, &self.master_addr, reconnect_port, self.world_size)?;
            id
        } else {
            Self::receive_id(&self.master_addr, reconnect_port)?
        };

        let mut new_comm: NcclComm = ptr::null_mut();
        let result = unsafe {
            nccl::ncclCommInitRank(
                &mut new_comm,
                self.world_size as i32,
                unique_id,
                self.rank as i32,
            )
        };
        nccl::check_nccl(result, "ncclCommInitRank (reconnect)")?;

        // Re-register recv buffer with the new communicator.
        if self.world_size == 2 && self.recv_buffer != 0 {
            let mut handle: *mut c_void = ptr::null_mut();
            let result = unsafe {
                nccl::ncclCommRegister(
                    new_comm,
                    self.recv_buffer as *mut c_void,
                    self.recv_capacity,
                    &mut handle,
                )
            };
            if result != NcclResult::Success {
                tracing::warn!(
                    "ncclCommRegister for recv_buffer after reconnect \
                     failed: {result:?}"
                );
            } else {
                tracing::info!("Re-registered recv_buffer after reconnect");
            }
        }

        // Clear stale registered buffer handles — the old handles are
        // invalid after abort. We cannot re-register external buffers
        // because we only stored handles, not (ptr, size) pairs.
        let mut handles = self.registered_handles.lock();
        if !handles.is_empty() {
            tracing::warn!(
                "Clearing {} stale registered buffer handles after reconnect",
                handles.len()
            );
            handles.clear();
        }
        drop(handles);

        *comm_guard = new_comm;
        self.unhealthy.store(false, Ordering::Release);
        self.reconnect_count.fetch_add(1, Ordering::Relaxed);

        tracing::info!(
            "NCCL reconnect successful (rank={}, total reconnects={})",
            self.rank,
            self.reconnect_count.load(Ordering::Relaxed),
        );

        Ok(())
    }

    /// 2-rank all-reduce using send/recv + local BF16 add.
    ///
    /// For world_size == 2:
    ///   1. Bounds-check the payload against the receive buffer
    ///   2. Group send+recv (one RDMA write each direction)
    ///   3. Local BF16 add: `ptr[i] += recv_buffer[i]`
    ///
    /// # Capacity invariant
    ///
    /// `bytes <= self.recv_capacity`, always. `ncclRecv` writes `bytes` into a
    /// buffer of exactly `recv_capacity` bytes, so a payload larger than the
    /// allocation would write past it — device-heap corruption, or
    /// `CUDA_ERROR_ILLEGAL_ADDRESS` if you are lucky enough for it to be fatal.
    ///
    /// The capacity is derived from the configured maximum transfer at
    /// construction, so in a correctly-configured serve this check never fires.
    /// It is retained as defense in depth: it is the only thing standing between
    /// a future caller with a larger payload and a silent out-of-bounds write,
    /// and the cost of being wrong here is not a bad number, it is corrupted
    /// memory that looks plausible.
    fn all_reduce_2rank(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        // Before ncclSend, before ncclRecv, before the add-kernel launch.
        ensure_payload_fits(bytes, self.recv_capacity, self.rank, self.world_size)?;

        // Nothing to reduce. Return before the kernel launch: `blocks` would be
        // 0, which cuLaunchKernel rejects with CUDA_ERROR_INVALID_VALUE. Both
        // ranks reduce the same shape, so this is symmetric and cannot desync
        // the NCCL group.
        if bytes == 0 {
            return Ok(());
        }

        let count = bytes / ALL_REDUCE_DTYPE_BYTES; // BF16 element count
        let partner = (1 - self.rank) as i32;
        let comm = *self.comm.lock();

        // Paired send/recv in a group (single NCCL launch).
        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")?;

        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend")?;

        let result = unsafe {
            nccl::ncclRecv(
                self.recv_buffer as *mut c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv")?;

        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")?;

        // Check for async errors after the group operation.
        self.check_async_error(comm);

        // Local BF16 addition: ptr[i] += recv_buffer[i]
        let kernel = self.add_kernel.load(Ordering::Relaxed);
        if kernel != 0 {
            let threads: u32 = 256;
            let blocks: u32 = (count as u32).div_ceil(threads);
            let mut p_dst = ptr;
            let mut p_src = self.recv_buffer;
            let mut p_n = count as i32;
            let mut params: [*mut c_void; 3] = [
                &mut p_dst as *mut u64 as *mut c_void,
                &mut p_src as *mut u64 as *mut c_void,
                &mut p_n as *mut i32 as *mut c_void,
            ];
            let status = unsafe {
                cuLaunchKernel(
                    kernel,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            if status != 0 {
                anyhow::bail!("cuLaunchKernel (bf16_add_inplace) failed: status {status}");
            }
        } else {
            anyhow::bail!("bf16_add_inplace kernel not set — call set_add_kernel() first");
        }

        Ok(())
    }

    fn generate_unique_id() -> Result<NcclUniqueId> {
        let mut id = NcclUniqueId {
            internal: [0u8; 128],
        };
        let result = unsafe { nccl::ncclGetUniqueId(&mut id) };
        nccl::check_nccl(result, "ncclGetUniqueId")?;
        Ok(id)
    }

    /// Rank 0: listen, accept (world_size - 1) connections, send the unique ID.
    fn distribute_id(id: &NcclUniqueId, addr: &str, port: u16, world_size: usize) -> Result<()> {
        let bind_addr = format!("0.0.0.0:{port}");
        let listener = TcpListener::bind(&bind_addr)
            .with_context(|| format!("Rank 0: failed to bind {bind_addr}"))?;
        tracing::info!(
            "Rank 0: waiting for {} worker(s) on {}",
            world_size - 1,
            bind_addr
        );

        for i in 0..(world_size - 1) {
            let (mut stream, peer_addr) = listener.accept().context("Rank 0: accept failed")?;
            stream
                .write_all(&id.internal)
                .context("Rank 0: failed to send unique ID")?;
            tracing::info!("Rank 0: sent unique ID to worker {} ({})", i + 1, peer_addr);
        }
        let _ = addr; // master_addr not used on rank 0 (we bind 0.0.0.0)
        Ok(())
    }

    /// Non-zero rank: connect to rank 0, receive the unique ID.
    fn receive_id(addr: &str, port: u16) -> Result<NcclUniqueId> {
        let target = format!("{addr}:{port}");
        tracing::info!("Rank N: connecting to master at {target}");

        // Retry with backoff — rank 0 may not be listening yet.
        // Large models (>100B) take 3-5 minutes to load weights before opening
        // the NCCL master port. The previous 30s ceiling consistently timed
        // out on 122B-ep2 / nemotron-super-120B-ep2 sweep rounds. 10 minutes
        // gives any model time to load shards on a Spark; if rank 0 actually
        // crashed the worker still surfaces the failure — just later.
        const MAX_ATTEMPTS: u32 = 600; // 10 minutes at 1s each
        let mut stream = None;
        for attempt in 0..MAX_ATTEMPTS {
            match TcpStream::connect(&target) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => {
                    if attempt + 1 < MAX_ATTEMPTS {
                        if attempt < 30 || attempt.is_multiple_of(30) {
                            tracing::info!(
                                "Connect attempt {}/{MAX_ATTEMPTS}: {e}, retrying in 1s (rank 0 may be loading weights)",
                                attempt + 1
                            );
                        }
                        std::thread::sleep(std::time::Duration::from_secs(1));
                    } else {
                        return Err(e).with_context(|| {
                            format!(
                                "Failed to connect to master at {target} \
                                 after {MAX_ATTEMPTS} attempts (~{} minutes)",
                                MAX_ATTEMPTS / 60
                            )
                        });
                    }
                }
            }
        }

        let mut id = NcclUniqueId {
            internal: [0u8; 128],
        };
        stream
            .unwrap()
            .read_exact(&mut id.internal)
            .context("Failed to receive unique ID from rank 0")?;
        tracing::info!("Received NCCL unique ID from master");
        Ok(id)
    }
}

impl Drop for NcclBackend {
    fn drop(&mut self) {
        let comm = *self.comm.lock();
        // Deregister all NCCL-registered buffers.
        let mut handles = self.registered_handles.lock();
        for handle in handles.drain(..) {
            unsafe { nccl::ncclCommDeregister(comm, handle) };
        }
        drop(handles);
        // Free recv buffer.
        if self.recv_buffer != 0 {
            unsafe { cuMemFree_v2(self.recv_buffer) };
        }
        nccl::destroy_event(self.compute_done_event);
        nccl::destroy_event(self.comm_done_event);
        nccl::destroy_stream(self.comm_stream);
        if !comm.is_null() {
            unsafe { nccl::ncclCommDestroy(comm) };
        }
    }
}

mod comm_impl;
