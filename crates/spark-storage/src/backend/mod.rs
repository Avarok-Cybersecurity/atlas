// SPDX-License-Identifier: AGPL-3.0-only
//
// Storage backend trait + impls for the high-speed-swap path.
//
// SBIO contract: tiled-attention / scratch-pool code never opens a file or
// issues a syscall. Every NVMe-touching operation flows through a
// `StorageBackend` impl, so the predictor / scratch / kernel layers can be
// tested with the deterministic POSIX backend and swap in the io_uring
// production backend transparently.

use anyhow::Result;

use crate::group::GroupKey;

pub mod io_uring;
pub mod posix;

pub use self::io_uring::IoUringBackend;
pub use posix::PosixBackend;

/// One read request: pull `group` from disk, land it at `dst_dev_ptr`.
#[derive(Clone, Copy, Debug)]
pub struct ReadRequest {
    pub group: GroupKey,
    pub dst_dev_ptr: u64,
}

pub trait StorageBackend: Send + Sync {
    /// Synchronously fulfil all `requests`, returning when the corresponding
    /// HBM destinations are populated and visible on `stream`. The backend
    /// chooses how to schedule (blocking POSIX `pread`, batched `io_uring`,
    /// etc.). At return, the `stream` has been synchronised so the caller
    /// can issue subsequent kernels that depend on the data.
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()>;

    /// One-shot sequential write — used at offload time to populate disk
    /// from a host-side K/V buffer.
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()>;
}
