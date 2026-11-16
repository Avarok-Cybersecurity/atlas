// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: one region of MAPPED (zero-copy) pinned host memory and the
//! device-visible pointer the GPU reads it through, plus the seam
//! ([`HostMapper`]) that lets the CPU tests stand in for the driver.
//!
//! # Why mapped, not "gather on the host before launch"
//!
//! The embedding gather is not only a host-driven copy. MTP drafts embed
//! token ids that live on the DEVICE (`embed_from_argmax`,
//! `layers/ops/sampling.rs:101`, called from `mtp_head/forward.rs`), and the
//! prefill and batched-decode paths hand `embed_tokens.weight` straight to a
//! kernel. A host-side gather before a graph launch cannot serve a token id
//! the host never sees. Mapped pinned memory can: every existing kernel and
//! copy reads the table through an ordinary device pointer, over PCIe, with
//! no code change at those sites.
//!
//! # The production mapper
//!
//! [`BackendMapper`] uses the backend's page-locked allocator
//! (`GpuBackend::alloc_host_pinned`, `cuMemAllocHost_v2`, zeroed) and its
//! `host_ptr_to_device` (`cuMemHostGetDevicePointer_v2`). With unified
//! addressing every page-locked allocation is mapped; a backend or a platform
//! that refuses the mapping (the Metal and mock backends, and possibly WSL2
//! for very large pinned regions, receipt Q5) makes [`MappedRegion::new_filled`]
//! fail, the caller keeps the table on the device, and the failure is counted
//! (`atlas_embed_host_refusals_total`).
//!
//! # Self-check
//!
//! After mapping, the first and the last row are read back THROUGH the device
//! view and compared with the host bytes. A mapping that hands back a pointer
//! the device cannot read correctly fails the load here, not as fluent
//! garbage at the first token.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::Arc;

/// How a region of host memory becomes visible to the device. Addresses are
/// `usize` so no public signature carries a raw pointer.
pub trait HostMapper: Send + Sync {
    /// `bytes` of page-locked, zeroed host memory; returns its address.
    fn alloc(&self, gpu: &dyn GpuBackend, bytes: usize) -> Result<usize>;
    /// The device-visible alias of the region at `host` (already filled).
    fn device_view(&self, gpu: &dyn GpuBackend, host: usize, bytes: usize) -> Result<DevicePtr>;
    /// Release both the alias and the host region.
    fn free(&self, gpu: &dyn GpuBackend, host: usize, view: DevicePtr, bytes: usize) -> Result<()>;
}

/// The driver's mapping: page-locked memory and its UVA device alias.
pub struct BackendMapper;

impl HostMapper for BackendMapper {
    fn alloc(&self, gpu: &dyn GpuBackend, bytes: usize) -> Result<usize> {
        Ok(gpu.alloc_host_pinned(bytes)? as usize)
    }

    fn device_view(&self, gpu: &dyn GpuBackend, host: usize, _bytes: usize) -> Result<DevicePtr> {
        gpu.host_ptr_to_device(host as *mut u8)
    }

    fn free(
        &self,
        gpu: &dyn GpuBackend,
        host: usize,
        _view: DevicePtr,
        bytes: usize,
    ) -> Result<()> {
        // The alias needs no release of its own: it dies with the allocation.
        gpu.free_host_pinned(host as *mut u8, bytes)
    }
}

/// A filled, mapped host region. Not `Drop`: releasing needs the backend, so
/// the owner calls [`MappedRegion::release`] (the model's teardown does, via
/// `port::embed_host::release_view`).
pub struct MappedRegion {
    host: usize,
    view: DevicePtr,
    bytes: usize,
    mapper: Arc<dyn HostMapper>,
}

impl std::fmt::Debug for MappedRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedRegion")
            .field("host", &format_args!("{:#x}", self.host))
            .field("view", &self.view)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl MappedRegion {
    /// Allocate `bytes`, let `fill` write every byte, then map the region and
    /// self-check the mapping on the rows at `check_offsets` (each
    /// `check_len` bytes). Every failure after the allocation frees it.
    pub fn new_filled(
        gpu: &dyn GpuBackend,
        mapper: Arc<dyn HostMapper>,
        bytes: usize,
        check: (&[usize], usize),
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        anyhow::ensure!(bytes > 0, "mapped host region: zero bytes requested");
        let host = mapper
            .alloc(gpu, bytes)
            .with_context(|| format!("mapped host region: pinning {bytes} bytes"))?;
        anyhow::ensure!(host != 0, "mapped host region: allocator returned null");
        let filled = {
            // SAFETY: `host` is a live, uniquely owned, zeroed allocation of
            // exactly `bytes` (the HostMapper contract), and nothing else holds
            // it until this function returns.
            let slice = unsafe { std::slice::from_raw_parts_mut(host as *mut u8, bytes) };
            fill(slice)
        };
        if let Err(e) = filled {
            let _ = mapper.free(gpu, host, DevicePtr::NULL, bytes);
            return Err(e);
        }
        let view = match mapper.device_view(gpu, host, bytes) {
            Ok(v) if !v.is_null() => v,
            Ok(_) => {
                let _ = mapper.free(gpu, host, DevicePtr::NULL, bytes);
                anyhow::bail!("mapped host region: the backend returned a null device alias");
            }
            Err(e) => {
                let _ = mapper.free(gpu, host, DevicePtr::NULL, bytes);
                return Err(e.context(
                    "mapped host region: the backend refused a device alias for page-locked \
                     memory (no unified addressing, a non-CUDA backend, or a pinned-size limit)",
                ));
            }
        };
        let region = Self {
            host,
            view,
            bytes,
            mapper,
        };
        if let Err(e) = region.self_check(gpu, check.0, check.1) {
            let _ = region.release(gpu);
            return Err(e);
        }
        Ok(region)
    }

    /// Read `len` bytes at each offset through the device alias and compare
    /// them with the host bytes.
    fn self_check(&self, gpu: &dyn GpuBackend, offsets: &[usize], len: usize) -> Result<()> {
        let host = self.host_bytes();
        let mut seen = vec![0u8; len];
        for &off in offsets {
            anyhow::ensure!(
                off + len <= self.bytes,
                "mapped host region: self-check range {off}+{len} past {} bytes",
                self.bytes
            );
            gpu.copy_d2h(self.view.offset(off), &mut seen)
                .context("mapped host region: reading through the device alias")?;
            anyhow::ensure!(
                seen == host[off..off + len],
                "mapped host region: the device alias at offset {off} does not read the host \
                 bytes; the mapping is not usable on this platform"
            );
        }
        Ok(())
    }

    /// The device-visible pointer every kernel and copy reads through.
    pub fn view(&self) -> DevicePtr {
        self.view
    }

    /// Size of the region.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The region's bytes, as the host sees them.
    pub fn host_bytes(&self) -> &[u8] {
        // SAFETY: the region is live until `release` consumes `self`, holds
        // exactly `bytes` initialized bytes (zeroed at allocation, then
        // filled), and is only written through the device alias by kernels
        // that write embedding rows (token overlays), never concurrently with
        // a host read made at load time.
        unsafe { std::slice::from_raw_parts(self.host as *const u8, self.bytes) }
    }

    /// Free the alias and the host memory.
    pub fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        self.mapper.free(gpu, self.host, self.view, self.bytes)
    }
}
