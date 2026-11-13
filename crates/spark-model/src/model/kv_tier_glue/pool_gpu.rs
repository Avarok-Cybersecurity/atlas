// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the pinned pool's handle to the model's GPU backend.
//!
//! Lane L25's `PinnedPool::new` takes an `Arc<dyn GpuBackend>` (its background pin thread binds
//! the backend's context and page-locks chunks through `host_register`; its `Drop` calls
//! `host_unregister`), but a `TransformerModel` owns its backend as a `Box`. [`PoolGpu`] is a
//! shared, detachable view of that box: every call runs under a read lock against a pointer to
//! the model's backend, and [`PoolGpu::detach`] (called by the tier teardown, which runs from
//! `release_pools` and from `Drop for TransformerModel` while the backend is alive) clears the
//! pointer under the write lock, so no call can outlive the backend: one that comes later gets
//! an error instead. Only the pinned pool holds this handle; it forwards the trait's required
//! methods and the three host-memory methods the pool uses, and keeps every other default.

use std::sync::Arc;

use anyhow::{Result, bail};
use parking_lot::RwLock;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::op_cache::OpCache;

/// A pointer to the model's boxed backend.
struct Target(*const (dyn GpuBackend + 'static));

// SAFETY: the pointee is `Send + Sync` (a supertrait of `GpuBackend`), and the pointer is
// dereferenced only under `PoolGpu::target`'s read lock while it is `Some`, which `detach`
// ends before the backend is dropped (see the module doc).
unsafe impl Send for Target {}
unsafe impl Sync for Target {}

/// The pinned pool's detachable handle to the model's backend.
pub(crate) struct PoolGpu {
    target: RwLock<Option<Target>>,
    /// `op_cache` must return a reference; the pool never uses it, so a private, empty cache
    /// keeps the handle from lending out the model's.
    op_cache: OpCache,
}

impl PoolGpu {
    /// A handle to `gpu`, valid until [`Self::detach`]. The caller guarantees `detach` runs
    /// before `gpu` is dropped.
    pub(crate) fn attach(gpu: &(dyn GpuBackend + 'static)) -> Arc<Self> {
        Arc::new(Self {
            target: RwLock::new(Some(Target(gpu as *const (dyn GpuBackend + 'static)))),
            op_cache: OpCache::default(),
        })
    }

    /// End every later call with an error. Waits for calls in flight.
    pub(crate) fn detach(&self) {
        *self.target.write() = None;
    }

    pub(crate) fn attached(&self) -> bool {
        self.target.read().is_some()
    }

    fn with<R>(&self, f: impl FnOnce(&dyn GpuBackend) -> Result<R>) -> Result<R> {
        let g = self.target.read();
        match g.as_ref() {
            // SAFETY: see `Target`: attached means the model's backend is alive, and the read
            // guard keeps `detach` (and so the model's teardown) waiting until `f` returns.
            Some(t) => f(unsafe { &*t.0 }),
            None => bail!("the model's GPU backend was released (host tier torn down)"),
        }
    }
}

impl GpuBackend for PoolGpu {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.with(|g| g.alloc(bytes))
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.with(|g| g.alloc_managed(bytes))
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.with(|g| g.free(ptr))
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.with(|g| g.copy_h2d(src, dst))
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.with(|g| g.copy_d2h(src, dst))
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.with(|g| g.copy_d2d(src, dst, bytes))
    }
    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.with(|g| g.launch(func, grid, block, shared_mem, stream, params))
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.with(|g| g.synchronize(stream))
    }
    fn default_stream(&self) -> u64 {
        self.with(|g| Ok(g.default_stream())).unwrap_or(0)
    }
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.with(|g| g.kernel(module, func_name))
    }
    fn has_module(&self, module: &str) -> bool {
        self.with(|g| Ok(g.has_module(module))).unwrap_or(false)
    }
    fn op_cache(&self) -> &OpCache {
        &self.op_cache
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.with(|g| g.memset(ptr, value, bytes))
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.with(|g| g.memset_async(ptr, value, bytes, stream))
    }
    fn total_memory(&self) -> Result<usize> {
        self.with(|g| g.total_memory())
    }
    fn free_memory(&self) -> Result<usize> {
        self.with(|g| g.free_memory())
    }
    fn sm_count(&self) -> Result<u32> {
        self.with(|g| g.sm_count())
    }
    // The three calls the pinned pool makes: bind the context on its pin thread, page-lock and
    // unlock its chunks.
    fn bind_to_thread(&self) -> Result<()> {
        self.with(|g| g.bind_to_thread())
    }
    fn host_register(&self, ptr: usize, bytes: usize) -> Result<()> {
        self.with(|g| g.host_register(ptr, bytes))
    }
    fn host_unregister(&self, ptr: usize) -> Result<()> {
        self.with(|g| g.host_unregister(ptr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn forwards_while_attached_and_refuses_after_detach() {
        let gpu: Box<dyn GpuBackend> = Box::new(MockGpuBackend::new());
        let h = PoolGpu::attach(gpu.as_ref());
        assert!(h.attached());
        let p = h.alloc(64).expect("forwarded alloc");
        h.memset(p, 7, 64).expect("forwarded memset");
        let mut back = [0u8; 64];
        h.copy_d2h(p, &mut back).expect("forwarded copy");
        assert!(back.iter().all(|&b| b == 7));
        h.bind_to_thread().expect("forwarded bind");
        h.detach();
        assert!(!h.attached());
        assert!(
            h.alloc(64).is_err(),
            "a detached handle never reaches the backend"
        );
        assert!(h.host_register(4096, 4096).is_err());
        assert!(h.bind_to_thread().is_err());
        assert!(!h.has_module("anything"));
        assert_eq!(h.default_stream(), 0);
        gpu.free(p).unwrap();
    }

    #[test]
    fn the_handle_is_a_shared_backend() {
        let gpu: Box<dyn GpuBackend> = Box::new(MockGpuBackend::new());
        let h = PoolGpu::attach(gpu.as_ref());
        let shared: Arc<dyn GpuBackend> = h.clone();
        assert_eq!(shared.default_stream(), gpu.default_stream());
        h.detach();
        assert!(shared.synchronize(0).is_err());
    }
}
