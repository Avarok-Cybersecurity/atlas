// SPDX-License-Identifier: AGPL-3.0-only

//! Small device-side helpers: BF16 buffer copy + optional debug dump.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::VisionEncoder;

impl VisionEncoder {
    /// Copy a BF16 device buffer to another device buffer via the
    /// existing `vision_bf16_copy` element kernel. The kernel takes a
    /// u32 element count (not bytes).
    pub(super) fn gpu_copy_bf16(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        n_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let n_elts = (n_bytes / 2) as u32;
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_elts, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(n_elts)
            .launch(stream)
    }

    /// Debug hook: when `ATLAS_DUMP_VIT=<dir>` is set, snapshot a GPU BF16
    /// buffer of `n` elements to `<dir>/<label>.bin`. Each file is plain
    /// little-endian BF16 with no header — python loader reads with
    /// `np.frombuffer(f.read(), dtype=np.uint16).view(np.float32[:8]>>16)`.
    /// Off by default; only for ViT-vs-HF reference comparison.
    pub(super) fn maybe_dump_buf(
        gpu: &dyn GpuBackend,
        ptr: DevicePtr,
        n_elements: usize,
        label: &str,
        stream: u64,
    ) -> Result<()> {
        let Ok(dir) = std::env::var("ATLAS_DUMP_VIT") else {
            return Ok(());
        };
        if dir.is_empty() {
            return Ok(());
        }
        gpu.synchronize(stream)?;
        let bytes = n_elements * 2; // BF16
        let mut buf = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut buf)?;
        let path = std::path::Path::new(&dir).join(format!("{label}.bin"));
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(&path, &buf).with_context(|| format!("write {}", path.display()))?;
        tracing::info!(
            "ATLAS_DUMP_VIT: wrote {} ({} elements)",
            path.display(),
            n_elements
        );
        Ok(())
    }
}
