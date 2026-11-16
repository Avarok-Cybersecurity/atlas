// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: run one [`AcceptRequest`] on the device: resolve the kernel once, upload the blob,
//! launch one CTA per (position, regime) job, read the results back.
//!
//! The kernel is looked up lazily through `try_target_kernel`, so a target that does not build
//! `spec_accept_sample_ada.cu` (every target but Ada) issues no lookup and returns `Ok(None)`: the
//! caller keeps the host pipeline. Nothing here runs unless the `mtp_device_accept` row is on and the
//! scheduler's planner accepted the row.

use anyhow::{Result, anyhow};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{AcceptRequest, BLOCK, ENTRY, JobResult, MODULE, SpecAcceptState, blob};

/// The device request and results buffers, allocated once per model at their largest size.
pub struct Buffers {
    blob: DevicePtr,
    out: DevicePtr,
}

/// Pick every job of `req` on the device. `Ok(None)`: this target has no device path.
pub fn run(
    state: &SpecAcceptState,
    gpu: &dyn GpuBackend,
    req: &AcceptRequest<'_>,
) -> Result<Option<Vec<JobResult>>> {
    super::validate(req).map_err(|e| anyhow!("spec_accept request rejected: {e}"))?;
    let kernel = *state.kernel.get_or_init(|| {
        let h = crate::layers::try_target_kernel(gpu, MODULE, ENTRY);
        if h.0 == 0 {
            tracing::info!(
                "mtp_device_accept: {MODULE}::{ENTRY} is not built for this target; \
                 verify picks stay on the host pipeline"
            );
        }
        h
    });
    if kernel.0 == 0 {
        return Ok(None);
    }

    let mut guard = state
        .bufs
        .lock()
        .map_err(|_| anyhow!("spec_accept buffers lock poisoned"))?;
    if guard.is_none() {
        let blob_buf = gpu.alloc(blob::capacity_words() * 4)?;
        let out_buf = match gpu.alloc(blob::results_bytes()) {
            Ok(p) => p,
            Err(e) => {
                let _ = gpu.free(blob_buf);
                return Err(e);
            }
        };
        *guard = Some(Buffers {
            blob: blob_buf,
            out: out_buf,
        });
    }
    let Some(bufs) = guard.as_ref() else {
        return Err(anyhow!("spec_accept buffers missing after allocation"));
    };

    let words = blob::encode(req);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, bufs.blob)?;

    let jobs = req.n_pos * req.regimes.len();
    let max_regime_edits = req.regimes.iter().map(Vec::len).max().unwrap_or(0);
    let smem = super::dyn_smem_bytes(req.vocab, max_regime_edits);
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, kernel)
        .grid([jobs as u32, 1, 1])
        .block([BLOCK, 1, 1])
        .shared_mem(smem as u32)
        .arg_ptr(req.logits)
        .arg_ptr(bufs.blob)
        .arg_ptr(bufs.out)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut out = vec![0u8; jobs * 8];
    gpu.copy_d2h(bufs.out, &mut out)?;
    Ok(Some(blob::decode_results(&out, jobs)))
}
