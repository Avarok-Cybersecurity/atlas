// SPDX-License-Identifier: AGPL-3.0-only

//! Teardown for [`SsmSnapshotPool`].
//!
//! Split from `ssm_snapshot.rs` to keep that file inside the 500-line cap.

use spark_runtime::gpu::GpuBackend;

use super::ssm_snapshot::SsmSnapshotPool;

/// Release every snapshot region, including the decode-rollback ring.
impl atlas_core::scope::ModelResource<dyn GpuBackend> for SsmSnapshotPool {
    fn label(&self) -> &'static str {
        "ssm snapshot pool"
    }

    fn release(&mut self, gpu: &dyn GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        for pool in [
            &mut self.h_snapshots,
            &mut self.conv_snapshots,
            &mut self.decode_h_snapshots,
            &mut self.decode_conv_snapshots,
        ] {
            for ptr in pool.drain(..) {
                if let Err(e) = gpu.free(ptr)
                    && first_error.is_none()
                {
                    first_error = Some(e);
                }
            }
        }
        // Host bookkeeping indexes into pools that are gone; a stale free slot
        // would hand out a pointer into freed memory after a swap.
        // parking_lot: no poisoning, so these cannot fail to clear.
        self.free_slots.lock().clear();
        self.session_tags.lock().clear();
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
