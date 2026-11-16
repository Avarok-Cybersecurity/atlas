// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the input embedding table in MAPPED (zero-copy) pinned host
//! memory instead of device memory (port lane L24, row `embed_host_offload`).
//!
//! # What it does
//!
//! The embedding table is read one row per token, so it does not need device
//! bandwidth. On Qwen3.8-27B it is `[248320, 5120]` BF16, 2.37 GiB: the single
//! most wasteful item on a 16 GB card. With the row
//! on, the loader (`weight_loader/qwen35_dense/loaders_b.rs::load_embedding`,
//! lane L23b) calls [`install_from_store`]: the table is copied from the
//! store's device tensor into page-locked host memory, the region's
//! device-visible alias is returned as the model's `embed_tokens` weight, and
//! the device copy is freed by [`after_load`] right after the loader's own
//! prune step. Every existing reader keeps working through the alias:
//!
//! * the gather kernels of prefill, batched decode and the MTP head
//!   (`embed_from_argmax`, `layers/ops/sampling.rs:101`, which embeds token ids
//!   that only exist on the device) read the rows over PCIe;
//! * the host-driven single-row copy (`model/impl_a3_embed.rs`) takes the one
//!   branch on the row: [`copy_host_row`] enqueues an ordinary pinned
//!   host-to-device copy of the row instead of a device-to-device copy.
//!
//! The bytes are the checkpoint's bytes either way, so the row is bit-identical
//! by construction; the receipt is a greedy token SHA equal with the row off
//! and on, and the freed VRAM in the boot ledger.
//!
//! # When it declines
//!
//! [`install_from_store`] returns `Ok(None)` and the loader keeps the device
//! table (exactly the row-off load) when the row is off, the tensor is not a
//! `[rows, hidden]` BF16 table, the checkpoint ties its LM head to the
//! embedding (the head would then be read over PCIe every token), or the
//! backend refuses mapped memory (Metal, the mock, and possibly WSL2 for a
//! region this large). A refusal is counted
//! (`atlas_embed_host_refusals_total`), and because a 16 GB card serving the
//! 27B with MTP has no room for the 2.37 GiB the row was meant to free,
//! [`after_load`] turns a refusal into a startup error when the serve is
//! speculative (the row refuses to start with `--speculative`).
//!
//! # Lifetime
//!
//! Installed tables live in a process registry keyed by their device alias
//! (`registry.rs`). The model's teardown calls [`release_view`] with its
//! `embed_tokens` pointer (a no-op for a device table), so a model swap does
//! not leak the pinned region. A load that fails between
//! [`install_from_store`] and the model's construction leaves its entry until
//! process exit; [`after_load`] never frees a store tensor that is not the one
//! the entry was copied from.

pub mod mapped;
pub mod registry;

#[cfg(test)]
mod tests;

use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::weight_map::DenseWeight;
pub use mapped::{BackendMapper, HostMapper, MappedRegion};
pub use registry::Registry;

/// An embedding table in mapped host memory: `[rows, hidden]` BF16, row-major.
#[derive(Debug)]
pub struct HostEmbedding {
    region: MappedRegion,
    rows: usize,
    hidden: usize,
}

impl HostEmbedding {
    /// The table `data` (`rows x hidden` BF16, the checkpoint's bytes) in
    /// mapped page-locked memory from the backend's allocator. The returned
    /// [`HostEmbedding::view`] is the pointer the existing embed path reads.
    pub fn from_bf16_host(
        gpu: &dyn GpuBackend,
        rows: usize,
        hidden: usize,
        data: &[u8],
    ) -> Result<Self> {
        Self::from_bf16_host_with(gpu, Arc::new(BackendMapper), rows, hidden, data)
    }

    /// [`HostEmbedding::from_bf16_host`] through an explicit mapper.
    pub fn from_bf16_host_with(
        gpu: &dyn GpuBackend,
        mapper: Arc<dyn HostMapper>,
        rows: usize,
        hidden: usize,
        data: &[u8],
    ) -> Result<Self> {
        let bytes = table_bytes(rows, hidden)?;
        anyhow::ensure!(
            data.len() == bytes,
            "embedding table: {rows} x {hidden} BF16 is {bytes} bytes, got {}",
            data.len()
        );
        Self::build(gpu, mapper, rows, hidden, |dst| {
            dst.copy_from_slice(data);
            Ok(())
        })
    }

    /// The table at device pointer `src` (`rows x hidden` BF16), copied into
    /// mapped page-locked memory. `src` is left untouched; freeing it is the
    /// caller's business ([`after_load`] for a store tensor).
    pub fn from_device_with(
        gpu: &dyn GpuBackend,
        mapper: Arc<dyn HostMapper>,
        rows: usize,
        hidden: usize,
        src: DevicePtr,
    ) -> Result<Self> {
        anyhow::ensure!(!src.is_null(), "embedding table: null device source");
        Self::build(gpu, mapper, rows, hidden, |dst| {
            gpu.copy_d2h(src, dst)
                .context("embedding table: copying the device table to host memory")
        })
    }

    fn build(
        gpu: &dyn GpuBackend,
        mapper: Arc<dyn HostMapper>,
        rows: usize,
        hidden: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        let bytes = table_bytes(rows, hidden)?;
        let row_bytes = hidden * 2;
        // First and last row: a mapping that misreads either end fails here.
        let check = [0usize, (rows - 1) * row_bytes];
        let region = MappedRegion::new_filled(gpu, mapper, bytes, (&check[..], row_bytes), fill)?;
        Ok(Self {
            region,
            rows,
            hidden,
        })
    }

    /// The device-visible alias every kernel and copy reads the table through.
    pub fn view(&self) -> DevicePtr {
        self.region.view()
    }

    /// The table as the `DenseWeight` the model binds as `embed_tokens`.
    pub fn as_dense(&self) -> DenseWeight {
        DenseWeight {
            weight: self.view(),
        }
    }

    /// Rows (the vocabulary the table covers).
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Hidden size (BF16 elements per row).
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Host bytes of the whole table.
    pub fn bytes(&self) -> usize {
        self.region.bytes()
    }

    /// Row `token`'s host bytes, or `None` past the table.
    pub fn row(&self, token: u32) -> Option<&[u8]> {
        let t = token as usize;
        if t >= self.rows {
            return None;
        }
        let row_bytes = self.hidden * 2;
        Some(&self.region.host_bytes()[t * row_bytes..(t + 1) * row_bytes])
    }

    /// Free the region. Nothing may read [`HostEmbedding::view`] afterwards.
    pub fn release(self, gpu: &dyn GpuBackend) -> Result<()> {
        self.region.release(gpu)
    }
}

fn table_bytes(rows: usize, hidden: usize) -> Result<usize> {
    anyhow::ensure!(
        rows > 0 && hidden > 0,
        "embedding table: empty shape [{rows}, {hidden}]"
    );
    rows.checked_mul(hidden)
        .and_then(|e| e.checked_mul(2))
        .context("embedding table: byte size overflows usize")
}

/// The process registry (tests build their own [`Registry`]).
pub fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(Registry::default)
}

/// The row as the process resolved it.
fn row_on() -> bool {
    crate::layers::ops::port_rows::embed_host_offload::row()
}

/// What the row frees on a load whose LM head is untied: the whole BF16 table.
/// For the pre-load weight estimate (lane L23b's `preflight/headroom.rs`) and
/// the consumer pool fit; 0 with the row off or a tied head.
pub fn predicted_freed_bytes(vocab: usize, hidden: usize, tied_head: bool) -> usize {
    predicted_freed_bytes_for(row_on(), vocab, hidden, tied_head)
}

/// [`predicted_freed_bytes`] with the row passed in.
pub fn predicted_freed_bytes_for(on: bool, vocab: usize, hidden: usize, tied_head: bool) -> usize {
    if on && !tied_head {
        vocab.saturating_mul(hidden).saturating_mul(2)
    } else {
        0
    }
}

/// Lane L23b's loader hook: with the row on,
/// move the store's embedding table `name` (`[rows, hidden]` BF16) to mapped
/// host memory and return its device alias as the `embed_tokens` weight.
/// `Ok(None)` = declined; the loader keeps the device table.
pub fn install_from_store(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    name: &str,
    rows: usize,
    hidden: usize,
) -> Result<Option<DenseWeight>> {
    if !row_on() {
        return Ok(None);
    }
    registry().install(gpu, Arc::new(BackendMapper), store, name, rows, hidden)
}

/// The factory's step after the loader's prune (`factory/build.rs`): free the
/// device copy of every table installed during this load, and fail the start
/// of a speculative serve whose table the backend refused to map.
pub fn after_load(store: &mut WeightStore, gpu: &dyn GpuBackend, speculative: bool) -> Result<()> {
    registry().after_load(store, gpu, speculative)
}

/// The model's teardown: free the host table behind `view` (its
/// `embed_tokens` pointer). A no-op for a device table.
pub fn release_view(gpu: &dyn GpuBackend, view: DevicePtr) -> Result<()> {
    registry().release_view(gpu, view)
}

/// The central single-row embed (`model/impl_a3_embed.rs`): when `table` is a
/// host-resident table, enqueue row `token` as a pinned host-to-device copy
/// into `out` and return `true`; `false` = a device table (the caller copies
/// device to device as before).
pub fn copy_host_row(
    gpu: &dyn GpuBackend,
    table: DevicePtr,
    token: u32,
    row_bytes: usize,
    out: DevicePtr,
    stream: u64,
) -> Result<bool> {
    registry().copy_host_row(gpu, table, token, row_bytes, out, stream)
}

/// The `/metrics` series of the process registry (zeros with the row off).
pub fn render_prometheus(out: &mut String) {
    registry().render(out);
}
