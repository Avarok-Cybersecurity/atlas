// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the installed host embedding tables of the process (row
//! `embed_host_offload`), their lifecycle (install, free the device copy after
//! the load, release at teardown), the host row copy of the central embed, and
//! the `atlas_embed_host_*` counters. The process-wide instance is
//! [`super::registry`]; tests build their own [`Registry`].

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use parking_lot::{Mutex, RwLock};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::{HostEmbedding, HostMapper};
use crate::weight_map::DenseWeight;

/// Every name a loader binds the LM head from (`loaders_b::load_lm_head`). A
/// store with none of them ties the head to the embedding table.
const LM_HEAD_KEYS: [&str; 3] = [
    "lm_head.weight",
    "language_model.lm_head.weight",
    "model.lm_head.weight",
];

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One installed table and where it came from.
struct Entry {
    table: HostEmbedding,
    /// The store tensor it was copied from, and that tensor's device pointer
    /// at copy time (so a later load's tensor of the same name is never freed).
    store_name: String,
    store_ptr: DevicePtr,
    device_freed: bool,
}

/// Installed tables, refusals of the load in progress, and the counters.
#[derive(Default)]
pub struct Registry {
    entries: RwLock<Vec<Entry>>,
    /// Store names whose install the backend refused since the last
    /// [`Registry::after_load`].
    refused: Mutex<Vec<String>>,
    resident_bytes: AtomicU64,
    refusals: AtomicU64,
    device_bytes_freed: AtomicU64,
    host_row_copies: AtomicU64,
}

impl Registry {
    /// [`super::install_from_store`] against this registry, through `mapper`.
    /// The caller has checked the row.
    pub fn install(
        &self,
        gpu: &dyn GpuBackend,
        mapper: Arc<dyn HostMapper>,
        store: &WeightStore,
        name: &str,
        rows: usize,
        hidden: usize,
    ) -> Result<Option<DenseWeight>> {
        let t = store.get(name)?;
        if t.dtype != WeightDtype::BF16 || t.shape != [rows, hidden] {
            tracing::warn!(
                "embed_host_offload: {name} is {:?} {:?}, not a [{rows}, {hidden}] BF16 table; \
                 the embedding stays on the device",
                t.dtype,
                t.shape
            );
            return Ok(None);
        }
        if !LM_HEAD_KEYS.iter().any(|k| store.contains(k)) {
            tracing::warn!(
                "embed_host_offload: the LM head is tied to {name}; moving the table would read \
                 the head over PCIe every token, so the embedding stays on the device"
            );
            return Ok(None);
        }
        let table = match HostEmbedding::from_device_with(gpu, mapper, rows, hidden, t.ptr) {
            Ok(table) => table,
            Err(e) => {
                self.refusals.fetch_add(1, Ordering::Relaxed);
                self.refused.lock().push(name.to_string());
                tracing::warn!(
                    "embed_host_offload: mapped host memory refused for {name} ({:.2} GiB): {e:#}; \
                     the embedding stays on the device",
                    (rows * hidden * 2) as f64 / GIB
                );
                return Ok(None);
            }
        };
        let dense = table.as_dense();
        let bytes = table.bytes() as u64;
        tracing::info!(
            "embed_host_offload: {name} [{rows}, {hidden}] BF16 ({:.2} GiB) served from mapped \
             host memory at {}; the device copy is freed after the load",
            bytes as f64 / GIB,
            dense.weight
        );
        self.resident_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.entries.write().push(Entry {
            table,
            store_name: name.to_string(),
            store_ptr: t.ptr,
            device_freed: false,
        });
        Ok(Some(dense))
    }

    /// [`super::after_load`] against this registry.
    pub fn after_load(
        &self,
        store: &mut WeightStore,
        gpu: &dyn GpuBackend,
        speculative: bool,
    ) -> Result<()> {
        let refused: Vec<String> = std::mem::take(&mut *self.refused.lock());
        if speculative && !refused.is_empty() {
            bail!(
                "embed_host_offload is on but the backend refused mapped host memory for {} \
                 (see the warning above). A speculative serve was sized to need the VRAM the row \
                 frees, so it does not start with the table on the device. Serve without \
                 --speculative, or set AVAROK_EMBED_HOST_OFFLOAD=0 and size the serve for the \
                 device table.",
                refused.join(", ")
            );
        }
        let mut entries = self.entries.write();
        for e in entries.iter_mut().filter(|e| !e.device_freed) {
            let current = store.get(&e.store_name).ok().map(|t| t.ptr);
            if current != Some(e.store_ptr) {
                continue;
            }
            let name = e.store_name.clone();
            let (count, bytes) = store.free_matching(gpu, |n| n == name).with_context(|| {
                format!("embed_host_offload: freeing the device copy of {name}")
            })?;
            if count > 0 {
                e.device_freed = true;
                self.device_bytes_freed
                    .fetch_add(bytes as u64, Ordering::Relaxed);
                tracing::info!(
                    "embed_host_offload: freed the device copy of {name} ({:.2} GiB of VRAM)",
                    bytes as f64 / GIB
                );
            }
        }
        Ok(())
    }

    /// [`super::release_view`] against this registry.
    pub fn release_view(&self, gpu: &dyn GpuBackend, view: DevicePtr) -> Result<()> {
        let entry = {
            let mut entries = self.entries.write();
            match entries.iter().position(|e| e.table.view() == view) {
                Some(i) => entries.swap_remove(i),
                None => return Ok(()),
            }
        };
        let bytes = entry.table.bytes() as u64;
        entry.table.release(gpu)?;
        self.resident_bytes.fetch_sub(bytes, Ordering::Relaxed);
        Ok(())
    }

    /// [`super::copy_host_row`] against this registry.
    pub fn copy_host_row(
        &self,
        gpu: &dyn GpuBackend,
        table: DevicePtr,
        token: u32,
        row_bytes: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let entries = self.entries.read();
        let Some(e) = entries.iter().find(|e| e.table.view() == table) else {
            return Ok(false);
        };
        anyhow::ensure!(
            row_bytes == e.table.hidden() * 2,
            "embed_host_offload: a {row_bytes}-byte row requested from a table of {}-element rows",
            e.table.hidden()
        );
        let Some(row) = e.table.row(token) else {
            bail!(
                "embed_host_offload: token id {token} is outside the {}-row embedding table",
                e.table.rows()
            );
        };
        // The region lives until `release_view`, which the model calls only at
        // teardown, so the source outlives the next synchronization.
        gpu.copy_h2d_async_retained(row, out, stream)?;
        self.host_row_copies.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    /// Host bytes of every installed table.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Relaxed)
    }

    /// Installs the backend refused.
    pub fn refusals(&self) -> u64 {
        self.refusals.load(Ordering::Relaxed)
    }

    /// Device bytes freed by moving tables to the host.
    pub fn device_bytes_freed(&self) -> u64 {
        self.device_bytes_freed.load(Ordering::Relaxed)
    }

    /// Single rows served from host memory by [`Registry::copy_host_row`].
    pub fn host_row_copies(&self) -> u64 {
        self.host_row_copies.load(Ordering::Relaxed)
    }

    /// Installed tables.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// No table is installed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `/metrics` text of this registry.
    pub fn render(&self, out: &mut String) {
        let series: [(&str, &str, &str, u64); 4] = [
            (
                "atlas_embed_host_resident_bytes",
                "gauge",
                "Host bytes of embedding tables served from mapped pinned memory (row embed_host_offload).",
                self.resident_bytes(),
            ),
            (
                "atlas_embed_host_device_bytes_freed_total",
                "counter",
                "Device bytes freed by moving embedding tables to mapped host memory.",
                self.device_bytes_freed(),
            ),
            (
                "atlas_embed_host_refusals_total",
                "counter",
                "Embedding tables the backend refused to map; each stayed on the device.",
                self.refusals(),
            ),
            (
                "atlas_embed_host_row_copies_total",
                "counter",
                "Single embedding rows copied host to device by the central embed path.",
                self.host_row_copies(),
            ),
        ];
        for (name, kind, help, v) in series {
            let _ = writeln!(
                out,
                "# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}"
            );
        }
    }
}
