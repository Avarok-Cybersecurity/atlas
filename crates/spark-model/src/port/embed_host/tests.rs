// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles for the embedding host offload (row
//! `embed_host_offload`) on the mock backend: the gather through the host
//! table reads exactly the bytes the device lookup reads, every decline keeps
//! the device table untouched, refusals and teardown free what they took, and
//! the device copy is freed only when it is the tensor the table came from.
//!
//! The mock cannot alias host memory, so [`FakeMapper`] stands in for the
//! driver: its "alias" is a mock device buffer holding the host bytes at
//! mapping time, which is what a correct mapping reads (nothing writes the
//! table after the load).

use std::alloc::Layout;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use parking_lot::Mutex;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::*;

const EMBED: &str = "model.embed_tokens.weight";
const ROWS: usize = 37;
const HIDDEN: usize = 24;

#[derive(Default)]
struct FakeMapper {
    /// The alias reads one wrong byte (a mapping the device cannot read).
    corrupt: bool,
    /// The backend refuses a device alias (no unified addressing).
    refuse_alias: bool,
    live: Mutex<HashMap<usize, Layout>>,
    frees: AtomicUsize,
}

impl HostMapper for FakeMapper {
    fn alloc(&self, _gpu: &dyn GpuBackend, bytes: usize) -> Result<usize> {
        let layout = Layout::from_size_align(bytes, 64)?;
        // SAFETY: a non-zero layout (callers never request 0 bytes).
        let p = unsafe { std::alloc::alloc_zeroed(layout) };
        anyhow::ensure!(!p.is_null(), "fake host alloc failed");
        self.live.lock().insert(p as usize, layout);
        Ok(p as usize)
    }

    fn device_view(&self, gpu: &dyn GpuBackend, host: usize, bytes: usize) -> Result<DevicePtr> {
        anyhow::ensure!(!self.refuse_alias, "fake mapper: alias refused");
        let view = gpu.alloc(bytes)?;
        // SAFETY: `host` is a live allocation of `bytes` from `alloc` above.
        let mut data = unsafe { std::slice::from_raw_parts(host as *const u8, bytes) }.to_vec();
        if self.corrupt {
            data[0] ^= 0xff;
        }
        gpu.copy_h2d(&data, view)?;
        Ok(view)
    }

    fn free(
        &self,
        gpu: &dyn GpuBackend,
        host: usize,
        view: DevicePtr,
        _bytes: usize,
    ) -> Result<()> {
        if !view.is_null() {
            gpu.free(view)?;
        }
        let layout = self
            .live
            .lock()
            .remove(&host)
            .context("fake mapper: freeing an address it never allocated")?;
        // SAFETY: allocated by `alloc` with exactly this layout, freed once.
        unsafe { std::alloc::dealloc(host as *mut u8, layout) };
        self.frees.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A deterministic table that differs in every row and column.
fn table(rows: usize, hidden: usize) -> Vec<u8> {
    let mut x: u32 = 0x9e37_79b9;
    (0..rows * hidden * 2)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 11) as u8
        })
        .collect()
}

fn device_tensor(
    gpu: &MockGpuBackend,
    bytes: &[u8],
    shape: Vec<usize>,
    dtype: WeightDtype,
) -> WeightTensor {
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, ptr).unwrap();
    WeightTensor { ptr, shape, dtype }
}

/// A store holding the embedding table and (unless `tied`) a separate head.
fn store_with(gpu: &MockGpuBackend, data: &[u8], tied: bool) -> WeightStore {
    let mut m = HashMap::new();
    m.insert(
        EMBED.to_string(),
        device_tensor(gpu, data, vec![ROWS, HIDDEN], WeightDtype::BF16),
    );
    if !tied {
        m.insert(
            "lm_head.weight".to_string(),
            device_tensor(gpu, data, vec![ROWS, HIDDEN], WeightDtype::BF16),
        );
    }
    WeightStore::from_map(m)
}

fn read(gpu: &MockGpuBackend, src: DevicePtr, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    gpu.copy_d2h(src, &mut out).unwrap();
    out
}

#[test]
fn the_host_table_holds_the_checkpoint_bytes_and_its_alias_reads_them() {
    let gpu = MockGpuBackend::new();
    let data = table(ROWS, HIDDEN);
    let mapper = Arc::new(FakeMapper::default());
    let t = HostEmbedding::from_bf16_host_with(&gpu, mapper.clone(), ROWS, HIDDEN, &data).unwrap();
    assert_eq!(t.bytes(), data.len());
    assert_eq!(read(&gpu, t.view(), data.len()), data);
    for token in [0u32, 1, 17, (ROWS - 1) as u32] {
        let r = token as usize * HIDDEN * 2;
        assert_eq!(
            t.row(token).unwrap(),
            &data[r..r + HIDDEN * 2],
            "row {token}"
        );
    }
    assert!(t.row(ROWS as u32).is_none());
    t.release(&gpu).unwrap();
    assert_eq!(mapper.frees.load(Ordering::Relaxed), 1);
    assert!(mapper.live.lock().is_empty());
}

#[test]
fn a_wrong_sized_host_table_is_refused_before_any_allocation() {
    let gpu = MockGpuBackend::new();
    let mapper = Arc::new(FakeMapper::default());
    let short = vec![0u8; ROWS * HIDDEN * 2 - 2];
    assert!(
        HostEmbedding::from_bf16_host_with(&gpu, mapper.clone(), ROWS, HIDDEN, &short).is_err()
    );
    assert!(HostEmbedding::from_bf16_host_with(&gpu, mapper.clone(), 0, HIDDEN, &[]).is_err());
    assert!(mapper.live.lock().is_empty());
}

/// The oracle the plan names: every gather through the installed table (the
/// kernels' device-pointer read, and the central embed's host row copy) is
/// byte-equal to the same lookup in the device table it replaced.
#[test]
fn every_gather_through_the_host_table_equals_the_device_lookup() {
    let gpu = MockGpuBackend::new();
    let data = table(ROWS, HIDDEN);
    let store = store_with(&gpu, &data, false);
    let device_table = store.get(EMBED).unwrap().ptr;
    let reg = Registry::default();
    let dense = reg
        .install(
            &gpu,
            Arc::new(FakeMapper::default()),
            &store,
            EMBED,
            ROWS,
            HIDDEN,
        )
        .unwrap()
        .expect("installed");
    assert_ne!(dense.weight, device_table);
    let row_bytes = HIDDEN * 2;
    let out_dev = gpu.alloc(row_bytes).unwrap();
    let out_host = gpu.alloc(row_bytes).unwrap();
    let out_ref = gpu.alloc(row_bytes).unwrap();
    for token in 0..ROWS as u32 {
        let off = token as usize * row_bytes;
        // Reference: the device table, as the row-off path reads it.
        gpu.copy_d2d_async(device_table.offset(off), out_ref, row_bytes, 0)
            .unwrap();
        // A kernel or a device-to-device copy through the alias.
        gpu.copy_d2d_async(dense.weight.offset(off), out_dev, row_bytes, 0)
            .unwrap();
        // The central embed's branch: a host-to-device copy of the row.
        assert!(
            reg.copy_host_row(&gpu, dense.weight, token, row_bytes, out_host, 0)
                .unwrap()
        );
        let want = read(&gpu, out_ref, row_bytes);
        assert_eq!(want, &data[off..off + row_bytes]);
        assert_eq!(read(&gpu, out_dev, row_bytes), want, "alias row {token}");
        assert_eq!(read(&gpu, out_host, row_bytes), want, "host row {token}");
    }
    assert_eq!(reg.host_row_copies(), ROWS as u64);
    assert_eq!(reg.resident_bytes(), data.len() as u64);
}

#[test]
fn a_device_table_is_not_served_by_the_host_branch() {
    let gpu = MockGpuBackend::new();
    let reg = Registry::default();
    let device = gpu.alloc(HIDDEN * 2).unwrap();
    let out = gpu.alloc(HIDDEN * 2).unwrap();
    assert!(
        !reg.copy_host_row(&gpu, device, 0, HIDDEN * 2, out, 0)
            .unwrap()
    );
    assert_eq!(reg.host_row_copies(), 0);
}

#[test]
fn an_out_of_range_token_or_row_width_is_an_error_not_a_stray_read() {
    let gpu = MockGpuBackend::new();
    let data = table(ROWS, HIDDEN);
    let store = store_with(&gpu, &data, false);
    let reg = Registry::default();
    let dense = reg
        .install(
            &gpu,
            Arc::new(FakeMapper::default()),
            &store,
            EMBED,
            ROWS,
            HIDDEN,
        )
        .unwrap()
        .unwrap();
    let out = gpu.alloc(HIDDEN * 2).unwrap();
    assert!(
        reg.copy_host_row(&gpu, dense.weight, ROWS as u32, HIDDEN * 2, out, 0)
            .is_err()
    );
    assert!(
        reg.copy_host_row(&gpu, dense.weight, 0, HIDDEN, out, 0)
            .is_err()
    );
}

#[test]
fn a_tied_head_keeps_the_table_on_the_device() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, &table(ROWS, HIDDEN), true);
    let reg = Registry::default();
    let mapper = Arc::new(FakeMapper::default());
    let got = reg
        .install(&gpu, mapper.clone(), &store, EMBED, ROWS, HIDDEN)
        .unwrap();
    assert!(got.is_none());
    assert!(reg.is_empty());
    assert!(
        mapper.live.lock().is_empty(),
        "nothing pinned for a declined table"
    );
    assert_eq!(reg.refusals(), 0, "a tied head is a decline, not a refusal");
}

#[test]
fn a_table_that_is_not_bf16_rows_by_hidden_is_declined() {
    let gpu = MockGpuBackend::new();
    let data = table(ROWS, HIDDEN);
    for (shape, dtype) in [
        (vec![ROWS, HIDDEN], WeightDtype::FP32),
        (vec![HIDDEN, ROWS], WeightDtype::BF16),
        (vec![ROWS * HIDDEN], WeightDtype::BF16),
    ] {
        let mut m = HashMap::new();
        m.insert(
            EMBED.to_string(),
            device_tensor(&gpu, &data, shape.clone(), dtype),
        );
        m.insert(
            "lm_head.weight".to_string(),
            device_tensor(&gpu, &data, vec![ROWS, HIDDEN], WeightDtype::BF16),
        );
        let store = WeightStore::from_map(m);
        let reg = Registry::default();
        let got = reg
            .install(
                &gpu,
                Arc::new(FakeMapper::default()),
                &store,
                EMBED,
                ROWS,
                HIDDEN,
            )
            .unwrap();
        assert!(got.is_none(), "{shape:?} {dtype:?}");
        assert!(reg.is_empty());
    }
}

#[test]
fn a_refused_mapping_frees_its_host_memory_and_is_counted() {
    for mapper in [
        FakeMapper {
            refuse_alias: true,
            ..Default::default()
        },
        FakeMapper {
            corrupt: true,
            ..Default::default()
        },
    ] {
        let gpu = MockGpuBackend::new();
        let store = store_with(&gpu, &table(ROWS, HIDDEN), false);
        let reg = Registry::default();
        let mapper = Arc::new(mapper);
        let got = reg
            .install(&gpu, mapper.clone(), &store, EMBED, ROWS, HIDDEN)
            .unwrap();
        assert!(got.is_none());
        assert_eq!(reg.refusals(), 1);
        assert!(reg.is_empty());
        assert!(mapper.live.lock().is_empty(), "the pinned region was freed");
        assert_eq!(mapper.frees.load(Ordering::Relaxed), 1);
        // The device table is untouched and still in the store.
        assert!(store.contains(EMBED));
    }
}

/// The production mapper on a backend without mapped memory (the mock, like
/// Metal) is a refusal, never an error that stops a plain serve.
#[test]
fn the_backend_mapper_on_a_backend_without_aliases_is_a_refusal() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, &table(ROWS, HIDDEN), false);
    let reg = Registry::default();
    let got = reg
        .install(&gpu, Arc::new(BackendMapper), &store, EMBED, ROWS, HIDDEN)
        .unwrap();
    assert!(got.is_none());
    assert_eq!(reg.refusals(), 1);
}

#[test]
fn after_load_frees_the_device_copy_once() {
    let gpu = MockGpuBackend::new();
    let data = table(ROWS, HIDDEN);
    let mut store = store_with(&gpu, &data, false);
    let reg = Registry::default();
    let dense = reg
        .install(
            &gpu,
            Arc::new(FakeMapper::default()),
            &store,
            EMBED,
            ROWS,
            HIDDEN,
        )
        .unwrap()
        .unwrap();
    let before = store.resident_bytes();
    reg.after_load(&mut store, &gpu, true).unwrap();
    assert!(!store.contains(EMBED));
    assert!(store.contains("lm_head.weight"));
    assert_eq!(before - store.resident_bytes(), data.len());
    assert_eq!(reg.device_bytes_freed(), data.len() as u64);
    // A second call frees nothing more, and the host table still serves.
    reg.after_load(&mut store, &gpu, true).unwrap();
    assert_eq!(reg.device_bytes_freed(), data.len() as u64);
    assert_eq!(read(&gpu, dense.weight, data.len()), data);
}

#[test]
fn after_load_never_frees_a_tensor_the_table_was_not_copied_from() {
    let gpu = MockGpuBackend::new();
    let data = table(ROWS, HIDDEN);
    let first = store_with(&gpu, &data, false);
    let reg = Registry::default();
    reg.install(
        &gpu,
        Arc::new(FakeMapper::default()),
        &first,
        EMBED,
        ROWS,
        HIDDEN,
    )
    .unwrap()
    .unwrap();
    // The load failed before `after_load`; the next load has a tensor of the
    // same name at another address.
    let mut second = store_with(&gpu, &data, false);
    reg.after_load(&mut second, &gpu, false).unwrap();
    assert!(second.contains(EMBED));
    assert_eq!(reg.device_bytes_freed(), 0);
}

#[test]
fn a_refusal_fails_a_speculative_start_and_not_a_plain_one() {
    let gpu = MockGpuBackend::new();
    let refuse = || {
        Arc::new(FakeMapper {
            refuse_alias: true,
            ..Default::default()
        })
    };
    let mut store = store_with(&gpu, &table(ROWS, HIDDEN), false);
    let reg = Registry::default();
    assert!(
        reg.install(&gpu, refuse(), &store, EMBED, ROWS, HIDDEN)
            .unwrap()
            .is_none()
    );
    let err = reg.after_load(&mut store, &gpu, true).unwrap_err();
    assert!(format!("{err:#}").contains("--speculative"), "{err:#}");
    assert!(store.contains(EMBED), "the device table is kept");

    let reg = Registry::default();
    assert!(
        reg.install(&gpu, refuse(), &store, EMBED, ROWS, HIDDEN)
            .unwrap()
            .is_none()
    );
    reg.after_load(&mut store, &gpu, false).unwrap();
    // The refusal list is per load: the next speculative load is not blamed.
    reg.after_load(&mut store, &gpu, true).unwrap();
}

#[test]
fn teardown_releases_the_host_table_and_ignores_device_tables() {
    let gpu = MockGpuBackend::new();
    let store = store_with(&gpu, &table(ROWS, HIDDEN), false);
    let reg = Registry::default();
    let mapper = Arc::new(FakeMapper::default());
    let dense = reg
        .install(&gpu, mapper.clone(), &store, EMBED, ROWS, HIDDEN)
        .unwrap()
        .unwrap();
    reg.release_view(&gpu, store.get(EMBED).unwrap().ptr)
        .unwrap();
    assert_eq!(reg.len(), 1, "a device table pointer releases nothing");
    reg.release_view(&gpu, dense.weight).unwrap();
    assert!(reg.is_empty());
    assert_eq!(reg.resident_bytes(), 0);
    assert!(mapper.live.lock().is_empty());
    let out = gpu.alloc(HIDDEN * 2).unwrap();
    assert!(
        !reg.copy_host_row(&gpu, dense.weight, 0, HIDDEN * 2, out, 0)
            .unwrap()
    );
}

#[test]
fn the_metrics_name_every_series_and_start_at_zero() {
    let reg = Registry::default();
    let mut out = String::new();
    reg.render(&mut out);
    for name in [
        "atlas_embed_host_resident_bytes",
        "atlas_embed_host_device_bytes_freed_total",
        "atlas_embed_host_refusals_total",
        "atlas_embed_host_row_copies_total",
    ] {
        assert!(out.contains(&format!("# TYPE {name} ")), "{name}\n{out}");
        assert!(out.contains(&format!("\n{name} 0\n")), "{name}\n{out}");
    }
}

#[test]
fn the_predicted_saving_is_the_bf16_table_only_with_the_row_on_and_an_untied_head() {
    // Qwen3.8-27B: 248,320 x 5,120 BF16 = 2.37 GiB.
    let b = predicted_freed_bytes_for(true, 248_320, 5_120, false);
    assert_eq!(b, 2_542_796_800);
    assert!((b as f64 / (1u64 << 30) as f64 - 2.368).abs() < 0.001);
    assert_eq!(predicted_freed_bytes_for(false, 248_320, 5_120, false), 0);
    assert_eq!(predicted_freed_bytes_for(true, 248_320, 5_120, true), 0);
}
