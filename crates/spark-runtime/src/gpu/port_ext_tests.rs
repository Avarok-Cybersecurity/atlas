// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: CPU tests of the port sprint's `GpuBackend` extensions (contract
//! 3.7.1): the topology signature against a hand-computed oracle, every trait default
//! (through a backend that overrides nothing optional). The mock's batched-copy, graph
//! and event emulation and the tier-owned host registry are in `port_ext_mock_tests.rs`.

use anyhow::Result;

use super::mock::MockGpuBackend;
use super::port_ext::{NODE_TYPE_KERNEL, NODE_TYPE_MEMCPY, NODE_TYPE_MEMSET, topology_signature};
use super::*;

/// A backend that implements only the REQUIRED trait methods, so every port extension
/// runs its default body. Device memory is the mock's.
struct DefaultsOnly(MockGpuBackend);

impl GpuBackend for DefaultsOnly {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.0.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.0.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.0.free(ptr)
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.0.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.0.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.0.copy_d2d(src, dst, bytes)
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
        self.0.launch(func, grid, block, shared_mem, stream, params)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.0.synchronize(stream)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.0.kernel(module, func_name)
    }
    fn has_module(&self, module: &str) -> bool {
        self.0.has_module(module)
    }
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        self.0.op_cache()
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.0.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.0.memset_async(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.0.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.0.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.0.sm_count()
    }
}

pub(super) fn pattern(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(seed))
        .collect()
}

/// Three device regions (one empty) filled with distinct patterns, and the entries that
/// copy each into its own host buffer.
pub(super) struct Scatter {
    pub(super) hosts: Vec<Vec<u8>>,
    pub(super) entries: Vec<CopyEntry>,
    pub(super) expected: Vec<Vec<u8>>,
}

pub(super) fn scatter_setup(g: &dyn GpuBackend) -> Scatter {
    let sizes = [96usize, 0, 1000];
    let base = g.alloc(4096).unwrap();
    let mut hosts: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0u8; n]).collect();
    let mut expected = Vec::new();
    let mut entries = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        let dev = base.offset(1024 * i + 8);
        let p = pattern(n, i as u8 * 50 + 1);
        if n > 0 {
            g.copy_h2d(&p, dev).unwrap();
        }
        let host = if n > 0 {
            hosts[i].as_mut_ptr() as usize
        } else {
            0
        };
        entries.push(CopyEntry {
            dev,
            host,
            bytes: n,
        });
        expected.push(p);
    }
    Scatter {
        hosts,
        entries,
        expected,
    }
}

// ── topology signature ──

#[test]
fn signature_matches_a_hand_computed_fnv() {
    // Oracle: the Cinference recurrence written out step by step, then the count.
    const OFF: u64 = 1_469_598_103_934_665_603;
    const P: u64 = 1_099_511_628_211;
    let k = (u64::from(NODE_TYPE_KERNEL) + 1).wrapping_mul(1_000_003) ^ 0xABCD;
    let m = u64::from(NODE_TYPE_MEMSET) + 1;
    let mut h = (OFF ^ k).wrapping_mul(P);
    h = (h ^ m).wrapping_mul(P);
    h = (h ^ 2).wrapping_mul(P);
    assert_eq!(
        topology_signature([(NODE_TYPE_KERNEL, 0xABCD), (NODE_TYPE_MEMSET, 0)]),
        h
    );
    // No nodes: the offset basis mixed with a count of 0 (x ^ 0 == x).
    assert_eq!(topology_signature(std::iter::empty()), OFF.wrapping_mul(P));
}

#[test]
fn signature_separates_count_type_function_and_order() {
    let a = [
        (NODE_TYPE_KERNEL, 7u64),
        (NODE_TYPE_MEMCPY, 0),
        (NODE_TYPE_KERNEL, 9),
    ];
    let base = topology_signature(a);
    assert_eq!(base, topology_signature(a), "deterministic");
    assert_ne!(base, topology_signature(a[..2].iter().copied()), "count");
    let other_fn = [
        (NODE_TYPE_KERNEL, 7u64),
        (NODE_TYPE_MEMCPY, 0),
        (NODE_TYPE_KERNEL, 8),
    ];
    assert_ne!(base, topology_signature(other_fn), "kernel function");
    let other_ty = [
        (NODE_TYPE_KERNEL, 7u64),
        (NODE_TYPE_MEMSET, 0),
        (NODE_TYPE_KERNEL, 9),
    ];
    assert_ne!(base, topology_signature(other_ty), "node type");
    let swapped = [
        (NODE_TYPE_KERNEL, 9u64),
        (NODE_TYPE_MEMCPY, 0),
        (NODE_TYPE_KERNEL, 7),
    ];
    assert_ne!(base, topology_signature(swapped), "order");
    // A non-kernel node's "function" is not part of the class.
    let noisy = [
        (NODE_TYPE_KERNEL, 7u64),
        (NODE_TYPE_MEMCPY, 1234),
        (NODE_TYPE_KERNEL, 9),
    ];
    assert_eq!(base, topology_signature(noisy), "memcpy payload ignored");
}

// ── trait defaults ──

#[test]
fn defaults_refuse_templates_and_never_update() {
    let g = DefaultsOnly(MockGpuBackend::new());
    assert!(g.end_capture_template(5).is_err());
    assert!(g.end_capture_graph(5).is_err());
    assert!(g.graph_instantiate(&GraphTemplate(1)).is_err());
    assert!(g.graph_topology_signature(&GraphTemplate(1)).is_err());
    assert!(
        !g.graph_exec_update(GraphHandle(1), &GraphTemplate(1))
            .unwrap()
    );
    g.graph_upload(GraphHandle(1), 5).unwrap();
    g.graph_template_destroy(GraphTemplate(1)).unwrap();
    assert!(!g.supports_graph_exec_update());
    assert!(!g.supports_copy_batch());
    assert!(
        g.create_timing_event().is_err(),
        "timing events: unsupported"
    );
    assert!(g.event_elapsed_ms(0, 0).is_err());
}

#[test]
fn default_batched_d2h_loops_and_moves_the_bytes() {
    let g = DefaultsOnly(MockGpuBackend::new());
    let sc = scatter_setup(&g);
    let before = g.0.d2h_blocking_count();
    g.copy_batch_d2h_async(&sc.entries, 7).unwrap();
    assert_eq!(
        sc.hosts, sc.expected,
        "every entry landed in its own host buffer"
    );
    // The default forwards to copy_d2h_async, whose default is the blocking copy: one per
    // non-empty entry, the empty one skipped.
    assert_eq!(g.0.d2h_blocking_count() - before, 2);
}

#[test]
fn default_batched_h2d_loops_and_moves_the_bytes() {
    let g = DefaultsOnly(MockGpuBackend::new());
    let dev = g.alloc(2048).unwrap();
    let a = pattern(300, 3);
    let b = pattern(5, 9);
    let entries = [
        CopyEntry {
            dev: dev.offset(1000),
            host: a.as_ptr() as usize,
            bytes: a.len(),
        },
        CopyEntry {
            dev,
            host: b.as_ptr() as usize,
            bytes: b.len(),
        },
    ];
    let before = g.0.h2d_count();
    g.copy_batch_h2d_async(&entries, 7, true).unwrap();
    let all = g.0.read_alloc(dev).unwrap();
    assert_eq!(&all[1000..1300], &a[..]);
    assert_eq!(&all[..5], &b[..]);
    assert_eq!(g.0.h2d_count() - before, 2);
}

#[test]
fn batches_reject_null_and_wrapping_entries() {
    let g = DefaultsOnly(MockGpuBackend::new());
    let dev = g.alloc(64).unwrap();
    let mut host = vec![0u8; 64];
    let h = host.as_mut_ptr() as usize;
    let null_dev = [CopyEntry {
        dev: DevicePtr::NULL,
        host: h,
        bytes: 8,
    }];
    let null_host = [CopyEntry {
        dev,
        host: 0,
        bytes: 8,
    }];
    let wraps = [CopyEntry {
        dev,
        host: usize::MAX - 2,
        bytes: 8,
    }];
    assert!(g.copy_batch_d2h_async(&null_dev, 1).is_err());
    assert!(g.copy_batch_d2h_async(&null_host, 1).is_err());
    assert!(g.copy_batch_h2d_async(&wraps, 1, false).is_err());
    // Empty entries pass even with null pointers: they are skipped.
    let empty = [CopyEntry {
        dev: DevicePtr::NULL,
        host: 0,
        bytes: 0,
    }];
    g.copy_batch_d2h_async(&empty, 1).unwrap();
    g.copy_batch_d2h_async(&[], 1).unwrap();
    assert!(
        host.iter().all(|&b| b == 0),
        "no refused batch wrote anything"
    );
}

#[test]
fn default_event_query_waits_and_reports_done() {
    let g = DefaultsOnly(MockGpuBackend::new());
    let ev = g.create_event().unwrap();
    g.record_event(ev, 3).unwrap();
    assert!(g.event_query(ev).unwrap());
}

#[test]
fn default_copy_stream_is_create_stream() {
    let g = DefaultsOnly(MockGpuBackend::new());
    assert_eq!(g.create_copy_stream().unwrap(), g.create_stream().unwrap());
}

#[test]
fn default_host_register_records_a_tier_owned_region() {
    let g = DefaultsOnly(MockGpuBackend::new());
    let v = vec![0u8; 4096];
    let p = v.as_ptr() as usize;
    g.host_register(p, v.len()).unwrap();
    assert!(crate::pinned_hosts::is_pinned(&v[10..20]));
    assert!(crate::pinned_hosts::is_tier_owned(&v));
    assert!(!crate::pinned_hosts::needs_transient_sync(&v));
    assert!(g.host_register(p + 100, 10).is_err(), "overlap refused");
    assert!(g.host_register(p, 0).is_err(), "empty refused");
    assert!(g.host_unregister(p + 1).is_err(), "not a base");
    g.host_unregister(p).unwrap();
    assert!(!crate::pinned_hosts::is_pinned(&v));
    assert!(g.host_unregister(p).is_err(), "already unregistered");
}
