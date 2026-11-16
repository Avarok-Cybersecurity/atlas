// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: CPU tests of the mock backend's emulation of the port sprint's `GpuBackend`
//! extensions (contract 3.7.1): batched copies, graph capture, update and fallback, held events,
//! copy streams and the tier-owned host registry. The topology-signature and trait-default tests
//! live in `port_ext_tests.rs`, whose `pattern` and `scatter_setup` helpers this file shares.

use super::mock::MockGpuBackend;
use super::port_ext::{NODE_TYPE_KERNEL, topology_signature};
use super::port_ext_tests::{pattern, scatter_setup};
use super::*;

#[test]
fn mock_batched_and_loop_paths_move_identical_bytes() {
    let looped = MockGpuBackend::new();
    let lp = scatter_setup(&looped);
    assert!(!looped.supports_copy_batch(), "off by default");
    looped.copy_batch_d2h_async(&lp.entries, 11).unwrap();
    assert_eq!(looped.d2h_async_count(), 2, "the loop shape");
    assert_eq!(looped.copy_batch_call_count(), 0);

    let batched = MockGpuBackend::new();
    batched.set_copy_batch_supported(true);
    let bt = scatter_setup(&batched);
    batched.copy_batch_d2h_async(&bt.entries, 11).unwrap();
    assert_eq!(batched.d2h_async_count(), 0, "no per-copy enqueue");
    assert_eq!(batched.copy_batch_call_count(), 1, "one call");
    assert_eq!(batched.copy_batch_entry_count(), 2, "empty entry skipped");
    assert_eq!(batched.copy_batch_streams(), vec![11]);
    assert_eq!(lp.hosts, lp.expected);
    assert_eq!(bt.hosts, lp.hosts, "same bytes either way");

    // And back up: H2D through both paths gives the same device image.
    let up = pattern(777, 42);
    let image = |g: &MockGpuBackend| {
        let dev = g.alloc(1024).unwrap();
        let e = [CopyEntry {
            dev: dev.offset(100),
            host: up.as_ptr() as usize,
            bytes: up.len(),
        }];
        g.copy_batch_h2d_async(&e, 2, true).unwrap();
        g.read_alloc(dev).unwrap()
    };
    assert_eq!(image(&looped), image(&batched));
    assert_eq!(batched.copy_batch_call_count(), 2);
}

#[test]
fn mock_batch_rejects_an_overrun() {
    let g = MockGpuBackend::new();
    g.set_copy_batch_supported(true);
    let dev = g.alloc(64).unwrap();
    let mut host = vec![0u8; 128];
    let e = [CopyEntry {
        dev: dev.offset(32),
        host: host.as_mut_ptr() as usize,
        bytes: 64,
    }];
    assert!(g.copy_batch_d2h_async(&e, 1).is_err());
    assert_eq!(
        g.copy_batch_call_count(),
        0,
        "a failed batch is not counted"
    );
}

#[test]
fn mock_graph_capture_update_and_fallback() {
    let g = MockGpuBackend::new();
    let s = 9;
    let capture = |kernels: &[u64]| {
        g.begin_capture(s).unwrap();
        for &k in kernels {
            g.launch_typed(KernelHandle(k), [1, 1, 1], [32, 1, 1], 0, s, &[])
                .unwrap();
        }
        // A launch on another stream is not part of this capture.
        g.launch_typed(KernelHandle(0xFFFF), [1, 1, 1], [1, 1, 1], 0, s + 1, &[])
            .unwrap();
    };

    capture(&[1, 2, 3]);
    let (exec, t1) = g.end_capture_template(s).unwrap();
    let expected = topology_signature([
        (NODE_TYPE_KERNEL, 1),
        (NODE_TYPE_KERNEL, 2),
        (NODE_TYPE_KERNEL, 3),
    ]);
    assert_eq!(g.graph_topology_signature(&t1).unwrap(), expected);
    assert_eq!(g.graph_instantiation_count(), 1);
    g.graph_upload(exec, s).unwrap();
    assert_eq!(g.graph_upload_count(), 1);

    // Same topology, captured without instantiating: the update succeeds.
    capture(&[1, 2, 3]);
    let t2 = g.end_capture_graph(s).unwrap();
    assert!(g.supports_graph_exec_update());
    assert!(g.graph_exec_update(exec, &t2).unwrap());
    assert_eq!(g.graph_instantiation_count(), 1, "no new instantiation");

    // A different topology is refused; so is a driver refusal on a matching one.
    capture(&[1, 2, 3, 4]);
    let t3 = g.end_capture_graph(s).unwrap();
    assert!(!g.graph_exec_update(exec, &t3).unwrap());
    g.fail_next_graph_update();
    assert!(!g.graph_exec_update(exec, &t2).unwrap());
    assert!(
        g.graph_exec_update(exec, &t2).unwrap(),
        "the refusal was one-shot"
    );
    assert_eq!(g.graph_update_count(), 2);
    assert_eq!(g.graph_update_reject_count(), 2);

    // Fallback as a cache would do it: instantiate the new template.
    let exec3 = g.graph_instantiate(&t3).unwrap();
    assert_eq!(g.live_graph_exec_count(), 2);

    for t in [t1, t2, t3] {
        g.graph_template_destroy(t).unwrap();
    }
    assert!(
        g.graph_template_destroy(GraphTemplate(exec3.0)).is_err(),
        "an exec id is not a template"
    );
    assert_eq!(g.live_graph_template_count(), 0);
    g.destroy_graph(exec).unwrap();
    g.destroy_graph(exec3).unwrap();
    assert_eq!(g.live_graph_exec_count(), 0);

    capture(&[1, 2, 3]);
    let t4 = g.end_capture_graph(s).unwrap();
    assert!(g.graph_exec_update(exec, &t4).is_err(), "destroyed exec");
    g.graph_template_destroy(t4).unwrap();
}

#[test]
fn mock_capture_misuse_fails_loudly() {
    let g = MockGpuBackend::new();
    assert!(g.end_capture_graph(3).is_err(), "not capturing");
    g.begin_capture(3).unwrap();
    let t = g.end_capture_graph(3).unwrap();
    assert_eq!(
        g.graph_topology_signature(&t).unwrap(),
        topology_signature(std::iter::empty()),
        "an empty capture has the empty signature"
    );
    g.graph_template_destroy(GraphTemplate(t.0)).unwrap();
    assert!(g.graph_template_destroy(t).is_err(), "double destroy");
    // The old `end_capture` default is untouched: GraphHandle(0), destroy a no-op.
    g.begin_capture(3).unwrap();
    let h = g.end_capture(3).unwrap();
    assert_eq!(h.0, 0);
    g.destroy_graph(h).unwrap();
}

#[test]
fn mock_update_unsupported_returns_false() {
    let g = MockGpuBackend::new();
    g.set_graph_update_supported(false);
    g.begin_capture(1).unwrap();
    let (exec, t) = g.end_capture_template(1).unwrap();
    assert!(!g.supports_graph_exec_update());
    assert!(!g.graph_exec_update(exec, &t).unwrap());
    g.graph_template_destroy(t).unwrap();
    g.destroy_graph(exec).unwrap();
}

#[test]
fn mock_held_events_stay_pending_until_completed() {
    let g = MockGpuBackend::new();
    // Defaults first: id 0 and immediately complete, as the trait default.
    let e0 = g.create_event().unwrap();
    g.record_event(e0, 1).unwrap();
    assert_eq!(e0, 0);
    assert!(g.event_query(e0).unwrap());

    g.set_unique_events(true);
    g.hold_events(true);
    let a = g.create_event().unwrap();
    let b = g.create_event().unwrap();
    assert!(a != 0 && b != 0 && a != b);
    g.record_event(a, 1).unwrap();
    g.record_event(b, 1).unwrap();
    assert!(!g.event_query(a).unwrap());
    assert_eq!(g.pending_event_count(), 2);
    g.complete_event(a);
    assert!(g.event_query(a).unwrap());
    assert!(!g.event_query(b).unwrap());
    g.event_synchronize(b).unwrap();
    assert!(g.event_query(b).unwrap());
    g.record_event(a, 1).unwrap();
    g.complete_all_events();
    assert_eq!(g.pending_event_count(), 0);
}

#[test]
fn mock_copy_streams_are_distinct_and_not_legacy() {
    let g = MockGpuBackend::new();
    let a = g.create_copy_stream().unwrap();
    let b = g.create_copy_stream().unwrap();
    assert!(a > 1 && b > 1 && a != b);
}

#[test]
fn mock_host_register_uses_the_tier_registry() {
    let g = MockGpuBackend::new();
    let v = vec![0u8; 8192];
    g.host_register(v.as_ptr() as usize, v.len()).unwrap();
    assert_eq!(crate::pinned_hosts::region_len(v.as_ptr()), Some(8192));
    assert!(crate::pinned_hosts::is_tier_owned(&v[4096..]));
    g.host_unregister(v.as_ptr() as usize).unwrap();
    assert_eq!(crate::pinned_hosts::region_len(v.as_ptr()), None);
}
