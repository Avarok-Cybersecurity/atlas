// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the error-path and scratch-rule half of the GDN W4
//! projection arm's mock-backend tests (`w4_proj.rs`): a pair that does not
//! fit the model, tensor-parallel refusal, the capture-needs-reserved-scratch
//! rule, the shared backend scratch region and the zero-row no-op. The
//! per-call-site tests are in `w4_proj_tests.rs`; the fixtures are in
//! `w4_proj_fake.rs`.

use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::fake::{HostW4, config, decode_ws, decodes, loader_layer, w4_layer, with_ctx};
use super::plan::W4Role;
use super::tests::arena;
use crate::layers::qwen3_ssm::tests::mk_state;
use crate::port::w4;

#[test]
fn a_pair_that_does_not_fit_the_model_errors_instead_of_launching() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let mut layer = loader_layer(&gpu, &c);
    let ((qn, qk), (on, ok)) = (W4Role::Qkvz.dims(&c), W4Role::OutProj.dims(&c));
    let q = HostW4::new(qn + 128, qk, 16, true, false);
    layer.set_w4_proj(q.clone(), HostW4::new(on, ok, 16, true, false));
    let buffers = arena(&gpu);
    let mut state = mk_state(&gpu, &layer, 0);
    let err = with_ctx(&gpu, &c, &buffers, true, false, |ctx| {
        layer
            .ssm_forward(buffers.norm_output(), &mut state, ctx, 0, false)
            .unwrap_err()
    });
    assert!(format!("{err:#}").contains("in_proj_qkvz"), "{err:#}");
    assert!(q.calls().is_empty());
}

#[test]
fn a_tensor_parallel_model_is_refused() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, _q, o) = w4_layer(&gpu, &c, true, false);
    let buffers = arena(&gpu);
    let mut tp = config();
    tp.tp_world_size = 2;
    let err = with_ctx(&gpu, &tp, &buffers, true, false, |ctx| {
        layer
            .w4_out_rows(ctx, buffers.ssm_qkvz(), buffers.moe_output(), 1, 0)
            .unwrap_err()
    });
    assert!(format!("{err:#}").contains("tensor-parallel"), "{err:#}");
    assert!(o.calls().is_empty());
}

#[test]
fn a_capture_needs_the_scratch_reserved_at_load() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, _q, o) = w4_layer(&gpu, &c, true, false);
    let buffers = arena(&gpu);
    let (x, out) = (buffers.ssm_qkvz(), buffers.moe_output());
    let err = with_ctx(&gpu, &c, &buffers, true, true, |ctx| {
        layer.w4_out_rows(ctx, x, out, 1, 0).unwrap_err()
    });
    assert!(format!("{err:#}").contains("prepare_w4_proj"), "{err:#}");
    assert!(o.calls().is_empty());
    let spec = layer.w4_proj_scratch_spec().unwrap();
    assert_eq!(layer.prepare_w4_proj(&gpu, 0).unwrap(), spec.device_bytes());
    let allocs = gpu.alloc_count();
    with_ctx(&gpu, &c, &buffers, true, true, |ctx| {
        layer.w4_out_rows(ctx, x, out, 1, 0).unwrap();
    });
    assert_eq!(
        gpu.alloc_count(),
        allocs,
        "a captured call allocates nothing"
    );
    assert_eq!(decodes(&o.calls()).len(), 1);
    assert_eq!(loader_layer(&gpu, &c).prepare_w4_proj(&gpu, 0).unwrap(), 0);
}

#[test]
fn the_scratch_is_the_shared_backend_region_and_never_moves() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (a, qa, _) = w4_layer(&gpu, &c, true, false);
    let (b, qb, _) = w4_layer(&gpu, &c, true, false);
    let buffers = arena(&gpu);
    let (x, out) = (buffers.norm_output(), buffers.ssm_deinterleaved());
    with_ctx(&gpu, &c, &buffers, true, false, |ctx| {
        for rows in [1, 16, 40] {
            a.w4_qkvz_rows(ctx, x, out, rows, 0).unwrap();
        }
        b.w4_qkvz_rows(ctx, x, out, 1, 0).unwrap();
    });
    let (wa, wb) = (decode_ws(&qa.calls()), decode_ws(&qb.calls()));
    assert!(
        wa.iter().all(|&p| p == wa[0] && p != DevicePtr::NULL),
        "{wa:?}"
    );
    assert_eq!(wb[0], wa[0], "two layers share the backend's regions");
    let spec = a.w4_proj_scratch_spec().unwrap();
    assert_eq!(
        w4::scratch::acquire(&gpu, &spec, 0).unwrap().ws,
        wa[0],
        "the regions every W4 consumer on the backend shares"
    );
}

#[test]
fn zero_rows_launch_nothing() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, q, o) = w4_layer(&gpu, &c, true, false);
    let buffers = arena(&gpu);
    let (launched, allocs) = (gpu.launches_snapshot().len(), gpu.alloc_count());
    with_ctx(&gpu, &c, &buffers, false, false, |ctx| {
        layer
            .w4_qkvz_prefill(
                ctx,
                buffers.norm_output(),
                buffers.ssm_deinterleaved(),
                0,
                0,
            )
            .unwrap();
        layer
            .w4_out_prefill(ctx, buffers.ssm_qkvz(), buffers.moe_output(), 0, 0)
            .unwrap();
    });
    assert!(q.calls().is_empty() && o.calls().is_empty());
    assert_eq!(gpu.launches_snapshot().len(), launched, "no launch");
    assert_eq!(gpu.alloc_count(), allocs, "and no scratch reservation");
}
