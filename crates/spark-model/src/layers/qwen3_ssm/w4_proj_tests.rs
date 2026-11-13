// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the GDN W4 projection arm
//! (`w4_proj.rs`), one per call site plus the setter, the scratch rules and
//! the error paths. The fixtures (a host-oracle `W4Linear`, sentinel BF16
//! handles) are in `w4_proj_fake.rs`; the row plan itself is pinned in
//! `w4_proj_plan_tests.rs`. The row-off byte identity of every existing arm is
//! pinned by the unchanged `fp8_route_dispatch_tests.rs`,
//! `gdn_batched_gemm_tests.rs` and `tests.rs`, plus the BF16 case below.

use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::fake::{
    Call, DEINTERLEAVE, DENSE_GEMV, DENSE_PIPE, HostW4, bf16_launches, bf16_layer, config, decodes,
    fill_rows, launches, loader_layer, read_bf16, w4_layer, with_ctx,
};
use super::plan::W4Role;
use crate::layers::qwen3_ssm::tests::{mk_state, run_batched_verify};
use crate::port::w4::{self, W4Format, W4Linear};

const BF16: usize = 2;

pub(super) fn arena(gpu: &MockGpuBackend) -> BufferArena {
    BufferArena::new(&config(), 64, 4096, 16, 32, gpu).unwrap()
}

#[test]
fn install_on_a_loader_built_layer_frees_nothing_and_reports_the_pair() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, q, o) = w4_layer(&gpu, &c, true, false);
    assert!(layer.w4_proj_installed());
    let affine = W4Format::Affine { bits: 4, group: 64 };
    assert_eq!(layer.w4_proj_formats(), Some((affine, affine)));
    assert_eq!(
        layer.w4_proj_device_bytes(),
        (q.n() * q.k() + o.n() * o.k()) / 2
    );
    let spec = layer.w4_proj_scratch_spec().unwrap();
    assert_eq!(spec.bf16_bytes, (q.n() * q.k()).max(o.n() * o.k()) * BF16);
    let bare = loader_layer(&gpu, &c);
    assert!(!bare.w4_proj_installed());
    assert_eq!(
        (bare.w4_proj_device_bytes(), bare.w4_proj_formats()),
        (0, None)
    );
    assert!(bare.w4_proj_scratch_spec().is_none());
}

#[test]
fn install_drops_the_bf16_slots_and_reports_their_bytes() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let mut layer = bf16_layer(&gpu, &c);
    let ((qn, qk), (on, ok)) = (W4Role::Qkvz.dims(&c), W4Role::OutProj.dims(&c));
    let freed = layer.set_w4_proj(
        HostW4::new(qn, qk, 16, true, false),
        HostW4::new(on, ok, 16, true, false),
    );
    assert_eq!(freed, (qn * qk + on * ok) * BF16);
    assert!(layer.ssm.in_proj_qkvz.weight.is_null() && layer.out_proj_dense.is_none());
    // A second pair is refused and the first one stays.
    let second = HostW4::new(qn, qk, 16, true, false);
    let e = layer
        .try_set_w4_proj(second.clone(), HostW4::new(on, ok, 16, true, false))
        .unwrap_err();
    assert!(format!("{e:#}").contains("already installed"), "{e:#}");
    assert_eq!(
        layer.set_w4_proj(second, HostW4::new(on, ok, 16, true, false)),
        0
    );
    assert!(layer.w4_proj_installed());
}

#[test]
fn a_refused_install_leaves_the_layer_as_it_was() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let ((qn, qk), (on, ok)) = (W4Role::Qkvz.dims(&c), W4Role::OutProj.dims(&c));
    // A native-FP8 layer: the loader's FP8-wins rule never gives it W4.
    let mut fp8 = crate::layers::qwen3_ssm::tests::native_fp8_gdn_layer(&gpu, &c, true, true);
    let e = fp8
        .try_set_w4_proj(
            HostW4::new(qn, qk, 16, true, false),
            HostW4::new(on, ok, 16, true, false),
        )
        .unwrap_err();
    assert!(format!("{e:#}").contains("FP8 QKVZ"), "{e:#}");
    assert!(!fp8.w4_proj_installed() && fp8.qkvz_fp8w.is_some() && fp8.out_proj_fp8w.is_some());
    // A swapped pair on a BF16 layer keeps its BF16 slots.
    let mut bf16 = bf16_layer(&gpu, &c);
    let slot = bf16.ssm.in_proj_qkvz.weight;
    let e = bf16
        .try_set_w4_proj(
            HostW4::new(on, ok, 16, true, false),
            HostW4::new(qn, qk, 16, true, false),
        )
        .unwrap_err();
    assert!(format!("{e:#}").contains("wrong order"), "{e:#}");
    assert_eq!(bf16.ssm.in_proj_qkvz.weight, slot);
    assert!(bf16.out_proj_dense.is_some() && !bf16.w4_proj_installed());
}

#[test]
fn m1_decode_runs_both_projections_on_w4_and_lands_the_rows() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, q, o) = w4_layer(&gpu, &c, true, true);
    let buffers = arena(&gpu);
    fill_rows(&gpu, buffers.norm_output(), 1, c.hidden_size);
    let mut state = mk_state(&gpu, &layer, 0);
    with_ctx(&gpu, &c, &buffers, true, false, |ctx| {
        layer
            .ssm_forward(buffers.norm_output(), &mut state, ctx, 0, false)
            .unwrap();
    });
    assert_eq!(
        decodes(&q.calls()),
        [(1, buffers.norm_output(), buffers.ssm_deinterleaved())]
    );
    let oc = decodes(&o.calls());
    assert!(
        matches!(oc[..], [(1, _, out)] if out == buffers.moe_output()),
        "{oc:?}"
    );
    assert_eq!(
        bf16_launches(&gpu),
        0,
        "no BF16 projection kernel on a W4 layer"
    );
    let x = read_bf16(&gpu, buffers.norm_output(), c.hidden_size).unwrap();
    let got = read_bf16(&gpu, buffers.ssm_deinterleaved(), c.ssm_qkvz_size()).unwrap();
    assert_eq!(
        got,
        q.oracle(&x),
        "the QKVZ rows land where the conv reads them"
    );
}

#[test]
fn without_a_pair_m1_decode_keeps_the_bf16_gemv() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let layer = bf16_layer(&gpu, &c);
    let buffers = arena(&gpu);
    let mut state = mk_state(&gpu, &layer, 0);
    with_ctx(&gpu, &c, &buffers, true, false, |ctx| {
        layer
            .ssm_forward(buffers.norm_output(), &mut state, ctx, 0, false)
            .unwrap();
    });
    assert!(!layer.w4_proj_installed());
    assert_eq!(
        launches(&gpu, DENSE_GEMV),
        2,
        "QKVZ and out_proj, as before the arm existed"
    );
}

/// The batched verify body at ragged ks: every row is a decode row, in
/// chunks of the window, with pointers advancing by whole rows.
#[test]
fn verify_rows_decode_whole_or_in_window_chunks() {
    let c = config();
    let (qn, qk) = W4Role::Qkvz.dims(&c);
    for (ks, want) in [
        (vec![2], vec![2]),
        (vec![4, 3], vec![7]),
        (vec![16, 16, 8], vec![16, 16, 8]),
    ] {
        let gpu = MockGpuBackend::new();
        let (layer, q, o) = w4_layer(&gpu, &c, true, false);
        run_batched_verify(&gpu, &c, &layer, &ks).unwrap();
        let qd = decodes(&q.calls());
        let rows: Vec<usize> = qd.iter().map(|d| d.0).collect();
        assert_eq!(rows, want, "QKVZ at ks={ks:?}");
        let od: Vec<usize> = decodes(&o.calls()).iter().map(|d| d.0).collect();
        assert_eq!(od, want, "out_proj at ks={ks:?}");
        assert_eq!(bf16_launches(&gpu), 0);
        let (x0, o0) = (qd[0].1, qd[0].2);
        let mut start = 0;
        for &(r, x, out) in &qd {
            assert_eq!(
                (x, out),
                (x0.offset(start * qk * BF16), o0.offset(start * qn * BF16))
            );
            start += r;
        }
        assert!(
            q.calls().iter().all(|c| matches!(c, Call::Decode { .. })),
            "never prefill"
        );
    }
}

/// The multi-sequence batched projections: n rows, decode chunks, the
/// buffers the batched mixer hands over, no BF16 launch.
#[test]
fn multi_sequence_projections_decode_every_row() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, q, o) = w4_layer(&gpu, &c, true, false);
    let buffers = arena(&gpu);
    let (h, vd) = (
        c.hidden_size,
        c.linear_num_value_heads * c.linear_value_head_dim,
    );
    with_ctx(&gpu, &c, &buffers, true, false, |ctx| {
        for n in [2usize, 40] {
            let tier = layer.batched_proj_tier(n);
            layer
                .ms_batched_qkvz(
                    ctx,
                    &tier,
                    n,
                    buffers.norm_output(),
                    buffers.ssm_deinterleaved(),
                    c.ssm_qkvz_size(),
                    h,
                    0,
                )
                .unwrap();
            layer
                .ms_batched_out_proj(
                    ctx,
                    &tier,
                    n,
                    buffers.ssm_qkvz(),
                    buffers.moe_output(),
                    h,
                    vd,
                    0,
                )
                .unwrap();
        }
    });
    let rows = |calls: Vec<Call>| decodes(&calls).iter().map(|d| d.0).collect::<Vec<_>>();
    assert_eq!(rows(q.calls()), [2, 16, 16, 8]);
    assert_eq!(rows(o.calls()), [2, 16, 16, 8]);
    assert_eq!(decodes(&q.calls())[0].2, buffers.ssm_deinterleaved());
    assert_eq!(decodes(&o.calls())[0].1, buffers.ssm_qkvz());
    assert_eq!(bf16_launches(&gpu), 0);
}

#[test]
fn prompts_use_prefill_with_the_rows_in_place() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, q, o) = w4_layer(&gpu, &c, true, true);
    let buffers = arena(&gpu);
    let (h, vd) = (
        c.hidden_size,
        c.linear_num_value_heads * c.linear_value_head_dim,
    );
    let nk = c.linear_num_key_heads;
    fill_rows(&gpu, buffers.norm_output(), 28, h);
    with_ctx(&gpu, &c, &buffers, false, false, |ctx| {
        let vpg = c.linear_num_value_heads / nk.max(1);
        layer
            .prefill_qkvz_proj(
                buffers.norm_output(),
                buffers.ssm_deinterleaved(),
                28,
                c.ssm_qkvz_size(),
                h,
                nk,
                c.linear_key_head_dim,
                vpg,
                c.linear_value_head_dim,
                ctx,
                0,
            )
            .unwrap();
        layer
            .prefill_out_proj_dispatch(
                ctx,
                buffers.norm_output(),
                buffers.hidden_states(),
                28,
                h,
                vd,
                0,
            )
            .unwrap();
    });
    assert!(
        matches!(q.calls()[..], [Call::Prefill { m: 28, .. }]),
        "{:?}",
        q.calls()
    );
    assert!(
        matches!(o.calls()[..], [Call::Prefill { m: 28, out, .. }] if out == buffers.hidden_states())
    );
    let x = read_bf16(&gpu, buffers.norm_output(), 28 * h).unwrap();
    let got = read_bf16(&gpu, buffers.ssm_deinterleaved(), 28 * c.ssm_qkvz_size()).unwrap();
    assert_eq!(got, q.oracle(&x));
    assert_eq!(bf16_launches(&gpu), 0);
}

#[test]
fn a_short_prompt_inside_the_window_is_one_decode() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, q, _o) = w4_layer(&gpu, &c, true, false);
    let buffers = arena(&gpu);
    with_ctx(&gpu, &c, &buffers, false, false, |ctx| {
        layer
            .w4_qkvz_prefill(
                ctx,
                buffers.norm_output(),
                buffers.ssm_deinterleaved(),
                9,
                0,
            )
            .unwrap();
    });
    assert_eq!(
        decodes(&q.calls()).iter().map(|d| d.0).collect::<Vec<_>>(),
        [9]
    );
}

#[test]
fn an_interleaved_layer_projects_into_ssm_qkvz_then_deinterleaves_once() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (mut layer, q, _o) = w4_layer(&gpu, &c, true, false);
    layer.sequential_qkvz = false;
    let buffers = arena(&gpu);
    with_ctx(&gpu, &c, &buffers, false, false, |ctx| {
        layer
            .w4_qkvz_prefill(
                ctx,
                buffers.norm_output(),
                buffers.ssm_deinterleaved(),
                28,
                0,
            )
            .unwrap();
    });
    assert!(
        matches!(q.calls()[..], [Call::Prefill { m: 28, out, .. }] if out == buffers.ssm_qkvz())
    );
    let d: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == DEINTERLEAVE.0)
        .collect();
    assert_eq!(d.len(), 1, "one deinterleave over all rows");
    assert_eq!(d[0].args[0], MockArg::Buffer(buffers.ssm_qkvz()));
    assert_eq!(d[0].args[1], MockArg::Buffer(buffers.ssm_deinterleaved()));
}

#[test]
fn a_declined_prefill_dequantizes_once_and_runs_the_bf16_gemm_per_chunk() {
    let (gpu, c) = (MockGpuBackend::new(), config());
    let (layer, _q, o) = w4_layer(&gpu, &c, false, false);
    let buffers = arena(&gpu);
    let (h, vd) = (
        c.hidden_size,
        c.linear_num_value_heads * c.linear_value_head_dim,
    );
    let before = w4::prefill_declines();
    // 3000 rows: two even prefill chunks of 1500; the fake touches no memory.
    with_ctx(&gpu, &c, &buffers, false, false, |ctx| {
        layer
            .prefill_out_proj_dispatch(
                ctx,
                buffers.norm_output(),
                buffers.hidden_states(),
                3000,
                h,
                vd,
                0,
            )
            .unwrap();
    });
    let calls = o.calls();
    assert_eq!(calls.len(), 3, "{calls:?}");
    assert!(
        matches!(calls[0], Call::Prefill { m: 1500, .. }),
        "{calls:?}"
    );
    let Call::Dequant { out: dq } = calls[1] else {
        panic!("{calls:?}")
    };
    assert!(
        matches!(calls[2], Call::Prefill { m: 1500, .. }),
        "{calls:?}"
    );
    assert!(
        w4::prefill_declines() >= before + 2,
        "each declined chunk is counted"
    );
    let gemms: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == DENSE_PIPE.0)
        .collect();
    assert_eq!(gemms.len(), 2, "one BF16 GEMM per declined chunk");
    let u = |v: usize| MockArg::Bytes((v as u32).to_le_bytes().to_vec());
    for (i, g) in gemms.iter().enumerate() {
        assert_eq!(
            g.args[0],
            MockArg::Buffer(buffers.norm_output().offset(i * 1500 * vd * BF16))
        );
        assert_eq!(g.args[1], MockArg::Buffer(dq));
        assert_eq!(
            g.args[2],
            MockArg::Buffer(buffers.hidden_states().offset(i * 1500 * h * BF16))
        );
        assert_eq!(g.args[3..6], [u(1500), u(h), u(vd)]);
    }
}
