// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the EXL3 `W4Linear`: what the upload writes, which kernels it
//! looks up (none on a build without the modules), the geometry and argument order of every launch, the
//! scratch carve-up, the decode / prefill / decline rules, and release.

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::format::{Exl3Codebook, F16_MINUS_ONE, F16_ONE, u16_bytes};
use super::kernels::{HADAMARD_MODULE, LINEAR_MODULE, ROT_IN_ENTRY, dequant_entry, linear_entry};
use super::plan::{DEQUANT_SMEM, MAX_KERNEL_ROWS, Plan, plan_for_device, scratch_layout, strides};
use super::tests::{Lcg, OwnedParts};
use super::w4linear::{Exl3Linear, exl3_of, release};
use crate::port::w4::scratch::ScratchSpec;
use crate::port::w4::{W4Format, W4Linear, W4Scratch};

fn i32b(v: i32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn u64b(v: u64) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn scratch(gpu: &MockGpuBackend, bytes: usize) -> W4Scratch {
    W4Scratch {
        ws: gpu.alloc(bytes).unwrap(),
        ws_bytes: bytes,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    }
}

fn launches_after(gpu: &MockGpuBackend, before: usize) -> Vec<MockLaunch> {
    gpu.launches_snapshot()[before..].to_vec()
}

#[test]
fn upload_copies_the_trellis_verbatim_and_expands_the_sign_words() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(21);
    let p = OwnedParts::random(&mut r, 256, 128, 6, Exl3Codebook::Mul1, true);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 0).unwrap();
    let w = *lin.weight();
    assert_eq!(gpu.read_alloc(w.trellis).unwrap(), p.trellis);
    let suh = gpu.read_alloc(w.suh).unwrap();
    assert_eq!(suh, u16_bytes(&p.parts().suh_bits()));
    assert!(
        suh.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .all(|b| b == F16_ONE || b == F16_MINUS_ONE)
    );
    assert_eq!(
        gpu.read_alloc(w.svh).unwrap(),
        u16_bytes(&p.parts().svh_bits())
    );
    let counters = gpu.read_alloc(w.counters).unwrap();
    assert_eq!(counters.len(), 32, "8 passes x 1 column block x 4 bytes");
    assert!(counters.iter().all(|&b| b == 0), "counters start at zero");
    assert_eq!(
        lin.device_bytes(),
        p.trellis.len() + 2 * 256 + 2 * 128 + counters.len()
    );
    assert_eq!(lin.format(), W4Format::Exl3 { bpw_x100: 300 });
    assert_eq!((lin.k(), lin.n(), lin.max_decode_rows()), (256, 128, 16));
    assert_eq!(
        w.plan,
        plan_for_device(256, 128, 48),
        "the mock reports 48 SMs"
    );
}

#[test]
fn the_upload_resolves_one_width_and_issues_no_lookup_without_the_modules() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(22);
    let p = OwnedParts::random(&mut r, 256, 1024, 8, Exl3Codebook::Mcg, false);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 0).unwrap();
    let wk = lin.weight().plan.wk;
    let lookups = gpu.kernel_lookups_snapshot();
    assert_eq!(
        lookups,
        vec![
            (HADAMARD_MODULE.to_owned(), ROT_IN_ENTRY.to_owned()),
            (
                LINEAR_MODULE.to_owned(),
                linear_entry(8, Exl3Codebook::Mcg, wk)
            ),
            (
                HADAMARD_MODULE.to_owned(),
                dequant_entry(8, Exl3Codebook::Mcg)
            ),
        ]
    );
    assert_eq!(
        linear_entry(8, Exl3Codebook::Mcg, 4),
        "exl3_ada_linear_8_1_4"
    );
    assert_eq!(
        dequant_entry(13, Exl3Codebook::Mul1),
        "exl3_ada_dequant_bf16_13_2"
    );

    for absent in [LINEAR_MODULE, HADAMARD_MODULE] {
        let other = MockGpuBackend::new();
        other.mark_module_absent(absent);
        let allocs = other.alloc_count();
        let err = Exl3Linear::upload_weight(&other, &p.parts(), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Ada build"), "{err}");
        assert!(
            other.kernel_lookups_snapshot().is_empty(),
            "no lookup on a target without {absent}"
        );
        assert_eq!(other.alloc_count(), allocs, "and no allocation");
    }
}

#[test]
fn a_missing_entry_fails_the_upload_and_frees_nothing_it_did_not_take() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(23);
    let p = OwnedParts::random(&mut r, 128, 128, 4, Exl3Codebook::Inst3, false);
    gpu.deny_kernel(HADAMARD_MODULE, &dequant_entry(4, Exl3Codebook::Inst3));
    let allocs = gpu.alloc_count();
    let err = Exl3Linear::upload_weight(&gpu, &p.parts(), 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exl3_ada_dequant_bf16_4_0"), "{err}");
    assert_eq!(gpu.alloc_count(), allocs);
}

#[test]
fn decode_launches_the_rotation_then_the_linear_with_the_shapes_plan() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(24);
    let (k, n) = (1280, 9216);
    let p = OwnedParts::random(&mut r, k, n, 7, Exl3Codebook::Mul1, false);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 3).unwrap();
    let w = *lin.weight();
    assert_eq!(
        w.plan,
        Plan { sk: 1, wk: 8 },
        "72 column blocks: wide, no split, 10 k tiles a warp"
    );
    let s = scratch(&gpu, lin.scratch_bytes(16));
    let (x, out) = (gpu.alloc(3 * k * 2).unwrap(), gpu.alloc(3 * n * 2).unwrap());
    let before = gpu.launches_snapshot().len();
    lin.decode(&gpu, x, 3, out, &s, 9).unwrap();
    let l = launches_after(&gpu, before);
    assert_eq!(l.len(), 2);
    let rot = &l[0];
    assert_eq!(
        (rot.grid, rot.block, rot.shared_mem, rot.stream),
        ([3, 3, 1], [128, 1, 1], 0, 9)
    );
    assert_eq!(
        rot.args,
        vec![
            MockArg::Buffer(x),
            MockArg::Buffer(w.suh),
            MockArg::Buffer(s.ws),
            i32b(3),
            i32b(k as i32)
        ]
    );
    let lin_l = &l[1];
    assert_eq!(lin_l.grid, [72, 1, 1]);
    assert_eq!(lin_l.block, [256, 1, 1]);
    assert_eq!(lin_l.shared_mem, 8 * 3 * 128 * 4);
    let (sk_words, nb_words) = strides(n, 7);
    assert_eq!(
        lin_l.args,
        vec![
            MockArg::Buffer(s.ws),
            MockArg::Buffer(w.trellis),
            u64b(sk_words),
            u64b(nb_words),
            MockArg::Buffer(w.svh),
            MockArg::Buffer(out),
            MockArg::Buffer(DevicePtr::NULL),
            MockArg::Buffer(w.counters),
            i32b(n as i32),
            i32b(1),
            i32b(3),
            i32b(k as i32),
        ]
    );
}

#[test]
fn a_split_plan_puts_the_partials_after_the_rotated_input() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(25);
    let (k, n) = (4096, 1024);
    let p = OwnedParts::random(&mut r, k, n, 4, Exl3Codebook::Mcg, false);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 0).unwrap();
    assert_eq!(
        lin.weight().plan,
        Plan { sk: 8, wk: 4 },
        "upstream's GB10 plan at 192 blocks"
    );
    let s = scratch(&gpu, lin.scratch_bytes(16));
    let (x, out) = (gpu.alloc(5 * k * 2).unwrap(), gpu.alloc(5 * n * 2).unwrap());
    let before = gpu.launches_snapshot().len();
    lin.decode(&gpu, x, 5, out, &s, 0).unwrap();
    let l = launches_after(&gpu, before);
    let layout = scratch_layout(Plan { sk: 8, wk: 4 }, 5, k, n);
    assert_eq!(l[1].grid, [8 * 8, 1, 1]);
    assert_eq!(l[1].args[6], MockArg::Buffer(s.ws.offset(layout.z_offset)));
    assert_eq!(l[1].args[9], i32b(8));
    assert!(layout.total <= lin.scratch_bytes(16));
}

#[test]
fn the_plan_never_depends_on_the_row_count() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(26);
    let p = OwnedParts::random(&mut r, 1024, 2560, 6, Exl3Codebook::Mul1, false);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 0).unwrap();
    let s = scratch(&gpu, lin.scratch_bytes(MAX_KERNEL_ROWS));
    let x = gpu.alloc(MAX_KERNEL_ROWS * 1024 * 2).unwrap();
    let out = gpu.alloc(MAX_KERNEL_ROWS * 2560 * 2).unwrap();
    let mut seen = Vec::new();
    for rows in [1usize, 2, 4, 5, 8, 16] {
        let before = gpu.launches_snapshot().len();
        lin.decode(&gpu, x, rows, out, &s, 0).unwrap();
        let l = launches_after(&gpu, before);
        seen.push((l[1].grid, l[1].block, l[1].args[9].clone()));
    }
    assert!(seen.windows(2).all(|w| w[0] == w[1]), "{seen:?}");
}

#[test]
fn decode_refuses_bad_rows_short_scratch_and_misaligned_buffers() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(27);
    let p = OwnedParts::random(&mut r, 256, 256, 8, Exl3Codebook::Mul1, false);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 0).unwrap();
    let s = scratch(&gpu, lin.scratch_bytes(16));
    let (x, out) = (
        gpu.alloc(16 * 256 * 2).unwrap(),
        gpu.alloc(16 * 256 * 2).unwrap(),
    );
    let before = gpu.launches_snapshot().len();
    assert!(lin.decode(&gpu, x, 0, out, &s, 0).is_err());
    assert!(lin.decode(&gpu, x, 17, out, &s, 0).is_err());
    let small = W4Scratch {
        ws_bytes: lin.scratch_bytes(4) - 1,
        ..s
    };
    assert!(lin.decode(&gpu, x, 4, out, &small, 0).is_err());
    assert!(lin.decode(&gpu, x.offset(2), 4, out, &s, 0).is_err());
    assert!(lin.decode(&gpu, x, 4, out.offset(4), &s, 0).is_err());
    assert_eq!(
        gpu.launches_snapshot().len(),
        before,
        "a refused call launches nothing"
    );
}

#[test]
fn prefill_runs_the_kernel_up_to_128_rows_and_declines_above() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(28);
    let (k, n) = (512, 256);
    let p = OwnedParts::random(&mut r, k, n, 5, Exl3Codebook::Mul1, false);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 0).unwrap();
    let s = scratch(&gpu, lin.scratch_bytes(2048));
    let x = gpu.alloc(2048 * k * 2).unwrap();
    let out = gpu.alloc(2048 * n * 2).unwrap();
    for (m, served, launches) in [
        (17, true, 2),
        (128, true, 2),
        (129, false, 0),
        (2048, false, 0),
        (0, true, 0),
    ] {
        let before = gpu.launches_snapshot().len();
        assert_eq!(
            lin.prefill(&gpu, x, m, out, &s, 0).unwrap(),
            served,
            "m={m}"
        );
        let l = launches_after(&gpu, before);
        assert_eq!(l.len(), launches, "m={m}");
        if launches == 2 {
            assert_eq!(l[0].grid[1], m as u32);
            assert_eq!(l[1].args[10], i32b(m as i32));
        }
    }
    // The workspace stops growing at one launch: a declined prefill needs none.
    assert_eq!(lin.scratch_bytes(2048), lin.scratch_bytes(MAX_KERNEL_ROWS));
    let spec = ScratchSpec::for_linear(&lin);
    assert_eq!(spec.ws_bytes, lin.scratch_bytes(MAX_KERNEL_ROWS));
    assert_eq!(spec.bf16_bytes, n * k * 2);
}

#[test]
fn the_dequant_launch_covers_every_block_with_the_opt_in_shared_memory() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(29);
    let (k, n) = (384, 640);
    let p = OwnedParts::random(&mut r, k, n, 16, Exl3Codebook::Inst3, false);
    let lin = Exl3Linear::upload_weight(&gpu, &p.parts(), 0).unwrap();
    let w = *lin.weight();
    let out = gpu.alloc(n * k * 2).unwrap();
    let before = gpu.launches_snapshot().len();
    lin.dequant_bf16(&gpu, out, 4).unwrap();
    let l = launches_after(&gpu, before);
    assert_eq!(l.len(), 1);
    assert_eq!(
        (l[0].grid, l[0].block, l[0].shared_mem, l[0].stream),
        ([5, 3, 1], [256, 1, 1], DEQUANT_SMEM, 4)
    );
    let (sk_words, nb_words) = strides(n, 16);
    assert_eq!(
        l[0].args,
        vec![
            MockArg::Buffer(w.trellis),
            u64b(sk_words),
            u64b(nb_words),
            MockArg::Buffer(w.suh),
            MockArg::Buffer(w.svh),
            MockArg::Buffer(out),
            i32b(n as i32),
            i32b(k as i32),
        ]
    );
    assert!(lin.dequant_bf16(&gpu, out.offset(2), 0).is_err());
}

#[test]
fn the_default_pair_is_two_decodes() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(30);
    let a = Exl3Linear::upload(
        &gpu,
        &OwnedParts::random(&mut r, 256, 256, 8, Exl3Codebook::Mul1, false).parts(),
        0,
    )
    .unwrap();
    let b = Exl3Linear::upload(
        &gpu,
        &OwnedParts::random(&mut r, 256, 384, 8, Exl3Codebook::Mul1, false).parts(),
        0,
    )
    .unwrap();
    let s = scratch(&gpu, a.scratch_bytes(16).max(b.scratch_bytes(16)));
    let x = gpu.alloc(4 * 256 * 2).unwrap();
    let (oa, ob) = (
        gpu.alloc(4 * 256 * 2).unwrap(),
        gpu.alloc(4 * 384 * 2).unwrap(),
    );
    let before = gpu.launches_snapshot().len();
    a.decode_pair(&*b, &gpu, x, 4, oa, ob, &s, 0).unwrap();
    let l = launches_after(&gpu, before);
    assert_eq!(l.len(), 4, "rotation + linear for each member");
    assert_eq!(l[1].args[5], MockArg::Buffer(oa));
    assert_eq!(l[3].args[5], MockArg::Buffer(ob));
}

#[test]
fn release_frees_once_and_later_calls_fail() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(31);
    let p = OwnedParts::random(&mut r, 128, 128, 4, Exl3Codebook::Mcg, false);
    let lin = Exl3Linear::upload(&gpu, &p.parts(), 0).unwrap();
    let w = exl3_of(&*lin).expect("a live EXL3 weight");
    assert!(release(&gpu, &*lin).unwrap());
    assert!(gpu.read_alloc(w.trellis).is_none() && gpu.read_alloc(w.counters).is_none());
    assert!(
        !release(&gpu, &*lin).unwrap(),
        "a second release is harmless"
    );
    assert!(exl3_of(&*lin).is_none());
    let s = scratch(&gpu, 1 << 20);
    let x = gpu.alloc(128 * 2).unwrap();
    assert!(lin.decode(&gpu, x, 1, x, &s, 0).is_err());
    assert!(lin.dequant_bf16(&gpu, x, 0).is_err());
}

/// A reader tensor (lane L20a's type) built from random parts.
fn reader_tensor(p: &OwnedParts, bias: bool) -> spark_runtime::weights::exl3::Exl3Tensor {
    use spark_runtime::weights::exl3::{Codebook, Exl3Tensor, ScaleForm, Width};
    let parts = p.parts();
    let cb = match p.codebook {
        Exl3Codebook::Inst3 => Codebook::Inst3,
        Exl3Codebook::Mcg => Codebook::Mcg,
        Exl3Codebook::Mul1 => Codebook::Mul1,
    };
    let trellis: Vec<i16> = parts.trellis_u16().into_iter().map(|w| w as i16).collect();
    Exl3Tensor::new(
        "model.layers.0.mlp.down_proj".to_owned(),
        cb,
        Width::from_halves(p.halves).unwrap(),
        p.k,
        p.n,
        trellis,
        parts.suh_bits(),
        parts.svh_bits(),
        bias.then(|| vec![0u16; p.n]),
        ScaleForm::Fp16,
    )
    .unwrap()
}

#[test]
fn the_reader_tensor_uploads_to_the_same_bytes() {
    let gpu = MockGpuBackend::new();
    let mut r = Lcg(32);
    let p = OwnedParts::random(&mut r, 256, 256, 7, Exl3Codebook::Mul1, true);
    let t = reader_tensor(&p, false);
    let bytes = super::tensor::TensorBytes::of(&t);
    let view = bytes.parts();
    view.check().unwrap();
    assert_eq!(view.trellis, &p.trellis[..]);
    assert_eq!(view.suh_bits(), p.parts().suh_bits());
    assert_eq!(view.svh_bits(), p.parts().svh_bits());
    assert_eq!((view.halves, view.codebook), (7, Exl3Codebook::Mul1));
    let lin = Exl3Linear::upload_weight(&gpu, &view, 0).unwrap();
    assert_eq!(gpu.read_alloc(lin.weight().trellis).unwrap(), p.trellis);
    assert_eq!(
        lin.format(),
        W4Format::Exl3 {
            bpw_x100: t.bpw_x100()
        }
    );
    // A biased reader tensor is refused (the contract has no bias term).
    let biased = super::tensor::TensorBytes::of(&reader_tensor(&p, true));
    assert!(Exl3Linear::upload(&gpu, &biased.parts(), 0).is_err());
}

#[test]
fn the_route_stays_closed_while_the_row_is_off() {
    // Every target declares weights_exl3 = false; only an explicit environment override opens it.
    if std::env::var_os(crate::layers::ops::port_rows::weights_exl3::ENV).is_some() {
        return;
    }
    let gpu = MockGpuBackend::new();
    assert!(!super::tensor::route_enabled(&gpu));
    assert!(
        super::kernels::available(&gpu),
        "the mock carries every module"
    );
    let mut r = Lcg(33);
    let t = reader_tensor(
        &OwnedParts::random(&mut r, 128, 128, 8, Exl3Codebook::Mcg, false),
        false,
    );
    let before = gpu.alloc_count();
    let err = super::tensor::from_exl3_tensors(&gpu, &t, 0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("weights_exl3"), "{err}");
    assert!(gpu.kernel_lookups_snapshot().is_empty() && gpu.alloc_count() == before);
}
