// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the Marlin `W4Linear` (`w4linear.rs`, `weight.rs`,
//! `mod.rs`): the uploaded image, the launch geometry and every one of the 17 kernel arguments, the
//! row-invariant decode plan, the prefill split and its declines, `dequant_bf16` against the CPU
//! dequant, release, and the module-absent refusal.
#![allow(clippy::useless_vec)]

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::config::{self, UNDERFILL_CONFIG};
use super::kernels;
use super::repack::to_host;
use super::repack_tests::random_logical;
use super::w4linear::DecodePolicy;
use super::{MarlinType, build, from_logical, release};
use crate::port::w4::{W4Format, W4Linear, W4Opts, W4Scratch};

const MOCK_SMS: usize = 48;

fn scratch(gpu: &MockGpuBackend, bytes: usize) -> W4Scratch {
    W4Scratch {
        ws: gpu.alloc(bytes).unwrap(),
        ws_bytes: bytes,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    }
}

fn bytes_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn int(a: &MockArg) -> i32 {
    match a {
        MockArg::Bytes(b) if b.len() == 4 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        other => panic!("not an int argument: {other:?}"),
    }
}

/// The three flags are 4-byte `int`s in `marlin_ada.cu`'s exported ABI (lane L18a), not C `bool`s.
fn flag(a: &MockArg) -> i32 {
    let v = int(a);
    assert!(v == 0 || v == 1, "a flag argument is 0 or 1, got {v}");
    v
}

fn ptr(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        other => panic!("not a pointer argument: {other:?}"),
    }
}

#[test]
fn the_upload_is_the_host_image_and_the_shape_is_reported() {
    let gpu = MockGpuBackend::new();
    for zp in [false, true] {
        let l = random_logical(256, 512, 128, zp, 61);
        let lin = build(&gpu, &l, 0).unwrap();
        let w = lin.weight();
        let host = to_host(&l).unwrap();
        assert_eq!(bytes_u32(&gpu.read_alloc(w.words).unwrap()), host.words);
        let scales: Vec<u16> = gpu
            .read_alloc(w.scales)
            .unwrap()
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(scales, host.scales);
        assert_eq!(w.zeros.is_null(), !zp);
        if zp {
            assert_eq!(
                Some(bytes_u32(&gpu.read_alloc(w.zeros).unwrap())),
                host.zeros
            );
        }
        assert!(gpu.read_alloc(w.locks).unwrap().iter().all(|&b| b == 0));
        assert_eq!(w.sms, MOCK_SMS);
        assert_eq!(w.decode_cfg, UNDERFILL_CONFIG);
        assert_eq!(
            lin.format(),
            W4Format::MarlinInt4 {
                group: 128,
                zero_point: zp
            }
        );
        assert_eq!((lin.n(), lin.k(), lin.max_decode_rows()), (256, 512, 64));
        let zbytes = if zp { 4 * 256 / 2 } else { 0 };
        assert_eq!(
            lin.device_bytes(),
            512 * 256 / 2 + 4 * 256 * 2 + zbytes + MOCK_SMS * 16
        );
        assert_eq!(lin.scratch_bytes(1), (MOCK_SMS + 1) * 16 * 256 * 4);
        assert_eq!(lin.scratch_bytes(64), (MOCK_SMS + 1) * 64 * 256 * 4);
    }
}

#[test]
fn decode_launches_one_kernel_with_every_argument_in_place() {
    let gpu = MockGpuBackend::new();
    let l = random_logical(256, 512, 128, false, 67);
    let lin = build(&gpu, &l, 0).unwrap();
    let w = lin.weight().clone();
    let s = scratch(&gpu, lin.scratch_bytes(64));
    let x = gpu.alloc(64 * 512 * 2).unwrap();
    let out = gpu.alloc(64 * 256 * 2).unwrap();
    let before = gpu.launches_snapshot().len();
    lin.decode(&gpu, x, 5, out, &s, 9).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), before + 1);
    let la = &launches[before];
    assert_eq!(la.grid, [MOCK_SMS as u32, 1, 1]);
    assert_eq!(la.block, [128, 1, 1]);
    assert_eq!(la.shared_mem as usize, config::ADA_MAX_SHARED_MEM_OPTIN);
    assert_eq!(la.stream, 9);
    let a = &la.args;
    assert_eq!(a.len(), kernels::KERNEL_PARAMS);
    assert_eq!(ptr(&a[0]), x);
    assert_eq!(ptr(&a[1]), w.words);
    assert_eq!(ptr(&a[2]), out);
    assert_eq!(ptr(&a[3]), s.ws);
    assert_eq!(ptr(&a[4]), DevicePtr::NULL, "no bias");
    assert_eq!(ptr(&a[5]), w.scales);
    assert_eq!(ptr(&a[6]), DevicePtr::NULL, "no global scale");
    assert_eq!(ptr(&a[7]), DevicePtr::NULL, "kU4B8 has no zero points");
    assert_eq!(
        (int(&a[8]), int(&a[9]), int(&a[10]), int(&a[11])),
        (5, 256, 512, 512)
    );
    assert_eq!(ptr(&a[12]), w.locks);
    assert_eq!((flag(&a[13]), flag(&a[14]), flag(&a[15])), (0, 0, 1));
    assert_eq!(int(&a[16]) as usize, config::ADA_MAX_SHARED_MEM_OPTIN);

    // Every row count takes the same configuration and never the 8-row tile.
    for rows in 1..=64 {
        let plan = lin.plan(rows, DecodePolicy::RowInvariant).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].cfg, w.decode_cfg);
        assert!(!plan[0].m_block_size_8);
    }
}

#[test]
fn decode_refuses_what_the_contract_excludes() {
    let gpu = MockGpuBackend::new();
    let l = random_logical(128, 256, 128, false, 71);
    let lin = build(&gpu, &l, 0).unwrap();
    let s = scratch(&gpu, lin.scratch_bytes(64));
    let x = gpu.alloc(65 * 256 * 2).unwrap();
    let out = gpu.alloc(65 * 128 * 2).unwrap();
    assert!(lin.decode(&gpu, x, 0, out, &s, 0).is_err());
    assert!(lin.decode(&gpu, x, 65, out, &s, 0).is_err());
    assert!(
        lin.decode(&gpu, x.offset(2), 1, out, &s, 0).is_err(),
        "unaligned input"
    );
    let small = scratch(&gpu, lin.scratch_bytes(1));
    assert!(
        lin.decode(&gpu, x, 17, out, &small, 0).is_err(),
        "17 rows need a 32-row buffer"
    );
    assert!(lin.decode(&gpu, x, 16, out, &small, 0).is_ok());
}

#[test]
fn prefill_splits_like_marlin_mm_and_declines_without_scratch() {
    let gpu = MockGpuBackend::new();
    let l = random_logical(256, 512, 128, true, 73);
    let lin = build(&gpu, &l, 0).unwrap();
    let w = lin.weight().clone();
    let (m, n, k) = (200usize, 256usize, 512usize);
    let x = gpu.alloc(m * k * 2).unwrap();
    let out = gpu.alloc(m * n * 2).unwrap();
    let s = scratch(&gpu, lin.scratch_bytes(m));
    let before = gpu.launches_snapshot().len();
    assert!(lin.prefill(&gpu, x, m, out, &s, 3).unwrap());
    let launches = gpu.launches_snapshot();
    let mine = &launches[before..];
    // N <= 4096: up to 128 slices, so 192 rows in one launch and the 8-row tail in another.
    assert_eq!(mine.len(), 2);
    assert_eq!(int(&mine[0].args[8]), 192);
    assert_eq!(int(&mine[1].args[8]), 8);
    assert_eq!(ptr(&mine[1].args[0]), x.offset(192 * k * 2));
    assert_eq!(ptr(&mine[1].args[2]), out.offset(192 * n * 2));
    assert_eq!(ptr(&mine[0].args[7]), w.zeros, "kU4 passes its zero points");
    assert_ne!(w.zeros, DevicePtr::NULL);

    let tiny = scratch(&gpu, 1024);
    let before = gpu.launches_snapshot().len();
    assert!(!lin.prefill(&gpu, x, m, out, &tiny, 3).unwrap());
    assert_eq!(
        gpu.launches_snapshot().len(),
        before,
        "a decline launches nothing"
    );
}

#[test]
fn dequant_bf16_writes_the_cpu_dequant_bit_for_bit() {
    let gpu = MockGpuBackend::new();
    for zp in [false, true] {
        let l = random_logical(128, 256, 128, zp, 79);
        let lin = build(&gpu, &l, 0).unwrap();
        let out = gpu.alloc(128 * 256 * 2).unwrap();
        lin.dequant_bf16(&gpu, out, 0).unwrap();
        let got: Vec<u16> = gpu
            .read_alloc(out)
            .unwrap()
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, l.dequant_nk(), "zp={zp}");
    }
}

#[test]
fn rtn_from_device_bf16_builds_a_g128_ku4b8_weight() {
    let gpu = MockGpuBackend::new();
    let (n, k) = (128usize, 256usize);
    let mut r = super::repack_tests::Rng::new(83);
    let w: Vec<u16> = (0..n * k)
        .map(|_| super::reference::f32_to_bf16(r.unit()))
        .collect();
    let src = gpu.alloc(n * k * 2).unwrap();
    let bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, src).unwrap();
    let opts = W4Opts {
        bits: 4,
        group: 128,
        clip_search: 0,
        int8_prefill: false,
    };
    let b = super::quantize_rtn_from_bf16(&gpu, src, n, k, &opts, 0).unwrap();
    assert_eq!(
        b.format(),
        W4Format::MarlinInt4 {
            group: 128,
            zero_point: false
        }
    );
    let out = gpu.alloc(n * k * 2).unwrap();
    b.dequant_bf16(&gpu, out, 0).unwrap();
    let want = super::quantize::rtn_int4_sym(
        &w,
        n,
        k,
        &super::quantize::RtnOpts {
            group: 128,
            clip_search: 0,
        },
    )
    .unwrap()
    .dequant_nk();
    let got: Vec<u16> = gpu
        .read_alloc(out)
        .unwrap()
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    assert_eq!(got, want);
    // The default affine group (64) and other widths are refused with a reason.
    let bad = W4Opts::default();
    assert!(super::quantize_rtn_from_bf16(&gpu, src, n, k, &bad, 0).is_err());
    let bits3 = W4Opts { bits: 3, ..opts };
    assert!(super::quantize_rtn_from_bf16(&gpu, src, n, k, &bits3, 0).is_err());
}

#[test]
fn release_frees_once_and_the_module_check_comes_first() {
    let gpu = MockGpuBackend::new();
    let l = random_logical(128, 256, 128, false, 89);
    let b = from_logical(&gpu, &l, 0).unwrap();
    let words = super::w4linear::MarlinLinear::new(
        // A descriptor clone only to learn the pointer; the boxed weight owns the buffers.
        b.as_any()
            .and_then(|a| a.downcast_ref::<super::MarlinLinear>())
            .map(|m| m.weight().clone())
            .expect("a MarlinLinear"),
    );
    let ptr = words.weight().words;
    std::mem::forget(words);
    assert!(gpu.read_alloc(ptr).is_some());
    assert!(release(&gpu, &*b).unwrap());
    assert!(gpu.read_alloc(ptr).is_none(), "the codes are freed");
    assert!(!release(&gpu, &*b).unwrap(), "a second release is a no-op");

    let absent = MockGpuBackend::new();
    absent.mark_module_absent(kernels::MODULE);
    let e = build(&absent, &l, 0).unwrap_err().to_string();
    assert!(e.contains(kernels::MODULE), "{e}");
    assert!(absent.kernel_lookups_snapshot().is_empty());
    assert!(absent.launches_snapshot().is_empty());
}

#[test]
fn shapes_without_an_instance_are_refused_before_any_allocation() {
    let gpu = MockGpuBackend::new();
    // Group 64: valid Marlin, but the pruned build carries g128 only.
    let l = random_logical(128, 256, 64, false, 97);
    assert!(build(&gpu, &l, 0).is_err());
    // One group (K == group) would need the channelwise kernel.
    let one = random_logical(128, 128, 128, false, 98);
    assert!(build(&gpu, &one, 0).is_err());
    assert_eq!(gpu.live_alloc_count(), 0, "nothing was allocated");
    assert_eq!(MarlinType::U4.label(), "u4");
}
