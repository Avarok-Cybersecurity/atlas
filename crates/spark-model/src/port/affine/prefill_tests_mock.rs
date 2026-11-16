// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the affine prompt GEMMs' host dispatch on the mock backend (lane L17b): launch
//! order and geometry of both routes, the device-side gate wiring (the flag the quantizer raises is
//! the one both GEMMs read), declines that launch nothing and look nothing up, the `_many`
//! descriptor's C layout, the fused down projection and the fallback counter.

use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;
use crate::port::w4::W4Scratch;

struct Bench {
    gpu: MockGpuBackend,
    w: Packed,
    x: DevicePtr,
    out: DevicePtr,
    s: W4Scratch,
}

fn bench(n: usize, k: usize, m: usize, int8_scratch: bool) -> Bench {
    let gpu = MockGpuBackend::new();
    let npad = n.div_ceil(128) * 128;
    let w = Packed {
        words: gpu.alloc(npad * k / 2).unwrap(),
        scales: gpu.alloc(k / GROUP * npad * 2).unwrap(),
        biases: gpu.alloc(k / GROUP * npad * 2).unwrap(),
        n,
        k,
        npad,
    };
    let (x, out) = (gpu.alloc(m * k * 2).unwrap(), gpu.alloc(m * n * 2).unwrap());
    let (xq, xs) = if int8_scratch {
        let xq = gpu.alloc(xq_bytes(m, k)).unwrap();
        (xq, gpu.alloc(xs_bytes(m, k)).unwrap())
    } else {
        (DevicePtr::NULL, DevicePtr::NULL)
    };
    let s = W4Scratch {
        ws: gpu.alloc(256).unwrap(),
        ws_bytes: 256,
        tickets: DevicePtr::NULL,
        xq,
        xs,
    };
    Bench { gpu, w, x, out, s }
}

fn last_ptr(l: &MockLaunch) -> DevicePtr {
    match l.args.last() {
        Some(MockArg::Buffer(p)) => *p,
        a => panic!("last arg {a:?} is not a pointer"),
    }
}

fn int_at(l: &MockLaunch, i: usize) -> i32 {
    match &l.args[i] {
        MockArg::Bytes(b) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        a => panic!("arg {i} is {a:?}"),
    }
}

#[test]
fn the_int8_route_quantizes_runs_int8_then_the_gated_bf16_gemm() {
    let b = bench(300, 256, 200, true);
    assert!(run_packed(&[(b.w, b.out)], &b.gpu, b.x, 200, &b.s, true, 7).unwrap());
    let l = b.gpu.launches_snapshot();
    assert_eq!(l.len(), 3);
    let f = flags(&b.gpu, 7).unwrap();
    assert_eq!(
        (l[0].grid, l[0].block, l[0].shared_mem),
        ([200, 1, 1], [128, 1, 1], 0)
    );
    assert_eq!(last_ptr(&l[0]), f, "the quantizer raises the flag");
    let int8 = (l[1].grid, l[1].block, l[1].shared_mem);
    assert_eq!(int8, ([6, 1, 1], [256, 1, 1], INT8_SMEM_BYTES));
    assert_eq!(last_ptr(&l[1]), f, "the int8 GEMM gates on it");
    assert_eq!(l[1].args.len(), 13);
    let dims: Vec<i32> = (7..12).map(|i| int_at(&l[1], i)).collect();
    assert_eq!(dims, vec![200, 300, 256, 384, 2], "m, n, k, npad, band");
    assert_eq!((l[2].grid, l[2].shared_mem), ([6, 1, 1], BF16_SMEM_BYTES));
    assert_eq!(last_ptr(&l[2]), f, "the BF16 GEMM runs only on the flag");
    assert!(l.iter().all(|x| x.stream == 7));
    let zeroed = b.gpu.read_alloc(f).unwrap();
    assert_eq!(zeroed, vec![0u8; FLAG_BYTES], "zeroed when created");
}

#[test]
fn the_bf16_route_is_one_ungated_launch() {
    let b = bench(256, 128, 64, true);
    assert!(run_packed(&[(b.w, b.out)], &b.gpu, b.x, 64, &b.s, false, 0).unwrap());
    let l = b.gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    let shape = (l[0].grid, l[0].shared_mem, l[0].args.len());
    assert_eq!(shape, ([2, 1, 1], BF16_SMEM_BYTES, 12));
    assert_eq!(last_ptr(&l[0]), DevicePtr::NULL);
    assert_eq!(int_at(&l[0], 9), 128, "ldx = k");
}

#[test]
fn int8_without_scratch_runs_the_bf16_gemm() {
    let b = bench(128, 64, 32, false);
    assert!(run_packed(&[(b.w, b.out)], &b.gpu, b.x, 32, &b.s, true, 0).unwrap());
    let l = b.gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(last_ptr(&l[0]), DevicePtr::NULL);
}

#[test]
fn an_absent_module_declines_with_no_lookup_and_no_launch() {
    let b = bench(128, 64, 32, true);
    b.gpu.mark_module_absent(launch::GEMM_MODULE);
    b.gpu.mark_module_absent(launch::ACT_MODULE);
    assert!(!run_packed(&[(b.w, b.out)], &b.gpu, b.x, 32, &b.s, true, 0).unwrap());
    assert!(b.gpu.launches_snapshot().is_empty());
    assert!(b.gpu.kernel_lookups_snapshot().is_empty());
    assert!(!prepare(&b.gpu, 0).unwrap());
    assert_eq!(take_fallbacks(&b.gpu, 0).unwrap(), 0);
    assert!(
        b.gpu.kernel_lookups_snapshot().is_empty(),
        "still no lookup"
    );
}

#[test]
fn unservable_shapes_decline() {
    let b = bench(128, 128, 32, true);
    let odd_k = Packed { k: 100, ..b.w };
    let bad_pad = Packed { npad: 256, ..b.w };
    let misaligned = Packed {
        words: b.w.words.offset(4),
        ..b.w
    };
    for w in [odd_k, bad_pad, misaligned] {
        let ran = run_packed(&[(w, b.out)], &b.gpu, b.x, 32, &b.s, true, 0).unwrap();
        assert!(!ran, "{w:?}");
    }
    assert!(b.gpu.launches_snapshot().is_empty());
    let ran = run_packed(&[(b.w, b.out)], &b.gpu, b.x, 0, &b.s, true, 0).unwrap();
    assert!(
        ran && b.gpu.launches_snapshot().is_empty(),
        "m = 0 is a no-op"
    );
}

#[test]
fn many_parts_encode_the_c_layout_and_take_the_many_entries() {
    let b = bench(300, 256, 200, true);
    let w2 = Packed {
        n: 100,
        npad: 128,
        ..b.w
    };
    let out2 = b.gpu.alloc(200 * 100 * 2).unwrap();
    let (bytes, blocks) = encode_parts(&[(b.w, b.out), (w2, out2)], 200).unwrap();
    assert_eq!(blocks, 6 + 2);
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let i32_at = |o: usize| i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let ptrs = [u64_at(0), u64_at(8), u64_at(16), u64_at(24)];
    assert_eq!(ptrs, [b.w.words.0, b.w.scales.0, b.w.biases.0, b.out.0]);
    assert_eq!([i32_at(32), i32_at(36), i32_at(40)], [300, 384, 0]);
    assert_eq!(u64_at(48 + 24), out2.0);
    assert_eq!(
        [i32_at(80), i32_at(84), i32_at(88)],
        [100, 128, 6],
        "n, npad, first CTA"
    );
    assert_eq!(i32_at(192), 2);
    assert!(
        bytes[96..192].iter().all(|v| *v == 0),
        "unused parts stay zero"
    );
    let other_k = Packed { k: 128, ..w2 };
    assert!(
        encode_parts(&[(b.w, b.out), (other_k, out2)], 200).is_err(),
        "k must match"
    );

    let members = [(b.w, b.out), (w2, out2)];
    assert!(run_packed(&members, &b.gpu, b.x, 200, &b.s, true, 0).unwrap());
    let l = b.gpu.launches_snapshot();
    assert_eq!(l.len(), 3);
    for (launch, args) in [(&l[1], 8), (&l[2], 7)] {
        assert_eq!((launch.grid, launch.args.len()), ([8, 1, 1], args));
        assert!(launch.args.contains(&MockArg::Bytes(bytes.to_vec())));
    }
}

#[test]
fn the_fused_down_projection_writes_h_then_gates_bf16_on_it() {
    let b = bench(2560, 512, 130, true);
    let gate = b.gpu.alloc(130 * 1024 * 2).unwrap();
    let up = b.gpu.alloc(130 * 1024 * 2).unwrap();
    // h written in place over gate (the gate|up halves of one [130, 1024] matrix).
    let run = |h: DevicePtr, ld_h: usize| {
        run_down_packed(&b.w, &b.gpu, gate, up, 1024, h, ld_h, 130, b.out, &b.s, 0).unwrap()
    };
    assert!(run(gate, 1024));
    let l = b.gpu.launches_snapshot();
    let f = flags(&b.gpu, 0).unwrap();
    assert_eq!(l.len(), 3);
    assert_eq!((l[0].grid, l[0].block), ([130, 2, 1], [128, 1, 1]));
    let strides = [int_at(&l[0], 3), int_at(&l[0], 4), int_at(&l[0], 8)];
    assert_eq!(strides, [1024, 1024, 512], "ld_in, ld_h, k");
    assert_eq!(last_ptr(&l[0]), f);
    assert_eq!((l[1].grid, l[1].shared_mem), ([40, 1, 1], INT8_SMEM_BYTES));
    assert_eq!(
        l[2].args[0],
        MockArg::Buffer(gate),
        "the BF16 fallback reads h"
    );
    assert_eq!(int_at(&l[2], 9), 1024, "with h's stride");
    assert_eq!(last_ptr(&l[2]), f);
    // h aliasing gate with another stride is refused; a null h declines.
    let act = Int8Act::carve(&b.s, 130, 512, f).unwrap();
    let alias = swiglu_quantize(&b.gpu, gate, up, 1024, gate, 512, 130, 512, &act, 0);
    assert!(alias.is_err());
    assert!(!run(DevicePtr::NULL, 512));
    assert_eq!(b.gpu.launches_snapshot().len(), 3);
}

#[test]
fn the_fallback_counter_is_read_and_cleared() {
    let gpu = MockGpuBackend::new();
    assert!(prepare(&gpu, 0).unwrap());
    let f = flags(&gpu, 0).unwrap();
    gpu.copy_h2d(&3u32.to_le_bytes(), f.offset(4)).unwrap();
    let before = int8_fallbacks_total();
    let declines = crate::port::w4::prefill_declines();
    assert_eq!(take_fallbacks(&gpu, 0).unwrap(), 3);
    assert!(int8_fallbacks_total() >= before + 3);
    assert!(
        crate::port::w4::prefill_declines() >= declines + 3,
        "each fallback is a declined prefill in the metric source"
    );
    assert_eq!(gpu.read_alloc(f).unwrap()[4..8], [0, 0, 0, 0]);
    assert_eq!(take_fallbacks(&gpu, 0).unwrap(), 0);
    let lookups = gpu.kernel_lookups_snapshot();
    assert_eq!(
        lookups.len(),
        ENTRIES.len(),
        "every entry resolved once (op cache)"
    );
}

#[test]
fn the_serve_harvest_moves_the_device_count_and_never_fails() {
    let gpu = MockGpuBackend::new();
    assert!(prepare(&gpu, 0).unwrap(), "prepare arms the harvest");
    let f = flags(&gpu, 0).unwrap();
    gpu.copy_h2d(&2u32.to_le_bytes(), f.offset(4)).unwrap();
    let before = int8_fallbacks_total();
    harvest_fallbacks(&gpu, 0);
    assert!(int8_fallbacks_total() >= before + 2);
    assert_eq!(gpu.read_alloc(f).unwrap()[4..8], [0, 0, 0, 0], "cleared");
    assert!(
        gpu.launches_snapshot().is_empty(),
        "a harvest launches nothing"
    );
    // A backend without the Ada modules: the armed harvest still touches nothing there.
    let other = MockGpuBackend::new();
    other.mark_module_absent(launch::GEMM_MODULE);
    other.mark_module_absent(launch::ACT_MODULE);
    harvest_fallbacks(&other, 0);
    assert!(other.kernel_lookups_snapshot().is_empty());
}

#[test]
fn a_flag_raised_on_the_device_is_seen_by_both_gemms_of_a_call() {
    // The quantizer's memset clears word 0 before it launches, so a raised flag never leaks into the next call.
    let b = bench(128, 128, 40, true);
    let f = flags(&b.gpu, 0).unwrap();
    b.gpu.copy_h2d(&1u32.to_le_bytes(), f).unwrap();
    assert!(run_packed(&[(b.w, b.out)], &b.gpu, b.x, 40, &b.s, true, 0).unwrap());
    assert_eq!(
        b.gpu.read_alloc(f).unwrap()[..4],
        [0, 0, 0, 0],
        "cleared before the quantizer"
    );
}
