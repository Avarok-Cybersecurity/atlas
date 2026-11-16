// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-backend tests of the NVFP4 `W4Linear` wrapper: which launcher each row count takes, with which
//! arguments, the decline rules, and the refusals.

use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

const N: usize = 96;
const K: usize = 128;
const SCALE2: f32 = 0.375;

fn weight(gpu: &MockGpuBackend, n: usize, k: usize) -> QuantizedWeight {
    QuantizedWeight {
        weight: gpu.alloc(n * k / 2).unwrap(),
        weight_scale: gpu.alloc(n * k / 16).unwrap(),
        weight_scale_2: SCALE2,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    }
}

fn scratch() -> W4Scratch {
    W4Scratch {
        ws: DevicePtr::NULL,
        ws_bytes: 0,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    }
}

fn u32_arg(l: &MockLaunch, i: usize) -> u32 {
    match &l.args[i] {
        MockArg::Bytes(b) => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        other => panic!("arg {i} is {other:?}, not a scalar"),
    }
}

fn f32_arg(l: &MockLaunch, i: usize) -> f32 {
    f32::from_bits(u32_arg(l, i))
}

fn ptr_arg(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a buffer"),
    }
}

#[test]
fn resolve_looks_up_the_family_and_nothing_where_the_modules_are_absent() {
    let gpu = MockGpuBackend::new();
    let k = Nvfp4Kernels::resolve(&gpu);
    let seen = gpu.kernel_lookups_snapshot();
    for (m, f) in [
        (GEMV_MODULE, "w4a16_gemv"),
        (GEMV_MODULE, "w4a16_gemv_batch2"),
        (GEMV_MODULE, "w4a16_gemv_batch3"),
        (GEMV_MODULE, "w4a16_gemv_batch16"),
        (GEMM_MODULE, "w4a16_gemm"),
        (DEQUANT_MODULE, "dequant_nvfp4_to_bf16"),
    ] {
        assert!(
            seen.contains(&(m.to_owned(), f.to_owned())),
            "{m}::{f} not looked up"
        );
    }
    assert!(k.decode_ready());
    assert_eq!(k.max_decode_rows(), 16);

    let bare = MockGpuBackend::new();
    for m in [GEMV_MODULE, GEMM_MODULE, DEQUANT_MODULE] {
        bare.mark_module_absent(m);
    }
    let k = Nvfp4Kernels::resolve(&bare);
    assert!(
        bare.kernel_lookups_snapshot().is_empty(),
        "a build without the modules sees no lookup (the boot audit)"
    );
    assert!(!k.decode_ready());
    assert_eq!(k.max_decode_rows(), 0);
}

#[test]
fn a_build_without_batch16_caps_the_window_at_8() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel(GEMV_MODULE, "w4a16_gemv_batch16");
    let k = Nvfp4Kernels::resolve(&gpu);
    assert_eq!(k.max_decode_rows(), 8);
    let w = Nvfp4Linear::new(weight(&gpu, N, K), N, K, k).unwrap();
    let x = gpu.alloc(16 * K * 2).unwrap();
    let out = gpu.alloc(16 * N * 2).unwrap();
    w.decode(&gpu, x, 8, out, &scratch(), 0).unwrap();
    assert!(w.decode(&gpu, x, 9, out, &scratch(), 0).is_err());
}

#[test]
fn every_row_count_launches_one_kernel_with_the_launcher_arguments() {
    let gpu = MockGpuBackend::new();
    let q = weight(&gpu, N, K);
    let w = Nvfp4Linear::new(q, N, K, Nvfp4Kernels::resolve(&gpu)).unwrap();
    let x = gpu.alloc(16 * K * 2).unwrap();
    let out = gpu.alloc(16 * N * 2).unwrap();
    for rows in 1..=16usize {
        let before = gpu.launches_snapshot().len();
        w.decode(&gpu, x, rows, out, &scratch(), 3).unwrap();
        let after = gpu.launches_snapshot();
        assert_eq!(after.len(), before + 1, "rows={rows}: one launch");
        let l = &after[before];
        assert_eq!(l.stream, 3);
        assert_eq!(l.block, [256, 1, 1]);
        assert_eq!(ptr_arg(l, 0), x);
        assert_eq!(ptr_arg(l, 1), q.weight);
        assert_eq!(ptr_arg(l, 2), q.weight_scale);
        assert_eq!(f32_arg(l, 3), SCALE2);
        assert_eq!(ptr_arg(l, 4), out);
        if rows == 1 {
            assert_eq!(l.grid[0], crate::layers::ops::w4a16_gemv_grid_x(N as u32));
        } else {
            assert_eq!(l.grid[0], N.div_ceil(4) as u32, "rows={rows}");
        }
        if rows <= 3 {
            // Fixed-M signature: (A, B, S, s2, C, N, K).
            assert_eq!(l.args.len(), 7, "rows={rows}");
            assert_eq!((u32_arg(l, 5), u32_arg(l, 6)), (N as u32, K as u32));
        } else {
            // M-guarded template: (A, B, S, s2, C, M, N, K).
            assert_eq!(l.args.len(), 8, "rows={rows}");
            assert_eq!(u32_arg(l, 5), rows as u32);
            assert_eq!((u32_arg(l, 6), u32_arg(l, 7)), (N as u32, K as u32));
        }
    }
}

#[test]
fn rows_outside_the_window_are_refused() {
    let gpu = MockGpuBackend::new();
    let w = Nvfp4Linear::new(weight(&gpu, N, K), N, K, Nvfp4Kernels::resolve(&gpu)).unwrap();
    let x = gpu.alloc(32 * K * 2).unwrap();
    let out = gpu.alloc(32 * N * 2).unwrap();
    assert!(w.decode(&gpu, x, 0, out, &scratch(), 0).is_err());
    assert!(w.decode(&gpu, x, 17, out, &scratch(), 0).is_err());
    assert!(
        gpu.launches_snapshot().is_empty(),
        "a refused call launches nothing"
    );
}

#[test]
fn prefill_runs_the_tile_gemm_and_declines_without_it() {
    let gpu = MockGpuBackend::new();
    let q = weight(&gpu, N, K);
    let w = Nvfp4Linear::new(q, N, K, Nvfp4Kernels::resolve(&gpu)).unwrap();
    let x = gpu.alloc(80 * K * 2).unwrap();
    let out = gpu.alloc(80 * N * 2).unwrap();
    assert!(w.prefill(&gpu, x, 80, out, &scratch(), 0).unwrap());
    let l = gpu.launches_snapshot().pop().unwrap();
    assert_eq!(l.grid, [N.div_ceil(64) as u32, 2, 1]);
    assert_eq!(l.block, [128, 1, 1]);
    assert_eq!(u32_arg(&l, 5), 80);
    assert!(
        w.prefill(&gpu, x, 16, out, &scratch(), 0).is_err(),
        "inside the decode window"
    );

    let no_gemm = MockGpuBackend::new();
    no_gemm.deny_kernel(GEMM_MODULE, "w4a16_gemm");
    let w = Nvfp4Linear::new(
        weight(&no_gemm, N, K),
        N,
        K,
        Nvfp4Kernels::resolve(&no_gemm),
    )
    .unwrap();
    assert!(!w.prefill(&no_gemm, x, 80, out, &scratch(), 0).unwrap());
    assert!(
        no_gemm.launches_snapshot().is_empty(),
        "a decline writes nothing"
    );

    // K = 48 is a whole number of 16-groups (decode serves it) but not of the GEMM's 64-wide K tile.
    let gpu = MockGpuBackend::new();
    let w = Nvfp4Linear::new(weight(&gpu, N, 48), N, 48, Nvfp4Kernels::resolve(&gpu)).unwrap();
    assert!(!w.prefill(&gpu, x, 80, out, &scratch(), 0).unwrap());
}

#[test]
fn dequant_passes_the_tensor_scale_as_the_combined_global() {
    let gpu = MockGpuBackend::new();
    let q = weight(&gpu, N, K);
    let w = Nvfp4Linear::new(q, N, K, Nvfp4Kernels::resolve(&gpu)).unwrap();
    let out = gpu.alloc(N * K * 2).unwrap();
    w.dequant_bf16(&gpu, out, 5).unwrap();
    let l = gpu.launches_snapshot().pop().unwrap();
    assert_eq!(l.grid, [N as u32, 1, 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(l.stream, 5);
    assert_eq!(ptr_arg(&l, 0), q.weight);
    assert_eq!(ptr_arg(&l, 1), q.weight_scale);
    assert_eq!(ptr_arg(&l, 2), out);
    assert_eq!(f32_arg(&l, 3), SCALE2);
    assert_eq!((u32_arg(&l, 4), u32_arg(&l, 5)), (N as u32, K as u32));
}

#[test]
fn from_quantized_refuses_what_the_kernels_cannot_serve() {
    let gpu = MockGpuBackend::new();
    let k = Nvfp4Kernels::resolve(&gpu);
    assert!(
        from_quantized(QuantizedWeight::null(), N, K, k).is_err(),
        "NULL weight"
    );
    assert!(
        from_quantized(weight(&gpu, N, 24), N, 24, k).is_err(),
        "K not a whole group"
    );
    assert!(
        from_quantized(weight(&gpu, N, K), 0, K, k).is_err(),
        "empty"
    );
    let mut prs = weight(&gpu, N, K);
    prs.weight_scale_2_vec = gpu.alloc(N * 4).unwrap();
    assert!(from_quantized(prs, N, K, k).is_err(), "per-row scale2");
    let bare = MockGpuBackend::new();
    bare.deny_kernel(DEQUANT_MODULE, "dequant_nvfp4_to_bf16");
    let not_ready = Nvfp4Kernels::resolve(&bare);
    assert!(
        from_quantized(weight(&gpu, N, K), N, K, not_ready).is_err(),
        "no dequant fallback"
    );
    assert!(from_quantized(weight(&gpu, N, K), N, K, k).is_ok());
}

#[test]
fn bytes_format_and_scratch() {
    // The 4B FFN gate [9216, 2560]: 11,796,480 code bytes + 1,474,560 scale bytes.
    assert_eq!(device_bytes_of(9216, 2560), 13_271_040);
    assert_eq!(
        device_bytes_of(9216, 2560),
        crate::weight_loader::fp8_residency::nvfp4_bytes(9216, 2560)
    );
    let gpu = MockGpuBackend::new();
    let w = from_quantized(weight(&gpu, N, K), N, K, Nvfp4Kernels::resolve(&gpu)).unwrap();
    assert_eq!(w.format(), W4Format::Nvfp4);
    assert_eq!((w.n(), w.k()), (N, K));
    assert_eq!(w.device_bytes(), N * K / 2 + N * K / 16);
    assert_eq!(w.scratch_bytes(16), 0);
    assert_eq!(w.max_decode_rows(), 16);
    assert!(
        w.as_any()
            .and_then(|a| a.downcast_ref::<Nvfp4Linear>())
            .is_some()
    );
}
