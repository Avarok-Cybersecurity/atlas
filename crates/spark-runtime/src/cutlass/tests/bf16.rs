// SPDX-License-Identifier: AGPL-3.0-only
//! Dense BF16 CUTLASS smoke/correctness + tile-variant & cuBLASLt algo benches.

use super::super::*;
use super::*;

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_bf16_ffi_smoke_computes_row_major_act_by_weight_t() {
    const M: usize = 128;
    const N: usize = 128;
    const K: usize = 32;

    let act_f32: Vec<f32> = (0..M * K)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.125)
        .collect();
    let weight_f32: Vec<f32> = (0..N * K).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    let act: Vec<u16> = act_f32.iter().copied().map(f32_to_bf16).collect();
    let weight: Vec<u16> = weight_f32.iter().copied().map(f32_to_bf16).collect();
    let mut out = vec![0u16; M * N];

    let act_dev;
    let weight_dev;
    let out_dev;
    unsafe {
        act_dev = device_alloc(act.len() * 2);
        weight_dev = device_alloc(weight.len() * 2);
        out_dev = device_alloc(out.len() * 2);
        copy_h2d(act_dev, &act);
        copy_h2d(weight_dev, &weight);
    }

    let result = bf16_gemm_act_weight_t(
        act_dev as u64,
        weight_dev as u64,
        out_dev as u64,
        M as u32,
        N as u32,
        K as u32,
        0,
    );
    assert!(result.is_ok(), "{result:?}");

    unsafe {
        cuda_check(cudaDeviceSynchronize(), "device synchronize");
        copy_d2h(&mut out, out_dev);
        cuda_check(cudaFree(act_dev), "free act");
        cuda_check(cudaFree(weight_dev), "free weight");
        cuda_check(cudaFree(out_dev), "free out");
    }

    for m in 0..M {
        for n in 0..N {
            let mut expected = 0.0f32;
            for k in 0..K {
                expected += bf16_to_f32(act[m * K + k]) * bf16_to_f32(weight[n * K + k]);
            }
            let actual = bf16_to_f32(out[m * N + n]);
            assert!(
                (actual - expected).abs() < 0.025,
                "m={m} n={n} actual={actual} expected={expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_bf16_ffi_computes_holo_ssm_qkvz_shape() {
    const M: usize = 3537;
    const N: usize = 12288;
    const K: usize = 2048;

    let mut act = vec![0u16; M * K];
    for m in 0..M {
        act[m * K + (m % K)] = f32_to_bf16(1.0);
    }
    let weight: Vec<u16> = (0..N * K)
        .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
        .collect();
    let mut out = vec![0u16; M * N];

    let act_dev;
    let weight_dev;
    let out_dev;
    unsafe {
        act_dev = device_alloc(act.len() * 2);
        weight_dev = device_alloc(weight.len() * 2);
        out_dev = device_alloc(out.len() * 2);
        copy_h2d(act_dev, &act);
        copy_h2d(weight_dev, &weight);
    }

    let result = bf16_gemm_act_weight_t(
        act_dev as u64,
        weight_dev as u64,
        out_dev as u64,
        M as u32,
        N as u32,
        K as u32,
        0,
    );
    assert!(result.is_ok(), "{result:?}");

    unsafe {
        cuda_check(cudaDeviceSynchronize(), "device synchronize");
        copy_d2h(&mut out, out_dev);
        cuda_check(cudaFree(act_dev), "free act");
        cuda_check(cudaFree(weight_dev), "free weight");
        cuda_check(cudaFree(out_dev), "free out");
    }

    for m in 0..M {
        let selected_k = m % K;
        for n in 0..N {
            let actual = bf16_to_f32(out[m * N + n]);
            let expected = bf16_to_f32(weight[n * K + selected_k]);
            assert_eq!(
                actual, expected,
                "m={m} n={n} selected_k={selected_k} actual={actual} expected={expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_bf16_holo_qkvz_bench_against_cublaslt() {
    const ITERS: usize = 100;

    let shapes = [
        ("ssm_qkvz", 3537usize, 12288usize, 2048usize),
        ("ssm_out", 3537, 2048, 4096),
        ("attn_q", 3537, 8192, 2048),
        ("attn_k", 3537, 512, 2048),
        ("attn_v", 3537, 512, 2048),
        ("attn_o", 3537, 2048, 4096),
        ("moe_gate_up_dense", 28296, 1024, 2048),
        ("moe_down_dense", 28296, 2048, 512),
    ];

    let variants: [(&str, CutlassVariant); 6] = [
        ("128x128", atlas_cutlass_bf16_gemm_act_weight_t),
        ("128x256", atlas_cutlass_bf16_gemm_act_weight_t_128x256),
        ("256x128", atlas_cutlass_bf16_gemm_act_weight_t_256x128),
        ("64x128", atlas_cutlass_bf16_gemm_act_weight_t_64x128),
        ("128x64", atlas_cutlass_bf16_gemm_act_weight_t_128x64),
        ("64x64", atlas_cutlass_bf16_gemm_act_weight_t_64x64),
    ];

    for (name, m, n, k) in shapes {
        let act: Vec<u16> = (0..m * k)
            .map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.03125))
            .collect();
        let weight: Vec<u16> = (0..n * k)
            .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
            .collect();

        let act_dev;
        let weight_dev;
        let cutlass_out;
        let cublas_out;
        unsafe {
            act_dev = device_alloc(act.len() * 2);
            weight_dev = device_alloc(weight.len() * 2);
            cutlass_out = device_alloc(m * n * 2);
            cublas_out = device_alloc(m * n * 2);
            copy_h2d(act_dev, &act);
            copy_h2d(weight_dev, &weight);
        }

        crate::cublaslt::bf16_gemm_act_weight_t(
            act_dev as u64,
            weight_dev as u64,
            cublas_out as u64,
            m as u32,
            n as u32,
            k as u32,
            0,
        )
        .unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
        }

        let mut best_variant = "";
        let mut best_cutlass_ms = f64::INFINITY;
        for (variant, f) in variants {
            run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k).unwrap();
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "cutlass warmup synchronize");
            }
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k).unwrap();
            }
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "cutlass synchronize");
            }
            let cutlass_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
            if cutlass_ms < best_cutlass_ms {
                best_cutlass_ms = cutlass_ms;
                best_variant = variant;
            }
        }

        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            crate::cublaslt::bf16_gemm_act_weight_t(
                act_dev as u64,
                weight_dev as u64,
                cublas_out as u64,
                m as u32,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
        }
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "cublas synchronize");
        }
        let cublas_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        eprintln!(
            "HOLO_DENSE_BENCH {name} M={m} N={n} K={k} iters={ITERS} best_cutlass={best_variant} cutlass_ms={best_cutlass_ms:.3} cutlass_tflops={:.1} cublaslt_ms={cublas_ms:.3} cublaslt_tflops={:.1} speedup_vs_cublas={:.3}",
            flop / (best_cutlass_ms / 1000.0) / 1.0e12,
            flop / (cublas_ms / 1000.0) / 1.0e12,
            cublas_ms / best_cutlass_ms
        );

        unsafe {
            cuda_check(cudaFree(act_dev), "free act");
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(cutlass_out), "free cutlass out");
            cuda_check(cudaFree(cublas_out), "free cublas out");
        }
    }
}

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cutlass_bf16_holo_decode_route_batch_shapes() {
    const ITERS: usize = 2000;
    let shapes = [
        ("moe_gate_up_routes_c1", 8usize, 1024usize, 2048usize),
        ("moe_gate_up_routes_c2", 16, 1024, 2048),
        ("moe_gate_up_routes_c4", 32, 1024, 2048),
        ("moe_gate_up_routes_c8", 64, 1024, 2048),
        ("moe_gate_up_routes_c16", 128, 1024, 2048),
        ("moe_down_routes_c1", 8, 2048, 512),
        ("moe_down_routes_c2", 16, 2048, 512),
        ("moe_down_routes_c4", 32, 2048, 512),
        ("moe_down_routes_c8", 64, 2048, 512),
        ("moe_down_routes_c16", 128, 2048, 512),
    ];
    let variants: [(&str, CutlassVariant); 6] = [
        ("128x128", atlas_cutlass_bf16_gemm_act_weight_t),
        ("128x256", atlas_cutlass_bf16_gemm_act_weight_t_128x256),
        ("256x128", atlas_cutlass_bf16_gemm_act_weight_t_256x128),
        ("64x128", atlas_cutlass_bf16_gemm_act_weight_t_64x128),
        ("128x64", atlas_cutlass_bf16_gemm_act_weight_t_128x64),
        ("64x64", atlas_cutlass_bf16_gemm_act_weight_t_64x64),
    ];

    for (name, m, n, k) in shapes {
        let act: Vec<u16> = (0..m * k)
            .map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.03125))
            .collect();
        let weight: Vec<u16> = (0..n * k)
            .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
            .collect();

        let act_dev;
        let weight_dev;
        let cutlass_out;
        let cublas_out;
        unsafe {
            act_dev = device_alloc(act.len() * 2);
            weight_dev = device_alloc(weight.len() * 2);
            cutlass_out = device_alloc(m * n * 2);
            cublas_out = device_alloc(m * n * 2);
            copy_h2d(act_dev, &act);
            copy_h2d(weight_dev, &weight);
        }

        crate::cublaslt::bf16_gemm_act_weight_t(
            act_dev as u64,
            weight_dev as u64,
            cublas_out as u64,
            m as u32,
            n as u32,
            k as u32,
            0,
        )
        .unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "warmup synchronize");
        }

        let mut best_variant = "";
        let mut best_cutlass_ms = f64::INFINITY;
        for (variant, f) in variants {
            run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k).unwrap();
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "cutlass warmup synchronize");
            }
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                run_cutlass_variant(variant, f, act_dev, weight_dev, cutlass_out, m, n, k).unwrap();
            }
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "cutlass synchronize");
            }
            let cutlass_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
            if cutlass_ms < best_cutlass_ms {
                best_cutlass_ms = cutlass_ms;
                best_variant = variant;
            }
        }

        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            crate::cublaslt::bf16_gemm_act_weight_t(
                act_dev as u64,
                weight_dev as u64,
                cublas_out as u64,
                m as u32,
                n as u32,
                k as u32,
                0,
            )
            .unwrap();
        }
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "cublas synchronize");
        }
        let cublas_ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;

        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        eprintln!(
            "HOLO_ROUTE_BATCH_BENCH {name} M={m} N={n} K={k} iters={ITERS} best_cutlass={best_variant} cutlass_us={:.3} cutlass_tflops={:.1} cublaslt_us={:.3} cublaslt_tflops={:.1} speedup_vs_cublas={:.3}",
            best_cutlass_ms * 1000.0,
            flop / (best_cutlass_ms / 1000.0) / 1.0e12,
            cublas_ms * 1000.0,
            flop / (cublas_ms / 1000.0) / 1.0e12,
            cublas_ms / best_cutlass_ms
        );

        unsafe {
            cuda_check(cudaFree(act_dev), "free act");
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(cutlass_out), "free cutlass out");
            cuda_check(cudaFree(cublas_out), "free cublas out");
        }
    }
}

#[test]
#[ignore = "requires a free CUDA device and CUTLASS_HOME build"]
fn cublaslt_bf16_holo_route_batch_algo_sweep() {
    const ITERS: usize = 2000;
    let shapes = [
        ("gate_up_c1", 8usize, 1024usize, 2048usize),
        ("gate_up_c2", 16, 1024, 2048),
        ("gate_up_c4", 32, 1024, 2048),
        ("gate_up_c8", 64, 1024, 2048),
        ("gate_up_c16", 128, 1024, 2048),
        ("down_c1", 8, 2048, 512),
        ("down_c2", 16, 2048, 512),
        ("down_c4", 32, 2048, 512),
        ("down_c8", 64, 2048, 512),
        ("down_c16", 128, 2048, 512),
    ];

    for (name, m, n, k) in shapes {
        let act: Vec<u16> = (0..m * k)
            .map(|i| f32_to_bf16(((i % 13) as f32 - 6.0) * 0.03125))
            .collect();
        let weight: Vec<u16> = (0..n * k)
            .map(|i| f32_to_bf16(((i % 17) as f32 - 8.0) * 0.0625))
            .collect();
        let act_dev;
        let weight_dev;
        let out_dev;
        unsafe {
            act_dev = device_alloc(act.len() * 2);
            weight_dev = device_alloc(weight.len() * 2);
            out_dev = device_alloc(m * n * 2);
            copy_h2d(act_dev, &act);
            copy_h2d(weight_dev, &weight);
        }

        let returned = run_cublaslt_algo(0, act_dev, weight_dev, out_dev, m, n, k).unwrap();
        unsafe {
            cuda_check(cudaDeviceSynchronize(), "algo warmup synchronize");
        }
        let mut best_algo = 0;
        let mut best_ms = f64::INFINITY;
        for algo in 0..returned.min(16) {
            if run_cublaslt_algo(algo, act_dev, weight_dev, out_dev, m, n, k).is_err() {
                continue;
            }
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "algo warmup synchronize");
            }
            let t0 = std::time::Instant::now();
            let mut ok = true;
            for _ in 0..ITERS {
                if run_cublaslt_algo(algo, act_dev, weight_dev, out_dev, m, n, k).is_err() {
                    ok = false;
                    break;
                }
            }
            unsafe {
                cuda_check(cudaDeviceSynchronize(), "algo synchronize");
            }
            if ok {
                let ms = t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64;
                if ms < best_ms {
                    best_ms = ms;
                    best_algo = algo;
                }
            }
        }
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        eprintln!(
            "HOLO_CUBLASLT_ALGO_BENCH {name} M={m} N={n} K={k} returned={returned} best_algo={best_algo} best_us={:.3} best_tflops={:.1}",
            best_ms * 1000.0,
            flop / (best_ms / 1000.0) / 1.0e12
        );
        unsafe {
            cuda_check(cudaFree(act_dev), "free act");
            cuda_check(cudaFree(weight_dev), "free weight");
            cuda_check(cudaFree(out_dev), "free out");
        }
    }
}
