// SPDX-License-Identifier: AGPL-3.0-only

//! [`super::quantize_to_nvfp4_rows`]: the global scale still spans every row,
//! only the first `rows_out` rows are materialized.

use super::{quantize_to_nvfp4, quantize_to_nvfp4_rows};
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

fn u32_arg(a: &MockArg) -> u32 {
    match a {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        MockArg::Buffer(_) => panic!("expected a scalar argument"),
    }
}

/// `(absmax total, quantize grid, quantize N, live bytes)` of one call.
fn run(rows_out: Option<usize>, n: usize, k: usize) -> (u32, u32, u32, usize) {
    let gpu = MockGpuBackend::new();
    let w = DenseWeight {
        weight: gpu.alloc(n * k * 2).unwrap(),
    };
    let absmax = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax").unwrap();
    let quant = gpu
        .kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")
        .unwrap();
    let stream = gpu.default_stream();
    let _q = match rows_out {
        Some(r) => quantize_to_nvfp4_rows(&w, r, n, k, &gpu, absmax, quant, stream).unwrap(),
        None => quantize_to_nvfp4(&w, n, k, &gpu, absmax, quant, stream).unwrap(),
    };
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 2, "one absmax pass, one quantize pass");
    let (a, q) = (&l[0], &l[1]);
    (
        u32_arg(&a.args[2]),
        q.grid[0],
        u32_arg(&q.args[4]),
        gpu.live_bytes().unwrap(),
    )
}

#[test]
fn a_row_prefix_keeps_the_full_global_scale_and_saves_the_rest() {
    // Shaped like the 4B's draft head at --mtp-vocab 100000, scaled down:
    // 64 of 160 rows, K = 256.
    let (n, v, k) = (160usize, 64usize, 256usize);
    let (full_total, full_grid, full_n, full_bytes) = run(None, n, k);
    let (pre_total, pre_grid, pre_n, pre_bytes) = run(Some(v), n, k);
    // The global absmax reads every element either way, so scale2 (and with
    // it every packed byte of the kept rows) is the same.
    assert_eq!(full_total as usize, n * k);
    assert_eq!(pre_total, full_total);
    // Only the prefix is quantized, and only its bytes are allocated.
    assert_eq!((full_grid, full_n), (n as u32, n as u32));
    assert_eq!((pre_grid, pre_n), (v as u32, v as u32));
    assert_eq!(full_bytes - pre_bytes, (n - v) * (k / 2 + k / 16));
}

#[test]
fn a_prefix_past_the_end_is_the_whole_tensor() {
    let (n, k) = (32usize, 128usize);
    assert_eq!(run(Some(n * 4), n, k), run(None, n, k));
}

#[test]
fn the_4b_draft_head_at_the_default_mtp_vocab_frees_203_mib() {
    // Qwen3.5-4B: vocab 248,070, hidden 2,560, `--mtp-vocab` default 100,000.
    let (vocab, v, h) = (248_070usize, 100_000usize, 2_560usize);
    let bytes = |rows: usize| rows * h / 2 + rows * h / 16;
    assert_eq!(bytes(vocab), 357_220_800); // 302.8 MiB packed + 37.9 MiB scales
    assert_eq!(bytes(vocab) - bytes(v), 213_220_800); // 203.3 MiB
}
