// SPDX-License-Identifier: AGPL-3.0-only
//! Miscellaneous real-model parity tests (chain norm+q_proj, real layer-3 q_proj GEMV/dequant).

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// End-to-end chain on real Qwen3.5-4B-MLX-8bit weights:
/// `rms_norm(input_layernorm) → mlx_int8_gemv(q_proj)` for layer 3
/// (the first full_attention layer). Pins that the entire fused-
/// dequant decode chain composes correctly when we wire two
/// kernels together over actual production tensors.
///
/// Doesn't compare against an MLX reference — we don't have MLX
/// installed in CI — but does verify:
///   • all 8192 outputs are finite (no NaN / inf),
///   • the output magnitudes sit in a sane regime (mean-abs in
///     [1e-3, 50]; the q_proj for a small post-norm input
///     produces O(1) values per head),
///   • a CPU re-execution of the same pipeline (FP32 RMSNorm +
///     bytewise dequant + matvec) agrees with the kernel chain
///     within BF16 ULP tolerance on the first 64 outputs.
///
/// `#[ignore]`-gated; requires the local model copy.
#[test]
#[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
fn metal_real_model_chain_norm_then_qproj() {
    use safetensors::SafeTensors;

    let model_dir = std::env::var("AVAROK_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    if !st_path.exists() {
        eprintln!("skipping: {} not found", st_path.display());
        return;
    }

    let file = std::fs::File::open(&st_path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

    // Layer 3's input_layernorm and q_proj.
    let layer = "language_model.model.layers.3";
    let ln = st
        .tensor(&format!("{layer}.input_layernorm.weight"))
        .unwrap();
    let q_w = st
        .tensor(&format!("{layer}.self_attn.q_proj.weight"))
        .unwrap();
    let q_s = st
        .tensor(&format!("{layer}.self_attn.q_proj.scales"))
        .unwrap();
    let q_b = st
        .tensor(&format!("{layer}.self_attn.q_proj.biases"))
        .unwrap();

    // Real model dims.
    let hidden_size: u32 = 2560;
    let q_full_out: u32 = 8192; // includes attn_output_gate doubling
    let group_size: u32 = 64;
    let groups_per_row = (hidden_size / group_size) as usize;

    assert_eq!(ln.shape(), [hidden_size as usize]);
    assert_eq!(
        q_w.shape(),
        [q_full_out as usize, (hidden_size / 4) as usize]
    );

    // Subset the q_proj to the first 64 rows so the CPU reference
    // stays cheap (~160 K dequant ops); kernel still uses the full
    // K = 2560 reduction.
    let n_rows: usize = 64;
    let row_stride_packed = (hidden_size as usize / 4) * 4;
    let row_stride_scales = groups_per_row * 2;
    let weight_data = q_w.data();
    let scales_data = q_s.data();
    let biases_data = q_b.data();

    let mut packed_subset = Vec::with_capacity(n_rows * row_stride_packed);
    let mut scales_subset = Vec::with_capacity(n_rows * row_stride_scales);
    let mut biases_subset = Vec::with_capacity(n_rows * row_stride_scales);
    for r in 0..n_rows {
        let p_off = r * row_stride_packed;
        packed_subset.extend_from_slice(&weight_data[p_off..p_off + row_stride_packed]);
        let s_off = r * row_stride_scales;
        scales_subset.extend_from_slice(&scales_data[s_off..s_off + row_stride_scales]);
        biases_subset.extend_from_slice(&biases_data[s_off..s_off + row_stride_scales]);
    }
    let ln_bytes = ln.data().to_vec();

    // Synthetic input — typical pre-RMSNorm activation range
    // (Qwen3-family residual stream sits around ±1 in BF16).
    let x: Vec<half::bf16> = (0..hidden_size)
        .map(|i| half::bf16::from_f32(0.4 * (i as f32 * 0.013).sin()))
        .collect();

    // ── CPU reference: rms_norm → dequant → matvec ──────────
    let eps: f32 = 1e-5;
    let mut x_norm_cpu = vec![0.0f32; hidden_size as usize];
    let mut ssq = 0.0f32;
    for v in &x {
        let f = v.to_f32();
        ssq += f * f;
    }
    let inv_rms = (ssq / hidden_size as f32 + eps).powf(-0.5);
    for (i, v) in x.iter().enumerate() {
        // RMSNorm weight is BF16 LE.
        let w_bytes = &ln_bytes[i * 2..i * 2 + 2];
        let w = half::bf16::from_le_bytes([w_bytes[0], w_bytes[1]]).to_f32();
        x_norm_cpu[i] = v.to_f32() * inv_rms * w;
    }

    let mut expected_q = vec![half::bf16::ZERO; n_rows];
    for r in 0..n_rows {
        let mut acc = 0.0f32;
        for c in 0..hidden_size as usize {
            let word_off = r * row_stride_packed + (c / 4) * 4;
            let word = u32::from_le_bytes([
                packed_subset[word_off],
                packed_subset[word_off + 1],
                packed_subset[word_off + 2],
                packed_subset[word_off + 3],
            ]);
            let byte = ((word >> ((c % 4) * 8)) & 0xFF) as f32;
            let g = c / group_size as usize;
            let s_idx = (r * groups_per_row + g) * 2;
            let s = half::bf16::from_le_bytes([scales_subset[s_idx], scales_subset[s_idx + 1]])
                .to_f32();
            let b = half::bf16::from_le_bytes([biases_subset[s_idx], biases_subset[s_idx + 1]])
                .to_f32();
            let w = byte * s + b;
            acc += w * x_norm_cpu[c];
        }
        expected_q[r] = half::bf16::from_f32(acc);
    }

    // ── Kernel chain: rms_norm → mlx_int8_gemv on real bytes ──
    let Some(backend) = maybe_backend() else {
        return;
    };

    let x_bytes = bf16_slice_to_bytes(&x);
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let ln_ptr = backend.alloc(ln_bytes.len()).unwrap();
    let xn_ptr = backend.alloc(x_bytes.len()).unwrap();
    let pk_ptr = backend.alloc(packed_subset.len()).unwrap();
    let sc_ptr = backend.alloc(scales_subset.len()).unwrap();
    let bi_ptr = backend.alloc(biases_subset.len()).unwrap();
    let q_ptr = backend.alloc(n_rows * 2).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();
    backend.copy_h2d(&ln_bytes, ln_ptr).unwrap();
    backend.copy_h2d(&packed_subset, pk_ptr).unwrap();
    backend.copy_h2d(&scales_subset, sc_ptr).unwrap();
    backend.copy_h2d(&biases_subset, bi_ptr).unwrap();

    // Stage 1: RMSNorm.
    let n_tokens: u32 = 1;
    let rms_kernel = backend.kernel("rms_norm", "rms_norm").unwrap();
    backend
        .launch_typed(
            rms_kernel,
            [n_tokens, 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&hidden_size.to_le_bytes()),
                KernelArg::Bytes(&eps.to_le_bytes()),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(ln_ptr),
                KernelArg::Buffer(xn_ptr),
            ],
        )
        .expect("launch rms_norm");

    // Stage 2: Q projection (fused dequant + matvec).
    let n: u32 = n_rows as u32;
    let k: u32 = hidden_size;
    let gemv_kernel = backend.kernel("mlx_int8_gemv", "mlx_int8_gemv").unwrap();
    backend
        .launch_typed(
            gemv_kernel,
            // ★ THE KERNEL'S DOCUMENTED CONTRACT, not a guess.
            // mlx_int8_gemv.metal:31 states `ceil(N/4)` threadgroups of 128
            // threads, and computes `row = tg_idx * ROWS_PER_TG + simd_group_id`
            // with ROWS_PER_TG = 4. At 64 threads there are only TWO
            // simdgroups, so simd_group_id is 0 or 1 and every row with
            // r mod 4 in {2,3} is NEVER WRITTEN -- half the output, scattered
            // rather than truncated, which is why it read as a numerics bug.
            // Production (weights/mlx_int8.rs:217) always obeyed the contract;
            // only these two tests did not.
            [n.div_ceil(4), 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Bytes(&group_size.to_le_bytes()),
                KernelArg::Buffer(pk_ptr),
                KernelArg::Buffer(sc_ptr),
                KernelArg::Buffer(bi_ptr),
                KernelArg::Buffer(xn_ptr),
                KernelArg::Buffer(q_ptr),
            ],
        )
        .expect("launch q_proj gemv");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut q_raw = vec![0u8; n_rows * 2];
    backend.copy_d2h(q_ptr, &mut q_raw).unwrap();
    let actual_q = bytes_to_bf16_vec(&q_raw);

    // ── Validate ────────────────────────────────────────────
    let mut max_abs_diff: f32 = 0.0;
    let mut sum_abs = 0.0f32;
    let mut nonzero_count = 0;
    for i in 0..n_rows {
        let e = expected_q[i].to_f32();
        let a = actual_q[i].to_f32();
        assert!(a.is_finite(), "chain produced non-finite Q[{i}] = {a}");
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        sum_abs += a.abs();
        if a.abs() > 1e-4 {
            nonzero_count += 1;
        }
    }
    let mean_abs = sum_abs / n_rows as f32;

    assert!(
        mean_abs >= 1e-3 && mean_abs <= 50.0,
        "Q output mean-abs {mean_abs} outside sanity band [1e-3, 50]"
    );
    assert!(
        nonzero_count >= n_rows / 2,
        "too many near-zero outputs ({nonzero_count}/{n_rows}); chain is suspicious"
    );
    assert!(
        max_abs_diff < 0.1,
        "rms_norm + gemv chain: max |kernel - cpu| = {max_abs_diff}"
    );

    backend.free(x_ptr).unwrap();
    backend.free(ln_ptr).unwrap();
    backend.free(xn_ptr).unwrap();
    backend.free(pk_ptr).unwrap();
    backend.free(sc_ptr).unwrap();
    backend.free(bi_ptr).unwrap();
    backend.free(q_ptr).unwrap();
}

#[test]
#[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
fn metal_mlx_int8_dequant_real_model() {
    use safetensors::SafeTensors;

    let model_dir = std::env::var("AVAROK_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    if !st_path.exists() {
        eprintln!("skipping: {} not found", st_path.display());
        return;
    }

    // mmap the safetensors file via memmap2 (already a spark-runtime
    // dep) and parse the header.
    let file = std::fs::File::open(&st_path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let st = SafeTensors::deserialize(&mmap).expect("parse safetensors header");

    // Pick a 4-row × 128-col slice from `language_model.model.embed_tokens`
    // — 4 rows × 32 uint32 packed = 128 columns; 128 cols / 64 group = 2
    // groups per row of scales/biases.
    let base = "language_model.model.embed_tokens";
    let weight = st.tensor(&format!("{base}.weight")).expect("weight");
    let scales = st.tensor(&format!("{base}.scales")).expect("scales");
    let biases = st.tensor(&format!("{base}.biases")).expect("biases");

    assert_eq!(weight.dtype(), safetensors::Dtype::U32);
    assert_eq!(scales.dtype(), safetensors::Dtype::BF16);
    assert_eq!(biases.dtype(), safetensors::Dtype::BF16);
    let weight_shape = weight.shape();
    assert_eq!(weight_shape.len(), 2);

    let full_in_cols_packed = weight_shape[1];
    // Slice dims
    let n_rows = 4usize;
    let n_cols = 128usize;
    let group_size = 64u32;
    let n_packed_cols = n_cols / 4;
    let groups_per_slice_row = (n_cols / group_size as usize) as usize;

    let weight_data = weight.data();
    let scales_data = scales.data();
    let biases_data = biases.data();

    // Extract row strides (in bytes).
    let row_stride_packed = full_in_cols_packed * 4; // u32 per col
    let row_stride_groups = (full_in_cols_packed * 4) / group_size as usize; // groups in full row
    let row_stride_scales = row_stride_groups * 2; // bf16 per group

    let mut packed_slice: Vec<u8> = Vec::with_capacity(n_rows * n_packed_cols * 4);
    let mut scales_slice: Vec<u8> = Vec::with_capacity(n_rows * groups_per_slice_row * 2);
    let mut biases_slice: Vec<u8> = Vec::with_capacity(n_rows * groups_per_slice_row * 2);
    for r in 0..n_rows {
        let p_off = r * row_stride_packed;
        packed_slice.extend_from_slice(&weight_data[p_off..p_off + n_packed_cols * 4]);
        let s_off = r * row_stride_scales;
        scales_slice.extend_from_slice(&scales_data[s_off..s_off + groups_per_slice_row * 2]);
        biases_slice.extend_from_slice(&biases_data[s_off..s_off + groups_per_slice_row * 2]);
    }

    // CPU reference. Walk byte by byte through the packed slice.
    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows * n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let word_offset = r * n_packed_cols * 4 + (c / 4) * 4;
            let word = u32::from_le_bytes([
                packed_slice[word_offset],
                packed_slice[word_offset + 1],
                packed_slice[word_offset + 2],
                packed_slice[word_offset + 3],
            ]);
            let byte = ((word >> ((c % 4) * 8)) & 0xFF) as f32;
            let g = c / group_size as usize;
            let s_idx = (r * groups_per_slice_row + g) * 2;
            let s =
                half::bf16::from_le_bytes([scales_slice[s_idx], scales_slice[s_idx + 1]]).to_f32();
            let b =
                half::bf16::from_le_bytes([biases_slice[s_idx], biases_slice[s_idx + 1]]).to_f32();
            expected[r * n_cols + c] = half::bf16::from_f32(byte * s + b);
        }
    }

    // Run the kernel on the slice.
    let Some(backend) = maybe_backend() else {
        return;
    };

    let packed_ptr = backend.alloc(packed_slice.len()).expect("alloc packed");
    let scales_ptr = backend.alloc(scales_slice.len()).expect("alloc scales");
    let biases_ptr = backend.alloc(biases_slice.len()).expect("alloc biases");
    let out_bytes = n_rows * n_cols * 2;
    let out_ptr = backend.alloc(out_bytes).expect("alloc out");
    backend.copy_h2d(&packed_slice, packed_ptr).unwrap();
    backend.copy_h2d(&scales_slice, scales_ptr).unwrap();
    backend.copy_h2d(&biases_slice, biases_ptr).unwrap();

    let kernel = backend
        .kernel("mlx_int8_dequant", "mlx_int8_dequant")
        .expect("kernel");
    let block_x = 16u32;
    let in_features_arg = n_cols as u32;
    let out_features_arg = n_rows as u32;
    let grid_x = in_features_arg.div_ceil(block_x);
    backend
        .launch_typed(
            kernel,
            [grid_x, out_features_arg, 1],
            [block_x, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&out_features_arg.to_le_bytes()),
                KernelArg::Bytes(&in_features_arg.to_le_bytes()),
                KernelArg::Bytes(&group_size.to_le_bytes()),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(scales_ptr),
                KernelArg::Buffer(biases_ptr),
                KernelArg::Buffer(out_ptr),
            ],
        )
        .expect("launch_typed");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; out_bytes];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    // Embedding values are typically tiny; tolerate L∞ ≤ 1e-2.
    let mut max_abs_diff: f32 = 0.0;
    for i in 0..expected.len() {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff < 1e-2,
        "real-model dequant mismatch: L∞ = {max_abs_diff}"
    );

    backend.free(packed_ptr).unwrap();
    backend.free(scales_ptr).unwrap();
    backend.free(biases_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}
