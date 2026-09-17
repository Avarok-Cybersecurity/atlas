// SPDX-License-Identifier: AGPL-3.0-only
//! Real-model GEMV parity test (decode-path matvec on real layer-3 q_proj weights).

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;

/// Real-data parity check for `mlx_int8_gemv`. Loads the actual
/// `language_model.model.layers.3.self_attn.q_proj` triplet (the
/// first full_attention layer's Q projection), subsets to the
/// first `N=128` output rows, runs the fused dequant+matvec on
/// a synthetic activation vector at the model's true hidden
/// dimension, and compares to a CPU reference that dequantizes
/// those exact bytes the same way.
///
/// `#[ignore]`-gated by default; requires the local model copy.
#[test]
#[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
fn metal_mlx_int8_gemv_real_model_q_proj() {
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

    let base = "language_model.model.layers.3.self_attn.q_proj";
    let weight = st.tensor(&format!("{base}.weight")).unwrap();
    let scales = st.tensor(&format!("{base}.scales")).unwrap();
    let biases = st.tensor(&format!("{base}.biases")).unwrap();

    // Real layer 3 q_proj shape: weight U32 [8192, 640], i.e.
    // out=8192, in_features=2560 (= 640 * 4 packed bytes).
    let weight_shape = weight.shape();
    let full_out = weight_shape[0];
    let in_packed_cols = weight_shape[1];
    let in_features = (in_packed_cols * 4) as u32;
    assert_eq!(
        in_features, 2560,
        "expected hidden_size=2560 for Qwen3.5-4B"
    );
    assert_eq!(
        full_out, 8192,
        "expected num_heads*head_dim*2=8192 for layer 3 q_proj (with attn output gate)"
    );

    // Subset to the first N=128 output rows so the test runs in
    // a few hundred ms on M-series rather than ~21 M dequant ops.
    let n_rows: usize = 128;
    let group_size: u32 = 64;
    let groups_per_row = (in_features / group_size) as usize;

    let weight_data = weight.data();
    let scales_data = scales.data();
    let biases_data = biases.data();

    let row_stride_packed = in_packed_cols * 4; // u32 per col
    let row_stride_scales = groups_per_row * 2; // bf16 per group

    let mut packed_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_packed);
    let mut scales_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_scales);
    let mut biases_slice: Vec<u8> = Vec::with_capacity(n_rows * row_stride_scales);
    for r in 0..n_rows {
        let p_off = r * row_stride_packed;
        packed_slice.extend_from_slice(&weight_data[p_off..p_off + row_stride_packed]);
        let s_off = r * row_stride_scales;
        scales_slice.extend_from_slice(&scales_data[s_off..s_off + row_stride_scales]);
        biases_slice.extend_from_slice(&biases_data[s_off..s_off + row_stride_scales]);
    }

    // Synthetic input activation in a typical post-norm range.
    let x_bf16: Vec<half::bf16> = (0..in_features)
        .map(|i| half::bf16::from_f32(0.05 + 0.001 * (i as f32).sin()))
        .collect();

    // CPU reference: dequant byte-by-byte then dot with x.
    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows];
    for r in 0..n_rows {
        let mut acc: f32 = 0.0;
        for c in 0..in_features as usize {
            let word_off = r * row_stride_packed + (c / 4) * 4;
            let word = u32::from_le_bytes([
                packed_slice[word_off],
                packed_slice[word_off + 1],
                packed_slice[word_off + 2],
                packed_slice[word_off + 3],
            ]);
            let byte = ((word >> ((c % 4) * 8)) & 0xFF) as f32;
            let g = c / group_size as usize;
            let s_idx = (r * groups_per_row + g) * 2;
            let s =
                half::bf16::from_le_bytes([scales_slice[s_idx], scales_slice[s_idx + 1]]).to_f32();
            let b =
                half::bf16::from_le_bytes([biases_slice[s_idx], biases_slice[s_idx + 1]]).to_f32();
            let w = byte * s + b;
            acc += w * x_bf16[c].to_f32();
        }
        expected[r] = half::bf16::from_f32(acc);
    }

    // Run the kernel on the same subset.
    let Some(backend) = maybe_backend() else {
        return;
    };

    let n: u32 = n_rows as u32;
    let k: u32 = in_features;

    let packed_ptr = backend.alloc(packed_slice.len()).unwrap();
    let scales_ptr = backend.alloc(scales_slice.len()).unwrap();
    let biases_ptr = backend.alloc(biases_slice.len()).unwrap();
    let x_bytes = bf16_slice_to_bytes(&x_bf16);
    let x_ptr = backend.alloc(x_bytes.len()).unwrap();
    let y_ptr = backend.alloc(n_rows * 2).unwrap();
    backend.copy_h2d(&packed_slice, packed_ptr).unwrap();
    backend.copy_h2d(&scales_slice, scales_ptr).unwrap();
    backend.copy_h2d(&biases_slice, biases_ptr).unwrap();
    backend.copy_h2d(&x_bytes, x_ptr).unwrap();

    let kernel = backend.kernel("mlx_int8_gemv", "mlx_int8_gemv").unwrap();
    backend
        .launch_typed(
            kernel,
            [n, 1, 1],
            [64, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&n.to_le_bytes()),
                KernelArg::Bytes(&k.to_le_bytes()),
                KernelArg::Bytes(&group_size.to_le_bytes()),
                KernelArg::Buffer(packed_ptr),
                KernelArg::Buffer(scales_ptr),
                KernelArg::Buffer(biases_ptr),
                KernelArg::Buffer(x_ptr),
                KernelArg::Buffer(y_ptr),
            ],
        )
        .expect("launch real-model gemv");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut y_raw = vec![0u8; n_rows * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
    let actual = bytes_to_bf16_vec(&y_raw);

    // 2560-element FP32 sum at output magnitudes ~0.1–1 has ULP
    // ≈ 0.005 at 1.0; tolerate 0.1 for ordering drift across
    // simdgroups versus the strictly-sequential CPU reference.
    let mut max_abs_diff: f32 = 0.0;
    let mut max_rel_diff: f32 = 0.0;
    for i in 0..n_rows {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        let rel = if e.abs() > 1e-3 { d / e.abs() } else { 0.0 };
        if rel > max_rel_diff {
            max_rel_diff = rel;
        }
        // Also assert no NaN / inf — the real signal that the
        // chain is wired correctly.
        assert!(
            a.is_finite(),
            "real-model gemv produced non-finite at row {i}: {a}"
        );
    }
    assert!(
        max_abs_diff < 0.1 || max_rel_diff < 0.05,
        "real-model gemv: max abs diff {max_abs_diff}, max rel diff {max_rel_diff}"
    );

    backend.free(packed_ptr).unwrap();
    backend.free(scales_ptr).unwrap();
    backend.free(biases_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(y_ptr).unwrap();
}
