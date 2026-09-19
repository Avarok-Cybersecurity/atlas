// SPDX-License-Identifier: AGPL-3.0-only
//! Real-model GEMV parity test (decode-path matvec on real layer-3 q_proj weights).

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
use crate::weights::mlx_int8::MlxInt8Weight;

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

    // CPU reference: dequant byte-by-byte then dot with x. `sum_abs_terms`
    // is the per-row sum of |w * x|, which sizes the fp32 reordering bound
    // below -- it is measured from the same bytes, not assumed.
    let mut expected: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows];
    let mut sum_abs_terms: Vec<f32> = vec![0.0; n_rows];
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
            let term = w * x_bf16[c].to_f32();
            acc += term;
            sum_abs_terms[r] += term.abs();
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

    // ★ THE PRODUCTION WRAPPER, not a hand-rolled launch. This test used to
    // dispatch `[n,1,1] x [64,1,1]` against a kernel whose contract
    // (mlx_int8_gemv.metal:31) is `ceil(N/4)` threadgroups of 128 threads, so
    // every row with r mod 4 in {2,3} was never written -- and the assertion
    // below let that through. A test that encodes its own geometry tests a
    // launch nothing ships; `MlxInt8Weight::gemv` is the launch that ships.
    let weight = MlxInt8Weight {
        packed: packed_ptr,
        scales: scales_ptr,
        biases: biases_ptr,
        out_features: n,
        in_features: k,
        group_size,
    };
    weight
        .gemv(&backend, x_ptr, y_ptr, backend.default_stream())
        .expect("launch real-model gemv");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut y_raw = vec![0u8; n_rows * 2];
    backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
    let actual = bytes_to_bf16_vec(&y_raw);

    // ★ THE BOUND IS DERIVED, NOT PICKED, AND BOTH HALVES MUST HOLD.
    //
    // The old assertion was `max_abs < 0.1 || max_rel < 0.05`. With this
    // input (x ~ 0.05) the outputs sit at |q| ~ 0.05, so a row that was NEVER
    // WRITTEN (d = |e| < 0.1) satisfied the first disjunct on its own and the
    // 100%-relative error never mattered. It passed with half the rows zero.
    //
    // Per row, kernel and reference both take the same fp32 products, sum them
    // in a different order (lane-strided + simd_sum vs sequential), and round
    // ONCE to bf16. So they may differ by:
    //   * one bf16 ulp of the value (the two roundings straddle a boundary), plus
    //   * the fp32 reordering allowance: each K-term summation carries at most
    //     (K-1)*eps*sum|terms| of rounding error, so two orders differ by at
    //     most twice that; +2 terms covers the fma-vs-mul-add contraction of
    //     `byte*s+b` that -ffast-math permits.
    // At these magnitudes that is ~1.5e-3 per row (vs 0.1), and a row left
    // unwritten trips it whenever |e| exceeds ~1.5e-3 -- about 97% of rows.
    // The cosine and norm-ratio gates below cover the aggregate: one zeroed
    // row of 128 costs ~0.8% of the energy and scores cos ~0.996 < 0.9999.
    let mut max_abs_diff: f32 = 0.0;
    let mut worst_ratio: f32 = 0.0;
    for i in 0..n_rows {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        assert!(
            a.is_finite(),
            "real-model gemv produced non-finite at row {i}: {a}"
        );
        let d = (e - a).abs();
        max_abs_diff = max_abs_diff.max(d);
        let fp32_allowance = (2.0 * k as f32 + 2.0) * f32::EPSILON * sum_abs_terms[i];
        let bound = bf16_ulp(e.abs().max(a.abs())) + fp32_allowance;
        worst_ratio = worst_ratio.max(d / bound);
        assert!(
            d <= bound,
            "real-model gemv row {i}: |kernel - cpu| = {d} exceeds the derived \
             bound {bound} (1 bf16 ulp + fp32 reorder allowance {fp32_allowance}); \
             expected {e}, got {a}"
        );
    }
    let cos = cosine_bf16(&expected, &actual);
    let mag = norm_ratio_bf16(&expected, &actual);
    eprintln!(
        "metal_mlx_int8_gemv_real_model_q_proj: rows={n_rows} K={k} \
         max_abs={max_abs_diff:.3e} worst_d/bound={worst_ratio:.3} \
         cos={cos:.7} norm_ratio={mag:.7}"
    );
    assert!(
        cos >= COSINE_GATE,
        "real-model gemv: cosine {cos} < {COSINE_GATE}"
    );
    assert!(
        mag >= COSINE_GATE,
        "real-model gemv: norm ratio {mag} < {COSINE_GATE}"
    );

    backend.free(packed_ptr).unwrap();
    backend.free(scales_ptr).unwrap();
    backend.free(biases_ptr).unwrap();
    backend.free(x_ptr).unwrap();
    backend.free(y_ptr).unwrap();
}
