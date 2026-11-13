// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the keep-packed 4-bit route's decisions and uploads (port lane L19). Nothing here
//! touches the process-wide switch: every test calls the `_with` forms, so the row stays off for every
//! other test in the binary.

use super::*;
use crate::gpu::mock::MockGpuBackend;

const FFN_DOWN: &str = "model.layers.3.mlp.down_proj.weight";
const ATTN_Q: &str = "model.layers.3.self_attn.q_proj.weight";
const GDN_Z: &str = "model.layers.0.linear_attn.in_proj_z.weight";
const GDN_QKV: &str = "model.layers.0.linear_attn.in_proj_qkv.weight";
const GDN_OUT: &str = "model.layers.0.linear_attn.out_proj.weight";
const NORM: &str = "model.layers.3.input_layernorm.weight";
const HEAD: &str = "lm_head.weight";

#[test]
fn the_row_is_off_by_default() {
    assert!(!enabled());
    assert!(!keep_head());
}

#[test]
fn packed_types_and_sizes() {
    assert_eq!(packed_type(ID_IQ4_XS), Some((WeightDtype::Iq4Xs, 136)));
    assert_eq!(packed_type(ID_Q4_K), Some((WeightDtype::Q4K, 144)));
    assert_eq!(packed_type(ID_Q6_K), Some((WeightDtype::Q6K, 210)));
    for id in [0u32, 1, 8, 10, 11, ID_Q5_K, ID_IQ4_NL, 30, 42] {
        assert_eq!(packed_type(id), None, "id {id}");
    }
}

#[test]
fn decision_table() {
    use Plan::*;
    let p = |id, hf| plan(id, hf, "qwen35", false);
    // The transform-free projections keep every packed type.
    for id in [ID_IQ4_XS, ID_Q4_K, ID_Q6_K] {
        let dt = packed_type(id).unwrap().0;
        assert_eq!(p(id, FFN_DOWN), Packed(dt));
        assert_eq!(p(id, ATTN_Q), Packed(dt));
        assert_eq!(
            p(id, GDN_Z),
            PackedRowReorder {
                dtype: dt,
                after_qk: false
            }
        );
        assert_eq!(
            p(id, GDN_QKV),
            PackedRowReorder {
                dtype: dt,
                after_qk: true
            }
        );
    }
    // A column reorder (out_proj) never stays packed; IQ4_XS there takes the existing CPU transform path.
    assert_eq!(p(ID_IQ4_XS, GDN_OUT), Existing);
    assert_eq!(p(ID_Q4_K, GDN_OUT), Existing);
    // The head only when the model side asked for it.
    assert_eq!(p(ID_Q6_K, HEAD), Existing);
    assert_eq!(
        plan(ID_Q6_K, HEAD, "qwen35", true),
        Packed(WeightDtype::Q6K)
    );
    // Only a Q6_K head is kept packed: an IQ4_XS head expands on the GPU, a Q4_K head takes the existing path.
    assert_eq!(plan(ID_IQ4_XS, HEAD, "qwen35", true), GpuDequant);
    assert_eq!(plan(ID_Q4_K, HEAD, "qwen35", true), Existing);
    // The MTP head stays BF16.
    assert_eq!(
        p(ID_IQ4_XS, "mtp.layers.0.mlp.down_proj.weight"),
        GpuDequant
    );
    assert_eq!(p(ID_Q4_K, "mtp.layers.0.mlp.down_proj.weight"), Existing);
    // Q5_K / IQ4_NL / IQ4_XS outside the packed set expand on the GPU, unless a value transform needs
    // the CPU.
    assert_eq!(p(ID_Q5_K, FFN_DOWN), GpuDequant);
    assert_eq!(p(ID_IQ4_NL, FFN_DOWN), GpuDequant);
    assert_eq!(p(ID_IQ4_XS, "model.embed_tokens.weight"), GpuDequant);
    assert_eq!(p(ID_Q5_K, NORM), Existing);
    // Other types and other architectures: the existing path.
    assert_eq!(p(8, FFN_DOWN), Existing);
    assert_eq!(p(10, FFN_DOWN), Existing);
    assert_eq!(plan(ID_IQ4_XS, FFN_DOWN, "llama", false), GpuDequant);
    assert_eq!(plan(ID_Q4_K, FFN_DOWN, "llama", false), Existing);
    assert_eq!(plan(ID_Q4_K, FFN_DOWN, "qwen35moe", false), Existing);
    assert_eq!(
        plan(ID_Q4_K, FFN_DOWN, "qwen3_5", false),
        Packed(WeightDtype::Q4K)
    );
}

#[test]
fn preflight_sizes_count_the_packed_bytes() {
    // The 27B FFN down projection, IQ4_XS: [5120, 17408] -> 5120 rows of 68 super-blocks.
    assert_eq!(
        est_bytes_with(ID_IQ4_XS, FFN_DOWN, "qwen35", &[5120, 17408], false),
        Some(5120 * 68 * 136)
    );
    assert_eq!(
        est_bytes_with(ID_Q4_K, ATTN_Q, "qwen35", &[2560, 2560], false),
        Some(2560 * 10 * 144)
    );
    // Not whole super-block rows, a 1-D tensor, or an expanded tensor: the caller counts BF16.
    assert_eq!(
        est_bytes_with(ID_Q4_K, FFN_DOWN, "qwen35", &[64, 100], false),
        None
    );
    assert_eq!(
        est_bytes_with(ID_Q4_K, FFN_DOWN, "qwen35", &[256], false),
        None
    );
    assert_eq!(
        est_bytes_with(ID_Q5_K, FFN_DOWN, "qwen35", &[64, 256], false),
        None
    );
    assert_eq!(
        est_bytes_with(ID_Q6_K, HEAD, "qwen35", &[1024, 512], false),
        None
    );
    assert_eq!(
        est_bytes_with(ID_Q6_K, HEAD, "qwen35", &[1024, 512], true),
        Some(1024 * 2 * 210)
    );
}

fn bytes(n: usize, seed: u8) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

#[test]
fn a_packed_projection_is_uploaded_byte_for_byte() {
    let gpu = MockGpuBackend::new();
    let raw = bytes(4 * 2 * 136, 7); // [4, 512] IQ4_XS
    let t = try_load_with(
        &gpu,
        ID_IQ4_XS,
        FFN_DOWN,
        "qwen35",
        &raw,
        &[4, 512],
        None,
        false,
        false,
    )
    .unwrap()
    .expect("kept packed");
    assert_eq!(t.dtype, WeightDtype::Iq4Xs);
    assert_eq!(t.shape, vec![4, 512]);
    assert_eq!(gpu.read_alloc(t.ptr).unwrap(), raw);
    assert!(
        gpu.launches_snapshot().is_empty(),
        "no kernel on the packed path"
    );
}

#[test]
fn a_size_mismatch_is_an_error_not_a_silent_upload() {
    let gpu = MockGpuBackend::new();
    let raw = bytes(4 * 2 * 136 - 1, 1);
    assert!(
        try_load_with(
            &gpu,
            ID_IQ4_XS,
            FFN_DOWN,
            "qwen35",
            &raw,
            &[4, 512],
            None,
            false,
            false
        )
        .is_err()
    );
    // A shape that is not whole super-block rows falls through to the existing path instead.
    let raw = bytes(4 * 144, 1);
    assert!(
        try_load_with(
            &gpu,
            ID_Q4_K,
            FFN_DOWN,
            "qwen35",
            &raw,
            &[4, 300],
            None,
            false,
            false
        )
        .unwrap()
        .is_none()
    );
}

fn dims() -> value_transform::GdnDims {
    value_transform::GdnDims {
        num_k_heads: 2,
        num_v_heads: 4,
        value_head_dim: 1,
        key_head_dim: 1,
    }
}

#[test]
fn a_gdn_input_projection_is_row_reordered_while_packed() {
    // in_proj_z: 4 value heads of one row each, rows of one Q4_K super-block. HF head hf = k*2 + r sits at
    // GGUF head r*2 + k (value_transform's tiled order), so HF rows [0,1,2,3] come from GGUF rows [0,2,1,3].
    let gpu = MockGpuBackend::new();
    let row = |r: u8| vec![r; 144];
    let raw: Vec<u8> = (0..4u8).flat_map(row).collect();
    let t = try_load_with(
        &gpu,
        ID_Q4_K,
        GDN_Z,
        "qwen35",
        &raw,
        &[4, 256],
        Some(dims()),
        false,
        false,
    )
    .unwrap()
    .expect("kept packed");
    assert_eq!(t.dtype, WeightDtype::Q4K);
    let got = gpu.read_alloc(t.ptr).unwrap();
    let firsts: Vec<u8> = got.chunks_exact(144).map(|c| c[0]).collect();
    assert_eq!(firsts, vec![0, 2, 1, 3]);
    // Without the GDN geometry the reorder cannot run: the existing path takes it.
    assert!(
        try_load_with(
            &gpu,
            ID_Q4_K,
            GDN_Z,
            "qwen35",
            &raw,
            &[4, 256],
            None,
            false,
            false
        )
        .unwrap()
        .is_none()
    );
}

#[test]
fn the_gpu_expansion_runs_only_where_the_ada_module_is_built() {
    let gpu = MockGpuBackend::new();
    let raw = bytes(3 * 176, 3); // three Q5_K blocks
    let t = try_load_with(
        &gpu,
        ID_Q5_K,
        FFN_DOWN,
        "qwen35",
        &raw,
        &[3, 256],
        None,
        false,
        false,
    )
    .unwrap()
    .expect("expanded on the GPU");
    assert_eq!(t.dtype, WeightDtype::BF16);
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!((l[0].grid, l[0].block), ([3, 1, 1], [256, 1, 1]));
    assert!(gpu.kernel_lookups_snapshot().contains(&(
        DEQUANT_MODULE.to_string(),
        "dequant_q5_k_to_bf16_ada".to_string()
    )));

    // `AVAROK_GGUF_FORCE_CPU=1` keeps the CPU decoder.
    assert!(
        try_load_with(
            &gpu,
            ID_Q5_K,
            FFN_DOWN,
            "qwen35",
            &raw,
            &[3, 256],
            None,
            true,
            false
        )
        .unwrap()
        .is_none()
    );
    // A target without the module (every non-Ada target): the CPU decoder, no lookup.
    let gb10 = MockGpuBackend::new();
    gb10.mark_module_absent(DEQUANT_MODULE);
    assert!(
        try_load_with(
            &gb10,
            ID_Q5_K,
            FFN_DOWN,
            "qwen35",
            &raw,
            &[3, 256],
            None,
            false,
            false
        )
        .unwrap()
        .is_none()
    );
    assert!(gb10.kernel_lookups_snapshot().is_empty());
}

#[test]
fn dequant_entries_match_the_block_geometry() {
    assert_eq!(
        dequant_entry(ID_IQ4_XS),
        Some(("dequant_iq4_xs_to_bf16_ada", 256, 136))
    );
    assert_eq!(
        dequant_entry(ID_IQ4_NL),
        Some(("dequant_iq4_nl_to_bf16_ada", 32, 18))
    );
    assert_eq!(
        dequant_entry(ID_Q5_K),
        Some(("dequant_q5_k_to_bf16_ada", 256, 176))
    );
    assert_eq!(dequant_entry(ID_Q4_K), None);
    let gpu = MockGpuBackend::new();
    assert!(dequant_to_bf16_gpu(&gpu, ID_IQ4_XS, &[0u8; 135], 256).is_err());
    assert!(dequant_to_bf16_gpu(&gpu, ID_Q4_K, &[0u8; 144], 256).is_err());
}

#[test]
fn the_row_off_path_never_maps_mtp_names_or_loads_anything() {
    // The public hooks with the process switch at its default (off).
    let gpu = MockGpuBackend::new();
    let raw = bytes(4 * 2 * 136, 7);
    assert!(
        try_load(
            &gpu,
            ID_IQ4_XS,
            FFN_DOWN,
            "qwen35",
            &raw,
            &[4, 512],
            None,
            false
        )
        .unwrap()
        .is_none()
    );
    assert_eq!(est_bytes(ID_IQ4_XS, FFN_DOWN, "qwen35", &[4, 512]), None);
    assert!(gpu.launches_snapshot().is_empty());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}
