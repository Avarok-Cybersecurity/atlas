// SPDX-License-Identifier: AGPL-3.0-only
#![allow(clippy::needless_range_loop)] // explicit indices read clearer in these tensor-layout tests

use super::*;

/// 27B GDN geometry.
fn dims() -> GdnDims {
    GdnDims {
        num_k_heads: 16,
        num_v_heads: 48,
        value_head_dim: 128,
        key_head_dim: 128,
    }
}

#[test]
fn gguf_head_perm_matches_reference() {
    let d = dims();
    // First 6 HF indices → GGUF indices: (i%3)*16 + i/3.
    let got: Vec<usize> = (0..6).map(|i| d.gguf_head(i)).collect();
    assert_eq!(got, vec![0, 16, 32, 1, 17, 33]);
    // Full permutation: bijective over 0..48.
    let mut seen = vec![false; 48];
    for i in 0..48 {
        let g = d.gguf_head(i);
        assert!(!seen[g], "gguf_head not injective at {i}");
        seen[g] = true;
    }
    assert!(seen.into_iter().all(|x| x));
}

#[test]
fn norm_offset_subtracts_one() {
    // GGUF attn_norm first4 → HF input_layernorm first4 (byte-verified). Values
    // stay in F32 through the subtract, so precision is preserved near 1.0.
    let mut buf = vec![1.0583f32, 0.9517, 0.9385, 0.9558];
    apply(
        "model.layers.0.input_layernorm.weight",
        &mut buf,
        &[4],
        &dims(),
    )
    .unwrap();
    let want = [0.0583, -0.0483, -0.0615, -0.0442];
    for (g, w) in buf.iter().zip(want) {
        assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
    }
}

#[test]
fn final_norm_and_qk_norm_offset_but_not_ssm_norm() {
    for name in [
        "model.norm.weight",
        "model.layers.5.self_attn.q_norm.weight",
        "model.layers.5.self_attn.k_norm.weight",
        "model.layers.5.post_attention_layernorm.weight",
    ] {
        let mut buf = vec![2.0f32, 3.0];
        apply(name, &mut buf, &[2], &dims()).unwrap();
        assert_eq!(buf, vec![1.0, 2.0], "name {name}");
    }
    // The GDN ssm norm must NOT be offset.
    assert!(!needs("model.layers.5.linear_attn.norm.weight"));
    let mut buf = vec![2.0f32, 3.0];
    apply(
        "model.layers.5.linear_attn.norm.weight",
        &mut buf,
        &[2],
        &dims(),
    )
    .unwrap();
    assert_eq!(buf, vec![2.0, 3.0]);
}

#[test]
fn a_log_recovers_and_reorders() {
    // ssm_a = -exp(A_log); 48 distinct heads so the reorder is observable.
    let n = 48;
    let ssm_a: Vec<f32> = (0..n)
        .map(|h| -((-((h as f32) + 1.0) / 10.0).exp()))
        .collect();
    let mut buf = ssm_a.clone();
    apply("model.layers.0.linear_attn.A_log", &mut buf, &[n], &dims()).unwrap();
    let d = dims();
    for hf in 0..n {
        // HF head hf pulls from GGUF head src; A_log = ln(-ssm_a[src]).
        let src = d.gguf_head(hf);
        let expect = (-ssm_a[src]).ln();
        assert!(
            (buf[hf] - expect).abs() < 1e-5,
            "hf {hf}: got {}, want {expect}",
            buf[hf]
        );
    }
}

#[test]
fn reorder_rows_one_row_per_head() {
    // dt_bias analogue: 48 heads, 1 element each, value == GGUF index.
    let n = 48;
    let mut buf: Vec<f32> = (0..n).map(|i| i as f32).collect();
    apply(
        "model.layers.0.linear_attn.dt_bias",
        &mut buf,
        &[n],
        &dims(),
    )
    .unwrap();
    let d = dims();
    for hf in 0..n {
        assert_eq!(buf[hf] as usize, d.gguf_head(hf), "hf {hf}");
    }
}

#[test]
fn reorder_rows_head_dim_block() {
    // in_proj_z analogue with tiny head_dim/hidden.
    let d = GdnDims {
        num_k_heads: 2,
        num_v_heads: 4,
        value_head_dim: 3,
        key_head_dim: 2,
    };
    let hidden = 2;
    let rows = d.num_v_heads * d.value_head_dim; // 12
    // Row r encodes its head index in every column.
    let mut buf = Vec::new();
    for r in 0..rows {
        let head = r / d.value_head_dim;
        for _ in 0..hidden {
            buf.push(head as f32);
        }
    }
    apply(
        "model.layers.0.linear_attn.in_proj_z.weight",
        &mut buf,
        &[rows, hidden],
        &d,
    )
    .unwrap();
    for hf in 0..d.num_v_heads {
        for sub in 0..d.value_head_dim {
            let r = hf * d.value_head_dim + sub;
            assert_eq!(
                buf[r * hidden] as usize,
                d.gguf_head(hf),
                "hf {hf} sub {sub}"
            );
        }
    }
}

#[test]
fn reorder_qkv_only_v_rows() {
    // Q|K rows untouched; only the V region reorders.
    let d = GdnDims {
        num_k_heads: 2,
        num_v_heads: 4,
        value_head_dim: 3,
        key_head_dim: 2,
    };
    let hidden = 1;
    let qk_rows = d.qk_rows(); // 2*2*2 = 8
    let v_rows = d.num_v_heads * d.value_head_dim; // 12
    let total = qk_rows + v_rows; // 20
    let mut buf = Vec::new();
    for r in 0..qk_rows {
        buf.push(r as f32);
    }
    for r in 0..v_rows {
        buf.push((100 + r / d.value_head_dim) as f32);
    }
    apply(
        "model.layers.0.linear_attn.in_proj_qkv.weight",
        &mut buf,
        &[total, hidden],
        &d,
    )
    .unwrap();
    for r in 0..qk_rows {
        assert_eq!(buf[r] as usize, r, "qk row {r} changed");
    }
    for hf in 0..d.num_v_heads {
        let r = qk_rows + hf * d.value_head_dim;
        assert_eq!(buf[r] as usize, 100 + d.gguf_head(hf), "v head {hf}");
    }
}

#[test]
fn reorder_out_cols_per_row() {
    let d = GdnDims {
        num_k_heads: 2,
        num_v_heads: 4,
        value_head_dim: 3,
        key_head_dim: 2,
    };
    let out_rows = 2;
    let value_dim = d.num_v_heads * d.value_head_dim; // 12
    let mut buf = Vec::new();
    for _ in 0..out_rows {
        for c in 0..value_dim {
            buf.push((c / d.value_head_dim) as f32);
        }
    }
    apply(
        "model.layers.0.linear_attn.out_proj.weight",
        &mut buf,
        &[out_rows, value_dim],
        &d,
    )
    .unwrap();
    for row in 0..out_rows {
        for hf in 0..d.num_v_heads {
            let c = row * value_dim + hf * d.value_head_dim;
            assert_eq!(buf[c] as usize, d.gguf_head(hf), "row {row} head {hf}");
        }
    }
}

#[test]
fn untouched_names_are_noops() {
    assert!(!needs("model.layers.0.self_attn.q_proj.weight"));
    assert!(!needs("model.layers.0.mlp.down_proj.weight"));
    assert!(!needs("model.embed_tokens.weight"));
    let mut buf = vec![1.0f32, 2.0, 3.0];
    apply(
        "model.layers.0.self_attn.q_proj.weight",
        &mut buf,
        &[3],
        &dims(),
    )
    .unwrap();
    assert_eq!(buf, vec![1.0, 2.0, 3.0]);
}

#[test]
fn to_bf16_bytes_round_trips() {
    let b = to_bf16_bytes(&[1.0, -2.0, 0.0]);
    assert_eq!(b.len(), 6);
    // 1.0 = 0x3F80, -2.0 = 0xC000, 0.0 = 0x0000 (LE).
    assert_eq!(&b[0..2], &[0x80, 0x3F]);
    assert_eq!(&b[2..4], &[0x00, 0xC0]);
    assert_eq!(&b[4..6], &[0x00, 0x00]);
}

#[test]
fn gdn_dims_geometry() {
    let d = dims();
    assert_eq!(d.num_v_per_k(), 3);
    assert_eq!(d.qk_rows(), 2 * 128 * 16);
}

#[test]
fn clip_arch_gate() {
    assert!(super::is_clip("clip"));
    assert!(!super::is_clip("qwen35"));
}

#[test]
fn vision_patch_frame_parsing() {
    use super::vision_patch_frame as f;
    assert_eq!(f("v.patch_embd.weight"), Some(0));
    assert_eq!(f("v.patch_embd.weight.1"), Some(1));
    assert_eq!(f("v.patch_embd.weight.2"), Some(2));
    assert_eq!(f("v.patch_embd.bias"), None);
    assert_eq!(f("v.post_ln.weight"), None);
    assert_eq!(f("v.blk.0.attn_qkv.weight"), None);
}

#[test]
fn patch_embed_concat_interleaves_temporal_within_channel() {
    // out_ch=2, in_ch=1, patch=2 → blk=4, frame_row=4, T=2, dst_row=8.
    let dims = super::VisionPatchDims {
        out_ch: 2,
        in_ch: 1,
        patch: 2,
    };
    let f0: Vec<f32> = (0..8).map(|x| x as f32).collect(); // o0:0..4  o1:4..8
    let f1: Vec<f32> = (10..18).map(|x| x as f32).collect(); // o0:10..14 o1:14..18
    let out = super::patch_embed_concat(&[&f0, &f1], &dims).unwrap();
    // row o: [t0 block (4)] then [t1 block (4)]  (single channel)
    assert_eq!(
        out,
        vec![
            0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, // o0
            4.0, 5.0, 6.0, 7.0, 14.0, 15.0, 16.0, 17.0, // o1
        ]
    );
}

#[test]
fn patch_embed_concat_channel_major_then_temporal() {
    // out_ch=1, in_ch=2, patch=1 → blk=1, K = c*T + t.
    let dims = super::VisionPatchDims {
        out_ch: 1,
        in_ch: 2,
        patch: 1,
    };
    let f0 = vec![10.0f32, 11.0]; // c0,c1 of frame0
    let f1 = vec![20.0f32, 21.0]; // c0,c1 of frame1
    let out = super::patch_embed_concat(&[&f0, &f1], &dims).unwrap();
    // K0=c0t0, K1=c0t1, K2=c1t0, K3=c1t1
    assert_eq!(out, vec![10.0, 20.0, 11.0, 21.0]);
}

#[test]
fn patch_embed_concat_rejects_wrong_frame_len() {
    let dims = super::VisionPatchDims {
        out_ch: 2,
        in_ch: 1,
        patch: 2,
    };
    let good = vec![0.0f32; 8];
    let bad = vec![0.0f32; 7];
    assert!(super::patch_embed_concat(&[&good, &bad], &dims).is_err());
}

#[test]
fn packed_reorder_rows_moves_whole_blocks_like_reorder_rows() {
    // Small GDN geometry: 2 k-heads, 6 v-heads (num_v_per_k=3), hd=4 rows/head.
    // K=8, group=4 → 2 blocks/row; block_bytes=6 → row_bytes=12.
    let d = GdnDims {
        num_k_heads: 2,
        num_v_heads: 6,
        value_head_dim: 4,
        key_head_dim: 4,
    };
    let (group, block_bytes) = (4usize, 6usize);
    let k = 8usize;
    let qk_rows = d.qk_rows(); // 2*4*2 = 16
    let v_rows = d.num_v_heads * d.value_head_dim; // 24
    let n = qk_rows + v_rows; // 40
    let row_bytes = (k / group) * block_bytes; // 12
    // Tag every byte of row r with value r (mod 256) so we can trace movement.
    let raw: Vec<u8> = (0..n)
        .flat_map(|r| std::iter::repeat_n(r as u8, row_bytes))
        .collect();
    let shape = [n, k];
    let out = super::reorder_packed_rows(&raw, &shape, &d, true, group, block_bytes).unwrap();

    // Q|K region (rows 0..qk_rows) is untouched.
    for r in 0..qk_rows {
        for b in 0..row_bytes {
            assert_eq!(out[r * row_bytes + b], r as u8, "Q|K row {r} moved");
        }
    }
    // V region: HF value-head hf gathers GGUF head gguf_head(hf); within a head
    // the 4 rows keep order. Reference: expected source row for dest row.
    for hf in 0..d.num_v_heads {
        let g = d.gguf_head(hf);
        for j in 0..d.value_head_dim {
            let dst = qk_rows + hf * d.value_head_dim + j;
            let src = qk_rows + g * d.value_head_dim + j;
            for b in 0..row_bytes {
                assert_eq!(
                    out[dst * row_bytes + b],
                    src as u8,
                    "V dest row {dst} should come from src {src}"
                );
            }
        }
    }
}

#[test]
fn packed_reorder_rows_classifier() {
    assert_eq!(
        super::packed_reorder_rows("model.layers.0.linear_attn.in_proj_qkv.weight"),
        Some(true)
    );
    assert_eq!(
        super::packed_reorder_rows("model.layers.0.linear_attn.in_proj_z.weight"),
        Some(false)
    );
    // Single-row reorders and column reorders are NOT packed-reorderable.
    assert_eq!(
        super::packed_reorder_rows("model.layers.0.linear_attn.in_proj_a.weight"),
        None
    );
    assert_eq!(
        super::packed_reorder_rows("model.layers.0.linear_attn.out_proj.weight"),
        None
    );
    assert_eq!(
        super::packed_reorder_rows("model.layers.0.self_attn.q_proj.weight"),
        None
    );
}
