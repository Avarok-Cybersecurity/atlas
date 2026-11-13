// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend dispatch tests of the single-sequence decode arms of the packed
//! (W4) attention projections (port lane L22b): k/v in `attention_forward_kv.rs` and o in
//! `attention_forward_oproj.rs` call the installed weights with one row at the production
//! pointers, and a layer without packed weights takes its BF16 GEMVs exactly as before.

use spark_runtime::gpu::mock::MockArg;

use crate::layers::qwen3_attention::w4_proj::fixture::{Fakes, H, HD, NKV, NQ, Rig, kv_dim};
use crate::port::w4::fake::Call;

#[test]
fn packed_kv_decode_is_one_row_per_projection_into_the_k_and_v_regions() {
    let rig = Rig::new(true);
    let mut layer = rig.bf16_layer(true);
    let fakes = Fakes::new(true, 16, false);
    fakes.install(&mut layer);
    layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
    let ctx = rig.ctx();
    let normed = rig.buffers.norm_output();
    let k_out = rig.buffers.qkv_output().offset(4096);
    let v_out = k_out.offset(kv_dim() * 2);
    let launches = rig.gpu.launch_count();
    let (nkv, hd, h) = (NKV as u32, HD as u32, H as u32);
    layer
        .attention_forward_kv(normed, k_out, v_out, nkv, hd, h, &ctx, 0)
        .unwrap();
    assert_eq!(
        fakes.calls(),
        vec![
            (
                "k",
                Call::Decode {
                    x: normed,
                    rows: 1,
                    out: k_out
                }
            ),
            (
                "v",
                Call::Decode {
                    x: normed,
                    rows: 1,
                    out: v_out
                }
            ),
        ]
    );
    assert_eq!(rig.gpu.launch_count(), launches, "no BF16 GEMV ran");
}

#[test]
fn packed_o_decode_writes_the_norm_output_buffer() {
    let rig = Rig::new(true);
    let mut layer = rig.bf16_layer(true);
    let fakes = Fakes::new(true, 16, false);
    fakes.install(&mut layer);
    layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
    let ctx = rig.ctx();
    let attn_out = rig.buffers.attn_output();
    let launches = rig.gpu.launch_count();
    let (nq, hd, h) = (NQ as u32, HD as u32, H as u32);
    let o_out = layer
        .attention_forward_oproj(attn_out, nq, hd, h, &ctx, 0)
        .unwrap();
    assert_eq!(o_out, rig.buffers.norm_output());
    assert_eq!(
        fakes.calls(),
        vec![(
            "o",
            Call::Decode {
                x: attn_out,
                rows: 1,
                out: o_out
            }
        )]
    );
    assert_eq!(rig.gpu.launch_count(), launches);
}

/// Row off: no packed weights, so the k/v and o decode arms launch their BF16 GEMVs on the BF16
/// weights, as on the base commit.
#[test]
fn without_packed_weights_the_bf16_gemvs_run_unchanged() {
    let rig = Rig::new(true);
    let layer = rig.bf16_layer(true);
    assert!(!layer.w4_proj_installed());
    let ctx = rig.ctx();
    let (k_w, v_w) = (layer.attn.k_proj.weight, layer.attn.v_proj.weight);
    let o_w = layer.o_dense_bf16.as_ref().unwrap().weight;
    let first = rig.gpu.launch_count();
    let normed = rig.buffers.norm_output();
    let k_out = rig.buffers.qkv_output().offset(4096);
    let v_out = k_out.offset(kv_dim() * 2);
    let (nq, nkv, hd, h) = (NQ as u32, NKV as u32, HD as u32, H as u32);
    layer
        .attention_forward_kv(normed, k_out, v_out, nkv, hd, h, &ctx, 0)
        .unwrap();
    layer
        .attention_forward_oproj(rig.buffers.attn_output(), nq, hd, h, &ctx, 0)
        .unwrap();
    let launches = rig.gpu.launches_snapshot()[first..].to_vec();
    for w in [k_w, v_w, o_w] {
        assert_eq!(
            launches
                .iter()
                .filter(|l| l.args.contains(&MockArg::Buffer(w)))
                .count(),
            1,
            "one BF16 GEMV over each dense weight"
        );
    }
}
