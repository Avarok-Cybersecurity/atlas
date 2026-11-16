// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend dispatch tests of the prefill arms of the packed (W4) attention
//! projections (port lane L22b): the chunk-0 q/k/v chain (`cache_skip_qkv.rs`), the later-chunk
//! q/k/v chain (`paged_qkv.rs`) and the O projection (`paged_oproj.rs`) hand the installed
//! weights the production input and output regions, prefill above the decode window and decode
//! inside it; a layer without packed weights takes its BF16 GEMMs as before.

use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockArg;

use crate::layers::qwen3_attention::w4_proj::fixture::{
    Fakes, H, HD, NKV, NQ, Rig, kv_dim, q_proj_dim,
};
use crate::port::w4::fake::Call;

/// The smoke prompt length of `alloc_tests.rs`: above a 16-row decode window.
const M: usize = 28;

/// The calls one q/k/v chain must make for `m` tokens: q into `qkv_output`, k and v back to back
/// in `ssm_qkvz`, each a prefill above the window and a decode inside it.
fn qkv_calls(rig: &Rig, m: usize, window: usize) -> Vec<(&'static str, Call)> {
    let x = rig.buffers.norm_output();
    let q_out = rig.buffers.qkv_output();
    let k_out = rig.buffers.ssm_qkvz();
    let v_out = k_out.offset(m * kv_dim() * 2);
    let call = |out: DevicePtr| {
        if m > window {
            Call::Prefill { x, m, out }
        } else {
            Call::Decode { x, rows: m, out }
        }
    };
    vec![("q", call(q_out)), ("k", call(k_out)), ("v", call(v_out))]
}

fn installed(
    rig: &Rig,
    gated: bool,
    window: usize,
) -> (crate::layers::qwen3_attention::Qwen3AttentionLayer, Fakes) {
    let mut layer = rig.bf16_layer(gated);
    let fakes = Fakes::new(gated, window, false);
    assert!(fakes.install(&mut layer) > 0);
    layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
    (layer, fakes)
}

#[test]
fn chunk_zero_qkv_takes_the_packed_weights() {
    for (m, gated) in [(M, true), (8, true), (M, false)] {
        let rig = Rig::new(gated);
        let (layer, fakes) = installed(&rig, gated, 16);
        let ctx = rig.ctx();
        let launches = rig.gpu.launch_count();
        layer
            .prefill_attention_cache_skip_qkv(
                rig.buffers.norm_output(),
                DevicePtr::NULL,
                m as u32,
                H as u32,
                NKV as u32,
                HD as u32,
                q_proj_dim(gated),
                kv_dim(),
                m,
                2,
                &ctx,
                0,
            )
            .unwrap();
        assert_eq!(fakes.calls(), qkv_calls(&rig, m, 16), "m={m} gated={gated}");
        assert_eq!(
            rig.gpu.launch_count(),
            launches,
            "no GEMM ran beside the packed weights"
        );
    }
}

#[test]
fn later_chunk_qkv_takes_the_packed_weights() {
    for m in [M, 16] {
        let rig = Rig::new(true);
        let (layer, fakes) = installed(&rig, true, 16);
        let ctx = rig.ctx();
        layer
            .prefill_attention_paged_qkv(
                rig.buffers.norm_output(),
                m as u32,
                H as u32,
                NKV as u32,
                HD as u32,
                q_proj_dim(true),
                kv_dim(),
                m,
                2,
                &ctx,
                0,
            )
            .unwrap();
        assert_eq!(fakes.calls(), qkv_calls(&rig, m, 16), "m={m}");
    }
}

#[test]
fn the_o_projection_prefill_takes_the_packed_weight() {
    let rig = Rig::new(true);
    let (layer, fakes) = installed(&rig, true, 16);
    let ctx = rig.ctx();
    let attn_out = rig.buffers.attn_output();
    let launches = rig.gpu.launch_count();
    let (nq, hd, h) = (NQ as u32, HD as u32, H as u32);
    let o_out = layer
        .prefill_attention_paged_oproj(attn_out, M as u32, h, nq, hd, &ctx, 0)
        .unwrap();
    assert_eq!(o_out, rig.buffers.norm_output());
    assert_eq!(
        fakes.calls(),
        vec![(
            "o",
            Call::Prefill {
                x: attn_out,
                m: M,
                out: o_out
            }
        )]
    );
    assert_eq!(rig.gpu.launch_count(), launches);
}

/// Row off: with no packed weights every chain runs its BF16 GEMM over the BF16 weights, once per
/// projection, as on the base commit.
#[test]
fn without_packed_weights_the_bf16_prefill_gemms_run_unchanged() {
    let rig = Rig::new(true);
    let layer = rig.bf16_layer(true);
    let ctx = rig.ctx();
    let dense = [
        layer.attn.q_proj.weight,
        layer.attn.k_proj.weight,
        layer.attn.v_proj.weight,
        layer.o_dense_bf16.as_ref().unwrap().weight,
    ];
    let first = rig.gpu.launch_count();
    let (nq, nkv, hd, h) = (NQ as u32, NKV as u32, HD as u32, H as u32);
    layer
        .prefill_attention_cache_skip_qkv(
            rig.buffers.norm_output(),
            DevicePtr::NULL,
            M as u32,
            h,
            nkv,
            hd,
            q_proj_dim(true),
            kv_dim(),
            M,
            2,
            &ctx,
            0,
        )
        .unwrap();
    layer
        .prefill_attention_paged_oproj(rig.buffers.attn_output(), M as u32, h, nq, hd, &ctx, 0)
        .unwrap();
    let launches = rig.gpu.launches_snapshot()[first..].to_vec();
    for w in dense {
        let n = launches
            .iter()
            .filter(|l| l.args.contains(&MockArg::Buffer(w)))
            .count();
        assert_eq!(n, 1, "one BF16 GEMM over each dense weight");
    }
}
