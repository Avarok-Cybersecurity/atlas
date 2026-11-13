// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend dispatch tests of the batched-decode arms of the packed (W4)
//! attention projections (port lane L22b): phase 2 (`qkv.rs`) decodes q/k/v into the staging
//! region and scatters them with three pitched copies, phase 6 (`o_proj.rs`) decodes o in windows
//! straight into `moe_output`; a layer without packed weights keeps its BF16 tiers.

use spark_runtime::gpu::mock::MockArg;

use super::super::super::ctx::MultiSeqCtx;
use crate::layers::qwen3_attention::w4_proj::STAGE_ROWS;
use crate::layers::qwen3_attention::w4_proj::fixture::{Fakes, H, Rig, kv_dim, q_dim, q_proj_dim};
use crate::port::w4::fake::Call;

#[test]
fn batched_qkv_decodes_into_the_stage_and_scatters_once_per_projection() {
    let rig = Rig::new(false);
    let mut layer = rig.bf16_layer(false);
    let fakes = Fakes::new(false, 16, false);
    fakes.install(&mut layer);
    layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
    let stage = layer.w4_ready(&rig.gpu, 0).unwrap().stage;
    let fwd = rig.ctx();
    let n = 5;
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        rig.buffers.hidden_states(),
        rig.buffers.residual(),
        n,
        16,
        0,
    );
    let copies = rig.gpu.d2d_2d_count();
    layer.ms_phase_qkv(&c).unwrap();
    let x = rig.buffers.norm_output();
    let q_bytes = q_proj_dim(false) * 2;
    let sk = stage.offset(STAGE_ROWS * q_bytes);
    let sv = sk.offset(STAGE_ROWS * kv_dim() * 2);
    assert_eq!(
        fakes.calls(),
        vec![
            (
                "q",
                Call::Decode {
                    x,
                    rows: n,
                    out: stage
                }
            ),
            (
                "k",
                Call::Decode {
                    x,
                    rows: n,
                    out: sk
                }
            ),
            (
                "v",
                Call::Decode {
                    x,
                    rows: n,
                    out: sv
                }
            ),
        ]
    );
    assert_eq!(
        rig.gpu.d2d_2d_count() - copies,
        3,
        "one pitched scatter per projection"
    );
}

#[test]
fn batched_o_decodes_in_windows_into_moe_output() {
    let rig = Rig::new(true);
    let mut layer = rig.bf16_layer(true);
    let fakes = Fakes::new(true, 16, false);
    fakes.install(&mut layer);
    layer.prepare_w4_proj(&rig.gpu, 0).unwrap();
    let fwd = rig.ctx();
    let n = 20;
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        rig.buffers.hidden_states(),
        rig.buffers.residual(),
        n,
        16,
        0,
    );
    let attn_out = rig.buffers.attn_output();
    let out = layer.ms_phase_o_proj(&c, attn_out).unwrap();
    assert_eq!(out, rig.buffers.moe_output());
    let row = |r: usize| (attn_out.offset(r * q_dim() * 2), out.offset(r * H * 2));
    let ((x0, o0), (x1, o1)) = (row(0), row(16));
    assert_eq!(
        fakes.calls(),
        vec![
            (
                "o",
                Call::Decode {
                    x: x0,
                    rows: 16,
                    out: o0
                }
            ),
            (
                "o",
                Call::Decode {
                    x: x1,
                    rows: 4,
                    out: o1
                }
            ),
        ]
    );
}

/// Row off: with no packed weights the batched tiers read the BF16 weights as before.
#[test]
fn without_packed_weights_the_batched_bf16_tiers_run_unchanged() {
    let rig = Rig::new(false);
    let layer = rig.bf16_layer(false);
    let fwd = rig.ctx();
    let n = 4;
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        rig.buffers.hidden_states(),
        rig.buffers.residual(),
        n,
        16,
        0,
    );
    let first = rig.gpu.launch_count();
    layer.ms_phase_qkv(&c).unwrap();
    layer
        .ms_phase_o_proj(&c, rig.buffers.attn_output())
        .unwrap();
    let launches = rig.gpu.launches_snapshot()[first..].to_vec();
    let reads = |w| {
        launches
            .iter()
            .filter(|l| l.args.contains(&MockArg::Buffer(w)))
            .count()
    };
    for w in [
        layer.attn.q_proj.weight,
        layer.attn.k_proj.weight,
        layer.attn.v_proj.weight,
        layer.o_dense_bf16.as_ref().unwrap().weight,
    ] {
        assert!(reads(w) >= 1, "a BF16 tier read every dense weight");
    }
}
