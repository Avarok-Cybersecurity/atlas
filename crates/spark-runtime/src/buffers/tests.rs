// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn mixed_dense_moe_sizes_for_widest_ffn() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.intermediate_size = 12_288;
    cfg.num_experts = 256;
    cfg.num_experts_per_tok = 10;
    cfg.moe_intermediate_size = 1_024;

    let sizes = BufferSizes::from_config(&cfg, 4, 4096, 16, 32);
    assert_eq!(sizes.expert_gate_out, 4 * 12_288 * 2);
    assert_eq!(sizes.expert_up_out, 4 * 12_288 * 2);
}
use crate::gpu::mock::MockGpuBackend;
use std::collections::HashSet;

#[test]
fn test_buffer_sizes_qwen3() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    // max_batch_size=32: the decode-meta rows floor — legacy byte-identical sizing.
    let sizes = BufferSizes::from_config(&cfg, 1, 4096, 16, 32);

    // hidden_states: 1 * 2048 * 2 = 4096 (BF16, 2 bytes/elem).
    // (Was FP32 = 8192 in earlier prototypes; NVFP4 path keeps the
    // residual stream in BF16, halving the buffer size.)
    assert_eq!(sizes.hidden_states, 4096);
    // qkv: 1 * (16*2 + 2*2) * 256 * 2 = 1 * 36 * 256 * 2 = 18432
    // Q+gate: 16*2*256, K: 2*256, V: 2*256
    assert_eq!(sizes.qkv_output, 18432);
    // attn: 1 * 16 * 256 * 2 = 8192
    assert_eq!(sizes.attn_output, 8192);
    // gate: 1 * 512 * 2 = 1024
    assert_eq!(sizes.gate_logits, 1024);
    // logits: 1 * 151936 * 2 = 303872
    assert_eq!(sizes.logits, 303872);
    // ssm_qkvz: 1 * 12288 * 2 = 24576
    // Q(16*128) + K(16*128) + V(32*128) + Z(32*128) = 12288
    assert_eq!(sizes.ssm_qkvz, 24576);
    // ssm_ba: max(1 * 64 * 2, 256) = 256 (minimum allocation)
    assert_eq!(sizes.ssm_ba, 256);
    // ssm_deinterleaved: same as ssm_qkvz = 24576
    assert_eq!(sizes.ssm_deinterleaved, 24576);
    // ssm_gates: 1 * 32 * 2 * 4 = 256 (FP32 gate + beta, scaled by M)
    assert_eq!(sizes.ssm_gates, 256);
}

#[test]
fn test_buffer_arena_alloc() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    // max_batch_size=32: the decode-meta rows floor — legacy byte-identical sizing.
    let arena = BufferArena::new(&cfg, 128, 4096, 16, 32, &gpu).unwrap();

    assert_eq!(arena.max_batch_tokens(), 128);
    let sizes = arena.sizes();
    let buffers = [
        ("hidden_states", arena.hidden_states(), sizes.hidden_states),
        ("residual", arena.residual(), sizes.residual),
        ("norm_output", arena.norm_output(), sizes.norm_output),
        ("qkv_output", arena.qkv_output(), sizes.qkv_output),
        ("attn_output", arena.attn_output(), sizes.attn_output),
        ("gate_logits", arena.gate_logits(), sizes.gate_logits),
        (
            "gate_logits_f32",
            arena.gate_logits_f32(),
            sizes.gate_logits_f32,
        ),
        (
            "moe_router_in_f32",
            arena.moe_router_in_f32(),
            sizes.moe_router_in_f32,
        ),
        ("moe_output", arena.moe_output(), sizes.moe_output),
        ("logits", arena.logits(), sizes.logits),
        ("ssm_qkvz", arena.ssm_qkvz(), sizes.ssm_qkvz),
        ("ssm_ba", arena.ssm_ba(), sizes.ssm_ba),
        (
            "ssm_deinterleaved",
            arena.ssm_deinterleaved(),
            sizes.ssm_deinterleaved,
        ),
        ("ssm_gates", arena.ssm_gates(), sizes.ssm_gates),
        (
            "ssm_conv_out_f32",
            arena.ssm_conv_out_f32(),
            sizes.ssm_conv_out_f32,
        ),
        ("scratch", arena.scratch(), sizes.scratch),
        (
            "expert_gate_out",
            arena.expert_gate_out(),
            sizes.expert_gate_out,
        ),
        ("expert_up_out", arena.expert_up_out(), sizes.expert_up_out),
        (
            "expert_down_out",
            arena.expert_down_out(),
            sizes.expert_down_out,
        ),
        (
            "splitk_workspace",
            arena.splitk_workspace(),
            sizes.splitk_workspace,
        ),
        ("o_latent", arena.o_latent(), sizes.o_latent),
        ("norm_unit_w", arena.norm_unit_w(), sizes.norm_unit_w),
        ("hc_streams", arena.hc_streams(), sizes.hc_streams),
        ("hc_post", arena.hc_post(), sizes.hc_post),
        ("hc_comb", arena.hc_comb(), sizes.hc_comb),
        ("ssd_scratch", arena.ssd_scratch(), sizes.ssd_scratch),
        (
            "gdn_fla_scratch",
            arena.gdn_fla_scratch(),
            sizes.gdn_fla_scratch,
        ),
        ("token_ids", arena.token_ids(), sizes.token_ids),
        ("ffn_act_q8", arena.ffn_act_q8(), sizes.ffn_act_q8),
        ("ffn_act_a", arena.ffn_act_a(), sizes.ffn_act_a),
        ("ffn_act_scale", arena.ffn_act_scale(), sizes.ffn_act_scale),
        ("fp8_act", arena.fp8_act(), sizes.fp8_act),
        ("fp8_act_scale", arena.fp8_act_scale(), sizes.fp8_act_scale),
        (
            "q2_dequant_scratch",
            arena.q2_dequant_scratch(),
            sizes.q2_dequant_scratch,
        ),
        ("q2_act_q8", arena.q2_act_q8(), sizes.q2_act_q8),
        ("lora_xa", arena.lora_xa(), sizes.lora_xa),
        ("lora_delta", arena.lora_delta(), sizes.lora_delta),
        ("lora_hact", arena.lora_hact(), sizes.lora_hact),
        ("lora_seq_slot", arena.lora_seq_slot(), sizes.lora_seq_slot),
    ];
    let mut allocated = HashSet::new();
    for (name, ptr, bytes) in buffers {
        if bytes == 0 {
            assert!(ptr.is_null(), "{name} must be null when disabled");
        } else {
            assert!(!ptr.is_null(), "{name} must be allocated");
            assert!(
                allocated.insert(ptr.0),
                "{name} aliases another arena buffer"
            );
            assert_eq!(gpu.read_alloc(ptr).unwrap().len(), bytes, "{name} size");
        }
    }
    assert_eq!(gpu.alloc_count(), allocated.len());
    assert!(
        gpu.read_alloc(arena.norm_unit_w())
            .unwrap()
            .iter()
            .all(|byte| *byte == 0),
        "unit RMSNorm weight must be zero-initialized"
    );
}

#[test]
fn q2_dequant_scratch_covers_largest_projection() {
    // The native keep-packed Q2_0 prefill reuses ONE BF16 dequant scratch for
    // every projection, so it must be sized to the widest `[N,K]` — otherwise a
    // later, larger dequant overruns the buffer. Every keep-packed projection
    // has one dim == hidden_size, so the bound is `max_other_dim * hidden * 2`.
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let bytes = q2_dequant_scratch_bytes(&cfg);
    let h = cfg.hidden_size;
    let ffn = cfg.intermediate_size * h * 2; // gate/up [inter,h] & down [h,inter]
    let qkvz = cfg.ssm_qkvz_size() * h * 2; // fused GDN in_proj_qkvz [qkvz,h]
    let q_mul = if cfg.attn_gated { 2 } else { 1 };
    let q = cfg.num_attention_heads * q_mul * cfg.head_dim * h * 2; // attn q_proj
    let kv = cfg.num_key_value_heads * cfg.head_dim * h * 2; // attn k/v_proj
    assert!(bytes >= ffn, "scratch {bytes} < FFN {ffn}");
    assert!(bytes >= qkvz, "scratch {bytes} < qkvz {qkvz}");
    assert!(bytes >= q, "scratch {bytes} < q_proj {q}");
    assert!(bytes >= kv, "scratch {bytes} < kv_proj {kv}");
    assert!(bytes > 0);
}

#[test]
fn q2_scratch_flags_are_explicit_partitions() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let m = 3;
    let h = cfg.hidden_size;
    let hd = cfg.head_dim;
    let dequant_bytes = q2_dequant_scratch_bytes(&cfg);
    let kmax = h
        .max(cfg.intermediate_size)
        .max(cfg.num_attention_heads * hd);
    let mmq_bytes = m * kmax.div_ceil(256) * 256 * 4 + (1 << 20);

    assert_eq!(
        sizes_q2::q2_scratch_sizes_for(&cfg, m, h, hd, false, false),
        (0, 0)
    );
    assert_eq!(
        sizes_q2::q2_scratch_sizes_for(&cfg, m, h, hd, true, false),
        (dequant_bytes, 0)
    );
    assert_eq!(
        sizes_q2::q2_scratch_sizes_for(&cfg, m, h, hd, false, true),
        (0, mmq_bytes)
    );
}

#[test]
fn test_buffer_sizes_scale_with_batch() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    // max_batch_size=32: the decode-meta rows floor — legacy byte-identical sizing.
    let s1 = BufferSizes::from_config(&cfg, 1, 4096, 16, 32);
    let s128 = BufferSizes::from_config(&cfg, 128, 4096, 16, 32);
    assert_eq!(s128.hidden_states, s1.hidden_states * 128);
    // logits does NOT scale with batch: BF16 rows (2 bytes/elem) capped at
    // 96 tokens — the batched-verify row cap (n=32 × k=3 rows, the wave-11
    // depth-at-width envelope, VERIFY_ROW_CAP; sizes.rs `logits_tokens`).
    // This assert was stale twice (16-row FP32 era, then unnoticed through
    // the 33-row bump) — it is the byte twin of the sizes.rs formula, so
    // update BOTH together.
    assert_eq!(s128.logits, 96 * cfg.vocab_size * 2);
}

/// bs=64 native boots (wave-14a): sizing must be BYTE-IDENTICAL to bs=32 —
/// the widened decode-meta layout (rows=64) still sits strictly inside the
/// 96-row verify scratch overlay (bt at 24*64=1536 < 2048, 64 < 96 rows)
/// and the 65-row logits need is under the 96-row cap. Only above those
/// bounds may sizes grow — asserted for the 128-row ceiling.
#[test]
fn test_buffer_sizes_decode_meta_widening() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let s32 = BufferSizes::from_config(&cfg, 8192, 4096, 16, 32);
    // bs 1..=32: rows floor 32 — identical sizing in every field.
    for bs in [1usize, 31, 32] {
        let s = BufferSizes::from_config(&cfg, 8192, 4096, 16, bs);
        assert_eq!(s.total_bytes(), s32.total_bytes(), "bs={bs}");
        assert_eq!(s.scratch, s32.scratch, "bs={bs}");
        assert_eq!(s.logits, s32.logits, "bs={bs}");
    }
    // bs 33..=64: layout widens but stays inside the verify envelope.
    for bs in [33usize, 64] {
        let s = BufferSizes::from_config(&cfg, 8192, 4096, 16, bs);
        assert_eq!(s.total_bytes(), s32.total_bytes(), "bs={bs}");
    }
    // bs=128 (the derived ceiling): logits go to 129 rows and the scratch
    // envelope must cover the decode layout term (24R + R*max_blocks*4).
    let s128 = BufferSizes::from_config(&cfg, 8192, 4096, 16, 128);
    assert_eq!(s128.logits, 129 * cfg.vocab_size * 2);
    let max_blocks = 4096 / 16 + 1;
    assert!(s128.scratch >= 32768 + 24 * 128 + 128 * max_blocks * 4);
}
