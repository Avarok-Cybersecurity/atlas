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

    assert!(!arena.hidden_states().is_null());
    assert!(!arena.logits().is_null());
    assert_eq!(arena.max_batch_tokens(), 128);
    // 27 allocations: main's 18 (12 data + 1 scratch + 3 expert + 2 splitk)
    // plus 9 added by the V4 foundation atop main:
    //   - 2 FP32-routing buffers (gate_logits_f32 + moe_router_in_f32),
    //   - 1 gdn_fla_scratch (allocated here: qwen3_next_80b has 128-dim linear
    //     heads, so sizes.gdn_fla_scratch > 0),
    //   - 2 V4-MLA buffers (o_latent + norm_unit_w, present non-zero for all
    //     configs via the .max(256) floor),
    //   - 3 HC buffers (hc_streams/hc_post/hc_comb, placeholder-sized 256 when
    //     hc_mult == 0 but still allocated unconditionally),
    //   - 1 token_ids buffer (hash-routing scratch, .max(256) floor so it is
    //     allocated unconditionally even for models without hash routing).
    // plus 2 added by the Holo-3.1/Ornith GB10 enablement (buffers.rs):
    //   - fp8_act + fp8_act_scale (persistent FP8 prefill-projection scratch,
    //     allocated unconditionally), plus the compact MoE work-list, counter,
    //     and overflow flag. 27 + 2 + 3 = 32.
    assert_eq!(gpu.alloc_count(), 32);
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
fn q2_dequant_scratch_zero_without_flag() {
    // Flag off (default): from_config must NOT size the buffer, so non-Q2
    // models allocate nothing extra (BufferArena skips the alloc on 0 → NULL).
    if std::env::var("ATLAS_GGUF_NATIVE_Q2").ok().as_deref() == Some("1") {
        return; // flag on in this environment — the sized path is covered above
    }
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let sizes = BufferSizes::from_config(&cfg, 1, 4096, 16, 32);
    assert_eq!(sizes.q2_dequant_scratch, 0);
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
