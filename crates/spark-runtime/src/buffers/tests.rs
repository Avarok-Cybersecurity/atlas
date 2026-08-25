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
    //     allocated unconditionally). 27 + 2 = 29.
    // plus 1 added by the keep-packed GGUF grouped MoE (06c89a33):
    //   - moe_grouped_q8 (q8_1 activation scratch in the arena so CUDA-graph
    //     replay sees a stable address; sized > 0 for MoE configs, and this
    //     config is MoE). 29 + 1 = 30.
    assert_eq!(gpu.alloc_count(), 30);
}

/// The keep-packed grouped-MoE q8_1 scratch is sized only for MoE configs; a
/// dense model must get 0 so `BufferArena::new` skips the alloc — cuMemAlloc(0)
/// is INVALID_VALUE and failed every dense-model boot when this was
/// unconditional (caught by the integration GPU smoke on the dense 27B).
#[test]
fn moe_grouped_q8_zero_for_dense_nonzero_for_moe() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(cfg.num_experts > 0);
    let moe_sizes = BufferSizes::from_config(&cfg, 4, 4096, 16, 32);
    assert!(moe_sizes.moe_grouped_q8 > 0);

    cfg.num_experts = 0;
    let dense_sizes = BufferSizes::from_config(&cfg, 4, 4096, 16, 32);
    assert_eq!(dense_sizes.moe_grouped_q8, 0);
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
    // logits does NOT scale freely with batch: BF16 rows (2 bytes/elem)
    // bounded by `m.min(160.max(rows+1))` — the batched-verify row cap
    // (VERIFY_ROW_CAP = 160; sizes.rs `logits_tokens`). At m=128 the
    // m-bound wins: 128 rows, still under the 160 cap. This assert was
    // stale three times (16-row FP32 era, the 33-row bump, then the 96
    // cap) — it is the byte twin of the sizes.rs formula, so update BOTH
    // together.
    assert_eq!(s128.logits, 128 * cfg.vocab_size * 2);
}

/// Native wide boots: sizing must be BYTE-IDENTICAL to bs=32 for every
/// batch whose decode-meta layout sits inside the 160-row verify scratch
/// overlay (VERIFY_ROW_CAP = 160; bt at 24*160 and the 160-row logits cap
/// both dominate until rows exceed 160). Only above that bound may sizes
/// grow — asserted at bs=160 (logits leave the cap) and bs=192 (the
/// decode layout term overtakes the bt overlay in scratch).
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
    // bs 33..=159: layout widens but stays inside the 160-row envelope —
    // logits stay at the 160-row cap and scratch stays bt-dominated.
    for bs in [33usize, 64, 128, 159] {
        let s = BufferSizes::from_config(&cfg, 8192, 4096, 16, bs);
        assert_eq!(s.total_bytes(), s32.total_bytes(), "bs={bs}");
    }
    // bs=160: the derived floor (rows+1 = 161) finally exceeds the cap.
    let s160 = BufferSizes::from_config(&cfg, 8192, 4096, 16, 160);
    assert_eq!(s160.logits, 161 * cfg.vocab_size * 2);
    // bs=192: the decode layout term (24R + R*max_blocks*4) overtakes the
    // bt overlay and scratch must cover it.
    let s192 = BufferSizes::from_config(&cfg, 8192, 4096, 16, 192);
    assert_eq!(s192.logits, 193 * cfg.vocab_size * 2);
    let max_blocks = 4096 / 16 + 1;
    assert!(s192.scratch >= 32768 + 24 * 192 + 192 * max_blocks * 4);
}
