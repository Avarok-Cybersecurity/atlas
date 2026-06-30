// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;

#[test]
fn test_buffer_sizes_qwen3() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let sizes = BufferSizes::from_config(&cfg, 1, 4096, 16);

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
    let arena = BufferArena::new(&cfg, 128, 4096, 16, &gpu).unwrap();

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
    assert_eq!(gpu.alloc_count(), 27);
}

#[test]
fn test_buffer_sizes_scale_with_batch() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let s1 = BufferSizes::from_config(&cfg, 1, 4096, 16);
    let s128 = BufferSizes::from_config(&cfg, 128, 4096, 16);
    assert_eq!(s128.hidden_states, s1.hidden_states * 128);
    // logits is capped at 16 tokens; FP32 sampling buffer (4 bytes/elem),
    // so s128.logits = 16 * vocab * 4 (not 128× the unbatched value).
    assert_eq!(s128.logits, 16 * cfg.vocab_size * 4);
}
