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
    // The DENSE intermediate (12288) is wider than the routed one
    // (top_k 10 x moe_intermediate 1024 = 10240), and it is the dense width
    // these buffers must hold. Rows are `max_batch_tokens` rounded up to 16 —
    // the cuBLASLt FP8 M-pad headroom the W8A8 dense-FFN prefill writes into
    // (#917/#928); see the sizing note on `k_max` in `sizes.rs`.
    let rows = 4_usize.div_ceil(16) * 16;
    assert!(cfg.intermediate_size > cfg.num_experts_per_tok * cfg.moe_intermediate_size);
    assert_eq!(sizes.expert_gate_out, rows * 12_288 * 2);
    assert_eq!(sizes.expert_up_out, rows * 12_288 * 2);
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
    // qkv: ceil16(1) * (16*2 + 2*2) * 256 * 2 = 16 * 36 * 256 * 2 = 294912
    // Q+gate: 16*2*256, K: 2*256, V: 2*256 — and the row extent is the
    // cuBLASLt M-pad, exactly as for `ssm_qkvz` / `ssm_deinterleaved` below:
    // the cache-skip Q/K/V prefill's cuBLASLt arm hands the library ceil16(M)
    // and WRITES the phantom rows (#928, `sizes.rs`'s `m_pad`). The 16x here
    // is an artifact of sizing at M=1; at a real prefill arena the pad is
    // <= 15 rows out of thousands.
    assert_eq!(sizes.qkv_output, 294912);
    // attn: 1 * 16 * 256 * 2 = 8192
    assert_eq!(sizes.attn_output, 8192);
    // gate: 1 * 512 * 2 = 1024
    assert_eq!(sizes.gate_logits, 1024);
    // logits: 1 * 151936 * 2 = 303872
    assert_eq!(sizes.logits, 303872);
    // ssm_qkvz: ceil16(1) * 12288 * 2 = 393216
    // Q(16*128) + K(16*128) + V(32*128) + Z(32*128) = 12288, and the row
    // extent is the cuBLASLt M-pad: the SSM QKVZ cuBLASLt arm hands the
    // library ceil16(M) and WRITES the phantom rows (#917, 2026-09-11). The
    // 16x here is an artifact of sizing at M=1; at a real prefill arena the
    // pad is <= 15 rows out of thousands.
    assert_eq!(sizes.ssm_qkvz, 393216);
    // ssm_ba: max(1 * 64 * 2, 256) = 256 (minimum allocation)
    assert_eq!(sizes.ssm_ba, 256);
    // ssm_deinterleaved: same as ssm_qkvz, same M-pad = 393216
    assert_eq!(sizes.ssm_deinterleaved, 393216);
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
        (
            "ffn_act_scale_kmajor",
            arena.ffn_act_scale_kmajor(),
            sizes.ffn_act_scale_kmajor,
        ),
        (
            "ffn_gate_up_fused",
            arena.ffn_gate_up_fused(),
            sizes.ffn_gate_up_fused,
        ),
        ("fp8_act", arena.fp8_act(), sizes.fp8_act),
        ("fp8_act_scale", arena.fp8_act_scale(), sizes.fp8_act_scale),
        (
            "fp8_act_scale_kmajor",
            arena.fp8_act_scale_kmajor(),
            sizes.fp8_act_scale_kmajor,
        ),
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
        // Added by the qwen4_exp work merged alongside this branch. The
        // assertion below is `alloc_count() == allocated.len()`, so a buffer the
        // arena allocates and this list omits fails the test rather than being
        // quietly uncounted — which is how the omission surfaced: 31 allocated,
        // 29 enumerated.
        (
            "hc_lowrank_scratch",
            arena.hc_lowrank_scratch(),
            sizes.hc_lowrank_scratch,
        ),
        (
            "qsa_select_scratch",
            arena.qsa_select_scratch(),
            sizes.qsa_select_scratch,
        ),
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

// ── Row-wise FP8 GDN prefill BF16-weight slab (#917) ──────────────────────
//
// H100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8`: the `ATLAS_FP8_ROWWISE` GDN arms
// dequantised their per-row FP8 weights to BF16 through a `gpu.alloc` memoised
// by weight pointer — `167772160` B per layer with NO entry here, so
// `--gpu-memory-utilization` could not see it and a 28-token prefill died at
// layer 36 with `cuMemAlloc_v2 failed: status 2`. These pin the entry that
// replaced it: present IFF the lever is armed, and exact at the 27B geometry.
//
// The lever is passed in rather than set: `set_var` is process-global and
// unsafe, and would race every other test in this binary.

/// `Qwen/Qwen3.8-27B-FP8` at the shapes `kernels/gb10/qwen3.8-27b/MODEL.toml`
/// declares (hidden 5120, 64 layers on a 4-cycle → 48 GDN), plus the GDN head
/// geometry `ModelConfig` reads from the checkpoint's own `config.json`
/// (16x128 key heads, 48x128 value heads). Same fixture as
/// `weight_loader::qwen35_dense::predicted_residency_tests::qwen38_27b`.
fn qwen38_27b() -> ModelConfig {
    use atlas_core::config::LayerType;
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 5120;
    c.num_hidden_layers = 64;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c.full_attention_interval = 4;
    c.layer_types = (0..64)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c
}

#[test]
fn rowwise_bf16_slab_is_sized_only_when_the_lever_is_armed() {
    let cfg = qwen38_27b();
    assert_eq!(
        ssm_rowwise_w_bf16_bytes_for(&cfg, false),
        0,
        "an unarmed ATLAS_FP8_ROWWISE must leave the default recipe's ledger \
         byte-identical — the arena allocates NULL for a 0-byte entry"
    );

    // in_proj_qkvz: (16*128 q + 16*128 k + 48*128 v + 48*128 z) = 16384 rows
    // x 5120 hidden x 2 B = 167772160 — the exact per-layer figure the H100
    // OOM receipt names. out_proj: [5120, 48*128] x 2 B = 62914560.
    assert_eq!(cfg.ssm_qkvz_size(), 16_384);
    let qkvz = 16_384 * 5_120 * 2;
    let out_proj = 5_120 * (48 * 128) * 2;
    assert_eq!(qkvz, 167_772_160);
    assert_eq!(ssm_rowwise_w_bf16_layer_bytes(&cfg), qkvz + out_proj);
    assert_eq!(cfg.num_ssm_layers(), 48);
    assert_eq!(
        ssm_rowwise_w_bf16_bytes_for(&cfg, true),
        48 * (qkvz + out_proj),
        "48 GDN layers x (in_proj_qkvz + out_proj) — 10.31 GiB, which is what \
         the preflight ring fitter now prices instead of discovering at \
         layer 36"
    );
}

/// The entry has to reach `total_bytes()`, because THAT is what preflight's
/// `headroom::post_load_yardstick` takes as its `arena` term. A field the
/// sum forgets is exactly as invisible as the `gpu.alloc` it replaced.
#[test]
fn rowwise_bf16_slab_is_counted_in_total_bytes() {
    let cfg = qwen38_27b();
    let mut sizes = BufferSizes::from_config(&cfg, 64, 4096, 16, 32);
    // Zeroed first, not assumed zero: `from_config` reads the ambient
    // environment, and a runner that happens to export ATLAS_FP8_ROWWISE=1
    // must not turn this into an assertion about nothing.
    sizes.ssm_rowwise_w_bf16 = 0;
    let before = sizes.total_bytes();
    // A SENTINEL, not the real slab size. This test is about the sum, and
    // reading the sizing function here would let a sizing bug that returns 0
    // make it vacuously true — which is exactly how a term goes missing.
    sizes.ssm_rowwise_w_bf16 = 4096;
    assert_eq!(sizes.total_bytes(), before + 4096);
}
