// SPDX-License-Identifier: AGPL-3.0-only
//! Post-BF16 + packed IQ2/IQ3 + F32 resident estimate for Kimi-K3 UD-Q2_K_XL.
//!
//! The old formula `802 GiB / tp + 12 GiB` is the on-disk lie (112.25 GiB at
//! tp=8). Live 8×H200 bind used ~130.3 GB/rank. This counts tensors after
//! Q8→BF16, packed experts that stay packed, F32 vectors, and KV for ctx 1024.

/// GiB. 896 at tp=8 is `hidden/tp` here, **not** `n_experts`.
const HIDDEN: usize = 7168;
const LATENT: usize = 3584;
const EXPERT_INTER: usize = 3072;
const N_EXPERTS: usize = 896;
const N_MOE: usize = 92;
const VOCAB: usize = 163_840;
const SHEXP_INTER: usize = 6144;
const DENSE_INTER: usize = 33_792;
const N_KDA: usize = 69;
const N_MLA: usize = 24;
const HEADS: usize = 96;
const Q_LORA: usize = 1536;
const KV_LORA: usize = 512;
const NOPE: usize = 128;
const ROPE: usize = 64;
const V: usize = 128;
const QK: usize = NOPE + ROPE;
const HEAD_DIM: usize = 128;
const IQ2_QK: usize = 256;
const IQ2_BLOCK: usize = 74;
/// Driver + NCCL + PTX + allocator. nvidia-smi − tensor bytes on the live bind.
const CUDA_WORKSPACE: usize = 15 * (1 << 30);

fn packed_iq2_bytes(n_weights: usize) -> usize {
    n_weights / IQ2_QK * IQ2_BLOCK
}

fn bf16_bytes(n: usize) -> usize {
    n.saturating_mul(2)
}

/// Rank-local bytes after TP slice. `tp` must divide hidden / expert_inter / shexp.
pub(super) fn estimate_resident_bytes(tp_world: usize) -> usize {
    let tp = tp_world.max(1);
    let packed = packed_expert_bytes(tp);
    let linears = q8_to_bf16_linear_bytes(tp);
    let f32s = f32_vector_bytes(tp);
    let kv = kv_bytes_ctx(tp, 1024);
    packed + linears + f32s + kv + CUDA_WORKSPACE
}

fn packed_expert_bytes(tp: usize) -> usize {
    // TP splits the 3072 expert-intermediate axis. All 896 experts stay present.
    let expert_local = EXPERT_INTER / tp;
    let per_expert = expert_local * LATENT; // w1 / w3
    let w2 = LATENT * expert_local;
    let n_w = N_EXPERTS * (per_expert * 2 + w2) * N_MOE;
    packed_iq2_bytes(n_w)
}

fn q8_to_bf16_linear_bytes(tp: usize) -> usize {
    let hidden_local = HIDDEN / tp; // 896 at tp=8: hidden/tp, not n_experts
    let heads_local = HEADS / tp;
    let qkv_local = heads_local * HEAD_DIM;
    let shexp_local = SHEXP_INTER / tp; // 768
    let dense_local = DENSE_INTER / tp; // 4224
    let embed = VOCAB * HIDDEN;
    let lm_head = VOCAB * HIDDEN;
    let routed = N_MOE * LATENT * hidden_local * 2; // down+up
    let shexp = N_MOE * (shexp_local * HIDDEN * 2 + HIDDEN * shexp_local);
    let dense0 = 3 * dense_local * HIDDEN;
    let kda = N_KDA
        * (qkv_local * HIDDEN * 4 // q k v g
            + HIDDEN * qkv_local // o
            + HEAD_DIM * HIDDEN // f_a replicated
            + qkv_local * HEAD_DIM // f_b
            + heads_local * HIDDEN); // b_proj
    let mla = N_MLA
        * (Q_LORA * HIDDEN
            + heads_local * QK * Q_LORA
            + (KV_LORA + ROPE) * HIDDEN
            + heads_local * KV_LORA * NOPE
            + heads_local * V * KV_LORA
            + heads_local * V * HIDDEN
            + HIDDEN * heads_local * V);
    let router = N_MOE * N_EXPERTS * HIDDEN; // 896 = n_experts
    bf16_bytes(embed + lm_head + routed + shexp + dense0 + kda + mla + router)
}

fn f32_vector_bytes(tp: usize) -> usize {
    let heads_local = HEADS / tp;
    // A_log, dt_bias, conv, norms, AttnRes, router bias. Upper bound.
    let kda_vec = N_KDA * (heads_local * HEAD_DIM * 2 + heads_local * 4 * HEAD_DIM);
    let norms = 93 * HIDDEN * 6;
    (kda_vec + norms) * 4
}

fn kv_bytes_ctx(tp: usize, ctx: usize) -> usize {
    let heads_local = HEADS / tp;
    // MLA device KV F32: 24 layers × seq × (kv_lora + v local).
    N_MLA * ctx * (KV_LORA + heads_local * V) * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const OLD_LIE: usize = 802 * (1 << 30) / 8 + 12 * (1 << 30);

    #[test]
    fn preflight_is_not_the_112_gib_disk_lie() {
        let est = estimate_resident_bytes(8);
        assert_ne!(est, OLD_LIE, "must not use disk/tp + 12 GiB");
        let gib = est as f64 / GIB;
        assert!(
            gib > 120.0 && gib <= 133.0,
            "tp=8 post-BF16+packed+F32+KV must sit near live ~130 GB, got {gib:.2} GiB"
        );
        assert!(
            packed_expert_bytes(8) < 100 * (1 << 30),
            "experts stay packed IQ2, not BF16 expand"
        );
        println!(
            "K3 tp=8 preflight: {:.2} GiB (packed IQ2 {:.2} + Q8→BF16 {:.2} + F32 {:.2} + KV1024 {:.2} + CUDA ws {:.2})",
            gib,
            packed_expert_bytes(8) as f64 / GIB,
            q8_to_bf16_linear_bytes(8) as f64 / GIB,
            f32_vector_bytes(8) as f64 / GIB,
            kv_bytes_ctx(8, 1024) as f64 / GIB,
            CUDA_WORKSPACE as f64 / GIB
        );
    }

    #[test]
    fn hidden_local_896_is_not_n_experts_in_the_byte_count() {
        assert_eq!(HIDDEN / 8, 896);
        assert_eq!(N_EXPERTS, 896);
        let down = LATENT * (HIDDEN / 8);
        assert_eq!(down, 3_211_264);
        assert_ne!(down, LATENT * HIDDEN, "on-rank down is not 3584×7168");
    }
}
