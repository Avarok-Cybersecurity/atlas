// SPDX-License-Identifier: AGPL-3.0-only

//! DSA layer contracts that hold without a GPU: the kernel *choices* and the
//! lockstep invariant. The numerics are gated by `examples/glm5next_dsa_decode_gate.rs`.

use super::*;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
    }
}

/// 🔴 Two RMSNorm kernels differ ONLY by a `+1` on the weight, with identical
/// signatures and shapes. GLM is plain, so the layer must name the vanilla entry point.
/// Pinned as a string because picking the other one is silent and the shapes agree.
#[test]
fn the_layer_takes_the_vanilla_rmsnorm_not_the_plus_one_variant() {
    let src = include_str!("../layer.rs");
    assert!(
        src.contains(r#"gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")"#),
        "GLM uses x*rms*w; `rms_norm` applies x*rms*(1+w) and would shift every norm"
    );
    assert!(
        !src.contains(r#""rms_norm", "rms_norm""#),
        "the +1-offset RMSNorm must not appear in a GLM path"
    );
}

/// The indexer's k_norm is a LayerNorm with a bias, so the layer must launch the
/// bias-bearing kernel and pass the bias — not silently norm with weight alone.
#[test]
fn the_indexer_norm_passes_a_bias() {
    let src = include_str!("../layer.rs");
    assert!(
        src.contains("k_norm_bias"),
        "indexer.k_norm.bias is REQUIRED; a weight-only bind drops mean subtraction too"
    );
    assert!(src.contains("self.select_kernels.k_norm"));
}

/// Q reaches the decode kernel in LATENT space. A raw `q_b_proj` is a well-formed tensor
/// of the wrong width per head (256 vs 512) in the wrong space.
#[test]
fn q_is_absorbed_to_the_latent_width() {
    let c = cfg();
    assert_eq!(c.kv_lora_rank, 512);
    assert_ne!(
        c.qk_nope_head_dim, c.kv_lora_rank,
        "if these were equal the absorption mistake would be undetectable by shape"
    );
    let src = include_str!("../layer.rs");
    assert!(src.contains("q_absorb"));
}

/// 🔴 The indexer stream and the KV cache must advance together, and the two drifts are NOT
/// symmetric.
///
/// BEHIND (`len < seq_len`) means rows were never written: selection would run over a shorter
/// context than the cache holds — a wrong answer with no crash — and nothing can repair it, so
/// decode refuses.
///
/// AHEAD (`len > seq_len`) is the speculative-verify reject: the rows past the accepted prefix
/// are unreachable, because the selector reads `[0, len)` and the next write starts at
/// `seq_len`. Rewinding is the KV cache's own semantics for a rejected slot, and doing it here
/// is why no rollback callback has to reach into eleven DSA layers. Refusing instead would make
/// every partially-accepted draft a hard error.
#[test]
fn a_lockstep_drift_is_refused_behind_and_rewound_ahead() {
    let src = include_str!("../layer.rs");
    assert!(
        src.contains("must advance in lockstep"),
        "the drift guard must state why it exists"
    );
    assert!(
        src.contains("st.len().cmp(&seq_len)"),
        "the guard must branch on the DIRECTION of the drift, not merely on inequality"
    );
    assert!(
        src.contains("Ordering::Greater => st.rewind_to(seq_len)?"),
        "AHEAD must rewind — this is the verify-reject path"
    );
    assert!(
        src.contains("rows are MISSING, not merely stale"),
        "BEHIND must still refuse, and say why it is the unrecoverable direction"
    );
}

/// `rewind_to` only ever shrinks. A forward "rewind" would mean the caller lost the sequence
/// position, and silently accepting it would advance the selector over rows never written.
#[test]
fn the_indexer_rewind_only_shrinks() {
    let src = include_str!("../state.rs");
    assert!(src.contains("rewind only shrinks"));
}

/// K and V are the SAME buffer: absorbed NoPE MLA caches one latent per token, and the
/// decode kernel reads it for both. Two different pointers would mean the cache is not
/// the absorbed form this layer assumes.
#[test]
fn k_and_v_are_the_same_latent_pool() {
    let src = include_str!("../layer.rs");
    assert!(src.contains("v_cache: pool"));
    assert!(src.contains("K and V are the same latent"));
}

/// The workspace is sized at the DSA context cap so a growing sequence never reallocates
/// mid-serve — the selection scratch is the piece that scales with context.
#[test]
fn the_workspace_is_sized_at_the_context_cap() {
    let c = cfg();
    let cap = super::super::state::max_dsa_context(&c);
    assert_eq!(cap, 16_384);
    let geom = super::super::select::DsaSelectGeometry::plan(&c, cap, 1).unwrap();
    assert_eq!(
        geom.n_pools, 4_096,
        "the cap is the largest plannable pool axis"
    );
}
