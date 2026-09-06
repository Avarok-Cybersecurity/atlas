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
        max_context: 16_384,
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

/// 🔴 A62: the indexer's capacity check must run BEFORE the first device write, not after.
/// `advance(1)` at the tail of `indexer_forward` refuses the 16,385th token only once the
/// GEMM has already written row `capacity` — 256 B past the buffer — and the resulting
/// `CUDA_ERROR_ILLEGAL_ADDRESS (700)` is sticky, so an over-length prompt downs the serve
/// for every later request instead of failing one of them.
#[test]
fn the_indexer_checks_capacity_before_it_writes() {
    let src = include_str!("../layer.rs");
    let body = src
        .split_once("pub fn indexer_forward")
        .expect("indexer_forward must exist")
        .1
        .split_once("fn select_row")
        .expect("select_row follows indexer_forward")
        .0;
    let guard = body
        .find("state.ensure_room(1)?")
        .expect("indexer_forward must precheck capacity");
    let first_write = body.find("gemm(").expect("indexer_forward writes via gemm");
    assert!(
        guard < first_write,
        "the capacity check must come before the first device write, not after"
    );
}

/// 🔴 A62, replay half. `sync_replayed_step` is a RECONCILE and runs after `launch_graph`
/// on purpose — so it cannot be what stops an out-of-bounds write. Every one of the four
/// replay branches must ask `check_replay_room` BEFORE it launches the graph; a replay
/// writes the indexer row from a device position with no host code in the loop, and the
/// resulting sticky CUDA 700 kills the context for every later request.
///
/// 🪤 Only the REPLAY launch needs it. The second `launch_graph` in each of these files
/// runs a graph just captured, and capture happens inside the eager path, whose
/// `indexer_forward` already prechecked — that is the prefill half of the same fix.
#[test]
fn every_graph_replay_checks_room_before_it_launches() {
    let paths: [(&str, &str); 4] = [
        (
            "decode_a",
            include_str!("../../../model/trait_impl/decode_a.rs"),
        ),
        (
            "verify_b",
            include_str!("../../../model/trait_impl/verify_b.rs"),
        ),
        (
            "verify_c",
            include_str!("../../../model/trait_impl/verify_c.rs"),
        ),
        (
            "verify_c2",
            include_str!("../../../model/trait_impl/verify_c2.rs"),
        ),
    ];
    for (name, src) in paths {
        let guard = src
            .find("layer.check_replay_room(")
            .unwrap_or_else(|| panic!("{name}: the replay branch must precheck capacity"));
        let launch = src
            .find("self.gpu.launch_graph(")
            .unwrap_or_else(|| panic!("{name}: expected a graph replay"));
        assert!(
            guard < launch,
            "{name}: the room check must precede the FIRST launch_graph — after it, the \
             write has already happened"
        );
        let sync = src
            .find("layer.sync_replayed_step(")
            .unwrap_or_else(|| panic!("{name}: the reconcile must still be there"));
        assert!(
            launch < sync,
            "{name}: the reconcile stays AFTER the launch — moving it would change the \
             A56 rewind semantics this fix must not touch"
        );
    }
}

/// The composite is what the model's layer vec holds, so a `check_replay_room` implemented
/// only on the inner `Glm5NextDsaLayer` would never run — the exact trap that left the
/// counter frozen for `sync_replayed_step`. Both impls must exist.
#[test]
fn the_composite_layer_implements_the_room_check_too() {
    let composite = include_str!("../../glm5next_layer/mod.rs");
    assert!(
        composite.contains("fn check_replay_room"),
        "the COMPOSITE layer must implement it — the inner impl is never reached"
    );
    assert!(
        include_str!("../layer.rs").contains("fn check_replay_room"),
        "the inner DSA layer implements it as well"
    );
}

/// Both A62 routes raise the same refusal text, so the replay guard tags its error. Without
/// the tag, "the replay route was proven at runtime" would rest on inference about which
/// guard fired, which is exactly the substitution this validation must not make.
#[test]
fn the_replay_refusal_is_distinguishable_from_the_prefill_one() {
    for src in [
        include_str!("../layer.rs"),
        include_str!("../../glm5next_layer/mod.rs"),
    ] {
        assert!(
            src.contains("DSA replay pre-check"),
            "the replay guard must tag its refusal so a log can name the route"
        );
    }
}

// ── Batched-selector execution scope ────────────────────────────────────────────────────
//
// The batched prefill selector was measured and qualified on PREFILL alone. These pin the
// phases it may and may not run in, because the earlier gate (`!graph_capture && k > 1`) let
// an EAGER verify in: `verify_a` builds its context with `graph_capture: false` outright, and
// `verify_b/c/c2/d/fused` set it from `use_graphs`, which is false under
// `ATLAS_GLM_VERIFY_GRAPHS=0`, under high-speed swap, and under `ATLAS_LORA_EAGER`.

use super::super::layer::batch_select_enabled;

#[test]
fn only_a_multi_row_eager_prefill_takes_the_batched_selector() {
    // (phase, workspace_ready, is_prefill, graph_capture, k, expected)
    let cases: &[(&str, bool, bool, bool, usize, bool)] = &[
        (
            "prefill sub-chunk, PREFILL_ROWS=16",
            true,
            true,
            false,
            16,
            true,
        ),
        ("prefill tail sub-chunk, k=2", true, true, false, 2, true),
        ("prefill tail sub-chunk, k=1", true, true, false, 1, false),
        ("chunked prefill, second chunk", true, true, false, 16, true),
        ("decode step, k=1, graphed", true, false, true, 1, false),
        ("decode step, k=1, eager", true, false, false, 1, false),
        ("graphed verify K=3", true, false, true, 3, false),
        ("graphed verify K=4", true, false, true, 4, false),
        // 🔴 the regression this table exists for
        (
            "EAGER verify K=3 (ATLAS_GLM_VERIFY_GRAPHS=0)",
            true,
            false,
            false,
            3,
            false,
        ),
        (
            "EAGER verify K=2 (verify_b, HSS engaged)",
            true,
            false,
            false,
            2,
            false,
        ),
        (
            "EAGER verify K=4 (verify_c2, LoRA eager)",
            true,
            false,
            false,
            4,
            false,
        ),
        (
            "verify_a generic N-token (graph_capture hard false)",
            true,
            false,
            false,
            5,
            false,
        ),
        (
            "kill-switch ATLAS_DSA_SELECT_ROWS=0, prefill",
            false,
            true,
            false,
            16,
            false,
        ),
        ("kill-switch, eager verify", false, false, false, 3, false),
    ];
    for &(phase, ws, pf, gc, k, want) in cases {
        assert_eq!(
            batch_select_enabled(ws, pf, gc, k),
            want,
            "{phase}: workspace={ws} is_prefill={pf} graph_capture={gc} k={k}"
        );
    }
}

/// `is_prefill` is only worth anything if every caller states it correctly. `forward_k` has
/// exactly two call sites — the prefill sub-chunk loop and the speculative verify — and the
/// verify one must pass `false`. A third caller has to come here and choose.
#[test]
fn forward_k_has_two_callers_and_the_verify_one_is_not_prefill() {
    let src = include_str!("../../glm5next_layer/mod.rs");
    assert_eq!(
        src.matches("self.forward_k(").count(),
        2,
        "a new forward_k caller must decide its own `is_prefill`, not inherit one"
    );
    assert!(
        src.contains(
            "// This IS the prefill sub-chunk caller.
                    true,"
        ),
        "the prefill sub-chunk must pass is_prefill = true"
    );
    assert!(
        src.contains("// A speculative verify, NOT a prefill sub-chunk"),
        "the speculative verify must pass is_prefill = false, and say why"
    );
}

/// The tail's first slot is `select_k * KP`, and `select_k` is a PER-PASS scalar. A batched
/// pass plans it from the group's FINAL length, so without a per-row clamp every earlier row
/// gets a wider `select_k` than its serial twin and the visible tail slides forward. That is
/// not cosmetic: `glm5next_dsa_mla_decode_fp8` splits the selection row into 8 warp slices
/// and merges their online softmaxes, so a tail token that slides across a slice boundary is
/// folded by the MERGE instead of by its warp's serial loop. Measured on the real kernel:
/// 14 of 18 crossing configurations differ, up to 2 BF16 ulp.
#[test]
fn expand_selection_clamps_select_k_to_the_row() {
    let src = include_str!("../../../../../../kernels/gb10/common/dsa_indexer.cu");
    assert!(
        src.contains("const unsigned int row_pools = (unsigned int)(q_pos[r] + 1) / KP;"),
        "the row's own pool count must come from its own q_pos"
    );
    assert!(
        src.contains("unsigned int base = row_select_k * KP;"),
        "the tail base must be the ROW's select_k, never the pass's"
    );
    assert!(
        !src.contains("unsigned int base = select_k * KP;"),
        "the pass-scalar tail base is the defect; it must not come back"
    );
}
