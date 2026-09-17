// SPDX-License-Identifier: AGPL-3.0-only

//! Slice-12 GATE 4 (algebra leg): **EP=2 MoE semantics, proven without collectives.**
//!
//! # What Atlas's EP path actually is
//!
//! Atlas does NOT dispatch tokens to the owning rank. It runs **masked-local +
//! all-reduce**:
//!
//! 1. Every rank runs the router over **all** `num_experts` and takes the same
//!    top-k (the router is replicated, so all ranks agree).
//! 2. Each rank holds only its owned experts; remote experts are
//!    `ExpertWeight::null()` (`weight_map/loaders_moe.rs:62-66`), and the
//!    grouped GEMM skips NULL, leaving zero
//!    (`kernels/gb10/common/moe_w4a16_grouped_gemm.cu:262`).
//! 3. Each rank weighted-sums its partial contribution.
//! 4. `all_reduce(SUM)` over the hidden vector reassembles the global result
//!    (`layers/moe/forward.rs:648-660`).
//! 5. The shared expert is **excluded** from the pre-reduce blend and added
//!    **once** after, or it would be counted `world_size` times
//!    (`forward.rs:621-630`).
//!
//! Because expert contributions enter the output as a **sum**, masked-local +
//! SUM-all-reduce is *algebraically identical* to true dispatch. It costs more
//! bandwidth (`O(tokens × hidden)` instead of `O(dispatched × hidden)`) but it
//! is not an approximation and not a "fallback" in the correctness sense.
//!
//! 🪤 `MoeLayer::forward_ep_dispatch` (`layers/moe/forward_ep.rs`) and
//! `layers/ep_dispatch.rs` are the *bandwidth* optimisation. `forward_ep_dispatch`
//! has **zero callers** — it is dead code. The shipped EP=4 397B path is the
//! masked-local + all-reduce path above. Slice 11 read the dead function's
//! doc-comment ("scaffolding … all-reduce fallback") as a statement about the
//! live path; it is not.
//!
//! # What this test proves, and what it does not
//!
//! PROVEN here (exact arithmetic, single process):
//! * ownership partitioning is complete and disjoint — every (token, expert)
//!   pair lands on exactly one rank: no drops, no duplicate execution;
//! * global expert ids and routing weights survive partitioning unchanged;
//! * summing the per-rank masked partials reproduces the single-process
//!   reference **bit-exactly**;
//! * the shared expert must be added exactly once, not once per rank;
//! * empty-send / empty-receive (a rank owning none of a token's experts).
//!
//! NOT proven here: NCCL transport. That needs a real 2-rank run on n1/n2 and
//! is the remaining leg of GATE 4.
//!
//! Expert outputs are small integers held in `f32`, so every sum below is exact
//! and `assert_eq!` on floats is legitimate — reassociation cannot bite.

use spark_model::layers::ep_dispatch::build_ep_routing_table;

const NUM_EXPERTS: usize = 8;
const EP_WORLD: usize = 2;
const HIDDEN: usize = 4;
const TOP_K: usize = 2;

/// Local expert range for `rank`, mirroring `ModelConfig::local_expert_range`.
fn range_of(rank: usize) -> (usize, usize) {
    let per = NUM_EXPERTS / EP_WORLD;
    let start = rank * per;
    let end = if rank == EP_WORLD - 1 {
        NUM_EXPERTS
    } else {
        start + per
    };
    (start, end)
}

/// Deterministic stand-in for expert `e`'s output: a distinct integer vector.
/// Distinct per expert so a mis-routed token is impossible to miss.
fn expert_out(e: u32) -> [f32; HIDDEN] {
    let base = (e as f32 + 1.0) * 10.0;
    [base, base + 1.0, base + 2.0, base + 3.0]
}

fn shared_out() -> [f32; HIDDEN] {
    [1.0, 2.0, 3.0, 4.0]
}

/// Single-process ground truth: weighted sum over every top-k expert, plus the
/// shared expert once.
fn reference(indices: &[u32], weights: &[f32], num_tokens: usize) -> Vec<[f32; HIDDEN]> {
    let mut out = vec![[0.0f32; HIDDEN]; num_tokens];
    for t in 0..num_tokens {
        for k in 0..TOP_K {
            let f = t * TOP_K + k;
            let e = expert_out(indices[f]);
            for h in 0..HIDDEN {
                out[t][h] += weights[f] * e[h];
            }
        }
        for h in 0..HIDDEN {
            out[t][h] += shared_out()[h];
        }
    }
    out
}

/// One rank's partial: routed experts it OWNS only. The shared expert is
/// deliberately excluded — it is added once after the reduce.
fn rank_partial(
    indices: &[u32],
    weights: &[f32],
    num_tokens: usize,
    rank: usize,
) -> (Vec<[f32; HIDDEN]>, usize) {
    let (start, end) = range_of(rank);
    let table = build_ep_routing_table(indices, weights, num_tokens, TOP_K, start, end);

    let mut out = vec![[0.0f32; HIDDEN]; num_tokens];
    for i in 0..table.local_count() {
        let t = table.local_token_indices[i] as usize;
        let e = table.local_expert_ids[i];
        // The owning rank must never be asked to run an expert it does not hold.
        assert!(
            (e as usize) >= start && (e as usize) < end,
            "rank {rank} asked to execute non-owned expert {e}"
        );
        let w = table.local_weights[i];
        let v = expert_out(e);
        for h in 0..HIDDEN {
            out[t][h] += w * v[h];
        }
    }
    (out, table.local_count())
}

/// all_reduce(SUM) across ranks, then the shared expert added exactly once.
fn reduce_and_finish(partials: Vec<Vec<[f32; HIDDEN]>>, num_tokens: usize) -> Vec<[f32; HIDDEN]> {
    let mut out = vec![[0.0f32; HIDDEN]; num_tokens];
    for p in &partials {
        for t in 0..num_tokens {
            for h in 0..HIDDEN {
                out[t][h] += p[t][h];
            }
        }
    }
    for t in 0..num_tokens {
        for h in 0..HIDDEN {
            out[t][h] += shared_out()[h];
        }
    }
    out
}

/// Run one routing case end-to-end and assert EP == single process.
fn assert_ep_matches_reference(indices: &[u32], weights: &[f32], case: &str) {
    let num_tokens = indices.len() / TOP_K;

    let mut partials = Vec::new();
    let mut executed = 0usize;
    for rank in 0..EP_WORLD {
        let (p, n) = rank_partial(indices, weights, num_tokens, rank);
        executed += n;
        partials.push(p);
    }

    // No dropped and no duplicated (token, expert) execution.
    assert_eq!(
        executed,
        num_tokens * TOP_K,
        "{case}: {} pairs executed across ranks, expected {}",
        executed,
        num_tokens * TOP_K
    );

    let got = reduce_and_finish(partials, num_tokens);
    let want = reference(indices, weights, num_tokens);
    assert_eq!(
        got, want,
        "{case}: EP=2 result diverged from single process"
    );
}

#[test]
fn mixed_local_and_remote_matches_single_process() {
    // t0: one expert per rank. t1: both remote. t2: both local. t3: split.
    let indices = vec![1u32, 5, 6, 7, 0, 3, 2, 4];
    let weights = vec![0.6f32, 0.4, 0.5, 0.5, 0.25, 0.75, 0.125, 0.875];
    assert_ep_matches_reference(&indices, &weights, "mixed");
}

#[test]
fn all_local_to_rank0_matches_single_process() {
    let indices = vec![0u32, 1, 2, 3];
    let weights = vec![0.5f32, 0.5, 0.25, 0.75];
    assert_ep_matches_reference(&indices, &weights, "all-rank0");
}

#[test]
fn all_local_to_rank1_matches_single_process() {
    let indices = vec![4u32, 5, 6, 7];
    let weights = vec![0.5f32, 0.5, 0.25, 0.75];
    assert_ep_matches_reference(&indices, &weights, "all-rank1");
}

/// Empty-send / empty-receive: rank 0 owns nothing here and must contribute an
/// exact zero partial, not garbage and not a skipped reduce.
#[test]
fn empty_partial_contributes_exact_zero() {
    let indices = vec![4u32, 7];
    let weights = vec![0.5f32, 0.5];
    let (p0, n0) = rank_partial(&indices, &weights, 1, 0);
    assert_eq!(n0, 0, "rank 0 owns no expert in this case");
    assert_eq!(p0, vec![[0.0f32; HIDDEN]], "empty partial must be zero");
    assert_ep_matches_reference(&indices, &weights, "empty-rank0");
}

/// Ownership partitioning: disjoint, complete, ids and weights preserved.
#[test]
fn ownership_is_disjoint_complete_and_value_preserving() {
    let indices = vec![1u32, 5, 6, 7, 0, 3, 2, 4];
    let weights = vec![0.6f32, 0.4, 0.5, 0.5, 0.25, 0.75, 0.125, 0.875];
    let num_tokens = indices.len() / TOP_K;

    // Collect (token, expert, weight) claimed by each rank as LOCAL.
    let mut claimed: Vec<(u32, u32, f32)> = Vec::new();
    for rank in 0..EP_WORLD {
        let (start, end) = range_of(rank);
        let t = build_ep_routing_table(&indices, &weights, num_tokens, TOP_K, start, end);

        // Every pair is accounted for exactly once as local-or-remote.
        assert_eq!(t.total_count(), num_tokens * TOP_K, "rank {rank} total");
        // What is remote here must be local on the other rank.
        for e in &t.remote_expert_ids {
            assert!(
                !((*e as usize) >= start && (*e as usize) < end),
                "rank {rank} classified an owned expert as remote"
            );
        }
        for i in 0..t.local_count() {
            claimed.push((
                t.local_token_indices[i],
                t.local_expert_ids[i],
                t.local_weights[i],
            ));
        }
    }

    // Exactly one owner per (token, expert) pair — no duplicates, no gaps.
    assert_eq!(claimed.len(), num_tokens * TOP_K);
    let mut sorted = claimed.clone();
    sorted.sort_by_key(|(t, e, _)| (*t, *e));
    sorted.dedup_by_key(|(t, e, _)| (*t, *e));
    assert_eq!(sorted.len(), claimed.len(), "a pair was claimed twice");

    // Global ids and weights survive the partition unchanged.
    for (t, e, w) in claimed {
        let mut found = false;
        for k in 0..TOP_K {
            let f = t as usize * TOP_K + k;
            if indices[f] == e {
                assert_eq!(w, weights[f], "weight altered for (t{t}, e{e})");
                found = true;
            }
        }
        assert!(
            found,
            "expert id {e} not in token {t}'s top-k — id was rewritten"
        );
    }
}

/// 🪤 The shared expert must be added ONCE, after the reduce. Adding it before
/// makes it appear `world_size` times. This pins the bug `forward.rs:621-630`
/// guards against.
#[test]
fn shared_expert_added_before_reduce_would_double_count() {
    let indices = vec![1u32, 5];
    let weights = vec![0.5f32, 0.5];

    // Wrong: each rank folds the shared expert into its own partial.
    let mut wrong = vec![[0.0f32; HIDDEN]; 1];
    for rank in 0..EP_WORLD {
        let (p, _) = rank_partial(&indices, &weights, 1, rank);
        for h in 0..HIDDEN {
            wrong[0][h] += p[0][h] + shared_out()[h];
        }
    }
    let right = reference(&indices, &weights, 1);
    for h in 0..HIDDEN {
        assert_eq!(
            wrong[0][h] - right[0][h],
            shared_out()[h] * (EP_WORLD - 1) as f32,
            "double-count control did not behave as predicted"
        );
    }
    assert_ne!(wrong, right, "the negative control must diverge");
}
