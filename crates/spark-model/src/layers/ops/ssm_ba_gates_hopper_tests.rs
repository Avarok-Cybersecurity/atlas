// SPDX-License-Identifier: AGPL-3.0-only

//! The BA-gates twin's index model and its selection grammar, on a CPU.
//!
//! `native_ssm_ba_gates_hopper_microtest` is the bitwise oracle, and it needs a
//! GPU. Everything below is the part of the twin's correctness that is
//! ARITHMETIC — which lane accumulates which `kv`, which warp slot holds which
//! partial, which BA output each thread owns, and when the launcher is allowed
//! to choose the twin at all — so it is graded here, in the gate that runs on
//! every push.

use super::*;

/// Qwen3.8-27B: `nv = 48`, so `N = ssm_ba_size = 96`, `K = hidden = 5120`.
const N: u32 = 96;
const K: u32 = 5120;
const VPG: u32 = 2;
const H100_SMS: u32 = 132;

// ── the reduction order ──

/// THE bit-identity argument, as a property: every lane's `kv` sequence is the
/// parent's, and the 64 sequences partition `0..K/8` exactly once. The twin
/// changes which CTA a lane lives in and nothing about this sequence.
#[test]
fn the_lane_kv_sweep_partitions_k_in_the_parents_order() {
    let k_vec = K / 8;
    let mut seen = vec![0u32; k_vec as usize];
    for lane in 0..BA_GATES_LANES {
        let kvs = ba_gates_lane_kv(lane, k_vec);
        assert_eq!(
            kvs.len(),
            (k_vec / BA_GATES_LANES) as usize,
            "lane {lane}: K/8 = {k_vec} is a multiple of {BA_GATES_LANES}, so \
             every lane takes the same count"
        );
        for (i, kv) in kvs.iter().enumerate() {
            assert_eq!(
                *kv,
                lane + i as u32 * BA_GATES_LANES,
                "lane {lane} step {i}: the stride IS the reduction order"
            );
            seen[*kv as usize] += 1;
        }
    }
    assert!(
        seen.iter().all(|c| *c == 1),
        "the 64 lane sweeps must cover each uint4 of the row exactly once"
    );
}

/// A ragged `K/8` (not a multiple of 64) still partitions, and the trailing
/// lanes simply take one fewer step. The guard refuses `K % 8 != 0`, not this.
#[test]
fn a_ragged_k_vec_still_partitions() {
    let k_vec = 100u32;
    let mut seen = vec![0u32; k_vec as usize];
    for lane in 0..BA_GATES_LANES {
        for kv in ba_gates_lane_kv(lane, k_vec) {
            seen[kv as usize] += 1;
        }
    }
    assert!(seen.iter().all(|c| *c == 1));
}

/// The cross-warp slot the parent writes and the slot the twin writes are the
/// same slot. The parent spells it `local_out * 2 + (lane / 32)`; the twin
/// spells it `threadIdx.x / 32`. They agree for every thread of the block, or
/// the twin sums two partials of different outputs.
#[test]
fn the_twins_warp_slot_is_the_parents_warp_slot() {
    for tid in 0..BA_GATES_BLOCK {
        let local_out = tid / BA_GATES_LANES;
        let lane = tid % BA_GATES_LANES;
        assert_eq!(
            ba_gates_warp(local_out, lane),
            tid / 32,
            "thread {tid}: the parent's smem index must be the block warp index"
        );
    }
}

/// The cross-warp sum is `even + odd`, in that order, and the two slots belong
/// to the output that reads them. Floating-point addition is not associative;
/// swapping these two is a one-ulp difference in `gate`, and `gate` multiplies
/// the GDN state at every one of 48 layers.
#[test]
fn the_cross_warp_pair_is_this_outputs_two_warps_in_order() {
    for local_out in 0..BA_GATES_OUTS {
        let (lo, hi) = ba_gates_cross_warp_pair(local_out);
        assert_eq!((lo, hi), (local_out * 2, local_out * 2 + 1));
        assert_eq!(ba_gates_warp(local_out, 0), lo, "lanes 0..31");
        assert_eq!(ba_gates_warp(local_out, 32), hi, "lanes 32..63");
        assert!(hi < BA_GATES_WARPS);
    }
}

// ── the output mapping ──

/// The twin's `(tile, group, local_out)` enumeration covers every BA output
/// exactly once, and each one lands on the same `n` the parent's
/// `(blockIdx.x, local_out)` gives it.
#[test]
fn the_tiled_group_sweep_covers_every_output_exactly_once() {
    let n_groups = N.div_ceil(BA_GATES_OUTS);
    let mut seen = vec![0u32; N as usize];
    let mut g0 = 0;
    while g0 < n_groups {
        for g in 0..BA_GATES_GROUPS {
            for local_out in 0..BA_GATES_OUTS {
                let group = g0 + g;
                if group >= n_groups {
                    continue;
                }
                let n = ba_gates_output(group, local_out);
                if n >= N {
                    continue;
                }
                // The parent's grid is `blockIdx.x = group`; same function.
                assert_eq!(n, group * BA_GATES_OUTS + local_out);
                seen[n as usize] += 1;
            }
        }
        g0 += BA_GATES_GROUPS;
    }
    assert!(
        seen.iter().all(|c| *c == 1),
        "every BA output must be produced exactly once: {seen:?}"
    );
}

/// The lever's whole claim, as arithmetic. The parent reads each token's
/// activation row ONCE PER BA OUTPUT — 24 CTAs, each of whose four 64-lane
/// groups sweeps all of K — so 96 reads at this geometry. The twin's 64-lane
/// groups sweep it once per output-group TILE, so `4 * ceil(24/8) = 12`.
#[test]
fn the_activation_row_reads_drop_from_ninety_six_to_twelve() {
    let n_groups = N.div_ceil(BA_GATES_OUTS);
    assert_eq!(n_groups, 24, "the parent's CTAs per token");
    let parent_reads = n_groups * BA_GATES_OUTS;
    let twin_reads = n_groups.div_ceil(BA_GATES_GROUPS) * BA_GATES_OUTS;
    assert_eq!(parent_reads, N, "one activation-row sweep per BA output");
    assert_eq!(twin_reads, 12);
    assert_eq!(parent_reads / twin_reads, BA_GATES_GROUPS, "the tile width");
}

/// The gate/beta split and the `vh` it resolves to are the parent's, for every
/// output of the real geometry: 48 gates at `[0, nv)` and 48 betas at
/// `[nv, 2nv)`, each `vh` produced exactly once.
#[test]
fn the_gate_and_beta_slots_are_the_parents() {
    let nv = N / 2;
    let mut gates = vec![0u32; nv as usize];
    let mut betas = vec![0u32; nv as usize];
    for n in 0..N {
        match ba_gates_slot(n, VPG) {
            Ok(vh) => gates[vh as usize] += 1,
            Err(vh) => betas[vh as usize] += 1,
        }
    }
    assert!(gates.iter().all(|c| *c == 1), "every head gets one gate");
    assert!(betas.iter().all(|c| *c == 1), "every head gets one beta");
    // The interleave the parent documents: per group of `2*vpg`, betas first.
    assert_eq!(ba_gates_slot(0, VPG), Err(0));
    assert_eq!(ba_gates_slot(1, VPG), Err(1));
    assert_eq!(ba_gates_slot(2, VPG), Ok(0));
    assert_eq!(ba_gates_slot(3, VPG), Ok(1));
}

// ── the selection grammar ──

/// The prefill shapes round 13 measured take the twin; the decode and
/// tail-chunk shapes do not. This is the table that decides whether promoting
/// the lever can cost a decode step, so it is written as the shapes themselves.
#[test]
fn the_guard_takes_the_prefill_shapes_and_declines_the_decode_ones() {
    let twin = KernelHandle(1);
    let parent = KernelHandle(2);
    let pick = |m| ba_gates_pick(true, parent, twin, m, N, K, K, H100_SMS);

    for m in [1168u32, 4576, 8176] {
        let p = pick(m);
        assert!(p.twin, "M={m} fills 132 SMs at one CTA per token");
        assert_eq!(p.kernel.0, twin.0);
        assert_eq!(p.reject, None);
    }
    for m in [1u32, 16, 17, 25, 263] {
        let p = pick(m);
        assert!(!p.twin, "M={m} must stay on the parent");
        assert_eq!(p.kernel.0, parent.0);
        assert_eq!(
            p.reject,
            Some("too few tokens to fill the device at one CTA per token")
        );
    }
    assert_eq!(
        ba_gates_min_tokens(H100_SMS),
        264,
        "two CTAs per SM on an H100"
    );
    assert!(pick(264).twin, "the floor itself is accepted");
}

/// Every other guard, each naming itself. A perf path that declines silently
/// measures as "no effect" — PR #296's lesson, and the reason these strings
/// exist at all.
#[test]
fn every_guard_names_itself() {
    let twin = KernelHandle(1);
    let parent = KernelHandle(2);
    let m = 4576u32;

    assert_eq!(
        ba_gates_pick(false, parent, twin, m, N, K, K, H100_SMS).reject,
        Some("not requested")
    );
    assert_eq!(
        ba_gates_pick(true, parent, KernelHandle(0), m, N, K, K, H100_SMS).reject,
        Some("kernel absent from this image (kernels/hopper only)")
    );
    assert_eq!(
        ba_gates_pick(true, parent, twin, m, 0, K, K, H100_SMS).reject,
        Some("empty BA projection")
    );
    assert_eq!(
        ba_gates_pick(true, parent, twin, m, N, 5124, 5124, H100_SMS).reject,
        Some("K is not a multiple of 8 (the uint4 K sweep would drop a tail)")
    );
    assert_eq!(
        ba_gates_pick(true, parent, twin, m, N, K, K - 8, H100_SMS).reject,
        Some("K_stride < K: the activation row is shorter than the reduction")
    );
    // A rejected pick is EXACTLY the pre-#928 launch: the parent's handle.
    let rejected = ba_gates_pick(false, parent, twin, m, N, K, K, H100_SMS);
    assert_eq!(rejected.kernel.0, parent.0);
    assert!(!rejected.twin);
}

/// A padded activation row (`K_stride > K`) is fine — the SSM prefill hands the
/// normed hidden state with `K_stride == K` today, but the twin indexes rows by
/// `K_stride` and reduces over `K`, exactly as the parent does.
#[test]
fn a_padded_activation_row_is_accepted() {
    let p = ba_gates_pick(
        true,
        KernelHandle(2),
        KernelHandle(1),
        4576,
        N,
        K,
        K + 128,
        H100_SMS,
    );
    assert!(p.twin);
}

/// The guard scales with the device rather than with a baked 132: on a 48-SM
/// GB10 the same M=96 shape would clear it. Inert there (the kernel is not in
/// that image), but the threshold must not be an H100 constant in disguise.
#[test]
fn the_guard_scales_with_the_device() {
    assert_eq!(ba_gates_min_tokens(48), 96);
    assert_eq!(ba_gates_min_tokens(132), 264);
    assert_eq!(
        ba_gates_min_tokens(0),
        MIN_CTAS_PER_SM,
        "never divide by nothing"
    );
}

/// The block shape is the reduction order's other half: 256 threads, 64 lanes
/// per output, 4 outputs, 8 warps. The kernel carries the same four constants.
#[test]
fn the_block_shape_matches_the_kernel() {
    assert_eq!(BA_GATES_BLOCK, 256);
    assert_eq!(BA_GATES_LANES, 64);
    assert_eq!(BA_GATES_OUTS, 4);
    assert_eq!(BA_GATES_WARPS, 8);
    assert_eq!(BA_GATES_OUTS * BA_GATES_LANES, BA_GATES_BLOCK);
    assert_eq!(
        BA_GATES_LANES / 32,
        2,
        "two warps per output is what makes the cross-warp sum a PAIR"
    );
}

// ── the route line's once-flags (H100 round 15 §3.2) ──

/// Replay a serve's verdict order through the once-set and collect the lines it
/// would actually say. One `bool` per slot — the CPU model of `ba_gates_log`'s
/// `[Once; BA_GATES_LOG_SLOTS]`.
fn replay(verdicts: &[(bool, BaGatesPick)]) -> Vec<BaGatesLogSlot> {
    let mut said = [false; BA_GATES_LOG_SLOTS];
    let mut lines = Vec::new();
    for (requested, pick) in verdicts {
        let Some(slot) = ba_gates_log_slot(pick, *requested) else {
            continue;
        };
        let idx = match slot {
            BaGatesLogSlot::Twin => BA_GATES_LOG_SLOTS - 1,
            BaGatesLogSlot::Reject(i) => i,
        };
        if !said[idx] {
            said[idx] = true;
            lines.push(slot);
        }
    }
    lines
}

/// THE round-15 defect. On five of six H100 serve cells the log carried
/// `the Hopper twin is NOT running at M=27` and nothing else for the life of
/// the process, while nsys showed `dense_gemm_ba_gates_prefill_hopper` running
/// 48x per prefill: the smoke test's 27-token request tripped a `Once` that
/// both branches shared. With a flag per branch the SAME call order says both
/// lines, each once.
#[test]
fn the_smoke_tests_m27_can_no_longer_silence_the_twins_line() {
    let twin = KernelHandle(1);
    let parent = KernelHandle(2);
    let pick = |m| ba_gates_pick(true, parent, twin, m, N, K, K, H100_SMS);

    // Smoke test (M=27) first, then 48 prefill layers at M=1168 and 48 more at
    // M=4576 — the A15 cell's order.
    let mut order = vec![(true, pick(27))];
    order.extend((0..48).map(|_| (true, pick(1168))));
    order.extend((0..48).map(|_| (true, pick(4576))));

    let too_few = BA_GATES_REJECTS
        .iter()
        .position(|r| *r == BA_GATES_TOO_FEW_TOKENS)
        .expect("the floor is in the slot table");
    assert_eq!(
        replay(&order),
        vec![BaGatesLogSlot::Reject(too_few), BaGatesLogSlot::Twin],
        "both branches say their line exactly once, in arrival order"
    );

    // And the nsys leg's order (warmup prefill first) says the same two lines.
    let mut reversed = vec![(true, pick(1168))];
    reversed.push((true, pick(27)));
    assert_eq!(
        replay(&reversed),
        vec![BaGatesLogSlot::Twin, BaGatesLogSlot::Reject(too_few)],
    );
}

/// Every guard the reject function can return owns a DISTINCT slot, so no two
/// refusals can silence each other either. Driven through `ba_gates_pick` so a
/// new guard string that never reached [`BA_GATES_REJECTS`] fails here.
#[test]
fn every_reject_reason_has_its_own_log_slot() {
    let twin = KernelHandle(1);
    let parent = KernelHandle(2);
    let m = 4576u32;
    let reachable = [
        ba_gates_pick(true, parent, KernelHandle(0), m, N, K, K, H100_SMS),
        ba_gates_pick(true, parent, twin, m, 0, K, K, H100_SMS),
        ba_gates_pick(true, parent, twin, m, N, 5124, 5124, H100_SMS),
        ba_gates_pick(true, parent, twin, m, N, K, K - 8, H100_SMS),
        ba_gates_pick(true, parent, twin, 27, N, K, K, H100_SMS),
    ];
    let mut slots: Vec<BaGatesLogSlot> = reachable
        .iter()
        .map(|p| {
            ba_gates_log_slot(p, true).unwrap_or_else(|| {
                panic!(
                    "guard {:?} has no log slot — add it to BA_GATES_REJECTS",
                    p.reject
                )
            })
        })
        .collect();
    let before = slots.len();
    slots.sort_by_key(|s| match s {
        BaGatesLogSlot::Twin => usize::MAX,
        BaGatesLogSlot::Reject(i) => *i,
    });
    slots.dedup();
    assert_eq!(
        slots.len(),
        before,
        "two guards share a once-flag: {slots:?}"
    );
    assert_eq!(BA_GATES_LOG_SLOTS, BA_GATES_REJECTS.len() + 1);
}

/// The lever being OFF is an answer, not a refusal: no line, on any target.
/// Every non-Hopper build takes this branch on every launch.
#[test]
fn a_lever_that_was_never_requested_says_nothing() {
    let pick = ba_gates_pick(
        false,
        KernelHandle(2),
        KernelHandle(1),
        4576,
        N,
        K,
        K,
        H100_SMS,
    );
    assert_eq!(pick.reject, Some("not requested"));
    assert_eq!(ba_gates_log_slot(&pick, false), None);
}
