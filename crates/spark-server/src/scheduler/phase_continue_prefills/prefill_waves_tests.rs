// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the VARLEN prefill wave planner and the per-stream chunk
//! geometry. Split from `prefill_waves.rs` for the 500-LoC cap; the #1002
//! round-13 replays roughly tripled the module.

use super::{WaveGeom, plan_prefill_waves, plan_stream_chunk, varlen_defer_pays};

fn g(chunk_start: usize, chunk_len: usize, is_last: bool) -> WaveGeom {
    WaveGeom {
        chunk_start,
        chunk_len,
        is_last,
    }
}

#[test]
fn flag_off_is_one_wave_with_every_stream_in_order() {
    // Byte-identical dispatch behaviour: the pre-wave scheduler made ONE
    // prefill_batch_chunk call with all streams, whatever their geometry
    // or total token count.
    let geoms = [g(0, 200, true), g(2048, 512, false), g(0, 4096, true)];
    assert_eq!(plan_prefill_waves(&geoms, false, 2048), vec![vec![0, 1, 2]]);
}

#[test]
fn empty_streams_no_waves() {
    assert!(plan_prefill_waves(&[], true, 2048).is_empty());
    assert!(plan_prefill_waves(&[], false, 2048).is_empty());
}

#[test]
fn ragged_chunk0_wave_packs_up_to_the_budget() {
    // Ten ~200-token fresh prompts against the 2048-token budget: the
    // first ten fit (Σ = 2000), the eleventh opens wave 2 — the measured
    // C=32 case (285 ms/prompt serial) becomes ⌈32/10⌉ dispatches.
    let geoms: Vec<WaveGeom> = (0..11).map(|_| g(0, 200, true)).collect();
    let waves = plan_prefill_waves(&geoms, true, 2048);
    assert_eq!(waves.len(), 2);
    assert_eq!(waves[0], (0..10).collect::<Vec<_>>());
    assert_eq!(waves[1], vec![10]);
}

#[test]
fn the_h100_c16_burst_packs_six_prompts_per_prefill_step() {
    // The #927 receipt, exactly: sixteen 1193-token prompts arriving
    // together against `--max-prefill-tokens 8192`. Measured behaviour
    // before this work was 32 prefill chunks for 32 requests — one prompt
    // per step, 220.8 ms each, budget never binding (2048 and 24576 both
    // changed nothing, ±0.4%). Six fit: 6 x 1193 = 7158 <= 8192, and a
    // seventh would be 8351.
    //
    // The tail pre-split in `run_batched_prefill_step` makes each prompt's
    // HEAD 1168 tokens (one KV block below the last boundary under 1193),
    // so seven heads is 8176 — still inside the budget. Both geometries
    // are pinned here because the planner sees whichever one the caller
    // built, and "how many prompts per step" is the whole claim.
    let whole: Vec<WaveGeom> = (0..16).map(|_| g(0, 1193, true)).collect();
    let waves = plan_prefill_waves(&whole, true, 8192);
    assert_eq!(waves.len(), 3, "16 prompts / 6 per wave = 3 waves");
    assert_eq!(waves[0].len(), 6);
    assert_eq!(waves[1].len(), 6);
    assert_eq!(waves[2].len(), 4);

    let heads: Vec<WaveGeom> = (0..16).map(|_| g(0, 1168, false)).collect();
    let waves = plan_prefill_waves(&heads, true, 8192);
    assert_eq!(waves[0].len(), 7, "1168-token heads pack 7 per step");

    // And the 25-token tails all share `chunk_start == 1168`, so the
    // planner puts every one of them in a SINGLE forward — against 32
    // standalone 25-token passes measured at 29.26 ms each (1170 µs/token,
    // 11.7% of prefill GPU time for 2.1% of the tokens).
    let tails: Vec<WaveGeom> = (0..16).map(|_| g(1168, 25, true)).collect();
    let waves = plan_prefill_waves(&tails, true, 8192);
    assert_eq!(waves, vec![(0..16).collect::<Vec<_>>()]);
}

#[test]
fn flag_off_still_admits_every_prompt_in_one_call() {
    // Control for the test above: with VARLEN off the planner must not
    // start splitting anything — the model-side dispatcher is what refuses
    // the batch, and it needs to see the same one-call shape it always did.
    let whole: Vec<WaveGeom> = (0..16).map(|_| g(0, 1193, true)).collect();
    assert_eq!(
        plan_prefill_waves(&whole, false, 8192),
        vec![(0..16).collect::<Vec<_>>()],
    );
}

#[test]
fn budget_cap_is_exact_not_off_by_one() {
    // 1024 + 1024 == cap exactly ⇒ same wave; +1 more opens a new one.
    let geoms = [g(0, 1024, true), g(0, 1024, true), g(0, 1, true)];
    let waves = plan_prefill_waves(&geoms, true, 2048);
    assert_eq!(waves, vec![vec![0, 1], vec![2]]);
}

#[test]
fn mixed_geometry_splits_into_compatible_waves() {
    // The model-side contract: chunk_start and is_last must match across
    // a batch (check_kernel_batched_eligible). A wave mixing them would
    // be rejected wholesale and every stream would fall back to serial —
    // the planner must never emit one.
    let geoms = [
        g(0, 200, true),     // fresh single-chunk
        g(0, 2048, false),   // fresh long prompt, chunk 0 of many
        g(0, 300, true),     // fresh single-chunk → wave of stream 0
        g(2048, 512, false), // mid-prefill continuation
        g(0, 250, true),     // fresh single-chunk → wave of stream 0
    ];
    let waves = plan_prefill_waves(&geoms, true, 2048);
    assert_eq!(waves, vec![vec![0, 2, 4], vec![1], vec![3]]);
    // Cross-check the invariant directly: uniform (chunk_start, is_last)
    // per wave, Σ ≤ cap for every multi-member wave.
    for wave in &waves {
        let head = geoms[wave[0]];
        let total: usize = wave.iter().map(|&i| geoms[i].chunk_len).sum();
        assert!(wave.len() == 1 || total <= 2048);
        for &i in wave {
            assert_eq!(geoms[i].chunk_start, head.chunk_start);
            assert_eq!(geoms[i].is_last, head.is_last);
        }
    }
}

#[test]
fn oversized_stream_gets_a_singleton_wave() {
    // chunk_len > cap must still dispatch (single-stream path); it must
    // not absorb siblings past the cap.
    let geoms = [g(0, 4096, true), g(0, 100, true)];
    let waves = plan_prefill_waves(&geoms, true, 2048);
    assert_eq!(waves, vec![vec![0], vec![1]]);
}

#[test]
fn every_stream_is_assigned_exactly_once() {
    let geoms: Vec<WaveGeom> = (0..37)
        .map(|i| g((i % 3) * 1024, 100 + i * 7, i % 2 == 0))
        .collect();
    let waves = plan_prefill_waves(&geoms, true, 1024);
    let mut seen = vec![0usize; geoms.len()];
    for wave in &waves {
        assert!(!wave.is_empty());
        for &i in wave {
            seen[i] += 1;
        }
    }
    assert!(
        seen.iter().all(|&c| c == 1),
        "each stream in exactly one wave"
    );
    // FIFO order preserved within each wave.
    for wave in &waves {
        assert!(wave.windows(2).all(|w| w[0] < w[1]));
    }
}

// ── #1002: the round-13 receipt, replayed over the planner ────────────

/// `prefill_chunk_dispatch`'s tail cut for a prompt of `total` tokens at
/// KV block size `bs` — one block below the last boundary under `total`.
/// Mirrors `Model::prefill_tail_cut` so these tests can drive the real
/// geometry function with no model. 1193 @ bs=16 -> 1168; 4593 -> 4576.
fn tail_cut(total: usize, bs: usize) -> Option<usize> {
    let cut = ((total - 1) / bs) * bs - bs;
    (cut > 0 && cut < total).then_some(cut)
}

/// Replay one stream's WHOLE chunk sequence through the geometry SSOT.
fn chunk_sequence(total: usize, bs: usize, budget: usize, varlen: bool) -> Vec<usize> {
    let mut offset = 0;
    let mut seq = Vec::new();
    while offset < total {
        let (len, is_last) = plan_stream_chunk(offset, total, budget, tail_cut(total, bs), varlen);
        assert!(len > 0, "a stream must always advance");
        seq.push(len);
        offset += len;
        if is_last {
            break;
        }
    }
    assert_eq!(offset, total, "the chunk sequence must cover the prompt");
    seq
}

#[test]
fn round13_short_shape_waves_are_7_7_2_heads_then_16_tails() {
    // THE round-13 cell-V shape: sixteen 1193-token prompts, block size
    // 16, `--max-prefill-tokens 8192`. The serve log's planner line for
    // the burst was `16 streams -> 3 wave(s), M per wave [50, 8176, 8176]`
    // — which is 2 already-split TAILS (2 x 25) plus 14 heads, because two
    // streams had arrived one tick earlier. Planned from a clean slate the
    // sixteen heads must pack 7 / 7 / 2 and the sixteen tails must share
    // ONE wave.
    let heads: Vec<WaveGeom> = (0..16)
        .map(|_| WaveGeom {
            chunk_start: 0,
            chunk_len: 1168,
            is_last: false,
        })
        .collect();
    let waves = plan_prefill_waves(&heads, true, 8192);
    assert_eq!(
        waves.iter().map(|w| w.len()).collect::<Vec<_>>(),
        vec![7, 7, 2],
        "7 x 1168 = 8176 <= 8192; an eighth would be 9344"
    );
    for w in &waves {
        let m: usize = w.iter().map(|&i| heads[i].chunk_len).sum();
        assert!(m <= 8192, "wave M {m} over the cap");
    }
    assert_eq!(
        waves
            .iter()
            .map(|w| w.iter().map(|&i| heads[i].chunk_len).sum::<usize>())
            .collect::<Vec<_>>(),
        vec![8176, 8176, 2336],
        "the nsys histogram: 8176 x2 and 2336 x1"
    );

    let tails: Vec<WaveGeom> = (0..16)
        .map(|_| WaveGeom {
            chunk_start: 1168,
            chunk_len: 25,
            is_last: true,
        })
        .collect();
    assert_eq!(
        plan_prefill_waves(&tails, true, 8192),
        vec![(0..16).collect::<Vec<_>>()],
    );
}

#[test]
fn chunk_zero_is_never_truncated_by_the_wave_budget() {
    // The hypothesis round 13 raised against the `50`-token wave: did the
    // leftover 16 tokens of an 8192 budget hand two streams a 25-token
    // CHUNK ZERO? It cannot. The budget the per-stream geometry sees is
    // `max_prefill_tokens`, and the WAVE cap is applied by the planner,
    // which opens a new wave rather than shrinking a member. (The `50` was
    // 2 x 25 TAILS from an earlier tick's streams — `[50, 8176, 8176]` is
    // 2 tails + 14 heads, and 2 x 25 + 14 x 1168 = 16402 exactly.)
    for n in 1..=16 {
        let geoms: Vec<WaveGeom> = (0..n)
            .map(|_| WaveGeom {
                chunk_start: 0,
                chunk_len: 1168,
                is_last: false,
            })
            .collect();
        for wave in plan_prefill_waves(&geoms, true, 8192) {
            for i in wave {
                assert_eq!(
                    geoms[i].chunk_len, 1168,
                    "the planner must never shrink a member's chunk_len"
                );
            }
        }
    }
    // And the geometry function itself: chunk 0 is the whole head, always.
    let (len, is_last) = plan_stream_chunk(0, 1193, 8192, tail_cut(1193, 16), true);
    assert_eq!((len, is_last), (1168, false));
}

#[test]
fn batched_geometry_equals_per_stream_geometry_on_the_round13_shapes() {
    // BF16 accumulation is not associative, so a stream must take the same
    // chunk sequence whoever it batches with. The VARLEN pre-split exists
    // to reproduce, in the scheduler, the split `prefill_chunk_dispatch`
    // performs inside a single per-stream call.
    assert_eq!(chunk_sequence(1193, 16, 8192, true), vec![1168, 25]);
    assert_eq!(chunk_sequence(4593, 16, 8192, true), vec![4576, 17]);
    // Flag OFF the scheduler hands the whole prompt over in one call and
    // the model splits it internally — same spans, one dispatch.
    assert_eq!(chunk_sequence(1193, 16, 8192, false), vec![1193]);
    assert_eq!(chunk_sequence(4593, 16, 8192, false), vec![4593]);
    // Every stream of the burst gets the identical sequence regardless of
    // wave membership: the geometry depends only on the prompt.
    let all: Vec<Vec<usize>> = (0..16)
        .map(|_| chunk_sequence(1193, 16, 8192, true))
        .collect();
    assert!(all.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn round13_long_shape_does_not_defer_because_no_two_heads_fit() {
    // 4593 pre-splits to 4576 + 17 and `2 x 4576 = 9152 > 8192`, so the
    // planner degenerates to one stream per wave — the measured
    // `16 streams -> 14 wave(s), M per wave [51, 4576 x13]`, which cost
    // +82.8% TTFT and -30.6% aggregate for zero batching. The admission
    // predicate must refuse the deferral outright.
    assert!(
        !varlen_defer_pays((0..16).map(|_| 4593), 8192),
        "no two 4593-token chunk-0s fit an 8192 wave"
    );
    assert!(
        !varlen_defer_pays((0..16).map(|_| 4576), 8192),
        "nor do two heads after the tail pre-split"
    );
    // The short shape still defers.
    assert!(varlen_defer_pays((0..16).map(|_| 1193), 8192));
    // And if it HAD deferred, this is the shape it would have got.
    let longs: Vec<WaveGeom> = (0..16)
        .map(|_| WaveGeom {
            chunk_start: 0,
            chunk_len: 4576,
            is_last: false,
        })
        .collect();
    assert_eq!(
        plan_prefill_waves(&longs, true, 8192).len(),
        16,
        "one stream per wave — not batching"
    );
}

#[test]
fn defer_pays_needs_two_candidates_and_respects_the_cap() {
    assert!(!varlen_defer_pays(std::iter::empty(), 8192));
    assert!(!varlen_defer_pays(std::iter::once(100), 8192), "alone");
    // Exactly the cap fits; one more token does not.
    assert!(varlen_defer_pays([4096, 4096].into_iter(), 8192));
    assert!(!varlen_defer_pays([4096, 4097].into_iter(), 8192));
    // The two SMALLEST are the pair that decides it.
    assert!(varlen_defer_pays([8000, 100, 90].into_iter(), 8192));
}

#[test]
fn replays_the_round13_cell_v_arrival_pattern_tick_by_tick() {
    // The arrival pattern recovered from the cell-V serve log
    // (`atlas-h100-best-V/serve.log`, 2026-09-11T18:54:30-31Z). Two
    // requests landed a tick ahead of the other fourteen, which is why the
    // logged wave line reads `[50, 8176, 8176]` and not `[8176, 8176,
    // 2336]`: the `50` is TWO 25-token TAILS, not a truncated chunk 0.
    //
    //   18:54:30.634  Varlen prefill waves: 2 streams  -> 1 wave(s)  [2336]
    //   18:54:31.262  Varlen prefill waves: 16 streams -> 3 wave(s)  [50, 8176, 8176]
    //   (next tick, via the mixed path)                             [350]
    //
    // and the nsys prefill histogram for the burst is exactly
    // `8176 x2, 2336 x1, 350 x1, 50 x1` = 19 088 = 16 x 1193.
    const HEAD: usize = 1168;
    const TAIL: usize = 25;
    let head = WaveGeom {
        chunk_start: 0,
        chunk_len: HEAD,
        is_last: false,
    };
    let tail = WaveGeom {
        chunk_start: HEAD,
        chunk_len: TAIL,
        is_last: true,
    };
    let m = |geoms: &[WaveGeom], waves: &[Vec<usize>]| -> Vec<usize> {
        waves
            .iter()
            .map(|w| w.iter().map(|&i| geoms[i].chunk_len).sum())
            .collect()
    };

    // Tick 1 — the two early arrivals, both fresh.
    let t1 = vec![head; 2];
    let w1 = plan_prefill_waves(&t1, true, 8192);
    assert_eq!(m(&t1, &w1), vec![2336]);

    // Tick 2 — those two are now at the tail; fourteen fresh join them.
    let mut t2 = vec![tail; 2];
    t2.extend(std::iter::repeat_n(head, 14));
    let w2 = plan_prefill_waves(&t2, true, 8192);
    assert_eq!(m(&t2, &w2), vec![50, 8176, 8176], "the logged wave line");
    assert_eq!(w2[0], vec![0, 1], "the 50 is the two TAILS");

    // Tick 3 — the fourteen reach their tails and batch into one forward.
    let t3 = vec![tail; 14];
    let w3 = plan_prefill_waves(&t3, true, 8192);
    assert_eq!(m(&t3, &w3), vec![350]);

    // Every prompt token accounted for, once: the nsys histogram.
    let total: usize = m(&t1, &w1).iter().sum::<usize>()
        + m(&t2, &w2).iter().sum::<usize>()
        + m(&t3, &w3).iter().sum::<usize>();
    assert_eq!(total, 16 * 1193);

    // And no wave ever mixes geometries or exceeds the cap.
    for (geoms, waves) in [(&t1, &w1), (&t2, &w2), (&t3, &w3)] {
        for w in waves {
            let head_geom = geoms[w[0]];
            assert!(w.iter().map(|&i| geoms[i].chunk_len).sum::<usize>() <= 8192);
            for &i in w {
                assert_eq!(geoms[i].chunk_start, head_geom.chunk_start);
                assert_eq!(geoms[i].is_last, head_geom.is_last);
            }
        }
    }
}
