// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: tests of the device acceptance hook (lane L12) against the host slow tail.
//!
//! The device is the CPU reference of the kernel (`spark_model::port::spec_accept::reference`), fed
//! the same BF16 rows the slow tail copies to the host. Greedy spans must equal
//! `pick_positions_from_host` bit for bit in every pipeline regime the planner accepts (penalties,
//! the `min_tokens` EOS mask, the A4 floor, post-close and tool masks, a span crossing `</think>`);
//! sampled spans must match its distribution (two-sample chi-square); every refused regime must name
//! its reason; nothing may mutate the sequence.

use super::device_accept::{picks_via, plan, walk};
use super::pick_positions::pick_positions_from_host;
use crate::grammar::tests::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::scheduler::logit_processors::{LogitsContext, SamplingLevers};
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;
use spark_model::port::spec_accept::metrics::{self, Fallback};
use spark_model::port::spec_accept::{JobResult, reference, status};
use spark_runtime::gpu::DevicePtr;

const VOCAB: usize = 160;
const THINK_END: u32 = 150;
const THINK_START: u32 = 151;
const TOOL_OPEN: u32 = 152;
const TOOL_CLOSE: u32 = 153;
const EOS: u32 = 154;

fn seq() -> ActiveSeq {
    let (mut a, _rx) = test_seq(vec![3, 9, 3, 40, 9, 3], 500, None, 20);
    a.finished = false;
    a.min_tokens = 0;
    a.eos_tokens = vec![EOS];
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.tool_call_start_token = Some(TOOL_OPEN);
    a
}

/// Deterministic rows with many exact ties (eight values on a 0.25 grid in [-1, 0.75]) and the named
/// hot ids, so the first-index-wins tie break is exercised on every row.
fn rows(k: usize, seed: u64, hot: &[(usize, u32, f32)]) -> Vec<Vec<f32>> {
    (0..k)
        .map(|r| {
            let mut v: Vec<f32> = (0..VOCAB)
                .map(|i| {
                    let h =
                        ((i as u64) ^ (seed + r as u64 * 7919)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    ((h >> 61) as f32) * 0.25 - 1.0
                })
                .collect();
            for &(row, id, val) in hot {
                if row == r {
                    v[id as usize] = val;
                }
            }
            v
        })
        .collect()
}

/// One row of distinct BF16-exact values (-2 + j/64, j a permutation of the ids), for the sampled
/// tests: the host's top-k keeps an arbitrary subset of an exact tie, so ties must not occur there.
fn distinct_row() -> Vec<Vec<f32>> {
    vec![
        (0..VOCAB)
            .map(|i| -2.0 + ((i * 37) % VOCAB) as f32 / 64.0)
            .collect(),
    ]
}

fn bf16_rows(rows: &[Vec<f32>]) -> Vec<u8> {
    rows.iter()
        .flat_map(|r| {
            r.iter()
                .flat_map(|&v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        })
        .collect()
}

fn with_ctx<R>(
    sampling: SamplingLevers,
    mid_word: Option<std::sync::Arc<[bool]>>,
    f: impl FnOnce(&LogitsContext) -> R,
) -> R {
    let scratch = crate::scheduler::sched_ctx::DecodeScratch::default();
    let dumps = crate::scheduler::dumps::RunDumps::default();
    let ctx = LogitsContext {
        scratch: &scratch,
        dumps: &dumps,
        stats: std::sync::Arc::new(crate::scheduler::spec_stats::SpecStats::new()),
        watchdog: crate::scheduler::helpers::WatchdogParams::default(),
        boundary_mask: None,
        mid_word_mask: mid_word,
        sampling,
        timing: std::sync::Arc::default(),
        think_end_token: Some(THINK_END),
        think_start_token: Some(THINK_START),
        tool_call_start_token: Some(TOOL_OPEN),
        tool_call_end_token: Some(TOOL_CLOSE),
        verify_pos: 0,
    };
    f(&ctx)
}

fn flags(a: &ActiveSeq) -> (bool, bool, bool) {
    (a.inside_thinking, a.think_ended, a.think_just_ended)
}

/// Host slow tail and device hook over the same rows; asserts the sequence is left untouched.
fn both(a: &mut ActiveSeq, rows: &[Vec<f32>], ctx: &LogitsContext) -> (Vec<u32>, Vec<u32>) {
    let buf = bf16_rows(rows);
    let k = rows.len();
    let before = (
        flags(a),
        a.output_tokens.clone(),
        a.consecutive_confident,
        a.sentence_defer_count,
    );
    let dev = picks_via(k, VOCAB, DevicePtr::NULL, a, ctx, |req| {
        Ok(Some(reference::run(&buf, req)))
    })
    .expect("the planner accepts this row");
    let after = (
        flags(a),
        a.output_tokens.clone(),
        a.consecutive_confident,
        a.sentence_defer_count,
    );
    assert_eq!(
        before, after,
        "the device hook must not mutate the sequence"
    );
    let host = pick_positions_from_host(&buf, VOCAB, 2, k, a, ctx);
    (host, dev)
}

#[test]
fn greedy_plain_rows_match_the_host_tail() {
    for seed in 0..20 {
        let mut a = seq();
        let r = rows(4, seed, &[]);
        let (host, dev) = with_ctx(SamplingLevers::default(), None, |c| both(&mut a, &r, c));
        assert_eq!(host, dev, "seed {seed}");
    }
}

#[test]
fn greedy_penalties_eos_mask_and_suppression_match_the_host_tail() {
    for seed in 0..20 {
        let mut a = seq();
        a.repetition_penalty = 1.15;
        a.repetition_penalty_window = 0;
        a.presence_penalty = 0.4;
        a.frequency_penalty = 0.2;
        a.min_tokens = 8; // 6 emitted: positions 0 and 1 mask EOS, 2 and 3 do not
        a.suppress_tool_call = true;
        let hot = [
            (0, 3, 5.0),
            (0, EOS, 6.0),
            (1, EOS, 6.0),
            (2, EOS, 6.0),
            (3, 9, 4.5),
            (3, TOOL_OPEN, 9.0),
        ];
        let r = rows(4, seed, &hot);
        let (host, dev) = with_ctx(SamplingLevers::default(), None, |c| both(&mut a, &r, c));
        assert_eq!(host, dev, "seed {seed}");
        assert_ne!(dev[0], EOS);
        assert_eq!(dev[2], EOS, "the min_tokens floor ends at position 2");
    }
}

#[test]
fn greedy_span_crossing_think_end_matches_the_host_tail() {
    for seed in 0..20 {
        let mut a = seq();
        a.inside_thinking = true;
        a.enable_thinking = true;
        a.thinking_tokens = 3; // under the A4 floor: `</think>` carries -8 in regime 0 only
        a.suppress_tool_call = true;
        let hot = [
            (0, THINK_END, 20.0),
            (1, THINK_START, 9.0),
            (1, TOOL_OPEN, 8.0),
            (2, THINK_END, 9.0),
            (3, TOOL_OPEN, 9.0),
        ];
        let r = rows(4, seed, &hot);
        let (host, dev) = with_ctx(SamplingLevers::default(), None, |c| both(&mut a, &r, c));
        assert_eq!(host, dev, "seed {seed}");
        assert_eq!(dev[0], THINK_END);
        assert_ne!(dev[1], THINK_START, "post-close mask after the crossing");
        assert!(a.inside_thinking, "state restored");
    }
}

#[test]
fn greedy_thinking_without_crossing_and_mid_word_mask_match_the_host_tail() {
    let mut mask = vec![false; VOCAB];
    mask[3] = true; // the last emitted token (3) ends mid-word: `</think>` is masked
    let mask: std::sync::Arc<[bool]> = mask.into();
    for seed in 0..10 {
        let mut a = seq();
        a.inside_thinking = true;
        a.thinking_tokens = 100;
        let hot = [(0, THINK_END, 9.0), (1, TOOL_OPEN, 9.0)];
        let r = rows(3, seed, &hot);
        let (host, dev) = with_ctx(SamplingLevers::default(), Some(mask.clone()), |c| {
            both(&mut a, &r, c)
        });
        assert_eq!(host, dev, "seed {seed}");
        assert_ne!(dev[0], THINK_END);
        assert_ne!(dev[1], TOOL_OPEN);
    }
}

#[test]
fn greedy_after_think_closed_matches_the_host_tail() {
    for seed in 0..10 {
        let mut a = seq();
        a.think_ended = true;
        a.think_just_ended = true;
        let hot = [(0, THINK_START, 9.0), (1, THINK_END, 9.0)];
        let r = rows(2, seed, &hot);
        let (host, dev) = with_ctx(SamplingLevers::default(), None, |c| both(&mut a, &r, c));
        assert_eq!(host, dev, "seed {seed}");
    }
}

fn sampling() -> SamplingLevers {
    SamplingLevers {
        mtp_verify_sample: true,
        mtp_minp: true,
        ..SamplingLevers::default()
    }
}

#[test]
fn sampled_rows_match_the_host_distribution() {
    const TRIALS: u64 = 20_000;
    let buf = bf16_rows(&distinct_row());
    let (mut dev, mut host) = (vec![0u64; VOCAB], vec![0u64; VOCAB]);
    with_ctx(sampling(), None, |c| {
        for n in 0..TRIALS {
            let mut a = seq();
            a.temperature = 0.8;
            a.top_k = 12;
            a.top_p = 0.9;
            a.min_p = 0.02;
            a.presence_penalty = 0.3;
            a.seed = Some(1_000_000 + n);
            let d = picks_via(1, VOCAB, DevicePtr::NULL, &mut a, c, |req| {
                Ok(Some(reference::run(&buf, req)))
            })
            .expect("planned");
            dev[d[0] as usize] += 1;
            host[pick_positions_from_host(&buf, VOCAB, 2, 1, &mut a, c)[0] as usize] += 1;
        }
    });
    let (mut stat, mut bins) = (0.0f64, 0usize);
    let (mut pa, mut pb) = (0.0f64, 0.0f64);
    for (&x, &y) in dev.iter().zip(&host) {
        let (x, y) = (x as f64, y as f64);
        if x + y >= 10.0 {
            stat += (x - y).powi(2) / (x + y);
            bins += 1;
        } else {
            pa += x;
            pb += y;
        }
    }
    if pa + pb > 0.0 {
        stat += (pa - pb).powi(2) / (pa + pb);
        bins += 1;
    }
    let df = bins.saturating_sub(1).max(1) as f64;
    let a = 2.0 / (9.0 * df);
    let crit = df * (1.0 - a + 4.75 * a.sqrt()).powi(3);
    assert!(
        stat < crit,
        "device vs host: chi2 {stat:.1} over {df} df (crit {crit:.1})"
    );
    for i in 0..VOCAB {
        assert!(
            dev[i] == 0 || host[i] > 0 || dev[i] < 5,
            "id {i} drawn {} times only on the device",
            dev[i]
        );
    }
}

#[test]
fn seeded_sampled_rows_are_deterministic() {
    let r = rows(4, 5, &[]);
    let buf = bf16_rows(&r);
    let run = || {
        with_ctx(sampling(), None, |c| {
            let mut a = seq();
            a.temperature = 1.0;
            a.top_k = 20;
            a.seed = Some(42);
            picks_via(4, VOCAB, DevicePtr::NULL, &mut a, c, |req| {
                Ok(Some(reference::run(&buf, req)))
            })
            .unwrap()
        })
    };
    assert_eq!(run(), run());
}

fn refused(setup: impl FnOnce(&mut ActiveSeq), levers: SamplingLevers) -> Fallback {
    let mut a = seq();
    setup(&mut a);
    with_ctx(levers, None, |c| plan::build(VOCAB, 4, &mut a, c)).expect_err("the planner refuses")
}

fn required_tool_grammar() -> GrammarState {
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[130]).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    GrammarState::new(&compiled, engine.vocab_size()).unwrap()
}

#[test]
fn the_planner_names_every_refused_regime() {
    let d = SamplingLevers::default();
    assert_eq!(
        refused(|a| a.grammar_state = Some(required_tool_grammar()), d),
        Fallback::Grammar
    );
    assert_eq!(
        refused(|a| a.logit_bias = vec![(1, 2.0)], d),
        Fallback::LogitBias
    );
    assert_eq!(refused(|a| a.lz_penalty = 0.5, d), Fallback::LzDry);
    assert_eq!(refused(|a| a.dry_multiplier = 0.8, d), Fallback::LzDry);
    assert_eq!(
        refused(
            |a| {
                a.inside_thinking = true;
                a.force_end_thinking = true;
            },
            d
        ),
        Fallback::ThinkStage
    );
    assert_eq!(
        refused(
            |a| {
                a.inside_thinking = true;
                a.thinking_tokens = 400;
            },
            d
        ),
        Fallback::ThinkStage
    );
    assert_eq!(
        refused(
            |a| {
                a.inside_thinking = true;
                a.require_tool_call = true;
            },
            d
        ),
        Fallback::ToolPin
    );
    let ftz = SamplingLevers {
        force_temp_zero: true,
        ..d
    };
    assert_eq!(refused(|_| {}, ftz), Fallback::Diagnostic);
    assert_eq!(
        refused(
            |a| {
                a.temperature = 0.7;
                a.top_n_sigma = 1.0;
            },
            sampling()
        ),
        Fallback::SamplerParams
    );
    assert_eq!(
        refused(
            |a| {
                a.temperature = 0.7;
                a.top_k = 100;
            },
            sampling()
        ),
        Fallback::SamplerParams
    );
    let mut a = seq();
    let r = with_ctx(d, None, |c| plan::build(VOCAB, 0, &mut a, c));
    assert_eq!(r, Err(Fallback::Positions));
}

#[test]
fn the_planner_builds_both_regimes_inside_thinking() {
    let mut a = seq();
    a.inside_thinking = true;
    a.thinking_tokens = 3;
    let p = with_ctx(SamplingLevers::default(), None, |c| {
        plan::build(VOCAB, 4, &mut a, c)
    })
    .unwrap();
    assert_eq!(p.regimes.len(), 2);
    let tool0 = p.regimes[0].iter().find(|e| e.token == TOOL_OPEN).unwrap();
    assert!(tool0.mask, "tool opener masked inside thinking");
    let end0 = p.regimes[0].iter().find(|e| e.token == THINK_END).unwrap();
    assert_eq!(
        (end0.n_bias, end0.bias[0]),
        (1, -8.0),
        "A4 floor in regime 0"
    );
    let end1 = p.regimes[1].iter().find(|e| e.token == THINK_END).unwrap();
    assert!(
        end1.mask && end1.n_bias == 0,
        "post-close mask, no A4 floor after the crossing"
    );
    assert!(
        p.regimes
            .iter()
            .all(|r| r.windows(2).all(|w| w[0].token < w[1].token))
    );
}

#[test]
fn the_walk_switches_regime_after_think_end_and_honors_fallbacks() {
    let ok = |pick| JobResult { pick, status: 0 };
    let res = [
        ok(1),
        ok(THINK_END),
        ok(2),
        ok(3),
        ok(9),
        ok(9),
        ok(7),
        ok(8),
    ];
    assert_eq!(
        walk(&res, 4, 2, VOCAB, Some(THINK_END)).unwrap(),
        vec![1, THINK_END, 7, 8]
    );
    assert_eq!(
        walk(&res[..4], 4, 1, VOCAB, Some(THINK_END)).unwrap(),
        vec![1, THINK_END, 2, 3]
    );
    let mut bad = res;
    bad[6].status = status::SUPPORT;
    assert_eq!(
        walk(&bad, 4, 2, VOCAB, Some(THINK_END)),
        Err(Fallback::Support)
    );
    // A fallback in a job the walk does not use is ignored.
    let mut unused = res;
    unused[4].status = status::EMPTY;
    assert!(walk(&unused, 4, 2, VOCAB, Some(THINK_END)).is_ok());
    assert_eq!(
        walk(&res[..3], 4, 1, VOCAB, None),
        Err(Fallback::DeviceError)
    );
    let oob = [ok(VOCAB as u32)];
    assert_eq!(walk(&oob, 1, 1, VOCAB, None), Err(Fallback::DeviceError));
}

#[test]
fn a_missing_or_failing_device_keeps_the_host_tail() {
    let before = metrics::fallbacks(Fallback::Unavailable);
    let mut a = seq();
    let none = with_ctx(SamplingLevers::default(), None, |c| {
        picks_via(2, VOCAB, DevicePtr::NULL, &mut a, c, |_| Ok(None))
    });
    assert!(none.is_none());
    assert!(metrics::fallbacks(Fallback::Unavailable) > before);
    let failed = with_ctx(SamplingLevers::default(), None, |c| {
        picks_via(2, VOCAB, DevicePtr::NULL, &mut a, c, |_| {
            Err(anyhow::anyhow!("launch failed"))
        })
    });
    assert!(failed.is_none());
}
