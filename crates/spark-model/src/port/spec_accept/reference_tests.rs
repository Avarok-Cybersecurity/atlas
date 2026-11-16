// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: CPU tests of the device acceptance model: the edit arithmetic against the host
//! penalty stage, the blob round trip, the support filters, the fallbacks, and chi-square tests over
//! 10^5 draws per configuration against both the exact distribution and the host sampler.

use super::reference::{self, Support};
use super::{AcceptMode, AcceptRequest, CAND_CAP, EditDesc, NO_DRAFT, SampleParams, status};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::{SamplingParams, apply_penalties_and_bias, sample_with_params_seeded};

pub(super) const DRAWS: usize = 100_000;

/// Deterministic logits: a smooth falloff plus 24-bit hash noise (distinct values, so the host's
/// arbitrary order among exact ties never matters), with a few masked ids.
pub(super) fn logits(vocab: usize, seed: u64, masked: &[usize]) -> Vec<f32> {
    let mut v: Vec<f32> = (0..vocab)
        .map(|i| {
            let h = (i as u64 ^ seed)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .rotate_left(17);
            let noise = (h >> 40) as f32 / 16_777_216.0;
            6.0 - (i as f32).sqrt() * 0.9 + noise * 3.0
        })
        .collect();
    for &m in masked {
        v[m] = f32::NEG_INFINITY;
    }
    v
}

fn host_params(p: &SampleParams, seed: u64) -> SamplingParams {
    SamplingParams {
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: 0.0,
        min_p: p.min_p,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: Some(seed),
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub(super) fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// Upper chi-square quantile by Wilson-Hilferty, at z standard deviations.
fn chi2_crit(df: usize, z: f64) -> f64 {
    let d = df as f64;
    let a = 2.0 / (9.0 * d);
    d * (1.0 - a + z * a.sqrt()).powi(3)
}

/// Goodness of fit of `counts` (over vocab ids) against `probs`; bins under 5 expected are pooled.
fn gof(counts: &[u64], probs: &[f64], n: usize) -> (f64, usize) {
    let (mut stat, mut bins) = (0.0, 0usize);
    let (mut pool_o, mut pool_e) = (0.0, 0.0);
    for (c, p) in counts.iter().zip(probs) {
        let e = p * n as f64;
        if e >= 5.0 {
            stat += (*c as f64 - e).powi(2) / e;
            bins += 1;
        } else {
            pool_o += *c as f64;
            pool_e += e;
        }
    }
    if pool_e >= 5.0 {
        stat += (pool_o - pool_e).powi(2) / pool_e;
        bins += 1;
    } else {
        assert!(
            pool_o <= 5.0 + 6.0 * pool_e.sqrt().max(1.0),
            "pooled tail drew {pool_o}"
        );
    }
    (stat, bins.saturating_sub(1).max(1))
}

/// Two-sample homogeneity of equal-size samples; bins under 10 combined are pooled.
fn two_sample(a: &[u64], b: &[u64]) -> (f64, usize) {
    let (mut stat, mut bins) = (0.0, 0usize);
    let (mut pa, mut pb) = (0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
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
    (stat, bins.saturating_sub(1).max(1))
}

fn exact_probs(s: &Support, vocab: usize) -> Vec<f64> {
    let total: f64 = s.w.iter().map(|&w| w as f64).sum();
    let mut p = vec![0.0; vocab];
    for (&i, &w) in s.ids.iter().zip(&s.w) {
        p[i as usize] = w as f64 / total;
    }
    p
}

/// 10^5 reference draws (fresh key each) against the exact distribution and against 10^5 host
/// sampler draws, for one configuration; `draft` exercises the accept/residual form.
fn check_config(vocab: usize, row: &[f32], p: SampleParams, draft: u32) {
    let s = reference::support(row, &p).expect("support closes");
    let probs = exact_probs(&s, vocab);
    let mut dev = vec![0u64; vocab];
    let mut host = vec![0u64; vocab];
    let mut accepted = 0u64;
    let bytes = f32_bytes(row);
    for n in 0..DRAWS {
        let r = reference::sample_row(row, &p, draft, 0xA5A5_0000 + n as u64, 7);
        assert_eq!(r.status & status::FALLBACK, 0);
        dev[r.pick as usize] += 1;
        if r.status & status::ACCEPTED != 0 {
            assert_eq!(r.pick, draft, "an accepted job returns its draft");
            accepted += 1;
        }
        let h = sample_with_params_seeded(&bytes, &host_params(&p, n as u64), &[], Some(n as u64));
        host[h as usize] += 1;
    }
    for (i, &c) in dev.iter().enumerate() {
        assert!(
            c == 0 || probs[i] > 0.0,
            "drew id {i} outside the support ({p:?})"
        );
    }
    let z = 4.75; // one-sided 1e-6
    let (stat, df) = gof(&dev, &probs, DRAWS);
    assert!(
        stat < chi2_crit(df, z),
        "reference vs exact: chi2 {stat:.1} df {df} ({p:?}, draft {draft})"
    );
    let (stat, df) = two_sample(&dev, &host);
    assert!(
        stat < chi2_crit(df, z),
        "reference vs host sampler: chi2 {stat:.1} df {df} ({p:?}, draft {draft})"
    );
    if draft != NO_DRAFT {
        let pd = probs[draft as usize];
        let sd = (pd * (1.0 - pd) / DRAWS as f64).sqrt().max(1e-9);
        let rate = accepted as f64 / DRAWS as f64;
        assert!(
            (rate - pd).abs() < 6.0 * sd + 1e-4,
            "acceptance {rate} vs p(d) {pd}"
        );
        assert_eq!(
            dev[draft as usize], accepted,
            "the residual never returns the draft"
        );
    }
}

#[test]
fn sampled_top_k_top_p_matches_host_distribution() {
    let v = logits(128, 3, &[5, 40]);
    let p = SampleParams {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.0,
    };
    check_config(128, &v, p, NO_DRAFT);
    check_config(128, &v, p, 2);
}

#[test]
fn sampled_full_vocabulary_matches_host_distribution() {
    let v = logits(128, 11, &[0]);
    let p = SampleParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
    };
    check_config(128, &v, p, NO_DRAFT);
    check_config(128, &v, p, 1);
}

#[test]
fn sampled_uncapped_top_p_and_min_p_match_host_distribution() {
    let v = logits(128, 29, &[]);
    let p = SampleParams {
        temperature: 0.6,
        top_k: 0,
        top_p: 0.9,
        min_p: 0.0,
    };
    check_config(128, &v, p, NO_DRAFT);
    let p = SampleParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.1,
    };
    check_config(128, &v, p, 3);
}

#[test]
fn sampled_all_filters_match_host_distribution() {
    let v = logits(128, 41, &[1, 2]);
    let p = SampleParams {
        temperature: 0.8,
        top_k: 5,
        top_p: 0.8,
        min_p: 0.05,
    };
    check_config(128, &v, p, NO_DRAFT);
    check_config(128, &v, p, 0);
}

#[test]
fn support_falls_back_when_it_cannot_close() {
    let flat = vec![1.0f32; 256];
    let wide_p = SampleParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 0.999,
        min_p: 0.0,
    };
    assert_eq!(reference::support(&flat, &wide_p), Err(status::SUPPORT));
    let wide_min_p = SampleParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.5,
    };
    assert_eq!(reference::support(&flat, &wide_min_p), Err(status::SUPPORT));
    let big_k = SampleParams {
        temperature: 1.0,
        top_k: CAND_CAP as u32 + 1,
        top_p: 1.0,
        min_p: 0.0,
    };
    assert_eq!(reference::support(&flat, &big_k), Err(status::SUPPORT));
    // A top-k at or above the finite count is no truncation at all.
    let all_k = SampleParams {
        temperature: 1.0,
        top_k: 1000,
        top_p: 1.0,
        min_p: 0.0,
    };
    assert_eq!(reference::support(&flat, &all_k).unwrap().ids.len(), 256);
    let dead = vec![f32::NEG_INFINITY; 8];
    assert_eq!(reference::support(&dead, &all_k), Err(status::EMPTY));
    // min_p above 1 retains nothing: the host returns id 0 by accident, the device defers.
    let silly = SampleParams {
        temperature: 1.0,
        top_k: 4,
        top_p: 1.0,
        min_p: 1.5,
    };
    assert_eq!(
        reference::support(&logits(32, 1, &[]), &silly),
        Err(status::SUPPORT)
    );
}

#[test]
fn ties_at_the_top_k_boundary_take_the_lowest_ids() {
    let mut v = vec![0.0f32; 16];
    for i in [3, 7, 9, 12] {
        v[i] = 2.0;
    }
    v[1] = 5.0;
    let p = SampleParams {
        temperature: 1.0,
        top_k: 3,
        top_p: 1.0,
        min_p: 0.0,
    };
    let s = reference::support(&v, &p).unwrap();
    assert_eq!(s.ids, vec![1, 3, 7]);
}

#[test]
fn edits_reproduce_the_host_penalty_stage_bit_for_bit() {
    let vocab = 64;
    let base = logits(vocab, 5, &[]);
    let history: Vec<u32> = vec![3, 9, 3, 60, 9, 3, 17, 2];
    let rep = 1.3f32;
    let (freq, pres) = (0.25f32, 0.5f32);
    let mut host = base.clone();
    let mut hp = host_params(
        &SampleParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        },
        0,
    );
    hp.repetition_penalty = rep;
    hp.frequency_penalty = freq;
    hp.presence_penalty = pres;
    hp.logit_bias = vec![(9, -8.0), (17, 0.75), (9, 0.125)];
    apply_penalties_and_bias(&mut host, &hp, &history);

    let mut dev = base.clone();
    for tok in 0..vocab as u32 {
        let c = history.iter().filter(|&&t| t == tok).count() as u32;
        let biases: Vec<f32> = hp
            .logit_bias
            .iter()
            .filter(|(t, _)| *t == tok)
            .map(|&(_, b)| b)
            .collect();
        if c == 0 && biases.is_empty() {
            continue;
        }
        let mut e = EditDesc {
            token: tok,
            rep_count: c,
            ..Default::default()
        };
        if c > 0 {
            e.delta = Some(freq * c as f32 + pres);
        }
        for (j, b) in biases.iter().enumerate() {
            e.bias[j] = *b;
        }
        e.n_bias = biases.len() as u8;
        dev[tok as usize] = reference::apply_edit(dev[tok as usize], &e, 0, 0, rep);
    }
    for i in 0..vocab {
        assert_eq!(dev[i].to_bits(), host[i].to_bits(), "id {i}");
    }
}

#[test]
fn masks_sub12_and_eos_follow_the_position() {
    let e = EditDesc {
        token: 0,
        eos: true,
        ..Default::default()
    };
    assert_eq!(reference::apply_edit(1.5, &e, 0, 2, 1.0), f32::NEG_INFINITY);
    assert_eq!(reference::apply_edit(1.5, &e, 1, 2, 1.0), f32::NEG_INFINITY);
    assert_eq!(reference::apply_edit(1.5, &e, 2, 2, 1.0), 1.5);
    let s = EditDesc {
        token: 0,
        sub12: true,
        ..Default::default()
    };
    assert_eq!(reference::apply_edit(20.0, &s, 0, 0, 1.0), 8.0);
    let m = EditDesc {
        token: 0,
        mask: true,
        sub12: true,
        ..Default::default()
    };
    assert_eq!(
        reference::apply_edit(20.0, &m, 0, 0, 1.0),
        f32::NEG_INFINITY
    );
}

pub(super) fn request<'a>(
    vocab: usize,
    n_pos: usize,
    mode: AcceptMode,
    regimes: &'a [Vec<EditDesc>],
    drafts: &'a [u32],
) -> AcceptRequest<'a> {
    AcceptRequest {
        logits: DevicePtr::NULL,
        vocab,
        n_pos,
        drafts,
        mode,
        rng_key: 0x1234_5678_9ABC_DEF0,
        pos_base: (1u64 << 33) + 5,
        eos_until: 1,
        rep_penalty: 1.1,
        regimes,
    }
}
