// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: CPU tests of the device acceptance model's greedy path, the blob round trip, request validation, the shared-memory budget and the uniform and order-key helpers.

use super::reference;
use super::reference_tests::{DRAWS, bf16_bytes, logits, request};
use super::{AcceptMode, EditDesc, NO_DRAFT, SampleParams, blob, status, validate};

#[test]
fn greedy_jobs_are_the_host_argmax_with_first_index_ties() {
    let vocab = 32;
    let mut rows = Vec::new();
    for pos in 0..3 {
        let mut v = logits(vocab, pos, &[]);
        v[4] = 9.0;
        v[20] = 9.0;
        v[11] = f32::NAN;
        rows.extend(bf16_bytes(&v));
    }
    let regimes = vec![
        vec![EditDesc {
            token: 4,
            mask: true,
            ..Default::default()
        }],
        vec![EditDesc {
            token: 30,
            eos: true,
            ..Default::default()
        }],
    ];
    let req = request(vocab, 3, AcceptMode::Greedy, &regimes, &[]);
    let out = reference::run(&rows, &req);
    assert_eq!(out.len(), 6);
    for r in &out[..3] {
        assert_eq!(r.pick, 20, "id 4 masked in regime 0");
        assert_ne!(r.status & status::NONFINITE, 0);
    }
    for r in &out[3..] {
        assert_eq!(r.pick, 4, "first of the tied maxima");
    }
}

#[test]
fn blob_round_trips_every_field() {
    let regimes = vec![
        vec![
            EditDesc {
                token: 1,
                mask: true,
                rep_count: 3,
                ..Default::default()
            },
            EditDesc {
                token: 7,
                eos: true,
                sub12: true,
                delta: Some(0.375),
                bias: [-8.0, 0.5],
                n_bias: 2,
                ..Default::default()
            },
        ],
        vec![EditDesc {
            token: 2,
            bias: [1.25, 0.0],
            n_bias: 1,
            ..Default::default()
        }],
    ];
    let drafts = [5u32, NO_DRAFT];
    let mode = AcceptMode::Sample(SampleParams {
        temperature: 0.6,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.01,
    });
    let req = request(16, 2, mode, &regimes, &drafts);
    validate(&req).unwrap();
    let w = blob::encode(&req);
    assert_eq!(w[blob::h::NEDIT0], 2);
    assert_eq!(w[blob::h::NEDIT1], 1);
    assert_eq!(w[blob::DRAFTS_AT], 5);
    assert_eq!(w[blob::DRAFTS_AT + 1], NO_DRAFT);
    assert_eq!(
        u64::from(w[blob::h::POS_LO]) | (u64::from(w[blob::h::POS_HI]) << 32),
        req.pos_base
    );
    let all: Vec<EditDesc> = regimes.iter().flatten().copied().collect();
    for (j, d) in all.iter().enumerate() {
        let at = blob::DESC_AT + j * blob::DESC_WORDS;
        assert_eq!(&blob::desc_from_words(&w[at..at + blob::DESC_WORDS]), d);
    }
    assert!(w.len() <= blob::capacity_words());
}

#[test]
fn validate_rejects_what_the_kernel_cannot_read() {
    let ok = vec![vec![EditDesc {
        token: 1,
        ..Default::default()
    }]];
    assert!(validate(&request(16, 1, AcceptMode::Greedy, &ok, &[])).is_ok());
    let unsorted = vec![vec![
        EditDesc {
            token: 3,
            ..Default::default()
        },
        EditDesc {
            token: 3,
            ..Default::default()
        },
    ]];
    assert!(validate(&request(16, 1, AcceptMode::Greedy, &unsorted, &[])).is_err());
    let oob = vec![vec![EditDesc {
        token: 16,
        ..Default::default()
    }]];
    assert!(validate(&request(16, 1, AcceptMode::Greedy, &oob, &[])).is_err());
    assert!(validate(&request(16, 0, AcceptMode::Greedy, &ok, &[])).is_err());
    assert!(
        validate(&request(
            16,
            super::MAX_POS + 1,
            AcceptMode::Greedy,
            &ok,
            &[]
        ))
        .is_err()
    );
    assert!(validate(&request(16, 2, AcceptMode::Greedy, &ok, &[1])).is_err());
    let cold = AcceptMode::Sample(SampleParams {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
    });
    assert!(validate(&request(16, 1, cold, &ok, &[])).is_err());
}

#[test]
fn shared_memory_budget_fits_the_qwen_vocabulary() {
    let cap = super::max_edits(248_320);
    assert!(cap >= 1024, "cap {cap}");
    assert!(super::dyn_smem_bytes(248_320, cap) + super::STATIC_SMEM_RESERVE <= super::SMEM_LIMIT);
    assert_eq!(super::max_edits(400_000), 0);
}

#[test]
fn uniform_is_a_pure_function_in_the_unit_interval() {
    let mut sum = 0.0f64;
    for n in 0..DRAWS as u64 {
        let u = reference::uniform(42, n, reference::PURPOSE_SAMPLE);
        assert!((0.0..1.0).contains(&u));
        sum += u as f64;
    }
    let mean = sum / DRAWS as f64;
    assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
    assert_eq!(reference::uniform(7, 9, 1), reference::uniform(7, 9, 1));
    assert_ne!(reference::uniform(7, 9, 1), reference::uniform(7, 9, 2));
    assert_ne!(reference::uniform(7, 9, 1), reference::uniform(7, 10, 1));
}

#[test]
fn order_key_is_monotone() {
    let xs = [
        f32::NEG_INFINITY,
        -3.5,
        -1.0,
        -0.0,
        0.0,
        1e-30,
        0.5,
        2.0,
        f32::INFINITY,
    ];
    for w in xs.windows(2) {
        assert!(
            reference::order_key(w[0]) < reference::order_key(w[1]),
            "{} < {}",
            w[0],
            w[1]
        );
    }
}
