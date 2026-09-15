// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Engram against the golden, one stage at a time, across prefill and both decode steps.

use super::super::{Golden, GoldenTensor, checksum, fixed_int};
use super::*;

/// Sample elementwise at `tol`, and the whole-tensor checksum against a bound scaled by the
/// index-weighted magnitude, so bf16 last-bit differences from accumulation order pass while a
/// wrong element fails.
#[track_caller]
fn check(what: &str, got: &[f64], g: &GoldenTensor, tol: f64, ck_rel: f64) {
    assert_eq!(got.len(), g.n, "{what}: numel");
    let sample: Vec<f64> = got.iter().step_by(g.stride).copied().collect();
    super::super::assert_close(what, &sample, &g.data, tol);
    let ck = checksum(got.iter().copied());
    let mag: f64 = got.iter().enumerate().map(|(i, v)| v.abs() * (i as f64 + 1.0)).sum();
    let bound = ck_rel * mag.max(1.0);
    assert!(
        (ck - g.ck).abs() <= bound,
        "{what}: checksum {ck} vs golden {} (|diff| {:e} > bound {:e})",
        g.ck,
        (ck - g.ck).abs(),
        bound
    );
}

fn bf16_tol(g: &GoldenTensor) -> f64 {
    let m = g.data.iter().fold(0f64, |a, v| a.max(v.abs()));
    2.0 * 2f64.powi(-8) * m + 1e-6
}

const BF16_CK_REL: f64 = 0.0078125; // 2 * 2^-8, one bf16 ulp either side
const EXACT: f64 = 1e-12;

struct Fx {
    g: Golden,
    t: EngramTables,
    regimes: Vec<String>,
    ids: Vec<i64>,
    prefill: usize,
    dim: usize,
    hc: usize,
    eps: f32,
}

fn fx() -> Fx {
    let g = Golden::load();
    let t = EngramTables::from_golden(&g);
    let regimes = g.regimes();
    let vocab = g.fixture_u64("vocab_size");
    let prefill = g.fixture_u64("prefill_len") as usize;
    let ids: Vec<i64> = (0..(prefill + 2) as u64).map(|i| fixed_int("input_ids", i, vocab) as i64).collect();
    let dim = g.fixture_u64("dim") as usize;
    let hc = g.fixture_u64("hc_mult") as usize;
    let eps = g.fixture_f64("norm_eps") as f32;
    Fx { g, t, regimes, ids, prefill, dim, hc, eps }
}

/// Runs the hash state through all three regimes, returning per-regime `[L, layers, cols]`.
fn hashes(f: &Fx) -> Vec<Vec<i64>> {
    let mut st = NgramHashState::new(&f.t, 1, f.g.fixture_u64("max_seq_len") as usize);
    let p = f.prefill;
    vec![
        st.forward(&f.ids[..p], p, 0),
        st.forward(&f.ids[p..p + 1], 1, p),
        st.forward(&f.ids[p + 1..p + 2], 1, p + 1),
    ]
}

#[test]
fn tables_match_the_fixture_geometry() {
    let f = fx();
    assert_eq!(f.t.layer_ids, vec![1, 4]);
    assert_eq!((f.t.max_ngram, f.t.n_heads, f.t.head_dim), (4, 2, 32));
    assert_eq!(f.t.n_hash_cols(), 6);
    assert_eq!(f.t.token_map.len(), 64);
    // rows == sum of that layer's primes, the identity Tier 2 proved on the real file
    for (li, rows) in f.t.num_embeddings.iter().enumerate() {
        let sum: i64 = f.t.primes[li].iter().flatten().sum();
        assert_eq!(*rows as i64, sum, "layer {} rows vs sum of primes", f.t.layer_ids[li]);
    }
    // offsets are the exclusive prefix sums of the flattened primes
    for (li, offs) in f.t.offsets.iter().enumerate() {
        let flat: Vec<i64> = f.t.primes[li].iter().flatten().copied().collect();
        let mut acc = 0;
        for (c, o) in offs.iter().enumerate() {
            assert_eq!(*o, acc, "layer {li} offset[{c}]");
            acc += flat[c];
        }
    }
}

#[test]
fn hash_is_exact_across_prefill_and_both_decode_steps() {
    let f = fx();
    let hs = hashes(&f);
    for (r, h) in f.regimes.iter().zip(&hs) {
        let got: Vec<f64> = h.iter().map(|&v| v as f64).collect();
        check(&format!("{r}.engram_hashes"), &got, &f.g.tensor(r, "engram_hashes"), EXACT, EXACT);
        // the per-layer slice the Engram module actually receives
        let nl = f.t.layer_ids.len();
        let cols = f.t.n_hash_cols();
        for (li, lid) in f.t.layer_ids.iter().enumerate() {
            let l_tokens = h.len() / (nl * cols);
            let slice: Vec<f64> = (0..l_tokens)
                .flat_map(|s| (0..cols).map(move |c| h[(s * nl + li) * cols + c] as f64))
                .collect();
            check(&format!("{r}.L{lid}.engram_hash_ids"), &slice, &f.g.tensor(r, &format!("L{lid}.engram_hash_ids")), EXACT, EXACT);
        }
    }
}

#[test]
fn e4m3_encoder_round_trips_the_fixture_tables() {
    // spot values
    assert_eq!(f32_to_e4m3_rne(1.0), 0x38);
    assert_eq!(f32_to_e4m3_rne(0.5), 0x30);
    assert_eq!(f32_to_e4m3_rne(448.0), 0x7E);
    assert_eq!(f32_to_e4m3_rne(2f32.powi(-6)), 0x08, "first normal");
    assert_eq!(f32_to_e4m3_rne(2f32.powi(-9)), 0x01, "smallest subnormal");
    assert_eq!(f32_to_e4m3_rne(1.0625), 0x38, "tie rounds to even (down)");
    assert_eq!(f32_to_e4m3_rne(1.1875), 0x3A, "tie rounds to even (up)");
    assert_eq!(f32_to_e4m3_rne(-1.5), 0xBC);
    for b in 0u8..=0xFF {
        let v = e4m3_to_f32(b);
        if v.is_nan() {
            continue;
        }
        assert_eq!(f32_to_e4m3_rne(v), b, "round trip of code {b:#04x} ({v})");
    }
    // the two fp8 tables against the checksums the generator committed for them
    let f = fx();
    let mut checked = 0;
    for w in f.g.weights_meta() {
        if w.kind != "fp8" {
            continue;
        }
        let ck = checksum((0..w.n as u64).map(|i| e4m3_to_f32(f32_to_e4m3_rne(fixed_value(&w.name, i, w.scale, w.offset))) as f64));
        assert!((ck - w.ck).abs() <= 1e-9 * w.ck.abs().max(1.0), "{}: {ck} vs {}", w.name, w.ck);
        checked += 1;
    }
    assert_eq!(checked, 2);
}

#[test]
fn e8m0_rounding_rule_matches_torch() {
    // the shipped scales are torch's e8m0 cast of the filler; recompute them with the rule
    let f = fx();
    let mut n = 0;
    for (li, lid) in f.t.layer_ids.iter().enumerate() {
        let name = format!("layers.{lid}.engram.embed.scale");
        for (i, want) in f.t.scales[li].iter().enumerate() {
            let raw = fixed_value(&name, i as u64, 1.0, 1.0);
            let got = f32_to_e8m0_rne(raw);
            assert_eq!(got, *want, "layer {lid} scale[{i}]: raw {raw} -> {got}, torch gave {want}");
            n += 1;
        }
    }
    assert!(n > 1000, "checked {n}");
}

#[test]
fn table_gather_is_exact() {
    let f = fx();
    let hs = hashes(&f);
    for (li, lid) in f.t.layer_ids.iter().enumerate() {
        let table = regen_table(&f.t, li);
        for (r, h) in f.regimes.iter().zip(&hs) {
            let nl = f.t.layer_ids.len();
            let cols = f.t.n_hash_cols();
            let l_tokens = h.len() / (nl * cols);
            let ids: Vec<i64> = (0..l_tokens).flat_map(|s| (0..cols).map(move |c| h[(s * nl + li) * cols + c])).collect();
            let got: Vec<f64> = embed_rows(&table, f.t.head_dim, &ids).iter().map(|&v| v as f64).collect();
            check(&format!("{r}.L{lid}.engram_embed"), &got, &f.g.tensor(r, &format!("L{lid}.engram_embed")), EXACT, EXACT);
        }
    }
}

#[test]
fn projection_matches_within_bf16() {
    let f = fx();
    let hs = hashes(&f);
    let in_f = f.t.n_hash_cols() * f.t.head_dim;
    let out_f = f.dim * (f.hc + 1);
    for (li, lid) in f.t.layer_ids.iter().enumerate() {
        let table = regen_table(&f.t, li);
        let w = regen_bf16_matrix(&format!("layers.{lid}.engram.wkv.weight"), out_f, in_f);
        for (r, h) in f.regimes.iter().zip(&hs) {
            let nl = f.t.layer_ids.len();
            let cols = f.t.n_hash_cols();
            let l_tokens = h.len() / (nl * cols);
            let ids: Vec<i64> = (0..l_tokens).flat_map(|s| (0..cols).map(move |c| h[(s * nl + li) * cols + c])).collect();
            let emb = embed_rows(&table, f.t.head_dim, &ids);
            let kv = linear_bf16(&emb, l_tokens, in_f, &w, out_f);
            let gt = f.g.tensor(r, &format!("L{lid}.engram_kv"));
            let got: Vec<f64> = kv.iter().map(|&v| v as f64).collect();
            check(&format!("{r}.L{lid}.engram_kv"), &got, &gt, bf16_tol(&gt), BF16_CK_REL);
        }
    }
}

#[test]
fn full_engram_matches_within_bf16() {
    let f = fx();
    let hs = hashes(&f);
    let in_f = f.t.n_hash_cols() * f.t.head_dim;
    let out_f = f.dim * (f.hc + 1);
    for (li, lid) in f.t.layer_ids.iter().enumerate() {
        let table = regen_table(&f.t, li);
        let w = regen_bf16_matrix(&format!("layers.{lid}.engram.wkv.weight"), out_f, in_f);
        let q = regen_bf16_qk(&format!("layers.{lid}.engram.q_weight"), f.hc * f.dim);
        let k = regen_bf16_qk(&format!("layers.{lid}.engram.k_weight"), f.hc * f.dim);
        for (r, h) in f.regimes.iter().zip(&hs) {
            let nl = f.t.layer_ids.len();
            let cols = f.t.n_hash_cols();
            let l_tokens = h.len() / (nl * cols);
            let ids: Vec<i64> = (0..l_tokens).flat_map(|s| (0..cols).map(move |c| h[(s * nl + li) * cols + c])).collect();
            let emb = embed_rows(&table, f.t.head_dim, &ids);
            let kv = linear_bf16(&emb, l_tokens, in_f, &w, out_f);
            let xin = f.g.tensor(r, &format!("L{lid}.engram_in"));
            assert_eq!(xin.stride, 1, "engram_in must be stored at full resolution");
            let x: Vec<f32> = xin.data.iter().map(|&v| v as f32).collect();
            let out = gate_and_add(&x, &kv, &q, &k, l_tokens, f.hc, f.dim, f.eps);
            let gt = f.g.tensor(r, &format!("L{lid}.engram_out"));
            let got: Vec<f64> = out.iter().map(|&v| v as f64).collect();
            check(&format!("{r}.L{lid}.engram_out"), &got, &gt, bf16_tol(&gt), BF16_CK_REL);
        }
    }
}
