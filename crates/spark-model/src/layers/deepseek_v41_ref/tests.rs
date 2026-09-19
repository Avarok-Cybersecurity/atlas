// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Proves the harness itself before any V4.1 graph code exists: the golden parses, the RNG-free
//! filler regenerates the generator's inputs and weights bit for bit, every capture the graph
//! will be checked against is present and internally consistent, and the comparator catches a
//! planted defect.

use super::*;

#[test]
fn golden_parses_and_fixture_reads() {
    let g = Golden::load();
    let regimes = g.regimes();
    assert_eq!(regimes.len(), 3, "prefill + two decode steps");
    assert!(regimes[0].starts_with("prefill"));
    assert_eq!(g.fixture_u64("vocab_size"), 64);
    assert_eq!(g.fixture_u64("dim"), 64);
    assert_eq!(g.fixture_u64("n_layers"), 6);
    assert_eq!(g.fixture_u64("hc_mult"), 4);
    assert_eq!(g.fixture_u64("batch"), 1);
}

#[test]
fn splitmix_probe_matches_python() {
    let g = Golden::load();
    let want = g.lcg_probe();
    assert_eq!(want.len(), 8);
    let got: Vec<u64> = (0..8).map(|i| fixed_raw("probe", i)).collect();
    assert_eq!(
        got, want,
        "splitmix64/fnv1a64 stream diverges from the generator"
    );
}

#[test]
fn input_ids_regenerate_bit_exact() {
    let g = Golden::load();
    let regimes = g.regimes();
    let vocab = g.fixture_u64("vocab_size");
    let prefill = g.fixture_u64("prefill_len") as usize;
    let batch = g.fixture_u64("batch") as usize;
    let total = batch * (prefill + 2);
    let ids: Vec<f64> = (0..total as u64)
        .map(|i| fixed_int("input_ids", i, vocab) as f64)
        .collect();

    let p = g.tensor(&regimes[0], "input_ids");
    assert_eq!(p.stride, 1, "int captures are stored densely");
    assert_eq!(p.data, ids[..prefill], "prefill input_ids");
    let d0 = g.tensor(&regimes[1], "input_ids");
    assert_eq!(d0.data, vec![ids[prefill]], "first decode token");
    let d1 = g.tensor(&regimes[2], "input_ids");
    assert_eq!(d1.data, vec![ids[prefill + 1]], "second decode token");
}

#[test]
fn f32_weights_regenerate_to_committed_checksums() {
    let g = Golden::load();
    let mut checked = 0usize;
    for w in g.weights_meta() {
        if w.kind != "f32" {
            continue; // fp8 / e8m0 table rows: regenerated when the engram module lands
        }
        let bf16 = w.dtype == "bfloat16";
        let ck = checksum((0..w.n as u64).map(|i| {
            let v = fixed_value(&w.name, i, w.scale, w.offset);
            (if bf16 { to_bf16_rne(v) } else { v }) as f64
        }));
        let tol = 1e-9 * w.ck.abs().max(1.0);
        assert!(
            (ck - w.ck).abs() <= tol,
            "{}: regenerated ck {ck} vs committed {} (dtype {}, n {})",
            w.name,
            w.ck,
            w.dtype,
            w.n
        );
        checked += 1;
    }
    assert!(checked >= 200, "only {checked} f32 parameters checked");
}

#[test]
fn every_capture_sample_length_is_consistent() {
    let g = Golden::load();
    for r in g.regimes() {
        for name in g.capture_names(&r) {
            let t = g.tensor(&r, &name);
            let numel: usize = t.shape.iter().product();
            assert_eq!(numel, t.n, "{r}.{name}: shape/n disagree");
            assert_eq!(
                t.data.len(),
                t.n.div_ceil(t.stride),
                "{r}.{name}: sample length"
            );
            assert!(
                t.data.iter().all(|x| x.is_finite()),
                "{r}.{name}: non-finite sample"
            );
            assert!(t.ck.is_finite(), "{r}.{name}: non-finite checksum");
        }
    }
}

#[test]
fn every_capture_the_graph_will_be_checked_against_is_present() {
    let g = Golden::load();
    let r = &g.regimes()[0];
    let have = g.capture_names(r);
    let mut want: Vec<String> = [
        "input_ids",
        "embed",
        "engram_hashes",
        "h_final",
        "head_in",
        "logits_full",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for l in 0..6 {
        for s in [
            "h_in",
            "pre_mix_in",
            "attn_pre",
            "attn_post",
            "attn_comb",
            "attn_in",
            "sa_q",
            "sa_kv",
            "sa_topk_idxs",
            "sa_o",
            "attn_out",
            "ffn_pre",
            "ffn_post",
            "ffn_comb",
            "ffn_in",
            "moe_weights",
            "moe_indices",
            "ffn_out",
            "h_out",
            "pre_mix_out",
        ] {
            want.push(format!("L{l}.{s}"));
        }
    }
    for l in [1, 4] {
        for s in ["engram_in", "engram_hash_ids", "engram_out"] {
            want.push(format!("L{l}.{s}"));
        }
    }
    for s in [
        "shared.L2.compress_kv",
        "shared.L2.index_k",
        "shared.L2.topk_idxs",
        "shared.L4.candidates",
        "shared.L4.compress_kv",
        "shared.L4.index_k",
        "shared.L4.topk_idxs",
        "shared.L5.topk_idxs",
        "shared.L5.candidates",
    ] {
        want.push(s.to_string());
    }
    let missing: Vec<&String> = want.iter().filter(|w| !have.contains(w)).collect();
    assert!(missing.is_empty(), "{r} is missing captures: {missing:?}");

    // the shared-attention story in the weights: 2 sources at ratio 2 give 12/2 = 6 rows,
    // the ratio-1 source gives 12
    assert_eq!(g.tensor(r, "shared.L2.compress_kv").shape, vec![1, 6, 32]);
    assert_eq!(g.tensor(r, "shared.L4.compress_kv").shape, vec![1, 12, 32]);
    assert_eq!(
        g.tensor(r, "engram_hashes").shape,
        vec![1, 12, 2, 6],
        "[B, L, n_engram_layers, (ngram-1)*heads]"
    );
    assert_eq!(g.tensor(r, "logits_full").shape, vec![1, 12, 64]);
}

#[test]
fn comparator_catches_a_planted_defect() {
    let g = Golden::load();
    let r = &g.regimes()[0];
    let t = g.tensor(r, "logits_full");
    assert_close("identity", &t.data, &t.data, 1e-12);
    let mut bad = t.data.clone();
    bad[5] += 1e-3;
    let caught = std::panic::catch_unwind(|| assert_close("planted", &bad, &t.data, 1e-6)).is_err();
    assert!(caught, "a 1e-3 planted defect must fail a 1e-6 comparison");
}
