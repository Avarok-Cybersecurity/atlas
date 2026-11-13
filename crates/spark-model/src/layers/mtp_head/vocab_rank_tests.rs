// SPDX-License-Identifier: AGPL-3.0-only

//! CPU oracles for the frequency-ranked draft vocabulary: the shipped table,
//! the row selection, the identity check, the grammar-mask translation, the
//! coverage the provenance records, and the GPU-free shape of the ranked head
//! build on the mock backend.

use super::*;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

/// The 4B / 0.8B runtime vocabulary (config.json's 248,320 capped to the
/// tokenizer's 248,070 by `cap_vocab_size_to_tokenizer`).
const QWEN35_VOCAB: usize = 248_070;

fn table() -> &'static RankTable {
    shipped().expect("the shipped rank table parses")
}

#[test]
fn the_shipped_table_is_a_permutation_prefix_of_the_vocabulary() {
    let t = table();
    assert_eq!(t.ids.len(), 131_072);
    let mut seen = vec![false; 248_320];
    for &id in &t.ids {
        assert!((id as usize) < 248_320, "id {id} past the vocabulary");
        assert!(!seen[id as usize], "id {id} appears twice");
        seen[id as usize] = true;
    }
}

#[test]
fn the_shipped_table_covers_every_special_id() {
    let t = table();
    // Qwen3.5's 26 added tokens, 248044..=248069: <|endoftext|>, <|im_start|>,
    // <|im_end|>, the tool-call and think tags among them.
    assert_eq!(t.forced, (248_044..=248_069).collect::<Vec<u32>>());
    for &f in &t.forced {
        assert!(t.ids.contains(&f), "forced id {f} missing from the table");
    }
    // With the first-N ids at the default 100,000 rows NONE of them is held,
    // so the id head can never draft an end of turn.
    let id_head: Vec<u32> = (0..100_000).collect();
    let bits = membership(&id_head, QWEN35_VOCAB);
    assert!(t.forced.iter().all(|&f| !is_member(&bits, f)));
}

#[test]
fn the_table_starts_with_the_upstream_top_tokens() {
    // ranking.train.manifest.json `top_tokens`: ".", ",", "\n", "，", "\n\n".
    assert_eq!(&table().ids[..5], &[13, 11, 198, 3709, 271]);
}

#[test]
fn provenance_names_the_source_and_verified_tokenizers() {
    let t = table();
    assert_eq!(t.vocab_sizes, vec![248_070, 248_320]);
    assert!(t.tokenizer_fnv1a64.contains(&0x349d_d3b6_7c2a_def5));
    let v: serde_json::Value = serde_json::from_str(PROVENANCE_JSON).unwrap();
    assert_eq!(v["source"]["license"], "Apache-2.0");
    assert_eq!(v["source"]["commit"], "383e5db");
    assert_eq!(v["tokenizer_check"]["mismatches"], 0);
}

#[test]
fn provenance_coverage_reproduces_the_held_out_numbers() {
    // Report 71 section C2: 98.1% ranked vs 86.2% first-N at 100,000 rows,
    // 96.0% vs 73.0% at 65,536, on the held-out counts.
    let v: serde_json::Value = serde_json::from_str(PROVENANCE_JSON).unwrap();
    let c = &v["coverage"]["ranking.heldout"];
    let at = |n: &str, mode: &str| c[n][mode]["total"].as_f64().unwrap();
    assert!((at("100000", "freq") - 0.981).abs() < 0.001);
    assert!((at("100000", "id") - 0.862).abs() < 0.001);
    assert!((at("65536", "freq") - 0.960).abs() < 0.001);
    assert!((at("65536", "id") - 0.730).abs() < 0.001);
}

/// Live recomputation from the upstream held-out counts, when they are on the
/// machine: `AVAROK_MTP_VOCAB_HELDOUT=<.../ranking.heldout.counts.i64>`.
#[test]
#[ignore = "reads an 11.9 MB upstream data file named by AVAROK_MTP_VOCAB_HELDOUT"]
fn coverage_recomputed_from_the_held_out_counts() {
    let path = std::env::var("AVAROK_MTP_VOCAB_HELDOUT").expect("set AVAROK_MTP_VOCAB_HELDOUT");
    let raw = std::fs::read(path).unwrap();
    let counts: Vec<u64> = raw
        .chunks_exact(8)
        .take(248_320)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()).max(0) as u64)
        .collect();
    let t = table();
    let freq = select_rows(&t.ids, QWEN35_VOCAB, 100_000, &t.forced);
    let id: Vec<u32> = (0..100_000).collect();
    assert!((coverage(&freq, &counts) - 0.981).abs() < 0.001);
    assert!((coverage(&id, &counts) - 0.862).abs() < 0.001);
}

#[test]
fn selection_is_unique_sized_and_forces_the_specials_in() {
    let t = table();
    for n in [
        1_000usize,
        32_768,
        65_536,
        100_000,
        131_072,
        200_000,
        QWEN35_VOCAB,
    ] {
        let rows = select_rows(&t.ids, QWEN35_VOCAB, n, &t.forced);
        assert_eq!(rows.len(), n.min(QWEN35_VOCAB), "n={n}");
        let mut seen = vec![false; QWEN35_VOCAB];
        for &r in &rows {
            assert!((r as usize) < QWEN35_VOCAB);
            assert!(!seen[r as usize], "n={n}: id {r} twice");
            seen[r as usize] = true;
        }
        for &f in &t.forced {
            assert!(seen[f as usize], "n={n}: forced id {f} dropped");
        }
    }
}

#[test]
fn selection_keeps_the_table_prefix_and_completes_in_id_order() {
    let table = [5u32, 2, 9, 2, 40];
    // 40 is past the vocabulary, 2 repeats: rows 5, 2, 9, then 0, 1, 3, ...
    assert_eq!(select_rows(&table, 12, 6, &[]), vec![5, 2, 9, 0, 1, 3]);
    // A forced id is held once, at the end, and takes a row from the tail.
    assert_eq!(
        select_rows(&table, 12, 6, &[11, 2]),
        vec![5, 9, 0, 1, 2, 11]
    );
    // More forced ids than rows: the lowest forced ids.
    assert_eq!(select_rows(&table, 12, 2, &[7, 3, 11]), vec![3, 7]);
    // The full width is a permutation of the vocabulary.
    let mut all = select_rows(&table, 12, 99, &[]);
    all.sort_unstable();
    assert_eq!(all, (0..12).collect::<Vec<u32>>());
}

#[test]
fn the_remap_is_the_identity_at_rank_id() {
    // rank=id: row r holds token r, so the device remap (ids[i] =
    // row_to_id[ids[i]]) maps every argmax back to itself.
    let id_rows: Vec<u32> = (0..100_000).collect();
    let remap = |r: u32| -> u32 { id_rows.get(r as usize).copied().unwrap_or(r) };
    for r in [0u32, 1, 13, 99_999, 100_000, 248_000] {
        assert_eq!(remap(r), r);
    }
    // And the ranked map is the kernel's oracle: row -> table id, out of
    // range left alone (draft_vocab_remap.cu's bound).
    let t = table();
    let rows = select_rows(&t.ids, QWEN35_VOCAB, 100_000, &t.forced);
    let kernel = |r: u32| -> u32 { rows.get(r as usize).copied().unwrap_or(r) };
    assert_eq!(kernel(0), 13);
    assert_eq!(kernel(100_000), 100_000);
}

#[test]
fn identity_check_requires_a_verified_tokenizer_and_vocabulary() {
    let t = table();
    let good = t.tokenizer_fnv1a64[0];
    assert!(table_applies(t, 248_070, Some(good)).is_ok());
    assert!(table_applies(t, 248_320, Some(good)).is_ok());
    assert!(
        table_applies(t, 151_936, Some(good)).is_err(),
        "Qwen3 vocabulary"
    );
    assert!(
        table_applies(t, 248_070, Some(good ^ 1)).is_err(),
        "other tokenizer"
    );
    assert!(
        table_applies(t, 248_070, None).is_err(),
        "no tokenizer.json"
    );
}

#[test]
fn fnv1a64_matches_the_reference_vectors() {
    assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
}

#[test]
fn parse_rejects_inconsistent_files() {
    assert!(
        parse_table(&[1, 2, 3], PROVENANCE_JSON).is_err(),
        "torn u32"
    );
    assert!(
        parse_table(&[0; 8], PROVENANCE_JSON).is_err(),
        "row count mismatch"
    );
    assert!(parse_table(&[0; 8], "not json").is_err());
    let ok = parse_table(&[7, 0, 0, 0, 9, 0, 0, 0], r#"{"rows":2,"forced_ids":[9]}"#).unwrap();
    assert_eq!(ok.ids, vec![7, 9]);
    assert_eq!(ok.forced, vec![9]);
    assert!(ok.tokenizer_fnv1a64.is_empty() && ok.vocab_sizes.is_empty());
}

fn bf16(x: f32) -> [u8; 2] {
    ((x.to_bits() >> 16) as u16).to_le_bytes()
}

#[test]
fn grammar_mask_is_translated_into_row_space() {
    // Head rows hold tokens 40, 3, 7; logits favor row 0 (token 40).
    let logits: Vec<u8> = [9.0f32, 1.0, 5.0].iter().flat_map(|&x| bf16(x)).collect();
    let rows = [40u32, 3, 7];
    let mut mask = vec![0i32; 2];
    let allow = |m: &mut Vec<i32>, t: u32| m[t as usize / 32] |= 1i32 << (t % 32);
    allow(&mut mask, 3);
    allow(&mut mask, 7);
    // Token 40 is masked off, so row 2 (token 7) wins, reported as a TOKEN id.
    assert_eq!(masked_argmax_rows(&logits, &rows, &mask), Some(7));
    allow(&mut mask, 40);
    assert_eq!(masked_argmax_rows(&logits, &rows, &mask), Some(40));
    // A mask that allows no token of the head: no draft.
    assert_eq!(masked_argmax_rows(&logits, &rows, &[0, 0]), None);
    // Ties: the first row wins, as on the id-ranked CPU path.
    let tie: Vec<u8> = [2.0f32, 2.0].iter().flat_map(|&x| bf16(x)).collect();
    assert_eq!(masked_argmax_rows(&tie, &[5, 6], &[0x60]), Some(5));
}

#[test]
fn membership_and_coverage_agree_on_a_small_head() {
    let rows = [1u32, 4, 63, 64];
    let bits = membership(&rows, 70);
    for id in 0..70u32 {
        assert_eq!(is_member(&bits, id), rows.contains(&id), "id {id}");
    }
    assert!(!is_member(&bits, 10_000), "past the bitmap");
    let counts = [0u64, 10, 0, 0, 30, 0, 60];
    assert!((coverage(&[1, 4], &counts) - 0.4).abs() < 1e-12);
    assert_eq!(coverage(&[1], &[0, 0]), 0.0);
}

#[test]
fn rank_mode_maps_row_values() {
    assert_eq!(RankMode::from_value("freq"), RankMode::Freq);
    assert_eq!(RankMode::from_value("id"), RankMode::Id);
    assert_eq!(
        RankMode::from_value("FREQ"),
        RankMode::Id,
        "values arrive normalized"
    );
}

#[test]
fn the_id_row_builds_no_ranked_head_and_issues_no_lookup() {
    // An empty environment resolves `mtp_vocab_rank` to its baseline `id` on
    // every target: the builder must return before touching the backend, so
    // the boot audit sees no lookup of the Ada-only module.
    if RankMode::resolved() != RankMode::Id {
        return; // the process environment overrides the row; nothing to pin
    }
    let gpu = MockGpuBackend::new();
    let lm = DenseWeight {
        weight: gpu.alloc(64 * 32 * 2).unwrap(),
    };
    let k = KernelHandle(1);
    let got = build_ranked_draft_head(&lm, 16, 64, 32, &gpu, k, k, 0).unwrap();
    assert!(got.is_none());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn metrics_render_both_series() {
    set_draft_rows(100_000);
    let mut s = String::new();
    render_prometheus(&mut s);
    assert!(s.contains("# TYPE atlas_mtp_draft_vocab_rows gauge\n"));
    assert!(s.contains("atlas_mtp_draft_vocab_rows 100000\n"));
    assert!(s.contains("# TYPE atlas_mtp_draft_out_of_head_total counter\n"));
    assert!(s.contains("atlas_mtp_draft_out_of_head_total "));
}
