// SPDX-License-Identifier: AGPL-3.0-only

//! Slice-9 acceptance: the 45-layer text-model skeleton is the topology the checkpoint actually
//! has, wired the way HF 5.16.1 wires it, and its structural weight contract closes exactly.
//!
//! Both fixtures come from `LibertAIDAI/GLM-5.3-Flash-NVFP4` snapshot `9e0d74e3…`:
//!   * `…-config.json` — the checkpoint's own config, parsed by Atlas's real `glm5_next` parser.
//!     Not a synthetic fixture: a parser that only ever sees a hand-written config proves nothing
//!     about the checkpoint.
//!   * `…-structural.txt` — every non-MLP text tensor name (real layer indices, not the `layers.N.`
//!     canonicalisation the Slice-1 pattern fixture uses), 1,047 rows.

use std::collections::BTreeSet;

use atlas_core::config::{LayerType, parse_config};
use spark_model::layers::glm5next_skeleton::{
    FinalStep, Glm5NextTextSkeleton, Mixer, Mlp, ResidualStep, Site, StateKind,
};

const CONFIG: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-config.json");
const STRUCTURAL: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-structural.txt");

const EXPECTED_STRUCTURAL: usize = 1_047;
const N_KDA: usize = 34;
const N_DSA: usize = 11;

fn skeleton() -> Glm5NextTextSkeleton {
    let cfg = parse_config(CONFIG).expect("the real checkpoint config parses");
    Glm5NextTextSkeleton::from_config(&cfg).expect("skeleton builds from the real config")
}

fn available() -> BTreeSet<String> {
    STRUCTURAL
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

#[test]
fn fixture_is_the_reference_checkpoint() {
    assert_eq!(
        available().len(),
        EXPECTED_STRUCTURAL,
        "structural fixture drifted"
    );
}

// ───────────────────────────────────────────────────────────── topology

/// 34 KDA + 11 DSA + layer 45 held separately. The zero is asserted: a nonzero count of any
/// other mixer kind means something was flattened again, which is how GLM-5.3 previously
/// round-tripped `deepseek_sparse_attention` into a lie.
#[test]
fn topology_is_34_kda_11_dsa_and_mtp_is_not_in_the_text_stack() {
    let s = skeleton();
    assert_eq!(s.layers.len(), 45, "the text stack is 45 layers, not 46");
    let kda = s.layers.iter().filter(|l| l.mixer == Mixer::Kda).count();
    let dsa = s.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count();
    assert_eq!((kda, dsa), (N_KDA, N_DSA));
    assert!(s.layers.iter().all(|l| !l.is_mtp));

    let mtp = s.mtp.expect("num_nextn_predict_layers = 1");
    assert_eq!(mtp.index, 45);
    assert!(mtp.is_mtp);
    // Layer 45's attention is DSA-identical; the MTP head is separate, so MTP needs no
    // attention implementation of its own.
    assert_eq!(mtp.mixer, Mixer::Dsa);
}

/// The sparse layers are the ones the checkpoint names, not `i % 4 == 3` arithmetic that
/// happens to fit.
#[test]
fn sparse_layers_are_the_checkpoint_s_own_list() {
    let s = skeleton();
    let dsa: Vec<usize> = s
        .layers
        .iter()
        .filter(|l| l.mixer == Mixer::Dsa)
        .map(|l| l.index)
        .collect();
    assert_eq!(dsa, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
}

/// 🪤 `first_k_dense_replace = 3` was read by NOTHING before this slice, so `mlp_only_layers`
/// came out empty and the whole stack looked routed.
#[test]
fn layers_0_to_2_are_dense_and_everything_after_routes() {
    let s = skeleton();
    let dense: Vec<usize> = s
        .layers
        .iter()
        .filter(|l| l.mlp == Mlp::Dense)
        .map(|l| l.index)
        .collect();
    assert_eq!(dense, vec![0, 1, 2]);
    assert_eq!(
        s.layers.iter().filter(|l| l.mlp == Mlp::RoutedMoe).count(),
        42
    );
    assert_eq!(s.mtp.unwrap().mlp, Mlp::RoutedMoe);
}

/// 🪤 Layer 45 carries ZERO `hc_*` tensors. All 270 live on 0..=44.
#[test]
fn hyper_connection_is_on_every_text_layer_and_on_no_mtp_layer() {
    let s = skeleton();
    assert!(s.layers.iter().all(|l| l.hyper_connection));
    assert!(!s.mtp.unwrap().hyper_connection);

    let hc = available().iter().filter(|n| n.contains(".hc_")).count();
    assert_eq!(hc, 270, "6 mHC tensors x 45 text layers");
    assert!(
        !available()
            .iter()
            .any(|n| n.starts_with("model.language_model.layers.45.hc_")),
        "the MTP layer must not carry a hyper-connection"
    );
}

// ───────────────────────────────────────────────────── wiring

/// The residual path, in order, exactly as `Glm5NextTextDecoderLayer::forward` runs it:
/// save residual → `hc_pre` → norm → sublayer → `hc_post`, once per site.
#[test]
fn residual_wiring_matches_the_reference_decoder_layer() {
    let s = skeleton();
    let l = s.layers[0];
    assert_eq!(
        s.residual_plan(&l),
        vec![
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Attn),
            ResidualStep::Norm("input_layernorm.weight"),
            ResidualStep::Mixer,
            ResidualStep::HcPost(Site::Attn),
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Ffn),
            ResidualStep::Norm("post_attention_layernorm.weight"),
            ResidualStep::Mlp,
            ResidualStep::HcPost(Site::Ffn),
        ]
    );
    // Every text layer has the same shape regardless of mixer or MLP kind — the hyper-connection
    // sites are structural, not conditional on what they wrap.
    for l in &s.layers {
        assert_eq!(s.residual_plan(l).len(), 10, "layer {}", l.index);
    }
}

/// No hyper-connection on the MTP layer means the mHC steps drop out entirely — they are not
/// silently run against absent weights.
#[test]
fn mtp_residual_path_has_no_mhc_steps() {
    let s = skeleton();
    let plan = s.residual_plan(&s.mtp.unwrap());
    assert!(
        !plan
            .iter()
            .any(|st| matches!(st, ResidualStep::HcPre(_) | ResidualStep::HcPost(_))),
        "{plan:?}"
    );
    assert_eq!(plan.len(), 6);
}

/// 🪤 The final collapse is an UNWEIGHTED MEAN. DeepSeek-V4's `hc_head` is a learned
/// sigmoid-weighted sum and Atlas's CUDA kernel implements that one; GLM carries zero
/// `hc_head` tensors, so reusing the kernel would read weights that do not exist.
#[test]
fn final_collapse_is_a_parameterless_mean_then_norm() {
    let s = skeleton();
    assert_eq!(
        s.final_plan(),
        [
            FinalStep::HyperHeadMean,
            FinalStep::Norm("model.language_model.norm.weight"),
            FinalStep::LmHead,
        ]
    );
    assert!(
        !available().iter().any(|n| n.contains("hc_head")),
        "a learned final collapse would need hc_head weights; the checkpoint has none"
    );
}

// ───────────────────────────────────────────────────── state plumbing

/// KDA state and KV blocks are not interchangeable, and admission needs both. 34 layers carry
/// recurrent state; 12 (11 text + MTP) consume KV blocks.
#[test]
fn state_plan_splits_recurrent_from_paged_and_covers_every_layer() {
    let s = skeleton();
    let plan = s.state_plan();
    assert_eq!(plan.len(), 46, "45 text layers + the MTP layer");
    assert_eq!(s.kda_state_layers().len(), N_KDA);
    assert_eq!(s.kv_cache_layers().len(), N_DSA + 1);
    assert_eq!(plan[&45], StateKind::SparseKv);
    assert_eq!(plan[&0], StateKind::KdaRecurrent);
    assert_eq!(plan[&3], StateKind::SparseKv);
    // No layer is both, and none is neither.
    assert_eq!(
        s.kda_state_layers().len() + s.kv_cache_layers().len(),
        plan.len()
    );
}

// ───────────────────────────────────────────────────── structural binding

/// THE acceptance criterion for this slice: the structural contract closes exactly against the
/// checkpoint — zero missing, zero unexpected, zero silent skips — while MLP/MoE stay
/// deliberately deferred rather than quietly ignored.
#[test]
fn structural_binding_closes_exactly() {
    let s = skeleton();
    let acc = s.account(&available());
    assert!(
        acc.missing.is_empty(),
        "{} structural tensor(s) absent from the checkpoint, first 10: {:?}",
        acc.missing.len(),
        &acc.missing[..acc.missing.len().min(10)]
    );
    assert!(
        acc.unexpected.is_empty(),
        "{} text tensor(s) the skeleton was never taught, first 10: {:?}",
        acc.unexpected.len(),
        &acc.unexpected[..acc.unexpected.len().min(10)]
    );
    assert_eq!(acc.bound, acc.required);
    assert!(acc.is_complete());
    // The fixture is the non-MLP text surface, so nothing here is deferred; the deferred count
    // is what will carry the MoE tensors when the next slice widens the fixture.
    assert_eq!(acc.deferred, 0);
    assert_eq!(acc.required, EXPECTED_STRUCTURAL);
}

/// Per-signature accounting, so a drift shows up as "which family changed" rather than a bare
/// total. 34 x 23 + 11 x 22 + 1 x 20 + 3 non-layer = 1,047.
#[test]
fn per_layer_structural_counts_match_the_three_measured_signatures() {
    let s = skeleton();
    for l in &s.layers {
        let n = s.structural_tensors(l).len();
        let want = match l.mixer {
            Mixer::Kda => 23,
            Mixer::Dsa => 22,
        };
        assert_eq!(n, want, "layer {} ({:?})", l.index, l.mixer);
    }
    assert_eq!(s.structural_tensors(&s.mtp.unwrap()).len(), 20);
    assert_eq!(
        N_KDA * 23 + N_DSA * 22 + 20 + 3,
        EXPECTED_STRUCTURAL,
        "the three signatures must account for the whole structural surface"
    );
}

/// A missing tensor must fail loudly. Dropping the indexer LayerNorm BIAS is the specific
/// silent-drop this model invites: every other norm here is a bias-free RMSNorm.
#[test]
fn a_dropped_indexer_bias_is_a_hard_failure_not_a_skip() {
    let s = skeleton();
    let mut avail = available();
    assert!(avail.remove("model.language_model.layers.3.self_attn.indexer.k_norm.bias"));
    let acc = s.account(&avail);
    assert!(!acc.is_complete());
    assert_eq!(acc.missing.len(), 1);
}

/// An unrecognised text tensor must be refused, never absorbed. `.mlp.` and vision names are
/// the ONLY things allowed to be deferred.
#[test]
fn an_untaught_text_tensor_is_refused_not_absorbed() {
    let s = skeleton();
    let mut avail = available();
    avail.insert("model.language_model.layers.9.self_attn.wat".to_string());
    let acc = s.account(&avail);
    assert!(!acc.is_complete());
    assert_eq!(acc.unexpected.len(), 1);

    // ...whereas an MoE tensor is deferred, and deferring is counted, not silent.
    let mut avail2 = available();
    avail2.insert("model.language_model.layers.9.mlp.experts.0.down_proj.weight".to_string());
    let acc2 = s.account(&avail2);
    assert!(acc2.is_complete());
    assert_eq!(acc2.deferred, 1);
}

/// The skeleton must refuse a config whose mixer set it does not have. A sparse layer bound as
/// dense full attention attends the whole cache and produces plausible output.
#[test]
fn an_unexpected_layer_type_is_refused() {
    let mut cfg = parse_config(CONFIG).expect("parses");
    cfg.layer_types[5] = LayerType::SlidingAttention;
    assert!(Glm5NextTextSkeleton::from_config(&cfg).is_err());
}

// ───────────────────────────────────────────────────── multi-EOS

/// Slice 9 decision 2, end to end on the REAL checkpoint config: all three stop tokens survive
/// parsing, with `<|endoftext|>` as the primary.
///
/// `154827 <|user|>` and `154829 <|observation|>` are turn terminators. Before this fix the
/// array made `config.json` fail to parse outright; the first repair collapsed it to one id,
/// which would have let an agent model run straight past the end of its own turn.
#[test]
fn all_three_glm_stop_tokens_survive_parsing() {
    let cfg = parse_config(CONFIG).expect("parses");
    assert_eq!(cfg.eos_token_id, 154820, "<|endoftext|> is the primary");
    assert_eq!(cfg.eos_ids(), vec![154820, 154827, 154829]);
    for id in [154820u32, 154827, 154829] {
        assert!(cfg.is_eos(id), "generation must stop on {id}");
    }
    assert!(!cfg.is_eos(154828));
}

// ───────────────────────────────────── Slice 11 gates 1 + 3

const FAMILIES: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-families.tsv");

fn families() -> Vec<(String, usize, usize)> {
    FAMILIES
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut p = l.split('\t');
            (
                p.next().unwrap().to_string(),
                p.next().unwrap().parse().unwrap(),
                p.next().unwrap().parse().unwrap(),
            )
        })
        .collect()
}

/// Gate 1: every one of the checkpoint's 113,074 tensors lands in exactly one family, and the
/// families that are deliberately NOT loaded are named rather than merely absent.
#[test]
fn the_whole_checkpoint_is_accounted_for_and_the_excluded_set_is_explicit() {
    let f = families();
    let total: usize = f.iter().map(|(_, c, _)| c).sum();
    assert_eq!(total, 113_074, "checkpoint tensor count drifted");
    let excluded: Vec<&str> = f
        .iter()
        .filter(|(n, ..)| n.contains("NOT LOADED"))
        .map(|(n, ..)| n.as_str())
        .collect();
    // The exclusions are the vision tower and EVERY layer-45 tensor — nothing else.
    assert!(excluded.iter().any(|n| n.starts_with("vision")));
    assert_eq!(excluded.len(), 7, "excluded families: {excluded:?}");
    let excl_t: usize = f
        .iter()
        .filter(|(n, ..)| n.contains("NOT LOADED"))
        .map(|(_, c, _)| c)
        .sum();
    assert_eq!(excl_t, 347 + 2592 + 14 + 4 + 3 + 2 + 2);
}

/// Gate 1: weight memory, from the MEASURED byte counts — never a formula.
///
/// 🔴 176.086 GiB of text weights against a 121 GB node is why Slice 11 cannot run EP=1.
#[test]
fn text_model_weight_memory_and_the_ep2_split() {
    let f = families();
    let text: usize = f
        .iter()
        .filter(|(n, ..)| !n.contains("NOT LOADED"))
        .map(|(_, _, b)| b)
        .sum();
    let experts: usize = f
        .iter()
        .filter(|(n, ..)| n == "moe_experts_NVFP4")
        .map(|(_, _, b)| b)
        .sum();
    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    assert!((gib(text) - 176.086).abs() < 0.01, "{}", gib(text));
    assert!((gib(experts) - 159.469).abs() < 0.01, "{}", gib(experts));
    // EP=2: only the routed experts shard; everything else replicates.
    let per_rank = experts / 2 + (text - experts);
    assert!((gib(per_rank) - 96.352).abs() < 0.01, "{}", gib(per_rank));
    // The node has 121 GB total / ~112 GB available. EP=1 does not fit; EP=2 does, barely.
    assert!(gib(text) > 121.0, "EP=1 would fit and this test is stale");
    assert!(gib(per_rank) < 112.0, "EP=2 no longer fits either");
}

/// Gate 3: the per-sequence state contract, split into what is fixed and what grows.
#[test]
fn per_sequence_state_budget_separates_fixed_from_growing() {
    let cfg = parse_config(CONFIG).expect("parses");
    let s = skeleton();
    let b = s.state_budget(&cfg, 0);
    // 🪤 fp32 recurrent state is mandatory; sizing it bf16 would halve this and be wrong.
    assert_eq!(b.kda_recurrent % 4, 0);
    assert!(b.fixed() > 0 && b.per_token() > 0);
    // Growth is linear and dominated by the MLA KV of the 11 sparse text layers.
    assert!(b.dsa_kv_per_token > b.dsa_indexer_per_token);
    let short = b.for_sequence(1);
    let long = b.for_sequence(4096);
    assert_eq!(long - short, 4095 * b.per_token());
    // EP=2 halves the attention state and leaves replicated scratch alone.
    let r = b.per_rank(2);
    assert_eq!(r.kda_recurrent, b.kda_recurrent / 2);
    assert_eq!(r.mhc_highway_per_token, b.mhc_highway_per_token);
}
