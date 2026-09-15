// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! Real-file oracle for DeepSeek-V4.1 Flash Q2_K: split-shard resolution and
//! `ModelConfig` construction straight from the seven-shard GGUF.
//!
//! Gated `#[ignore]` so CI without the 246 GiB checkpoint is unaffected. Run:
//!   ATLAS_SKIP_BUILD=1 cargo test -p spark-runtime -- --ignored deepseek_v41
//!
//! Every constant below was read out of the published file, not from the HF
//! `config.json`, so a converter change that alters the GGUF breaks this test
//! rather than silently producing a wrong graph.

use std::path::Path;

use crate::weights::{config_from_gguf_dir, find_gguf, find_gguf_shards};

const MODEL_DIR: &str = "/home/rstesiak/models/dsv41-q2k";

fn dir() -> Option<&'static Path> {
    let p = Path::new(MODEL_DIR);
    p.is_dir().then_some(p)
}

/// All seven shards resolve from shard 0, and the per-shard tensor counts sum
/// to `split.tensors.count`. This is the check that catches a lost shard: each
/// remaining file is individually valid and checksum-clean, so only the sum
/// notices.
#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_resolves_all_seven_shards() {
    let Some(d) = dir() else {
        panic!("model dir {MODEL_DIR} not present");
    };
    let first = find_gguf(d).expect("find shard 0");
    assert!(
        first.to_string_lossy().contains("00001-of-00007"),
        "find_gguf returned {first:?}, expected the -00001-of-00007 shard"
    );

    let set = find_gguf_shards(&first).expect("resolve shard set");
    assert_eq!(set.paths.len(), 7, "shard count");
    assert_eq!(set.total_tensors, 1046, "split.tensors.count");
    for (i, p) in set.paths.iter().enumerate() {
        let want = format!("{:05}-of-00007", i + 1);
        assert!(
            p.to_string_lossy().contains(&want),
            "shard {i} is {p:?}, expected to contain {want}"
        );
    }
}

/// The GGUF alone yields a complete V4.1 `ModelConfig`: no `config.json`, no
/// sidecar, no defaults standing in for architecture facts.
#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_config_from_gguf_metadata() {
    let Some(d) = dir() else {
        panic!("model dir {MODEL_DIR} not present");
    };
    let c = config_from_gguf_dir(d).expect("build ModelConfig from GGUF");

    assert_eq!(c.model_type, "deepseek_v41");
    // Core dimensions.
    assert_eq!(c.hidden_size, 5120);
    assert_eq!(c.num_hidden_layers, 40);
    assert_eq!(c.vocab_size, 129_280);
    assert_eq!(c.num_attention_heads, 64);
    assert_eq!(c.num_key_value_heads, 1, "MLA: one latent KV head");
    assert_eq!(c.head_dim, 512);
    assert_eq!(c.max_position_embeddings, 1_048_576);

    // MoE.
    assert_eq!(c.num_experts, 384);
    assert_eq!(c.num_experts_per_tok, 6);
    assert_eq!(c.moe_intermediate_size, 2304);
    assert_eq!(c.scoring_func, "sqrtsoftplus");
    assert!(c.norm_topk_prob);
    assert!((c.routed_scaling_factor - 1.5).abs() < 1e-9);
    // Pure MoE: the file ships no dense feed_forward_length.
    assert_eq!(c.intermediate_size, 0);

    // MLA geometry.
    assert_eq!(c.q_lora_rank, 1280);
    assert_eq!(c.o_lora_rank, 1024);
    assert_eq!(c.o_groups, 8);
    assert_eq!(c.kv_lora_rank, 512);
    assert_eq!(c.rotary_dim, 64, "partial RoPE over a 512-wide head");

    // Sparse attention.
    assert_eq!(c.index_n_heads, 32);
    assert_eq!(c.index_head_dim, 128);
    assert_eq!(c.index_topk, 512);
    assert_eq!(c.sliding_window, 128);
    // 43 entries for 40 blocks: the three DSpark stages carry their own ratio.
    assert_eq!(c.compress_ratios.len(), 43);
    assert!((c.compress_rope_theta - 160_000.0).abs() < 1.0);

    // Hyper-connections.
    assert_eq!(c.hc_mult, 4);
    assert_eq!(c.hc_sinkhorn_iters, 20);

    // V4.1 has no hash-routed layers (V4 did).
    assert_eq!(c.num_hash_layers, 0);

    // Engram: two tables, and the hash parameters travel in the file.
    assert_eq!(c.engram_layer_ids, vec![1, 14]);
    assert_eq!(c.engram_max_ngram_size, 4);
    assert_eq!(c.engram_n_heads, 8);
    assert_eq!(c.engram_head_dim, 256);
    assert_eq!(c.engram_pad_token_id, 2);
    assert_eq!(c.engram_multipliers.len(), 8);
    assert_eq!(c.engram_primes.len(), 48);
    assert_eq!(c.engram_offsets.len(), 48);
}

/// EVERY tensor in the real seven-shard file translates. An unmapped name is a
/// silently-missing weight at load time, so the assertion is total: 1,046 of
/// 1,046, with the offenders printed rather than a bare count mismatch.
#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_every_tensor_name_translates() {
    use crate::weights::gguf::names::{self, GgufName};
    use crate::weights::gguf::sidecar;

    let Some(d) = dir() else {
        panic!("model dir {MODEL_DIR} not present");
    };
    let first = find_gguf(d).expect("find shard 0");
    let set = find_gguf_shards(&first).expect("resolve shard set");

    let mut total = 0usize;
    let mut unmapped: Vec<String> = Vec::new();
    let mut stacks = 0usize;
    let mut direct = 0usize;
    for p in &set.paths {
        let (_f, _m, g) = sidecar::open_gguf(p).expect("open shard");
        for t in &g.tensors {
            total += 1;
            match names::translate(&t.name, "deepseek41") {
                Some(GgufName::ExpertStack { .. }) => stacks += 1,
                Some(GgufName::Direct(_)) => direct += 1,
                Some(GgufName::Drop) => {}
                None => unmapped.push(t.name.clone()),
            }
        }
    }

    assert_eq!(total, 1046, "tensor total across all shards");
    assert!(
        unmapped.is_empty(),
        "{} of {total} tensor names have no deepseek41 translation: {:?}",
        unmapped.len(),
        &unmapped[..unmapped.len().min(20)]
    );
    // 3 stacked expert projections per block x 40 blocks.
    assert_eq!(stacks, 120, "stacked expert tensors");
    assert_eq!(direct + stacks, total, "every name is Direct or ExpertStack");
}

/// The tensors that exist on only SOME layers appear exactly as often as the
/// config says they should. This is the cross-check that the shared-attention
/// story in the metadata matches the weights actually shipped.
#[test]
#[ignore = "requires the on-disk DeepSeek-V4.1-Flash Q2_K shards"]
fn deepseek_v41_sparse_tensors_match_source_layer_counts() {
    use crate::weights::gguf::sidecar;
    use std::collections::HashMap;

    let Some(d) = dir() else {
        panic!("model dir {MODEL_DIR} not present");
    };
    let first = find_gguf(d).expect("find shard 0");
    let set = find_gguf_shards(&first).expect("resolve shard set");

    let mut counts: HashMap<String, usize> = HashMap::new();
    for p in &set.paths {
        let (_f, _m, g) = sidecar::open_gguf(p).expect("open shard");
        for t in &g.tensors {
            if let Some(rest) = t.name.strip_prefix("blk.")
                && let Some((_, suffix)) = rest.split_once('.')
            {
                *counts.entry(suffix.to_string()).or_default() += 1;
            }
        }
    }
    let n = |k: &str| counts.get(k).copied().unwrap_or(0);

    // Every block has its own MLA attention and its own MoE.
    assert_eq!(n("attn_q_a.weight"), 40);
    assert_eq!(n("ffn_gate_inp.weight"), 40);
    // ...but KV is produced by four layers and shared (kv_source_layer_ids).
    assert_eq!(n("attn_compressor_kv.weight"), 4, "kv_source_layer_ids");
    assert_eq!(n("attn_compressor_norm.weight"), 4);
    // The indexer's K side is shared four ways, its Q side eight.
    assert_eq!(n("indexer.attn_k.weight"), 4);
    assert_eq!(n("indexer.k_norm.weight"), 4);
    assert_eq!(n("indexer.attn_q_b.weight"), 8, "index_source_layer_ids");
    assert_eq!(n("indexer.proj.weight"), 8);
    // Engram lives on exactly two layers.
    assert_eq!(n("engram_embd.weight"), 2, "engram_layer_ids");
    assert_eq!(n("engram_q.weight"), 2);
    assert_eq!(n("engram_k.weight"), 2);
    assert_eq!(n("engram_wkv.weight"), 2);
    // The vision routing bias ships on every layer even in this text quant.
    assert_eq!(n("exp_probs_b_vl.bias"), 40);
}
