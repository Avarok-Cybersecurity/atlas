// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 DSA tensor-parallel shard plan — **MLA heads sharded, indexer replicated**.
//!
//! Shapes measured from `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3` layer 3.
//! Same pure-plan approach as [`crate::layers::glm5next_kda::tp`]: the GPU copy is
//! `tp_shard` reuse, but a wrong head range yields a running model with mixed-up
//! heads, so the row arithmetic is data and gets proven without a GPU.
//!
//! # 🔴 Why the indexer is REPLICATED, not sharded
//!
//! `index_n_heads = 32` divides cleanly at TP=2, so sharding looks free. It is not.
//!
//! The indexer emits a **token selection**, not a partial sum. `index_scores`
//! accumulates `weights[h] * relu(scale · dot)` as a plain sum over heads, so a
//! head-sharded indexer gives each rank only a *partial* score. If each rank then
//! takes its own top-k, **the two ranks attend to different tokens** — no crash, no
//! shape error, just a wrong answer. Sharding is therefore only correct with an
//! all-reduce of the score tensor *before* the top-k.
//!
//! That reduce is the problem. Scores are `[q_rows, n_pools]` with
//! `n_pools = seq / kpool`, so at a 262 144-token context one decode token costs
//! `65 536 × 4 B = 256 KB` **per layer** — about 2.8 MB/token across the 11 DSA
//! layers, on a critical path the DS4F performance review measured as
//! **latency-bound** (86 × 8 KB collectives/token). The weight saved by sharding the
//! indexer is a small fraction of one 249.8 MB layer, once. The trade loses.
//!
//! A second hazard argues the same way: the reference pins a deterministic tiebreak
//! (higher score, then **smaller** pool index) precisely because `torch.topk`'s tie
//! order is undefined. An all-reduced score changes the summation order and can flip
//! a tie that straddles the cutoff.
//!
//! So: indexer replicated, MLA heads sharded, `o_proj` row-parallel with the single
//! all-reduce `qwen3_attention` already performs.
//!
//! # 🪤 Traps this module encodes
//!
//! * **`kv_a_proj_with_mqa` is REPLICATED.** It produces the shared latent KV that
//!   every head decompresses from — it is the MQA part of MLA and has no head axis.
//!   Sharding it starves each rank of half the latent.
//! * **`q_a_proj` / `kv_a_layernorm` / `q_a_layernorm` are replicated** — low-rank
//!   down-projections and their norms, no head structure. Same failure mode as
//!   KDA's `f_a`/`g_a`.
//! * **`q_b_proj` and `kv_b_proj` shard by head**, at `qk_head_dim` and
//!   `nope + v_dim` per head respectively. The two strides differ; using one for the
//!   other silently mixes heads.
//! * **`o_proj` is row-parallel** on `heads * v_head_dim`. Column-slicing gives a
//!   plausible, wrong output.

use anyhow::{Result, bail};

use super::Glm5NextDsaConfig;

/// How one DSA tensor maps onto TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsaShard {
    /// Every rank holds the whole tensor: the indexer, the latent KV projection,
    /// and every low-rank down-projection / norm.
    Replicated,
    /// Leading dim is `heads * per_head` — slice by this rank's head range.
    HeadRows,
    /// Trailing (input) dim is `heads * v_head_dim` — row-parallel GEMM, slice the
    /// input dim, then **all-reduce**.
    HeadCols,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaTensorPlan {
    pub name: &'static str,
    pub kind: DsaShard,
    pub elem_bytes: usize,
    pub full_rows: usize,
    pub full_row_elems: usize,
    pub local_rows: usize,
    pub local_row_elems: usize,
    pub src_row_offset: usize,
    pub src_col_offset: usize,
}

impl DsaTensorPlan {
    pub fn local_bytes(&self) -> usize {
        self.local_rows * self.local_row_elems * self.elem_bytes
    }
    pub fn full_bytes(&self) -> usize {
        self.full_rows * self.full_row_elems * self.elem_bytes
    }
}

const BF16: usize = 2;

/// Per-rank shard plan for one DSA block.
#[derive(Debug, Clone)]
pub struct DsaTpPlan {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub full_heads: usize,
    pub local_heads: usize,
    pub tensors: Vec<DsaTensorPlan>,
}

impl DsaTpPlan {
    /// `cfg.local_heads` is already per-rank (topology divides before loaders run),
    /// so the full count is reconstructed as `local * tp_size` — the same convention
    /// `TpGdnDims::from_config` uses.
    pub fn new(tp_rank: usize, tp_size: usize, cfg: &Glm5NextDsaConfig) -> Result<Self> {
        if tp_rank >= tp_size {
            bail!("tp_rank {tp_rank} >= tp_size {tp_size}");
        }
        cfg.validate()?;
        let local_heads = cfg.local_heads;
        let full_heads = local_heads * tp_size;

        let h = cfg.hidden;
        let qk = cfg.qk_head_dim();
        let kvb_per_head = cfg.qk_nope_head_dim + cfg.v_head_dim;
        let ihd = cfg.index_head_dim;

        let mk = |name, kind, full_rows: usize, full_row_elems: usize| {
            let (local_rows, local_row_elems, src_row_offset, src_col_offset) = match kind {
                DsaShard::Replicated => (full_rows, full_row_elems, 0, 0),
                DsaShard::HeadRows => {
                    let per = full_rows / tp_size;
                    (per, full_row_elems, tp_rank * per, 0)
                }
                DsaShard::HeadCols => {
                    let per = full_row_elems / tp_size;
                    (full_rows, per, 0, tp_rank * per)
                }
            };
            DsaTensorPlan {
                name,
                kind,
                elem_bytes: BF16,
                full_rows,
                full_row_elems,
                local_rows,
                local_row_elems,
                src_row_offset,
                src_col_offset,
            }
        };

        let tensors = vec![
            // ── MLA ──
            mk("q_a_proj", DsaShard::Replicated, cfg.q_lora_rank, h),
            mk("q_a_layernorm", DsaShard::Replicated, cfg.q_lora_rank, 1),
            mk(
                "q_b_proj",
                DsaShard::HeadRows,
                full_heads * qk,
                cfg.q_lora_rank,
            ),
            // 🪤 the shared MQA latent — no head axis, must replicate.
            mk(
                "kv_a_proj_with_mqa",
                DsaShard::Replicated,
                cfg.kv_cache_dim(),
                h,
            ),
            mk("kv_a_layernorm", DsaShard::Replicated, cfg.kv_lora_rank, 1),
            mk(
                "kv_b_proj",
                DsaShard::HeadRows,
                full_heads * kvb_per_head,
                cfg.kv_lora_rank,
            ),
            mk("o_proj", DsaShard::HeadCols, h, full_heads * cfg.v_head_dim),
            // ── indexer: replicated, see the module docs ──
            mk(
                "indexer.wq_b",
                DsaShard::Replicated,
                cfg.index_heads * ihd,
                cfg.q_lora_rank,
            ),
            mk("indexer.wk", DsaShard::Replicated, ihd, h),
            mk("indexer.k_norm.weight", DsaShard::Replicated, ihd, 1),
            // 🪤 LayerNorm, not RMSNorm — the bias is real and is the tell.
            mk("indexer.k_norm.bias", DsaShard::Replicated, ihd, 1),
            mk(
                "indexer.weights_proj",
                DsaShard::Replicated,
                cfg.index_heads,
                h,
            ),
            mk(
                "indexer.index_kpool_compress_gate",
                DsaShard::Replicated,
                ihd,
                h,
            ),
            mk(
                "indexer.index_kpool_compress_ape",
                DsaShard::Replicated,
                cfg.index_kpool,
                ihd,
            ),
        ];

        Ok(Self {
            tp_rank,
            tp_size,
            full_heads,
            local_heads,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&DsaTensorPlan> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn local_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.local_bytes()).sum()
    }
    pub fn full_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.full_bytes()).sum()
    }
    /// `o_proj` is row-parallel, so its partial output needs the reduce.
    pub fn needs_output_all_reduce(&self) -> bool {
        self.tp_size > 1
    }
    /// Bytes replicated on every rank — the part TP cannot remove.
    pub fn replicated_bytes(&self) -> usize {
        self.tensors
            .iter()
            .filter(|t| t.kind == DsaShard::Replicated)
            .map(|t| t.full_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests;
