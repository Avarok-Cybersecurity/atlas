// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 KDA tensor-parallel shard plan — **head-parallel, one all-reduce**.
//!
//! Scoped to `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`. Shapes below are measured
//! from the checkpoint's safetensors headers, not inferred.
//!
//! # Why a pure plan
//!
//! The GPU copy is three lines of [`crate::tp_shard`] reuse. The part that is easy
//! to get silently wrong is *which rows belong to this rank* — and a wrong head
//! range produces a running model with quietly mixed-up heads, not a crash. So the
//! row arithmetic lives here as pure data that can be proven without a GPU, exactly
//! like the EP residency proof.
//!
//! # The pattern this follows
//!
//! Qwen3.5 GDN HeadParallel (`weight_loader/qwen35/load_layers/linear_attn_arms.rs`,
//! helpers in `tp_shard/gdn.rs`): each rank owns a contiguous head range, `out_proj`
//! is row-parallel, and **one** all-reduce follows it. KDA differs from GDN in two
//! ways that matter here:
//!
//! * KDA's `q/k/v_proj` and `q/k/v_conv1d` are **separate tensors on disk** — GDN
//!   fuses them into `in_proj_qkv` / `conv1d`. So KDA needs no segmented slice: each
//!   tensor is sliced independently and the 3-segment trap
//!   (`crate::tp_shard::gdn::segment_copy_plan`) simply does not arise.
//! * KDA has **no `Z` tensor**. The output gate is low-rank `g_a`/`g_b`.
//!
//! # 🪤 Traps this module encodes
//!
//! * **`a_log` is per-HEAD `[64]`; `dt_bias` is per-CHANNEL `[8192]`.** The KDA
//!   module doc already calls this "the highest-risk line". Under TP they shard at
//!   different granularity — heads vs heads×head_dim. Slicing `dt_bias` by head
//!   count silently keeps 1/128th of the right data.
//! * **`o_norm` is `[head_dim]`, NOT `[heads*head_dim]`** — it is per-channel-within-
//!   a-head and therefore **replicated**, never sharded. It is 256 B; a "shard
//!   everything that looks per-head" rule corrupts it.
//! * **`f_a`/`g_a` are down-projections `[rank, hidden]` — replicated.** Only the
//!   `_b` up-projections carry head structure. Sharding an `_a` splits the low-rank
//!   bottleneck and every head reads a truncated gate.
//! * **`o_proj` is row-parallel**: `[hidden, heads*head_dim]` sliced on its INPUT
//!   dim. Each rank produces a partial `[hidden]` that is only correct after the
//!   all-reduce. Column-slicing it instead yields a plausible, wrong output.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;

/// How one KDA tensor maps onto TP ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdaShard {
    /// Every rank holds the whole tensor.
    Replicated,
    /// Leading dim is `heads` — slice by head range.
    HeadRows,
    /// Leading dim is `heads * head_dim` — slice by channel range.
    ChannelRows,
    /// Trailing (input) dim is `heads * head_dim` — row-parallel GEMM, slice the
    /// input dim, then **all-reduce** the output.
    ChannelCols,
}

/// One tensor's placement. `row_elems` is the width of a row in elements; for
/// [`KdaShard::ChannelCols`] the roles invert and `row_elems` is the sharded axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdaTensorPlan {
    pub name: &'static str,
    pub kind: KdaShard,
    pub elem_bytes: usize,
    /// Rows of the full, on-disk tensor.
    pub full_rows: usize,
    /// Elements per row of the full tensor.
    pub full_row_elems: usize,
    /// Rows this rank keeps.
    pub local_rows: usize,
    /// Elements per row this rank keeps (differs from full only for `ChannelCols`).
    pub local_row_elems: usize,
    /// Offset, in rows, of this rank's slice into the full tensor. Always 0 for
    /// `Replicated` and for `ChannelCols` (which slices columns, not rows).
    pub src_row_offset: usize,
    /// Offset, in elements, of this rank's column slice. Non-zero only for
    /// `ChannelCols`.
    pub src_col_offset: usize,
}

impl KdaTensorPlan {
    /// Bytes this rank stores for this tensor.
    pub fn local_bytes(&self) -> usize {
        self.local_rows * self.local_row_elems * self.elem_bytes
    }
    /// Bytes the full tensor occupies on disk.
    pub fn full_bytes(&self) -> usize {
        self.full_rows * self.full_row_elems * self.elem_bytes
    }
}

const BF16: usize = 2;
const F32: usize = 4;

/// The complete per-rank shard plan for one KDA block.
#[derive(Debug, Clone)]
pub struct KdaTpPlan {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub hidden: usize,
    pub head_dim: usize,
    /// Pre-shard head count (all ranks combined).
    pub full_heads: usize,
    /// Heads this rank owns.
    pub local_heads: usize,
    pub conv_kernel: usize,
    /// Low-rank width of the `f`/`g` gate bottleneck.
    pub gate_rank: usize,
    pub tensors: Vec<KdaTensorPlan>,
}

impl KdaTpPlan {
    /// Build from a `ModelConfig` whose linear-head counts are already **per-rank
    /// local** — `serve_phases::topology` divides them before any loader runs, and
    /// `TpGdnDims::from_config` reconstructs `full = local * tp_size` the same way.
    ///
    /// `gate_rank` is not a config key; it is the `f_a_proj` row count read from the
    /// checkpoint (128 for GLM-5.3).
    pub fn from_config(config: &ModelConfig, gate_rank: usize) -> Result<Self> {
        let tp_size = config.tp_world_size.max(1);
        Self::new(
            config.tp_rank,
            tp_size,
            config.hidden_size,
            config.linear_key_head_dim,
            config.linear_num_key_heads * tp_size,
            config.linear_conv_kernel_dim,
            gate_rank,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tp_rank: usize,
        tp_size: usize,
        hidden: usize,
        head_dim: usize,
        full_heads: usize,
        conv_kernel: usize,
        gate_rank: usize,
    ) -> Result<Self> {
        if tp_rank >= tp_size {
            bail!("tp_rank {tp_rank} >= tp_size {tp_size}");
        }
        if tp_size == 0 || head_dim == 0 || full_heads == 0 {
            bail!("degenerate KDA TP geometry: heads={full_heads} head_dim={head_dim}");
        }
        if !full_heads.is_multiple_of(tp_size) {
            bail!("KDA TP requires heads ({full_heads}) divisible by tp_size ({tp_size})");
        }
        let local_heads = full_heads / tp_size;

        // The fused conv+L2 kernel's contract, re-checked on the LOCAL width: it
        // reads 2 heads per 256-thread block over the q|k channels only.
        let local_qk_channels = 2 * local_heads * head_dim;
        if !local_qk_channels.is_multiple_of(256) {
            bail!(
                "KDA TP: local qk_channels ({local_qk_channels}) must be a multiple of 256 \
                 (heads={local_heads}, head_dim={head_dim}); causal_conv1d_update_l2norm \
                 hardcodes 2 heads per 256-thread block"
            );
        }

        let full_ch = full_heads * head_dim;
        let local_ch = local_heads * head_dim;
        let ch_off = tp_rank * local_ch;
        let head_off = tp_rank * local_heads;

        let rows = |name, kind, elem_bytes, full_rows, full_row_elems| {
            let (local_rows, local_row_elems, src_row_offset, src_col_offset) = match kind {
                KdaShard::Replicated => (full_rows, full_row_elems, 0, 0),
                KdaShard::HeadRows => (full_rows / tp_size, full_row_elems, head_off, 0),
                KdaShard::ChannelRows => (full_rows / tp_size, full_row_elems, ch_off, 0),
                // Row-parallel: slice the INPUT (column) dim, keep every row.
                KdaShard::ChannelCols => (full_rows, full_row_elems / tp_size, 0, ch_off),
            };
            KdaTensorPlan {
                name,
                kind,
                elem_bytes,
                full_rows,
                full_row_elems,
                local_rows,
                local_row_elems,
                src_row_offset,
                src_col_offset,
            }
        };

        let tensors = vec![
            // [heads*head_dim, hidden] — column-parallel by output channel.
            rows("q_proj", KdaShard::ChannelRows, BF16, full_ch, hidden),
            rows("k_proj", KdaShard::ChannelRows, BF16, full_ch, hidden),
            rows("v_proj", KdaShard::ChannelRows, BF16, full_ch, hidden),
            // [heads*head_dim, conv_kernel] each — SEPARATE on disk, so each slices
            // independently. (GDN fuses these; KDA does not.)
            rows(
                "q_conv1d",
                KdaShard::ChannelRows,
                BF16,
                full_ch,
                conv_kernel,
            ),
            rows(
                "k_conv1d",
                KdaShard::ChannelRows,
                BF16,
                full_ch,
                conv_kernel,
            ),
            rows(
                "v_conv1d",
                KdaShard::ChannelRows,
                BF16,
                full_ch,
                conv_kernel,
            ),
            // Low-rank forget/output gates: `_a` down-projects (replicate),
            // `_b` up-projects into channel space (shard).
            rows("f_a_proj", KdaShard::Replicated, BF16, gate_rank, hidden),
            rows("f_b_proj", KdaShard::ChannelRows, BF16, full_ch, gate_rank),
            rows("g_a_proj", KdaShard::Replicated, BF16, gate_rank, hidden),
            rows("g_b_proj", KdaShard::ChannelRows, BF16, full_ch, gate_rank),
            // beta — one row per HEAD.
            rows("b_proj", KdaShard::HeadRows, BF16, full_heads, hidden),
            // 🪤 per-HEAD ...
            rows("A_log", KdaShard::HeadRows, F32, full_heads, 1),
            // 🪤 ... versus per-CHANNEL. Different granularity, same block.
            rows("dt_bias", KdaShard::ChannelRows, F32, full_ch, 1),
            // 🪤 [head_dim] — within-head, so REPLICATED.
            rows("o_norm", KdaShard::Replicated, BF16, head_dim, 1),
            // [hidden, heads*head_dim] — row-parallel, all-reduce after.
            rows("o_proj", KdaShard::ChannelCols, BF16, hidden, full_ch),
        ];

        Ok(Self {
            tp_rank,
            tp_size,
            hidden,
            head_dim,
            full_heads,
            local_heads,
            conv_kernel,
            gate_rank,
            tensors,
        })
    }

    pub fn get(&self, name: &str) -> Option<&KdaTensorPlan> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Total bytes this rank stores for one KDA block.
    pub fn local_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.local_bytes()).sum()
    }

    /// Total bytes one KDA block occupies on disk.
    pub fn full_bytes(&self) -> usize {
        self.tensors.iter().map(|t| t.full_bytes()).sum()
    }

    /// Whether the layer must all-reduce after `o_proj`. False at `tp_size == 1`,
    /// where the row-parallel slice is the whole tensor and the reduce is a no-op.
    pub fn needs_output_all_reduce(&self) -> bool {
        self.tp_size > 1
    }
}

#[cfg(test)]
mod tests;
