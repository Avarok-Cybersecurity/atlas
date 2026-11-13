// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: resolve the port sprint's (`port/consumer-opt`) serving
//! rows, **baked first, environment second**, and print them as one boot line.
//!
//! The rows are the `port` sub-table of [`avarok_kernels::TargetDefaults`]
//! ([`avarok_kernels::target_defaults_port::PortDefaults`]). Each row's
//! resolver lives in its own file, `layers/ops/port_rows/<row>.rs`, owned by
//! the lane that adds the row; this file only collects them. The override
//! grammar and the [`Resolved`] / [`Source`] types are
//! [`super::target_defaults`]'s, re-exported here so a row file needs one
//! import path.
//!
//! [`resolved_port`] is a separate `OnceLock` from
//! [`super::target_defaults::resolved`], with the same environment reader, so
//! the pre-sprint table and its boot line are untouched by the sprint. A lane
//! with a CLI flag publishes it from `spark-server`'s `port_apply::apply`,
//! which runs before any model is built and before this table is first read.
//!
//! Every row is OFF (or its baseline) on every target when it lands, so a
//! binary with an empty environment serves exactly as it did before the row.
//!
//! Rows arrive through the coordinator's fragment apply pass: each `@port:`
//! marker below is an insertion point, and text is inserted immediately BEFORE
//! the marker line.

pub use super::target_defaults::{Resolved, Source};

/// Every port row this target declares, resolved against the environment.
///
/// Field order is the order the boot line prints them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortLevers {
    /// `kernels/<hw>` this binary was compiled from, for the log line only.
    pub hw: &'static str,
    /// Collect the per-request logprob confidence summary for every chat request and feed the `atlas_conf_*` series on /metrics.
    pub conf_summary: Resolved<bool>,
    /// D-Cut v2 for the batched MTP verify: keep the number of optional draft rows that maximizes expected tokens per second instead of a fixed 0.75.
    pub mtp_dcut_cost_model: Resolved<bool>,
    /// Update a full batched decode or verify CUDA-graph cache's least-recently-used executable in place with `cuGraphExecUpdate` instead of destroying and re-instantiating it.
    pub graph_exec_update: Resolved<bool>,
    /// GDN write-on-accept for the batched speculative verify: fold the accepted rows into the GDN state once after the verdict.
    pub gdn_woa_mtp: Resolved<bool>,
    /// Which LM-head rows the MTP draft-only NVFP4 head holds: `id` (the first token ids) or `freq` (the most frequent tokens).
    pub mtp_vocab_rank: Resolved<&'static str>,
    /// Run the MTP propose chain device-resident and read every draft back once per round instead of one blocking readback per draft.
    pub mtp_propose_graph: Resolved<bool>,
    /// Pick MTP verify positions on the device instead of copying the verify logits to the host and running the pre-sample pipeline there.
    pub mtp_device_accept: Resolved<bool>,
    /// Copy drafter for MTP speculative rounds: which copy-drafter plans may be used (`off`, `chain`, `rounds`, `auto`).
    pub copy_draft: Resolved<&'static str>,
    /// The dense FFN's FP8 speculative-verify rows (M = 2..8) on the sm_89 batch-M GEMVs: one pass over each FP8 weight for all M rows.
    pub ffn_fp8_verify_gemv: Resolved<bool>,
    /// The sm_89 FP8 decode and verify GEMVs load their weights with an explicit evict-first L2 policy.
    pub l2_evict_first_weights: Resolved<bool>,
    /// The prefill FFN-norm of a Qwen3 / Qwen3.5 layer also writes the dense FFN's W8A8 gate/up activation (E4M3 codes and scales) in the same launch.
    pub rmsnorm_fp8_quant_fused: Resolved<bool>,
    /// Serve every 4-bit BF16 load route's Marlin-servable projection by round-to-nearest int4 at group 128 on vLLM's Marlin W4A16 GEMM.
    pub marlin_w4a16: Resolved<bool>,
    /// A dense Qwen3.5-family GGUF keeps its IQ4_XS, Q4_K and Q6_K projections as the file's raw super-blocks in VRAM for the sm_89 keep-packed GEMVs and MMQ GEMMs.
    pub gguf_native_iq4: Resolved<bool>,
    /// EXL3 trellis matrices are served from their packed codes on sm_89.
    pub weights_exl3: Resolved<bool>,
    /// On a plain-BF16 dense Qwen3.5 checkpoint, quantize the dense FFN's gate, up and down once at load into MLX affine codes and serve them through W4Linear.
    pub dense_affine4_from_bf16: Resolved<bool>,
    /// On a plain-BF16 dense Qwen3.5 checkpoint, quantize each GDN layer's in-projection and out_proj once at load to MLX affine codes.
    pub gdn_affine4_from_bf16: Resolved<bool>,
    /// On a plain-BF16 dense Qwen3.5 checkpoint, quantize each full-attention layer's q, k, v and o once at load to MLX affine codes.
    pub attn_affine4_from_bf16: Resolved<bool>,
    /// The affine routes' prompt GEMM quantizes the activation rows to int8 per group and runs the W4A8 int8 tensor-core GEMM over the same 4-bit codes.
    pub affine4_prefill_int8: Resolved<bool>,
    /// On a plain-BF16 dense Qwen3.5 checkpoint, quantize the dense FFN's gate, up and down once at load with the NVFP4 quantizer and serve them through W4Linear.
    pub dense_nvfp4_from_bf16: Resolved<bool>,
    /// The input embedding table is copied once at load into page-locked host memory and read through the region's device-visible alias.
    pub embed_host_offload: Resolved<bool>,
    /// Batched sequence swap: gather a swapped-out sequence's image with batched async copies into one reused page-locked staging buffer.
    pub kv_batched_swap: Resolved<bool>,
    /// SSM tier blob store eviction order: a capped host-RAM store gives up the least recently used blob instead of the oldest insert.
    pub ssm_store_lru: Resolved<bool>,
    /// Host-RAM KV tier: copy every full prompt block the prefix cache newly holds to a pinned host pool and restore evicted prefix blocks from it.
    pub kv_host_tier: Resolved<bool>,
    /// Asynchronous host-tier KV restore: park a restoring request outside the batch while its host-to-device copy runs.
    pub kv_restore_async: Resolved<bool>,
    /// Disk tier and warm restart for the host KV tier: demote evicted host blocks to one record file each and adopt a previous run's records.
    pub kv_tier_disk: Resolved<bool>,
    /// Radix prefix-cache eviction takes its victim from an ordered (last_access, node id) index of evictable leaves instead of scanning every node.
    pub radix_lru_index: Resolved<bool>,
    /// A prefix-cache hit may end inside a KV block: the chunked prefill copies that block into a private block instead of sharing the donor's writable tail.
    pub prefix_cow_tail: Resolved<bool>,
    // @port:fields
}

/// The port table, as a pure function of the baked declaration and a variable
/// lookup, so a CPU test can resolve it for any target without touching the
/// process environment.
#[allow(unused_variables, unused_mut)]
pub fn resolve(
    defaults: &avarok_kernels::TargetDefaults,
    mut var: impl FnMut(&str) -> Option<String>,
) -> PortLevers {
    let d = &defaults.port;
    PortLevers {
        hw: defaults.hw,
        conf_summary: super::port_rows::conf_summary::resolve(
            d.conf_summary,
            var(super::port_rows::conf_summary::ENV).as_deref(),
        ),
        mtp_dcut_cost_model: super::port_rows::mtp_dcut_cost_model::resolve(
            d.mtp_dcut_cost_model,
            var(super::port_rows::mtp_dcut_cost_model::ENV).as_deref(),
        ),
        graph_exec_update: super::port_rows::graph_exec_update::resolve(
            d.graph_exec_update,
            var(super::port_rows::graph_exec_update::ENV).as_deref(),
        ),
        gdn_woa_mtp: super::port_rows::gdn_woa_mtp::resolve(
            d.gdn_woa_mtp,
            var(super::port_rows::gdn_woa_mtp::ENV).as_deref(),
        ),
        mtp_vocab_rank: super::port_rows::mtp_vocab_rank::resolve(
            d.mtp_vocab_rank,
            var(super::port_rows::mtp_vocab_rank::ENV).as_deref(),
        ),
        mtp_propose_graph: super::port_rows::mtp_propose_graph::resolve(
            d.mtp_propose_graph,
            var(super::port_rows::mtp_propose_graph::ENV).as_deref(),
        ),
        mtp_device_accept: super::port_rows::mtp_device_accept::resolve(
            d.mtp_device_accept,
            var(super::port_rows::mtp_device_accept::ENV).as_deref(),
        ),
        copy_draft: super::port_rows::copy_draft::resolve(
            d.copy_draft,
            var(super::port_rows::copy_draft::ENV).as_deref(),
        ),
        ffn_fp8_verify_gemv: super::port_rows::ffn_fp8_verify_gemv::resolve(
            d.ffn_fp8_verify_gemv,
            var(super::port_rows::ffn_fp8_verify_gemv::ENV).as_deref(),
        ),
        l2_evict_first_weights: super::port_rows::l2_evict_first_weights::resolve(
            d.l2_evict_first_weights,
            var(super::port_rows::l2_evict_first_weights::ENV).as_deref(),
        ),
        rmsnorm_fp8_quant_fused: super::port_rows::rmsnorm_fp8_quant_fused::resolve(
            d.rmsnorm_fp8_quant_fused,
            var(super::port_rows::rmsnorm_fp8_quant_fused::ENV).as_deref(),
        ),
        marlin_w4a16: super::port_rows::marlin_w4a16::resolve(
            d.marlin_w4a16,
            var(super::port_rows::marlin_w4a16::ENV).as_deref(),
        ),
        gguf_native_iq4: super::port_rows::gguf_native_iq4::resolve(
            d.gguf_native_iq4,
            var(super::port_rows::gguf_native_iq4::ENV).as_deref(),
        ),
        weights_exl3: super::port_rows::weights_exl3::resolve(
            d.weights_exl3,
            var(super::port_rows::weights_exl3::ENV).as_deref(),
        ),
        dense_affine4_from_bf16: super::port_rows::dense_affine4_from_bf16::resolve(
            d.dense_affine4_from_bf16,
            var(super::port_rows::dense_affine4_from_bf16::ENV).as_deref(),
        ),
        gdn_affine4_from_bf16: super::port_rows::gdn_affine4_from_bf16::resolve(
            d.gdn_affine4_from_bf16,
            var(super::port_rows::gdn_affine4_from_bf16::ENV).as_deref(),
        ),
        attn_affine4_from_bf16: super::port_rows::attn_affine4_from_bf16::resolve(
            d.attn_affine4_from_bf16,
            var(super::port_rows::attn_affine4_from_bf16::ENV).as_deref(),
        ),
        affine4_prefill_int8: super::port_rows::affine4_prefill_int8::resolve(
            d.affine4_prefill_int8,
            var(super::port_rows::affine4_prefill_int8::ENV).as_deref(),
        ),
        dense_nvfp4_from_bf16: super::port_rows::dense_nvfp4_from_bf16::resolve(
            d.dense_nvfp4_from_bf16,
            var(super::port_rows::dense_nvfp4_from_bf16::ENV).as_deref(),
        ),
        embed_host_offload: super::port_rows::embed_host_offload::resolve(
            d.embed_host_offload,
            var(super::port_rows::embed_host_offload::ENV).as_deref(),
        ),
        kv_batched_swap: super::port_rows::kv_batched_swap::resolve(
            d.kv_batched_swap,
            var(super::port_rows::kv_batched_swap::ENV).as_deref(),
        ),
        ssm_store_lru: super::port_rows::ssm_store_lru::resolve(
            d.ssm_store_lru,
            var(super::port_rows::ssm_store_lru::ENV).as_deref(),
        ),
        kv_host_tier: super::port_rows::kv_host_tier::resolve(
            d.kv_host_tier,
            var(super::port_rows::kv_host_tier::ENV).as_deref(),
        ),
        kv_restore_async: super::port_rows::kv_restore_async::resolve(
            d.kv_restore_async,
            var(super::port_rows::kv_restore_async::ENV).as_deref(),
        ),
        kv_tier_disk: super::port_rows::kv_tier_disk::resolve(
            d.kv_tier_disk,
            var(super::port_rows::kv_tier_disk::ENV).as_deref(),
        ),
        radix_lru_index: super::port_rows::radix_lru_index::resolve(
            d.radix_lru_index,
            var(super::port_rows::radix_lru_index::ENV).as_deref(),
        ),
        prefix_cow_tail: super::port_rows::prefix_cow_tail::resolve(
            d.prefix_cow_tail,
            var(super::port_rows::prefix_cow_tail::ENV).as_deref(),
        ),
        // @port:resolve
    }
}

/// The process-wide resolution of the port rows.
///
/// `OnceLock`-cached for the reason [`super::target_defaults::resolved`] is:
/// rows are read on hot paths, and a route must stay constant across CUDA-graph
/// replays.
pub fn resolved_port() -> &'static PortLevers {
    static LEVERS: std::sync::OnceLock<PortLevers> = std::sync::OnceLock::new();
    LEVERS.get_or_init(|| {
        resolve(&avarok_kernels::TARGET_DEFAULTS, |name| {
            std::env::var(name).ok()
        })
    })
}

/// `port defaults (<hw>): …`, one line naming every resolved port row and
/// which came from the environment. Logged by `spark-server` right after the
/// `target defaults` line.
pub fn port_boot_line() -> String {
    format_port(resolved_port())
}

/// [`port_boot_line`] over a table the caller already has (pure, so a CPU test
/// can grade the line without sealing the `OnceLock`).
pub fn format_port(l: &PortLevers) -> String {
    let hw = if l.hw.is_empty() { "unknown" } else { l.hw };
    let mut s = format!("port defaults ({hw}):");
    let header = s.len();
    push(&mut s, "conf_summary", &l.conf_summary);
    push(&mut s, "mtp_dcut_cost_model", &l.mtp_dcut_cost_model);
    push(&mut s, "graph_exec_update", &l.graph_exec_update);
    push(&mut s, "gdn_woa_mtp", &l.gdn_woa_mtp);
    push(&mut s, "mtp_vocab_rank", &l.mtp_vocab_rank);
    push(&mut s, "mtp_propose_graph", &l.mtp_propose_graph);
    push(&mut s, "mtp_device_accept", &l.mtp_device_accept);
    push(&mut s, "copy_draft", &l.copy_draft);
    push(&mut s, "ffn_fp8_verify_gemv", &l.ffn_fp8_verify_gemv);
    push(&mut s, "l2_evict_first_weights", &l.l2_evict_first_weights);
    push(
        &mut s,
        "rmsnorm_fp8_quant_fused",
        &l.rmsnorm_fp8_quant_fused,
    );
    push(&mut s, "marlin_w4a16", &l.marlin_w4a16);
    push(&mut s, "gguf_native_iq4", &l.gguf_native_iq4);
    push(&mut s, "weights_exl3", &l.weights_exl3);
    push(
        &mut s,
        "dense_affine4_from_bf16",
        &l.dense_affine4_from_bf16,
    );
    push(&mut s, "gdn_affine4_from_bf16", &l.gdn_affine4_from_bf16);
    push(&mut s, "attn_affine4_from_bf16", &l.attn_affine4_from_bf16);
    push(&mut s, "affine4_prefill_int8", &l.affine4_prefill_int8);
    push(&mut s, "dense_nvfp4_from_bf16", &l.dense_nvfp4_from_bf16);
    push(&mut s, "embed_host_offload", &l.embed_host_offload);
    push(&mut s, "kv_batched_swap", &l.kv_batched_swap);
    push(&mut s, "ssm_store_lru", &l.ssm_store_lru);
    push(&mut s, "kv_host_tier", &l.kv_host_tier);
    push(&mut s, "kv_restore_async", &l.kv_restore_async);
    push(&mut s, "kv_tier_disk", &l.kv_tier_disk);
    push(&mut s, "radix_lru_index", &l.radix_lru_index);
    push(&mut s, "prefix_cow_tail", &l.prefix_cow_tail);
    // @port:boot
    if s.len() == header {
        s.push_str(" none");
    }
    s
}

/// A value a port row can hold, as the boot line prints it.
pub trait PortValue: Copy {
    /// The printed form: `on`/`off`, a number (`max` for `u32::MAX`), or the
    /// string itself.
    fn show(self) -> String;
}

impl PortValue for bool {
    fn show(self) -> String {
        (if self { "on" } else { "off" }).to_string()
    }
}

impl PortValue for u32 {
    fn show(self) -> String {
        // `u32::MAX` is a no-cap baseline, not a chosen bound.
        if self == u32::MAX {
            "max".to_string()
        } else {
            self.to_string()
        }
    }
}

impl PortValue for &'static str {
    fn show(self) -> String {
        self.to_string()
    }
}

/// Append ` <name>=<value>` to the boot line, with ` (env)` when the
/// environment overrode the target's declaration.
pub fn push<T: PortValue>(s: &mut String, name: &str, r: &Resolved<T>) {
    s.push(' ');
    s.push_str(name);
    s.push('=');
    s.push_str(&r.value.show());
    s.push_str(r.source.tag());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_prints_each_value_type_and_its_source() {
        let mut s = String::new();
        let on_env = Resolved {
            value: true,
            source: Source::Env,
        };
        push(&mut s, "a", &on_env);
        let cap = Resolved {
            value: u32::MAX,
            source: Source::Target,
        };
        push(&mut s, "b", &cap);
        let word = Resolved {
            value: "off",
            source: Source::Target,
        };
        push(&mut s, "c", &word);
        assert_eq!(s, " a=on (env) b=max c=off");
    }

    #[test]
    fn the_line_names_its_target() {
        let l = resolve(&avarok_kernels::TARGET_DEFAULTS, |_| None);
        let line = format_port(&l);
        let hw = if l.hw.is_empty() { "unknown" } else { l.hw };
        assert!(
            line.starts_with(&format!("port defaults ({hw}):")),
            "{line}"
        );
    }

    #[test]
    fn an_empty_environment_resolves_every_row_from_the_target() {
        let l = resolve(&avarok_kernels::TARGET_DEFAULTS, |_| None);
        assert!(!format_port(&l).contains(" (env)"));
    }
}
