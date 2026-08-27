// SPDX-License-Identifier: AGPL-3.0-only

//! Which cross-architecture kernel families this model's config says exist.
//!
//! `Qwen3AttentionLayer` is the shared attention stack for a dozen
//! architectures, so its constructor used to probe for MLA, hyper-connection,
//! DeepSeek sparse attention and the HDIM>256 arms UNCONDITIONALLY — a
//! `try_kernel` per family, on every model, with a documented fallback when it
//! came back handle 0.
//!
//! Those probes are correct at runtime and wrong as an interface. "Ask and see"
//! is an implicit default (PCND): the config already states, exactly, whether
//! the model has MLA (`kv_lora_rank`), hyper-connections (`hc_mult`) or
//! compressed attention (`compress_ratios`), so the answer is knowable before
//! the lookup. Issuing it anyway put ~24 permanently-unresolvable lookups into
//! every 27B boot audit, and a report that is 24 parts noise is a report in
//! which four genuinely-missing GDN kernels are invisible.
//!
//! Same shape as the `config.final_logit_softcapping > 0.0` gate in
//! `model::impl_a1`: derive the gate from config, then do not look up what this
//! model cannot use.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

use atlas_core::config::ModelConfig;

/// Per-family presence flags, derived once per layer construction.
#[derive(Clone, Copy, Debug)]
pub(super) struct ArchProbes {
    /// Multi-head Latent Attention (DeepSeek-V2+/-V4, Mistral Small 4).
    /// `kv_lora_rank` IS the compressed-KV latent dimension: 0 means the model
    /// has no latent cache, so nothing can dispatch an MLA kernel.
    pub mla: bool,
    /// Manifold-Constrained Hyper-Connections (DeepSeek-V4, `hc_mult` = 4).
    /// 0 on every other model.
    pub hyper_connection: bool,
    /// DeepSeek sparse / hybrid compressed attention. `compress_ratios` is
    /// per-layer and empty when every layer is full attention.
    pub compressed_attn: bool,
    /// The HDIM>128 kernel arms (`*_512`). The dispatch sites gate on
    /// `head_dim > 256`, and `config.head_dim` is the MAX across layers for
    /// heterogeneous models (Gemma-4 sizes buffers from it), so this is a
    /// superset of the per-layer overrides and can never gate a live path off.
    /// MLA carries its own >256 shapes (576-dim compressed KV), hence the or.
    pub wide_head_dim: bool,
}

impl ArchProbes {
    pub(super) fn from_config(config: &ModelConfig) -> Self {
        Self {
            mla: config.kv_lora_rank > 0,
            hyper_connection: config.hc_mult > 0,
            compressed_attn: config.compress_ratios.iter().any(|&r| r > 0),
            wide_head_dim: config.head_dim > 256 || config.kv_lora_rank > 0,
        }
    }
}

/// `try_kernel` when `enabled`, otherwise `KernelHandle(0)` with NO lookup.
///
/// Skipping the lookup — rather than discarding its result — is the whole
/// point: a lookup that is never issued leaves no failed row in the audit, so
/// what remains in the audit is what someone has to act on.
///
/// `#[track_caller]` so the audit still names the dispatch site in `init.rs`
/// rather than this line.
#[track_caller]
pub(super) fn gated(enabled: bool, gpu: &dyn GpuBackend, module: &str, func: &str) -> KernelHandle {
    if enabled {
        crate::layers::try_kernel(gpu, module, func)
    } else {
        KernelHandle(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plain GQA model (qwen3.6-27B shape) must claim NONE of the
    /// cross-architecture families — that is the ~24-lookup saving.
    #[test]
    fn a_plain_gqa_model_probes_for_nothing() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.head_dim = 128;
        let p = ArchProbes::from_config(&cfg);
        assert!(!p.mla);
        assert!(!p.hyper_connection);
        assert!(!p.compressed_attn);
        assert!(!p.wide_head_dim);
    }

    #[test]
    fn each_architecture_signal_enables_only_its_kernel_families() {
        for (name, configure, expected) in [
            (
                "MLA",
                (512usize, 0usize, Vec::new(), 128usize),
                [true, false, false, true],
            ),
            (
                "hyper connection",
                (0, 4, Vec::new(), 128),
                [false, true, false, false],
            ),
            (
                "compressed attention",
                (0, 0, vec![0, 8, 8], 128),
                [false, false, true, false],
            ),
            (
                "wide head",
                (0, 0, Vec::new(), 512),
                [false, false, false, true],
            ),
        ] {
            let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
            (
                cfg.kv_lora_rank,
                cfg.hc_mult,
                cfg.compress_ratios,
                cfg.head_dim,
            ) = configure;
            let p = ArchProbes::from_config(&cfg);
            assert_eq!(
                [
                    p.mla,
                    p.hyper_connection,
                    p.compressed_attn,
                    p.wide_head_dim
                ],
                expected,
                "{name}"
            );
        }
    }

    /// Gemma-4 sizes buffers from the MAX per-layer head_dim, so the config
    /// value is a superset of what any layer overrides to. Gating on it can
    /// never switch a live dispatch path off.
    #[test]
    fn a_heterogeneous_model_gates_on_its_max_head_dim() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.head_dim = 512;
        assert!(ArchProbes::from_config(&cfg).wide_head_dim);
    }

    /// An all-full-attention `compress_ratios` (every entry 0) is not
    /// compressed attention — the vector's presence is not the signal.
    #[test]
    fn all_zero_compress_ratios_is_not_compressed_attention() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.compress_ratios = vec![0, 0, 0];
        assert!(!ArchProbes::from_config(&cfg).compressed_attn);
    }
}
