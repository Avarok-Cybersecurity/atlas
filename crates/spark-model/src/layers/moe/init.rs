// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::new constructor.

use super::*;

impl MoeLayer {
    pub fn new(
        weights: MoeWeights,
        num_experts: usize,
        gate_nvfp4: Option<QuantizedWeight>,
        gpu: &dyn GpuBackend,
        config: &avarok_core::config::ModelConfig,
    ) -> Result<Self> {
        Self::new_with_hash(weights, num_experts, gate_nvfp4, None, gpu, config)
    }
}

/// Check the routing config against what the topk kernels can actually
/// index and store, at load time rather than as silent NaN routing or an
/// out-of-bounds shared-memory write on the first token.
fn check_routing_bounds(
    config: &avarok_core::config::ModelConfig,
    num_experts: usize,
) -> Result<()> {
    // Sanity-check the routing config: top-k that exceeds the
    // expert count would index OOB in the topk kernel and produce
    // silent NaN routing. Catch the misconfiguration at load time.
    anyhow::ensure!(
        config.num_experts_per_tok <= num_experts && num_experts > 0,
        "MoE config invalid: num_experts_per_tok={} must be in 1..={}",
        config.num_experts_per_tok,
        num_experts,
    );
    // The check above bounds top-k by the expert count, which is the OOB
    // the topk kernel can READ. It says nothing about the OOB the kernel
    // can WRITE: the sigmoid routing kernels stage their top-K in a
    // fixed-size shared array, and `top_k` was passed to them unbounded.
    anyhow::ensure!(
        config.num_experts_per_tok <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
            && num_experts <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        "MoE config exceeds the routing kernels' fixed shared-memory bounds: \
         num_experts_per_tok={} (max {}), num_experts={} (max {}). Raise \
         MAX_TOP_K / MAX_EXPERTS in kernels/gb10/common/moe_topk_sigmoid.cu \
         and their mirrors in layers::ops together.",
        config.num_experts_per_tok,
        crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
        num_experts,
        crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
    );
    Ok(())
}

#[path = "init_with_hash.rs"]
mod init_with_hash;
