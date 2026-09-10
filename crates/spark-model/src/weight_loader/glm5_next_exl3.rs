// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash routed experts kept PACKED as EXL3 trellis.
//!
//! # Which checkpoint this is for
//!
//! `vcruz305/GLM-5.3-Flash-EXL3-K2` declares
//!
//! ```text
//! quant_method exl3   bits 2   codebook mcg   head_bits 16
//! scope glm53_routed_experts_only
//! ```
//!
//! and that `scope` is the whole reason this file is small: **only the routed
//! experts are quantized.** The dense MLP, attention, `lm_head` and embeddings
//! all stay BF16, so GLM's existing binders keep serving them unchanged and
//! nothing here touches the shared expert or the router. The tensors are
//!
//! ```text
//! model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.trellis  I16 [in/16, out/16, 16*K]
//! ...                                                        .suh      F16 [in]
//! ...                                                        .svh      F16 [out]
//! ...                                                        .mcg      I32 [1]  = 0xCBAC1FED
//! ```
//!
//! # Why the pack is worth this file
//!
//! 96 GB total against the NVFP4 checkpoint's 98.59 GB **per rank**, so this is
//! the difference between one GB10 and two.
//!
//! # What it is NOT
//!
//! A quantization path. Nothing is dequantized at load: the trellis stays
//! packed in VRAM and is decoded in-kernel by `exl3_mgemm`. This file only
//! resolves pointers and builds the global-id-indexed tables the routed forward
//! reads.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;
use spark_runtime::weights::exl3::{Exl3Weight, is_exl3_linear};

use crate::layers::moe::{Exl3ExpertPtrTable, build_exl3_ptr_table};

/// The three projections' EXL3 tables for one routed site, indexed by GLOBAL
/// expert id (`None` slots inside the table mark ids another EP rank owns).
#[derive(Debug)]
pub struct Glm5NextExl3Experts {
    /// `[gate, up, down]`, in the order `exl3_moe_decode_routed` expects.
    pub tables: [Exl3ExpertPtrTable; 3],
    /// The process-shared mgemm launch state. An `Arc` per site, ONE allocation
    /// for the whole model — see `Exl3MoeState::shared`.
    pub state: std::sync::Arc<crate::layers::moe::Exl3MoeState>,
}

/// Relative name of one expert projection, e.g. `mlp.experts.7.gate_proj`.
fn expert_leaf(id: usize, proj: &str) -> String {
    format!("mlp.experts.{id}.{proj}")
}

/// Is this layer's routed-expert set stored as EXL3 trellis?
///
/// Probes `probe_id`'s `gate_proj` for the `.trellis`/`.suh`/`.svh` triplet.
/// Any single expert answers for the layer because the pack's scope is
/// "routed experts only": either every routed expert of the layer is packed or
/// none is, and a partially-packed layer fails the per-expert resolve in
/// [`bind_experts_exl3`] by name rather than being silently half-loaded.
///
/// # `probe_id` must be an expert THIS RANK OWNS
///
/// Under expert parallelism the weight store is sharded, so a rank holds only
/// its own slice: with EP=2 and 288 experts, rank 1's store starts at 144 and
/// expert 0's tensors are simply absent. Probing a fixed expert 0 therefore
/// reported "not EXL3" on every rank but 0, which dropped the layer to the
/// dense binder and surfaced -- confusingly far from the cause -- as
/// `Weight 'model.language_model.layers.3.mlp.experts.144.gate_proj.weight'
/// not found in store`, a BF16 name the EXL3 pack never contained. Pass
/// `local_expert_range().start`.
pub fn layer_is_exl3(
    store: &WeightStore,
    qualify: &dyn Fn(&str) -> String,
    probe_id: usize,
) -> bool {
    is_exl3_linear(store, &qualify(&expert_leaf(probe_id, "gate_proj")))
}

/// Bind one layer's routed experts from their packed trellis tensors.
///
/// `local` is the EP-local half-open range of GLOBAL expert ids this rank owns;
/// ids outside it are left `None` so the built table carries a null pointer and
/// the kernel writes nothing for that slot (the caller's pre-zeroed routed row
/// stands). That is the same contract `Glm5NextExpertPtrTable` already uses for
/// NVFP4, so the routed sum is unchanged under EP.
pub fn bind_experts_exl3(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    qualify: &dyn Fn(&str) -> String,
    num_experts: usize,
    local: (usize, usize),
    // MoE geometry for the shared slabs: hidden, one expert's intermediate, top_k.
    geom: (usize, usize, usize),
) -> Result<Glm5NextExl3Experts> {
    let (local_start, local_end) = local;
    ensure!(
        local_end > local_start && local_end <= num_experts,
        "GLM EXL3 routed experts: invalid local range [{local_start}, {local_end}) \
         of {num_experts}"
    );

    // Resolve the codebook ONCE for the whole layer.
    //
    // 🔴 `from_store` reads the `.mcg`/`.mul1` flag with a BLOCKING 4-byte
    // `copy_d2h` (spark-runtime weights/exl3.rs:168 -> gpu_copy.rs:82). Per
    // projection that is 144 local experts x 3 = 432 synchronous stream syncs
    // PER MoE LAYER, ~18,000 over a full EP=2 load, every one of them fetching
    // the SAME constant. Measured cost is inside the unattributed boot window;
    // `weight_map/moe_exl3.rs:90` already avoided it this way.
    //
    // Safe because the codebook is uniform per layer by construction, and the
    // (K, cb) check below still enforces exactly that: the fused MoE kernel is
    // instantiated for ONE (K, codebook) pair per launch, so a mixed layer is
    // refused there rather than silently decoded with the probed value.
    let cb_probe = qualify(&expert_leaf(local_start, "gate_proj"));
    let layer_cb = Exl3Weight::from_store(gpu, store, &cb_probe)
        .with_context(|| format!("GLM EXL3 codebook probe {cb_probe}"))?
        .cb;

    let load_proj = |proj: &str| -> Result<Vec<Option<Exl3Weight>>> {
        let mut out: Vec<Option<Exl3Weight>> = (0..num_experts).map(|_| None).collect();
        for id in local_start..local_end {
            let prefix = qualify(&expert_leaf(id, proj));
            let w = Exl3Weight::from_store_with_cb(gpu, store, &prefix, layer_cb)
                .with_context(|| format!("GLM EXL3 routed expert {prefix}"))?;
            out[id] = Some(w);
        }
        Ok(out)
    };

    let gate = load_proj("gate_proj")?;
    let up = load_proj("up_proj")?;
    let down = load_proj("down_proj")?;

    // Uniformity is a hard requirement, not a nicety: the fused MoE kernel is
    // instantiated for ONE (K, codebook) pair per launch, so a layer whose
    // experts disagree would decode part of the routed sum with the wrong
    // codebook and produce plausible garbage. Fail here, where the name is
    // still in hand.
    for (name, set) in [("gate", &gate), ("up", &up), ("down", &down)] {
        let mut first: Option<(u32, u32)> = None;
        for (id, w) in set.iter().enumerate() {
            let Some(w) = w else { continue };
            let this = (w.k_bits, w.cb as u32);
            match first {
                None => first = Some(this),
                Some(f) => ensure!(
                    f == this,
                    "GLM EXL3 {name}: expert {id} is (K={}, cb={}) but the layer started \
                     (K={}, cb={}) — the fused MoE kernel decodes one pair per launch",
                    this.0,
                    this.1,
                    f.0,
                    f.1
                ),
            }
        }
    }

    let (hidden, inter, top_k) = geom;
    Ok(Glm5NextExl3Experts {
        state: crate::layers::moe::Exl3MoeState::shared(gpu, hidden, inter, top_k, num_experts)?,
        tables: [
            build_exl3_ptr_table(&gate, gpu).context("GLM EXL3 gate table")?,
            build_exl3_ptr_table(&up, gpu).context("GLM EXL3 up table")?,
            build_exl3_ptr_table(&down, gpu).context("GLM EXL3 down table")?,
        ],
    })
}

#[cfg(test)]
#[path = "glm5_next_exl3_tests.rs"]
mod tests;
