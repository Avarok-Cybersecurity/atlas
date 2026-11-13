// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the load-time BF16 -> NVFP4 route for the dense FFN (`[defaults] dense_nvfp4_from_bf16`):
//! whether a load takes it, which layers qualify, the quantization, and what it frees.
//!
//! **WHAT.** On a plain-BF16 dense checkpoint (`Nvfp4Variant::Bf16Raw`: Holo 3.1 0.8B and 4B, Qwen3.5 dense)
//! each layer's gate, up and down are quantized ONCE at load with the existing NVFP4 quantizer
//! (`quantize_to_nvfp4`: E2M1 codes, E4M3 per-16 scales, one F32 tensor scale), the BF16 sources are freed,
//! and the three weights are installed as `W4Linear`s (`port::w4::nvfp4::Nvfp4Linear`) through
//! `DenseFfnLayer::set_w4` (by `w4_install.rs`). The layer's 4-bit dispatch (lane L21) then serves decode,
//! verify and prefill. No new kernel and no new format: the `w4a16_*` family every CUDA target compiles.
//!
//! **WHY.** Single-stream decode streams every weight once per token. NVFP4 is 4.5 bits a weight, so the
//! 4B's FFN (4.53 GB of BF16) becomes 1.27 GB, 3.26 GB less per token and in VRAM, against FP8's 2.26 GB.
//! Whether the software-dequant GEMV reaches the bandwidth that makes that a win on Ada (no FP4 datapath)
//! is the never-measured question of queue row Q2. An expectation, not a measurement.
//!
//! **THE PRICE.** E2M1 with per-16 E4M3 scales is a numerics change well below FP8, so the route is a target
//! declaration, false on every target, with `AVAROK_DENSE_NVFP4_FROM_BF16=0|1` as the A/B. The flip receipt
//! (master plan 3.5): Q2 at >= 85% of the measured bandwidth, then the D7 gate (NLL delta <= 2x the FP8
//! route's, full-vocab KL, coherence, recall).
//!
//! **PRECEDENCE.** The FP8 FFN route wins, then the affine 4-bit FFN route; this route serves the FFN only
//! when both are off (the boot line says which won).
//!
//! **LOOKUPS.** With the row off nothing is looked up; with it on, the kernels are looked up only once
//! every other clause holds, so a declined load leaves the boot audit as it was.

use anyhow::Result;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::WeightStore;

use super::fp8_from_bf16::layer_shapes_ok;
use crate::layers::ops::target_defaults::Resolved;
use crate::port::w4::W4Box;
use crate::port::w4::nvfp4::{self, Nvfp4Kernels};
use crate::weight_map::{DenseWeight, quantize_to_nvfp4};

/// Module of the NVFP4 quantizer (`nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`).
pub const QUANT_MODULE: &str = "quantize_nvfp4";

/// Ledger label of the packed E2M1 codes.
pub const LABEL_CODES: &str = "ffn nvfp4 from bf16";
/// Ledger label of the E4M3 group scales.
pub const LABEL_SCALES: &str = "ffn nvfp4 from bf16 group scale";

/// Everything the per-load decision reads, as data, so the CPU table test pins each clause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteInputs {
    /// `[defaults] dense_nvfp4_from_bf16` after `AVAROK_DENSE_NVFP4_FROM_BF16`.
    pub lever: bool,
    /// The checkpoint is plain BF16 (`Nvfp4Variant::Bf16Raw`).
    pub bf16_checkpoint: bool,
    /// `tp_world_size == 1`: the 4-bit dispatch has no shard path.
    pub single_gpu: bool,
    /// A LoRA adapter is pending: the adapter install refuses a packed FFN, so the route declines.
    pub lora: bool,
    /// The quantizer and the GEMV family (rows 1..=4 plus the dequant fallback) resolve.
    pub kernels: bool,
    /// The BF16 -> FP8 FFN route is on for this load: FP8 wins.
    pub fp8_ffn: bool,
    /// The affine 4-bit FFN route is on for this load: affine wins.
    pub affine_ffn: bool,
}

/// The decision table. Pure.
pub fn route_selected(i: RouteInputs) -> bool {
    why_declined(i).is_none()
}

/// The first clause that declines the route, in table order, or `None` when it runs.
pub fn why_declined(i: RouteInputs) -> Option<&'static str> {
    if !i.lever {
        Some("row off")
    } else if !i.bf16_checkpoint {
        Some("not a plain BF16 checkpoint")
    } else if !i.single_gpu {
        Some("tensor parallel")
    } else if i.lora {
        Some("a LoRA adapter is pending")
    } else if i.fp8_ffn {
        Some("the FP8 FFN route wins")
    } else if i.affine_ffn {
        Some("the affine 4-bit FFN route wins")
    } else if !i.kernels {
        Some("the NVFP4 quantizer or GEMV family is missing from this build")
    } else {
        None
    }
}

/// Device bytes of one layer's three NVFP4 weights (gate and up `[inter, hidden]`, down `[hidden, inter]`).
#[cfg_attr(not(test), allow(dead_code))] // the residency credit prices NVFP4 through `w4_residency`
pub fn layer_device_bytes(hidden: usize, inter: usize) -> usize {
    2 * nvfp4::device_bytes_of(inter, hidden) + nvfp4::device_bytes_of(hidden, inter)
}

/// Bytes one qualifying layer hands back: its BF16 gate, up and down minus the NVFP4 weights that replace them.
#[cfg_attr(not(test), allow(dead_code))] // the residency credit prices NVFP4 through `w4_residency`
pub fn layer_freed_bytes(hidden: usize, inter: usize) -> usize {
    (3 * hidden * inter * 2).saturating_sub(layer_device_bytes(hidden, inter))
}

/// Per-layer qualification: the FP8 route's (`[n, k]` BF16 on the store, both extents whole 16-element
/// chunks), which is exactly the NVFP4 group rule. A layer that fails keeps its BF16 path.
pub fn layer_qualifies(store: &WeightStore, lp: &str, hidden: usize, inter: usize) -> bool {
    layer_shapes_ok(store, lp, hidden, inter)
}

/// The route, resolved once per load: `Some` when it runs.
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4FromBf16 {
    absmax_k: KernelHandle,
    quant_k: KernelHandle,
    kernels: Nvfp4Kernels,
}

impl Nvfp4FromBf16 {
    /// Resolve the route from the resolved lever (passed in so a test can drive it). Silent: the caller
    /// (`w4_install.rs`) prints the boot line from the returned inputs. Looks nothing up with the lever off
    /// or when a clause before the kernel clause already declines.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve(
        gpu: &dyn GpuBackend,
        lever: Resolved<bool>,
        bf16_checkpoint: bool,
        tp_world_size: usize,
        lora: bool,
        fp8_ffn: bool,
        affine_ffn: bool,
    ) -> (Option<Self>, RouteInputs) {
        let mut inputs = RouteInputs {
            lever: lever.value,
            bf16_checkpoint,
            single_gpu: tp_world_size.max(1) == 1,
            lora,
            // Assumed present until looked up, so the table names the first real clause.
            kernels: true,
            fp8_ffn,
            affine_ffn,
        };
        if why_declined(inputs).is_some() {
            return (None, inputs);
        }
        let absmax_k = crate::layers::try_target_kernel(gpu, QUANT_MODULE, "nvfp4_global_absmax");
        let quant_k = crate::layers::try_target_kernel(gpu, QUANT_MODULE, "quantize_bf16_to_nvfp4");
        let kernels = Nvfp4Kernels::resolve(gpu);
        inputs.kernels = absmax_k.0 != 0 && quant_k.0 != 0 && kernels.decode_ready();
        let me = Self {
            absmax_k,
            quant_k,
            kernels,
        };
        (route_selected(inputs).then_some(me), inputs)
    }

    /// Quantize one layer's gate, up and down, in that order. Each projection's two NVFP4 buffers are adopted
    /// into the store's derived ledger (#736) and its BF16 source is freed right after its own quantize (the
    /// quantizer synchronizes its stream before returning, so nothing still reads the source), so the load
    /// peak is the checkpoint plus one projection. Returns `[gate, up, down]`. The caller tallies residency.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_ffn(
        &self,
        store: &WeightStore,
        lp: &str,
        hidden: usize,
        inter: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<[W4Box; 3]> {
        let one = |name: &str, n: usize, k: usize| -> Result<W4Box> {
            let src = store.get(&format!("{lp}.mlp.{name}.weight"))?;
            let bf16 = DenseWeight { weight: src.ptr };
            let q = quantize_to_nvfp4(&bf16, n, k, gpu, self.absmax_k, self.quant_k, stream)?;
            let d = store.derived();
            d.adopt(LABEL_CODES, q.weight, n * k / 2);
            d.adopt(LABEL_SCALES, q.weight_scale, n * k / nvfp4::GROUP);
            // The checkpoint tensor itself, freed as the FP8 route frees it: the store keeps the (dead) map
            // entry and nothing reads it again (the BF16 overlay is detached by `set_w4`).
            gpu.free(src.ptr)?;
            nvfp4::from_quantized(q, n, k, self.kernels)
        };
        Ok([
            one("gate_proj", inter, hidden)?,
            one("up_proj", inter, hidden)?,
            one("down_proj", hidden, inter)?,
        ])
    }
}

#[cfg(test)]
#[path = "nvfp4_from_bf16_tests.rs"]
mod tests;
