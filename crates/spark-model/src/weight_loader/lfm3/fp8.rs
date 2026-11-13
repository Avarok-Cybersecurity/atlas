// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. The route structure follows weight_loader/qwen35_dense/{fp8_from_bf16,attn_fp8_from_bf16}.rs (Atlas).

//! Responsibility: the FP8 weight routes for LFM3: the vendor checkpoint's
//! native E4M3 projections, and the opt-in load-time BF16 -> FP8 conversion.
//!
//! **Vendor checkpoint** (`LiquidAI/LFM3-1.7B-FP8`, compressed-tensors
//! `float-quantized`, 128x128 weight blocks): the FFN and attention
//! projections are E4M3 with a `weight_scale` block grid beside each; the conv
//! projections, the head gate, the norms, the embedding and the head stay BF16.
//! A tensor's dtype decides: an E4M3 `.weight` is loaded through
//! `load_fp8_block_scaled_as_fp8weight`, which accepts the `weight_scale`
//! spelling (the 2-D grid) as well as `weight_scale_inv`. Nothing is
//! dequantized to BF16: that would put the checkpoint's 1.3 GB saving back.
//!
//! **Load-time BF16 -> FP8** (a numerics change, so opt-in): on a BF16
//! checkpoint, the projections of a class are quantized once to the same
//! 128x128 block-scaled E4M3 format (`quantize_to_fp8_blockscaled`). Classes,
//! and what turns each on:
//!
//! | class | projections | turned on by | BF16 source |
//! |---|---|---|---|
//! | FFN | w1, w3, w2 | `lfm3_fp8_all` or the existing `dense_fp8_from_bf16` | freed |
//! | attention | q, k, v, o | `lfm3_fp8_all` or the existing `attn_fp8_from_bf16` | freed |
//! | conv | in_proj, out_proj | `lfm3_fp8_all` only | KEPT: lane L38's layer decodes from the E4M3 copies (M up to 16) and prefills from BF16 |
//!
//! So the conv class buys decode bandwidth, not memory (+0.22 GB of E4M3
//! copies across the 13 layers). The head gate, the norms and the conv taps are never quantized. The
//! route declines (whole, never half) for tensor parallelism, a pending LoRA
//! adapter, or a missing quantizer / M=1 FP8 GEMV, and a single projection
//! declines when an extent is not a whole 128-block.
//!
//! Row: `lfm3_fp8_all` (`AVAROK_LFM3_FP8_ALL`), false everywhere; receipt:
//! KL < 0.003 against the BF16 serve on the lfm3_qual panel, and the
//! comparison with Liquid's own FP8 checkpoint (`lfm3_qual.py`).

use anyhow::{Context, Result, ensure};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::weight_loader::fp8_from_bf16::QUANT_MODULE;
use crate::weight_map::{
    DenseWeight, Fp8Weight, dense_auto_fp8_or_bf16, load_fp8_block_scaled_as_fp8weight,
    quantize_to_fp8_blockscaled,
};

/// One block of the E4M3 scale grid is 128 x 128 weights.
const BLOCK: usize = 128;

/// Which projection family a tensor of an overlay group belongs to. (The conv
/// projections are not a class: they are loaded as a pair, `load_conv_pair`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Ffn,
    Attn,
}

/// What the load-time BF16 -> FP8 route will convert.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Routes {
    pub ffn: bool,
    pub attn: bool,
    pub conv: bool,
}

impl Routes {
    pub fn wants(self, class: Class) -> bool {
        match class {
            Class::Ffn => self.ffn,
            Class::Attn => self.attn,
        }
    }
}

/// Everything the route decision reads, as data, so a CPU table test pins each
/// clause without a GPU or the process environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Inputs {
    /// `[defaults] lfm3_fp8_all` after `AVAROK_LFM3_FP8_ALL`.
    pub all: bool,
    /// `[defaults] dense_fp8_from_bf16` (the existing FFN lever).
    pub ffn_lever: bool,
    /// `[defaults] attn_fp8_from_bf16` (the existing attention lever).
    pub attn_lever: bool,
    /// `tp_world_size == 1`.
    pub single_gpu: bool,
    /// A LoRA adapter is pending (`adapter_max_rank > 0`).
    pub lora: bool,
    /// The quantizer and the FP8 M=1 decode GEMV both resolve.
    pub kernels: bool,
}

/// The decision table. Pure.
pub fn routes(i: Inputs) -> Routes {
    let admitted = i.single_gpu && !i.lora && i.kernels;
    Routes {
        ffn: admitted && (i.all || i.ffn_lever),
        attn: admitted && (i.all || i.attn_lever),
        conv: admitted && i.all,
    }
}

/// FP8 bytes plus the FP32 scale grid one quantized `[n, k]` weight costs.
pub fn fp8_weight_bytes(n: usize, k: usize) -> (usize, usize) {
    (n * k, n.div_ceil(BLOCK) * k.div_ceil(BLOCK) * 4)
}

/// A projection can take the E4M3 block format: both extents are whole
/// 128-blocks (the scale grid and the W8A8 GEMMs; 128 also covers the GEMV's
/// 16-element K chunk).
pub fn qualifies(n: usize, k: usize) -> bool {
    n > 0 && k > 0 && n.is_multiple_of(BLOCK) && k.is_multiple_of(BLOCK)
}

/// One loaded projection.
#[derive(Clone, Copy, Debug)]
pub enum Proj {
    Bf16(DenseWeight),
    Fp8(Fp8Weight),
}

impl Proj {
    pub fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8(_))
    }
}

/// The conv layer's projections: BF16 for prefill, optional E4M3 decode copies
/// of both (see [`Fp8Loader::load_conv_pair`]).
#[derive(Clone, Copy, Debug)]
pub struct ConvProjs {
    pub in_bf16: DenseWeight,
    pub out_bf16: DenseWeight,
    /// `(in_proj, out_proj)` E4M3, or `None`.
    pub fp8: Option<(Fp8Weight, Fp8Weight)>,
}

/// A group of projections that must share one precision (an FFN's three, an
/// attention's four): the layer overlays are all-or-nothing.
#[derive(Clone, Copy, Debug)]
pub enum Group<const N: usize> {
    Bf16([DenseWeight; N]),
    Fp8([Fp8Weight; N]),
}

/// The route, resolved once per load, plus the handles it needs.
pub struct Fp8Loader<'a> {
    store: &'a WeightStore,
    gpu: &'a dyn GpuBackend,
    stream: u64,
    routes: Routes,
    quant_k: KernelHandle,
}

impl<'a> Fp8Loader<'a> {
    /// Resolve the route. Looks the quantizer and the GEMV up only when a
    /// lever is on, so a target that declares none issues no extra lookup.
    pub fn resolve(
        store: &'a WeightStore,
        gpu: &'a dyn GpuBackend,
        config: &ModelConfig,
        all: bool,
        ffn_lever: bool,
        attn_lever: bool,
    ) -> Self {
        let wanted = all || ffn_lever || attn_lever;
        let (quant_k, kernels) = if wanted {
            let q = crate::layers::try_target_kernel(gpu, QUANT_MODULE, QUANT_MODULE);
            let gemv = crate::layers::dense_ffn::ada_kernels::Fp8FfnKernels::m1_gemv(gpu);
            (q, q.0 != 0 && gemv.0 != 0)
        } else {
            (KernelHandle(0), false)
        };
        let inputs = Inputs {
            all,
            ffn_lever,
            attn_lever,
            single_gpu: config.tp_world_size.max(1) == 1,
            lora: config.adapter_max_rank > 0,
            kernels,
        };
        let routes = routes(inputs);
        if wanted {
            tracing::info!(
                "[avarok] LFM3 BF16 -> FP8 at load: ffn={} attn={} conv={} (lfm3_fp8_all={} \
                 dense_fp8_from_bf16={} attn_fp8_from_bf16={}; single_gpu={} lora={} kernels={}). \
                 128x128 block-scaled E4M3; AVAROK_LFM3_FP8_ALL=0 keeps BF16.",
                routes.ffn,
                routes.attn,
                routes.conv,
                all,
                ffn_lever,
                attn_lever,
                inputs.single_gpu,
                inputs.lora,
                inputs.kernels,
            );
        }
        Self::with_routes(store, gpu, routes, quant_k)
    }

    /// The loader for an already-decided route (tests drive this directly).
    pub fn with_routes(
        store: &'a WeightStore,
        gpu: &'a dyn GpuBackend,
        routes: Routes,
        quant_k: KernelHandle,
    ) -> Self {
        Self {
            store,
            gpu,
            stream: gpu.default_stream(),
            routes,
            quant_k,
        }
    }

    pub fn routes(&self) -> Routes {
        self.routes
    }

    /// The vendor's E4M3 projection, kept native. Only the widened FP32 scale
    /// grid is a fresh allocation (the E4M3 bytes are store-owned), so it is
    /// adopted by the store's ledger (#736).
    fn native(&self, prefix: &str, n: usize, k: usize) -> Result<Fp8Weight> {
        let scale = self
            .store
            .get(&format!("{prefix}.weight_scale"))
            .or_else(|_| self.store.get(&format!("{prefix}.weight_scale_inv")))
            .with_context(|| format!("lfm3 FP8 projection {prefix} has no block scale tensor"))?;
        // A 2-D grid is `[N/128, K/128]`; anything else (a scalar, a per-row
        // vector) is not Liquid's format and would be broadcast silently.
        ensure!(
            scale.shape == [n.div_ceil(BLOCK), k.div_ceil(BLOCK)],
            "lfm3 FP8 projection {prefix}: block scale shape {:?}, expected [{}, {}]",
            scale.shape,
            n.div_ceil(BLOCK),
            k.div_ceil(BLOCK)
        );
        let w = load_fp8_block_scaled_as_fp8weight(self.store, prefix, self.gpu)?;
        let (_, scale_bytes) = fp8_weight_bytes(n, k);
        self.store
            .derived()
            .adopt("lfm3 fp8 block scale (widened)", w.row_scale, scale_bytes);
        Ok(w)
    }

    /// Quantize a BF16 projection now. With `free_source` its BF16 bytes are
    /// freed once the quantize is issued (same stream), so the load peak is
    /// not the BF16 checkpoint plus every FP8 copy; the conv projections keep
    /// their BF16 (the layer's prefill reads it) and pass `false`.
    fn quantize_inner(
        &self,
        prefix: &str,
        n: usize,
        k: usize,
        free_source: bool,
    ) -> Result<Fp8Weight> {
        let src = self.store.get(&format!("{prefix}.weight"))?;
        let bf16 = DenseWeight { weight: src.ptr };
        let w = quantize_to_fp8_blockscaled(&bf16, n, k, self.gpu, self.quant_k, self.stream)?;
        let (wb, sb) = fp8_weight_bytes(n, k);
        let d = self.store.derived();
        d.adopt("lfm3 fp8 from bf16", w.weight, wb);
        d.adopt("lfm3 fp8 from bf16 block scale", w.row_scale, sb);
        if free_source {
            self.gpu.free(src.ptr)?;
        }
        Ok(w)
    }

    fn quantize(&self, prefix: &str, n: usize, k: usize) -> Result<Fp8Weight> {
        self.quantize_inner(prefix, n, k, true)
    }

    /// One conv projection: the BF16 weight (always, the layer's prefill reads
    /// it) and an optional E4M3 decode copy. A BF16 tensor gets the copy only
    /// when the conv route is on; an E4M3 tensor (not Liquid's layout, which
    /// keeps the convs BF16) is its own decode copy and its BF16 is a one-time
    /// dequantization.
    fn conv_one(
        &self,
        prefix: &str,
        n: usize,
        k: usize,
    ) -> Result<(DenseWeight, Option<Fp8Weight>)> {
        let name = format!("{prefix}.weight");
        let t = self.store.get(&name)?;
        ensure!(
            t.shape == [n, k],
            "{name} has shape {:?}, expected [{n}, {k}]",
            t.shape
        );
        match t.dtype {
            WeightDtype::BF16 => {
                let bf16 = DenseWeight { weight: t.ptr };
                let copy = (self.routes.conv && qualifies(n, k))
                    .then(|| self.quantize_inner(prefix, n, k, false))
                    .transpose()?;
                Ok((bf16, copy))
            }
            WeightDtype::FP8E4M3 => {
                let native = self.native(prefix, n, k)?;
                let bf16 = dense_auto_fp8_or_bf16(self.store, prefix, self.gpu)?;
                self.store
                    .derived()
                    .adopt("lfm3 conv bf16 from fp8", bf16.weight, n * k * 2);
                Ok((bf16, Some(native)))
            }
            other => anyhow::bail!("{name}: unsupported projection dtype {other:?}"),
        }
    }

    /// The conv layer's two projections (`in_proj [3h, h]`, `out_proj [h, h]`):
    /// BF16 for both, and E4M3 decode copies for both or neither (the layer
    /// installs them as a pair). A pair half that is missing is completed by
    /// quantizing it when the quantizer is available, else the copies are
    /// dropped and the layer decodes from BF16.
    pub fn load_conv_pair(
        &self,
        in_prefix: &str,
        out_prefix: &str,
        hidden: usize,
    ) -> Result<ConvProjs> {
        let (in_bf16, in_fp8) = self.conv_one(in_prefix, 3 * hidden, hidden)?;
        let (out_bf16, out_fp8) = self.conv_one(out_prefix, hidden, hidden)?;
        let can_complete = self.quant_k.0 != 0 && qualifies(hidden, hidden);
        let fp8 = match (in_fp8, out_fp8) {
            (Some(a), Some(b)) => Some((a, b)),
            (Some(a), None) if can_complete => {
                Some((a, self.quantize_inner(out_prefix, hidden, hidden, false)?))
            }
            (None, Some(b)) if can_complete => Some((
                self.quantize_inner(in_prefix, 3 * hidden, hidden, false)?,
                b,
            )),
            _ => None,
        };
        Ok(ConvProjs {
            in_bf16,
            out_bf16,
            fp8,
        })
    }

    /// Load one projection `{prefix}.weight` of shape `[n, k]`.
    ///
    /// E4M3 on disk: native. BF16: converted when the class's route is on and
    /// the extents qualify, else kept BF16.
    pub fn load(&self, class: Class, prefix: &str, n: usize, k: usize) -> Result<Proj> {
        let name = format!("{prefix}.weight");
        let t = self.store.get(&name)?;
        ensure!(
            t.shape == [n, k],
            "{name} has shape {:?}, expected [{n}, {k}]",
            t.shape
        );
        match t.dtype {
            WeightDtype::FP8E4M3 => Ok(Proj::Fp8(self.native(prefix, n, k)?)),
            WeightDtype::BF16 => {
                if self.routes.wants(class) && qualifies(n, k) {
                    Ok(Proj::Fp8(self.quantize(prefix, n, k)?))
                } else {
                    Ok(Proj::Bf16(DenseWeight { weight: t.ptr }))
                }
            }
            other => anyhow::bail!("{name}: unsupported projection dtype {other:?}"),
        }
    }

    /// Load a group that must share one precision. If the checkpoint mixes
    /// E4M3 and BF16 inside the group (it should not), the BF16 members are
    /// converted when the quantizer is available, else the load fails loudly:
    /// a half-FP8 overlay would silently read the wrong weight.
    pub fn load_group<const N: usize>(
        &self,
        class: Class,
        items: [(&str, usize, usize); N],
    ) -> Result<Group<N>> {
        let loaded: Vec<Proj> = items
            .iter()
            .map(|&(p, n, k)| self.load(class, p, n, k))
            .collect::<Result<_>>()?;
        if loaded.iter().all(|p| !p.is_fp8()) {
            let mut out = [DenseWeight {
                weight: spark_runtime::gpu::DevicePtr::NULL,
            }; N];
            for (slot, p) in out.iter_mut().zip(&loaded) {
                if let Proj::Bf16(w) = p {
                    *slot = *w;
                }
            }
            return Ok(Group::Bf16(out));
        }
        let mut out = [Fp8Weight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
            row_scale: spark_runtime::gpu::DevicePtr::NULL,
            n: 0,
            k: 0,
            scale_format: crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
        }; N];
        for (i, (slot, p)) in out.iter_mut().zip(&loaded).enumerate() {
            let (prefix, n, k) = items[i];
            *slot = match *p {
                Proj::Fp8(w) => w,
                Proj::Bf16(_) => {
                    ensure!(
                        self.quant_k.0 != 0 && qualifies(n, k),
                        "lfm3: {prefix} is BF16 but its group is FP8 and the FP8 quantizer is \
                         unavailable or the extents [{n}, {k}] are not 128-blocks"
                    );
                    self.quantize(prefix, n, k)?
                }
            };
        }
        Ok(Group::Fp8(out))
    }
}
