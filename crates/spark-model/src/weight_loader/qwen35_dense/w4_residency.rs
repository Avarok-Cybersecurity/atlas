// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the residency arithmetic of the 4-bit (`W4Linear`) routes: what one weight of each
//! format occupies, what a layer of each type hands back when its BF16 projections are replaced, and the
//! pre-load and post-load credits the server's memory budgets read.
//!
//! **WHY.** The SSM decode-ring and Marconi budgets are a share of the memory beside the weights, and the
//! post-load audit measures free memory before `load_layers` runs. A load-time route that replaces BF16
//! weights with smaller ones frees memory while the layers are built, after both measurements, so each
//! route must be credited in advance or the budgets under-state what fits (the FP8 FFN route's credit,
//! `fp8_from_bf16::{predicted_free_bytes, deferred_free_bytes}`, exists for the same reason). This module is
//! the 4-bit half; those two functions add it, so `serve_load.rs` and `preflight/headroom.rs` need no new
//! call sites.
//!
//! **CONSERVATIVE BY CONSTRUCTION.** The credit is what the BF16 weights cost minus what the 4-bit weights
//! cost, minus the shared W4 scratch the routes reserve (`port::w4::scratch`: the int8 activations, the
//! group sums, the BF16 dequant target and the tickets; the per-format workspace is left out, so this is a
//! floor of the scratch and the credit errs low). Over-crediting would let the ring take memory the KV
//! cache needs; under-crediting only leaves headroom unused.
//!
//! **PRECEDENCE.** The formats come from `w4_install::plan_formats`, which applies the same precedence the
//! loader does (FP8 wins over affine 4-bit, affine over NVFP4, per layer type).
//!
//! **MARLIN.** With the `marlin_w4a16` row on (lane L18b), an affine 4-bit projection that Marlin serves
//! keeps the Marlin layout (4.125 bits a weight plus `sms * 16` lock bytes), not the affine one, so the
//! credit prices it that way ([`projection_w4_bytes`]); a projection Marlin declines keeps the affine
//! price, as the install does per projection. The row off adds no lookup and no device query.

use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::attn_fp8_from_bf16::AttnShapes;
use super::gdn_fp8_from_bf16::GdnShapes;
use crate::port::w4::W4Format;
use crate::port::w4::scratch::{ScratchSpec, xq_bytes, xs_bytes};
pub use crate::port::w4_stats::W4Role;

/// Device bytes one `[n, k]` weight occupies in `format`.
///
/// Affine 4-bit and NVFP4 are the layouts their quantizers allocate (`port::affine::Layout` plus the split-K
/// tickets; `port::w4::nvfp4::device_bytes_of`). The packed formats are their on-disk block sizes (GGUF:
/// 136 / 144 / 210 bytes per 256 weights for IQ4_XS / Q4_K / Q6_K; Marlin int4: codes, f16 group scales,
/// packed zero points for AWQ; EXL3: codes at the stated rate plus the two f16 sign-scale vectors). A
/// format's actual `W4Linear::device_bytes()` is what the metrics report; this is the prediction.
pub fn weight_bytes(format: W4Format, n: usize, k: usize) -> usize {
    match format {
        W4Format::Affine { bits, group } => {
            let g = (group as usize).max(1);
            let l = crate::port::affine::Layout::of(n, k, g);
            let tickets = 4 * n.div_ceil(crate::port::affine::weight::TILE_N).max(1);
            if bits == 4 {
                l.total() + tickets
            } else {
                l.npad * k * bits as usize / 8 + 2 * l.scales + tickets
            }
        }
        W4Format::Nvfp4 => crate::port::w4::nvfp4::device_bytes_of(n, k),
        W4Format::MarlinInt4 { group, zero_point } => {
            let groups = if group == 0 {
                1
            } else {
                k.div_ceil(group as usize)
            };
            n * k / 2 + groups * n * 2 + if zero_point { groups * n / 2 } else { 0 }
        }
        W4Format::GgufIq4Xs => n * k / 256 * 136,
        W4Format::GgufQ4K => n * k / 256 * 144,
        W4Format::GgufQ6K => n * k / 256 * 210,
        W4Format::Exl3 { bpw_x100 } => n * k * bpw_x100 as usize / 800 + (n + k) * 2,
    }
}

/// The input-group size a format's codes need K to be a whole multiple of.
pub fn k_group(format: W4Format) -> usize {
    match format {
        W4Format::Affine { group, .. } => (group as usize).max(1),
        W4Format::Nvfp4 => crate::port::w4::nvfp4::GROUP,
        W4Format::MarlinInt4 { group, .. } => (group as usize).max(16),
        W4Format::GgufIq4Xs | W4Format::GgufQ4K | W4Format::GgufQ6K => 256,
        W4Format::Exl3 { .. } => 16,
    }
}

/// The `[n, k]` projections one layer of `role` carries, in the loader's order.
pub fn projections(config: &ModelConfig, role: W4Role) -> Vec<(usize, usize)> {
    let h = config.hidden_size;
    match role {
        W4Role::Ffn => {
            let inter = ffn_inter(config);
            vec![(inter, h), (inter, h), (h, inter)]
        }
        W4Role::Gdn => {
            let s = GdnShapes::of(config);
            vec![(s.qkvz, s.hidden), (s.hidden, s.value)]
        }
        W4Role::Attn => AttnShapes::of(config).projections().to_vec(),
        W4Role::LmHead => vec![(config.vocab_size, h)],
    }
}

/// Layers of `role` in the model.
pub fn layers_of(config: &ModelConfig, role: W4Role) -> usize {
    match role {
        W4Role::Ffn => config.num_hidden_layers,
        W4Role::Gdn => config.num_ssm_layers(),
        W4Role::Attn => config.num_attention_layers(),
        W4Role::LmHead => 1,
    }
}

/// Every projection non-empty and its K a whole number of the format's groups.
pub fn qualifies(format: W4Format, projs: &[(usize, usize)]) -> bool {
    let g = k_group(format);
    !projs.is_empty()
        && projs
            .iter()
            .all(|&(n, k)| n > 0 && k > 0 && k.is_multiple_of(g))
}

/// BF16 bytes of a layer's projections.
pub fn layer_bf16_bytes(projs: &[(usize, usize)]) -> usize {
    projs.iter().map(|&(n, k)| n * k * 2).sum()
}

/// The SM count that sizes a Marlin weight's lock buffer, when the `marlin_w4a16` row will quantize this
/// load's affine 4-bit projections (lane L18b's request R2): `Some` only when some layer type takes an
/// affine 4-bit format, the row is on and the target carries the Marlin kernels. With the row off this
/// is one boolean read: no kernel lookup, no device query.
pub fn marlin_sms(gpu: &dyn GpuBackend, f: &W4Formats) -> Option<usize> {
    let affine4 = f
        .on()
        .any(|(_, fmt)| matches!(fmt, W4Format::Affine { bits: 4, .. }));
    if !affine4 || !crate::port::marlin::row_enabled() || !crate::port::marlin::available(gpu) {
        return None;
    }
    gpu.sm_count().ok().map(|s| s as usize)
}

/// Device bytes one `[n, k]` projection keeps on the route: `weight_bytes`, except that an affine 4-bit
/// projection Marlin serves (`marlin_sms` given and `route::shape_ok` holds, the install's own per-projection
/// test) keeps the Marlin layout, 4.125 bits a weight plus the locks. The install falls back to the affine
/// quantizer per projection, so the prediction does too.
pub fn projection_w4_bytes(
    format: W4Format,
    n: usize,
    k: usize,
    marlin_sms: Option<usize>,
) -> usize {
    if let (W4Format::Affine { bits: 4, .. }, Some(sms)) = (format, marlin_sms)
        && crate::port::marlin::route::shape_ok(n, k, crate::port::marlin::MarlinType::U4B8).is_ok()
    {
        return crate::port::marlin::route::marlin_bytes(n, k, false, sms);
    }
    weight_bytes(format, n, k)
}

/// 4-bit bytes of a layer's projections in `format`.
#[cfg_attr(not(test), allow(dead_code))] // the serve path calls the `_with` form
pub fn layer_w4_bytes(format: W4Format, projs: &[(usize, usize)]) -> usize {
    layer_w4_bytes_with(format, projs, None)
}

/// [`layer_w4_bytes`] with the Marlin row's SM count (see [`projection_w4_bytes`]).
pub fn layer_w4_bytes_with(
    format: W4Format,
    projs: &[(usize, usize)],
    marlin_sms: Option<usize>,
) -> usize {
    projs
        .iter()
        .map(|&(n, k)| projection_w4_bytes(format, n, k, marlin_sms))
        .sum()
}

/// What one qualifying layer hands back: BF16 minus 4-bit, never negative.
#[cfg_attr(not(test), allow(dead_code))] // the serve path calls the `_with` form
pub fn layer_freed_bytes(format: W4Format, projs: &[(usize, usize)]) -> usize {
    layer_freed_bytes_with(format, projs, None)
}

/// [`layer_freed_bytes`] with the Marlin row's SM count (see [`projection_w4_bytes`]).
pub fn layer_freed_bytes_with(
    format: W4Format,
    projs: &[(usize, usize)],
    marlin_sms: Option<usize>,
) -> usize {
    layer_bf16_bytes(projs).saturating_sub(layer_w4_bytes_with(format, projs, marlin_sms))
}

/// The 4-bit format each layer type takes from BF16 on this load (`None` = not 4-bit).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct W4Formats {
    pub ffn: Option<W4Format>,
    pub gdn: Option<W4Format>,
    pub attn: Option<W4Format>,
}

impl W4Formats {
    /// `(role, format)` for every layer type that is 4-bit.
    pub fn on(&self) -> impl Iterator<Item = (W4Role, W4Format)> + '_ {
        [
            (W4Role::Ffn, self.ffn),
            (W4Role::Gdn, self.gdn),
            (W4Role::Attn, self.attn),
        ]
        .into_iter()
        .filter_map(|(r, f)| f.map(|f| (r, f)))
    }

    #[cfg_attr(not(test), allow(dead_code))] // test-side convenience over `on()`
    pub fn any(&self) -> bool {
        self.on().next().is_some()
    }
}

/// The shared W4 scratch the routes reserve (`scratch::acquire`), without the per-format workspace: the
/// int8 activations and group sums of one 2,048-row prefill chunk at the widest K, the BF16 dequant target
/// of the largest matrix, and the ticket region. 0 when no route is on.
pub fn scratch_floor_bytes(config: &ModelConfig, f: W4Formats) -> usize {
    let projs: Vec<(usize, usize)> = f.on().flat_map(|(r, _)| projections(config, r)).collect();
    if projs.is_empty() {
        return 0;
    }
    let rows = crate::port::w4::select::prefill_chunk_cap(16);
    let k_max = projs.iter().map(|&(_, k)| k).max().unwrap_or(0);
    let bf16_max = projs.iter().map(|&(n, k)| n * k * 2).max().unwrap_or(0);
    ScratchSpec {
        ws_bytes: 0,
        xq_bytes: xq_bytes(rows, k_max),
        xs_bytes: xs_bytes(rows, k_max),
        bf16_bytes: bf16_max,
    }
    .device_bytes()
}

/// Pre-load credit: every layer of each 4-bit type counted (a dense Qwen3.5 layer's projections have the
/// config's shapes by construction, as `fp8_from_bf16::predicted_free_bytes_for` assumes), minus the scratch.
#[cfg_attr(not(test), allow(dead_code))] // the serve path calls the `_with` form
pub fn predicted_free_bytes_for(config: &ModelConfig, f: W4Formats) -> usize {
    predicted_free_bytes_with(config, f, None)
}

/// [`predicted_free_bytes_for`] with the Marlin row's SM count (see [`projection_w4_bytes`]).
pub fn predicted_free_bytes_with(
    config: &ModelConfig,
    f: W4Formats,
    marlin_sms: Option<usize>,
) -> usize {
    let freed: usize = f
        .on()
        .filter(|&(r, fmt)| qualifies(fmt, &projections(config, r)))
        .map(|(r, fmt)| {
            layers_of(config, r) * layer_freed_bytes_with(fmt, &projections(config, r), marlin_sms)
        })
        .sum();
    freed.saturating_sub(scratch_floor_bytes(config, f))
}

/// Post-load credit: as [`predicted_free_bytes_for`], with the FFN layers counted by what the store holds
/// (the BF16 `[n, k]` tensors `fp8_from_bf16::layer_shapes_ok` admits), the loader's own per-layer test.
#[cfg_attr(not(test), allow(dead_code))] // the serve path calls the `_with` form
pub fn deferred_free_bytes_for(store: &WeightStore, config: &ModelConfig, f: W4Formats) -> usize {
    deferred_free_bytes_with(store, config, f, None)
}

/// [`deferred_free_bytes_for`] with the Marlin row's SM count (see [`projection_w4_bytes`]).
pub fn deferred_free_bytes_with(
    store: &WeightStore,
    config: &ModelConfig,
    f: W4Formats,
    marlin_sms: Option<usize>,
) -> usize {
    let freed: usize = f
        .on()
        .map(|(r, fmt)| {
            let projs = projections(config, r);
            if !qualifies(fmt, &projs) {
                return 0;
            }
            let layers = match r {
                W4Role::Ffn => {
                    let (h, i) = (config.hidden_size, ffn_inter(config));
                    (0..config.num_hidden_layers)
                        .filter(|&l| {
                            super::fp8_from_bf16::layer_shapes_ok(
                                store,
                                &config.layer_prefix(l),
                                h,
                                i,
                            )
                        })
                        .count()
                }
                _ => layers_of(config, r),
            };
            layers * layer_freed_bytes_with(fmt, &projs, marlin_sms)
        })
        .sum();
    freed.saturating_sub(scratch_floor_bytes(config, f))
}

/// [`predicted_free_bytes_for`] over this load's resolved routes, BEFORE the checkpoint is read (the plain
/// BF16 test is "the config declares no `quantization_config`", as for the FP8 route). Silent: no boot line.
pub fn predicted_free_bytes(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    tp_world_size: usize,
    lora: bool,
) -> usize {
    if !super::w4_install::dense_loader_takes(config) {
        return 0;
    }
    let bf16 = config.quantization_config.is_none();
    let f = super::w4_install::plan_formats(gpu, config, bf16, tp_world_size, lora);
    // `marlin_w4a16` on: the projections Marlin serves keep its smaller layout (lane L18b's R2).
    let marlin = marlin_sms(gpu, &f);
    predicted_free_bytes_with(config, f, marlin)
}

/// [`deferred_free_bytes_for`] over this load's resolved routes, after the checkpoint is on the device.
pub fn deferred_free_bytes(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    config: &ModelConfig,
    tp_world_size: usize,
    lora: bool,
) -> usize {
    if !super::w4_install::dense_loader_takes(config) {
        return 0;
    }
    let bf16 = matches!(
        crate::weight_map::detect_nvfp4_variant(store, config),
        crate::weight_map::Nvfp4Variant::Bf16Raw
    );
    let f = super::w4_install::plan_formats(gpu, config, bf16, tp_world_size, lora);
    let marlin = marlin_sms(gpu, &f);
    deferred_free_bytes_with(store, config, f, marlin)
}

/// The FFN width, with the loader's fallback (`qwen35_dense::ffn_inter`).
pub fn ffn_inter(config: &ModelConfig) -> usize {
    if config.intermediate_size > 0 {
        config.intermediate_size
    } else {
        config.moe_intermediate_size
    }
}

#[cfg(test)]
#[path = "w4_residency_tests.rs"]
mod tests;
