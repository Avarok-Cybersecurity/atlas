// SPDX-License-Identifier: AGPL-3.0-only

//! Packed NVFP4 → `f32`, on the host, for the tensors this loader reaches
//! through [`super::LayerSource::f32`].
//!
//! # Why this exists
//!
//! The reference checkpoint this port was built against
//! (`LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`) leaves the three dense MLP
//! layers (`first_k_dense_replace = 3`) **unquantised**: `layers.{0,1,2}.mlp.
//! {gate,up,down}_proj.weight` are plain BF16. NVIDIA's official export,
//! `nvidia/GLM-5.3-Flash-NVFP4` (ModelOpt), quantises them:
//!
//! ```text
//! model.language_model.layers.{0,1,2}.mlp.gate_proj.weight
//!   LibertAIDAI : BF16 [12288, 4096]
//!   nvidia      : U8   [12288, 2048]  (+ .weight_scale F8_E4M3 [12288, 256],
//!                                       .weight_scale_2 F32 scalar)
//! ```
//!
//! `LayerSource::f32` bailed on U8, and the dense MLP builder
//! ([`crate::layers::glm5next_mlp::build::build_dense_mlp`]) only ever sees
//! `f32`. Dequantising here keeps the builder, the weight structs and the
//! kernels untouched: the dense MLP stays BF16 on the device for both exports,
//! which is what `Glm5NextDenseMlpWeights` promises and what the GEMM tier
//! selection assumes.
//!
//! # The format, and where the convention comes from
//!
//! `value = E2M1[nibble] * e4m3(block_scale) * weight_scale_2`, the ModelOpt
//! "direct multiplier" convention — the same product
//! `weight_map::fp8_lut::dequant_nvfp4_to_bf16` folds for the dense NVFP4 path
//! and the same multiply ORDER the routed-expert kernel uses
//! (`kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu`:
//! `E2M1_LUT[nibble] * (float)fp8 * scale2`).
//!
//! Three conventions have to agree with the kernels that read the SAME
//! checkpoint, so all three are read from the runtime's own tables rather than
//! restated here:
//!
//! * the codebook — [`NVFP4_E2M1_LUT`], which mirrors
//!   `kernels/gb10/common/paged_decode_attn_nvfp4.cu`;
//! * the block width — [`NVFP4_GROUP_SIZE`] (16), NOT the 32 of DeepSeek-V4's
//!   E8M0 microscaling variant;
//! * the scale decode — [`e4m3_lut`].
//!
//! 🪤 **Nibble order.** Even flat index = LOW nibble, odd = HIGH. This is the
//! ModelOpt/CUTLASS packing and it is what `avarok_core::mxfp4_e8m0` and the
//! CUTLASS comparator (`cutlass/tests/nvfp4.rs`: `if kk % 2 == 0 { byte & 0x0f
//! } else { byte >> 4 }`) both read. Swapping the two halves produces a
//! well-formed tensor of the right shape with every pair of columns exchanged —
//! no shape error, no NaN, just a wrong model.
//!
//! # What is and is not dequantised
//!
//! Entered only when the tensor is **U8 on disk** and carries the
//! `.weight_scale` / `.weight_scale_2` siblings. A BF16 tensor never reaches
//! this module, so `LibertAIDAI/GLM-5.3-Flash-NVFP4` allocates nothing here and
//! takes the identical code path it did before this module existed.

use anyhow::{Result, bail};
use spark_runtime::kv_dequant::{NVFP4_E2M1_LUT, NVFP4_GROUP_SIZE, e4m3_lut};

/// Packed NVFP4 `[rows, cols/2]` + E4M3 block scales `[rows, cols/16]` + one
/// global `f32` → row-major `f32 [rows, cols]`.
///
/// `packed_shape` is the tensor's ON-DISK shape, so the logical column count is
/// `2 * packed_shape[1]`. Every length is checked against it: a truncated shard
/// is an error here rather than a shorter tensor the MLP builder then rejects
/// with an element-count message that points at the wrong thing.
pub(super) fn dequant_nvfp4_to_f32(
    what: &str,
    packed: &[u8],
    packed_shape: &[usize],
    scales: &[u8],
    scale_2: f32,
) -> Result<Vec<f32>> {
    let [rows, packed_cols] = packed_shape[..] else {
        bail!("{what}: packed NVFP4 must be 2-D, got shape {packed_shape:?}");
    };
    if packed.len() != rows * packed_cols {
        bail!(
            "{what}: {} B of packed NVFP4 for shape [{rows}, {packed_cols}]",
            packed.len()
        );
    }
    let cols = packed_cols * 2;
    if !cols.is_multiple_of(NVFP4_GROUP_SIZE) {
        bail!(
            "{what}: {cols} logical columns is not a whole number of \
             {NVFP4_GROUP_SIZE}-element NVFP4 blocks"
        );
    }
    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    if scales.len() != rows * groups_per_row {
        bail!(
            "{what}: {} block scales, expected {rows} x {groups_per_row} = {}",
            scales.len(),
            rows * groups_per_row
        );
    }
    let e4m3 = e4m3_lut();
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let prow = &packed[r * packed_cols..(r + 1) * packed_cols];
        let srow = &scales[r * groups_per_row..(r + 1) * groups_per_row];
        let orow = &mut out[r * cols..(r + 1) * cols];
        for (g, &sb) in srow.iter().enumerate() {
            // 🪤 The LUT is the full 256-entry E4M3 table, so every byte
            // INDEXES — `0x7F` is NaN, `0x80..` are the negatives, and both
            // would silently produce a tensor of NaNs or a sign-flipped block
            // instead of an error. A block scale is an amax ratio: non-
            // negative and finite by construction, i.e. `0x00..=0x7E`. A byte
            // outside that means the `.weight_scale` sibling is not the E4M3
            // block-scale tensor this code thinks it is — a different export
            // convention, a misaligned read, or the wrong tensor entirely —
            // and saying so here beats debugging NaN logits.
            if sb >= 0x7F {
                bail!(
                    "{what}: block scale byte 0x{sb:02X} at row {r}, block {g} is not a \
                     finite non-negative E4M3 value (0x7F is NaN, 0x80.. are negative). \
                     The `.weight_scale` sibling is not ModelOpt E4M3 block scales — check \
                     that it is F8_E4M3 of shape [{rows}, {groups_per_row}] and not, say, a \
                     compressed-tensors `weight_global_scale`."
                );
            }
            // 🪤 The kernel multiplies `(E2M1 * fp8) * scale2`, in that order.
            // Folding `fp8 * scale2` first is a different f32 rounding and the
            // difference is visible in a byte-identity check.
            let s = e4m3[sb as usize];
            let base = g * NVFP4_GROUP_SIZE;
            for i in 0..NVFP4_GROUP_SIZE {
                let c = base + i;
                let byte = prow[c / 2];
                let nibble = if c.is_multiple_of(2) {
                    byte & 0x0F
                } else {
                    byte >> 4
                };
                orow[c] = NVFP4_E2M1_LUT[nibble as usize] * s * scale_2;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "nvfp4_dequant_tests.rs"]
mod tests;
