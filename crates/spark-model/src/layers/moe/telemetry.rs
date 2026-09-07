// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request MoE expert-activation telemetry: the device-side staging half.
//!
//! Records, for one forward pass, which experts every routed token selected
//! and with what weight, so a request can report the set of experts it
//! actually used. The `expert-categories` benchmark aggregates that over a
//! category's prompts; `--expert-category` later loads only those experts.
//!
//! ## Why staging instead of reading the router's own buffers
//!
//! The router already writes `[n * top_k]` expert ids and weights into
//! `ctx.buffers.scratch()` on every path. Two problems with reading them
//! where they are written:
//!
//!  1. **Scratch is reused.** The very next step of the same MoE forward
//!     overwrites those bytes with sorted token ids. Anything that reads
//!     them must do so before that, i.e. inside the layer, per layer.
//!  2. **A D2H there costs a sync per layer.** `union_stats` does exactly
//!     that and is why it samples 1-in-64. Sixty-one syncs per forward is
//!     not something a request can opt into.
//!
//! So each layer instead issues one **device-to-device** copy into a
//! per-layer slot of a buffer that lives for the model's lifetime, and the
//! host reads the whole thing once, after the pass, where the trait impl
//! already synchronizes to read logits.
//!
//! ## CUDA graphs
//!
//! A D2H copy or a stream sync inside graph capture invalidates the capture
//! (CUDA 901) and wedges the serve — the constraint documented on
//! [`super::union_stats`]. A D2D copy on the captured stream does not: it
//! records as a memcpy node and re-executes on every replay. That is what
//! makes graphed decode observable here at all, since a graph replay runs no
//! host code and any host-side tap would silently record nothing.
//!
//! The staging buffer's address is fixed at allocation for the same reason
//! `moe_row_adapter_buf` is: a captured node holds the pointer it was
//! recorded with, so the destination cannot be re-allocated per step.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Device staging for one model's expert-routing telemetry.
///
/// Layout is `[layer][row][top_k]`, indexed by ABSOLUTE layer index so a
/// model whose MoE layers are interleaved with dense ones needs no
/// remapping — dense layers simply never write their slot. Ids and weights
/// are separate allocations of the same shape.
pub struct ExpertTelemetryStaging {
    indices: DevicePtr,
    weights: DevicePtr,
    /// Rows a single pass may stage per layer. A pass wider than this stages
    /// its first `max_rows` rows and reports the overflow (see
    /// [`Self::rows_for`]) rather than writing out of bounds.
    max_rows: usize,
    top_k: usize,
    num_layers: usize,
}

impl ExpertTelemetryStaging {
    /// Allocate staging for `num_layers` layers × `max_rows` rows × `top_k`.
    ///
    /// Sized for the widest pass the serve can issue; at 61 layers, 2048
    /// rows and top-k 8 that is 4 MiB per array. Allocated only when expert
    /// telemetry is enabled for the model, so a serve without it pays
    /// nothing.
    pub fn new(
        gpu: &dyn GpuBackend,
        num_layers: usize,
        max_rows: usize,
        top_k: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            num_layers > 0 && max_rows > 0 && top_k > 0,
            "expert telemetry staging needs non-zero dimensions, got \
             layers={num_layers} rows={max_rows} top_k={top_k}"
        );
        let elems = num_layers * max_rows * top_k;
        let indices = gpu.alloc(elems * 4)?;
        let weights = gpu.alloc(elems * 4)?;
        // Zeroed once so a slot never read this pass cannot present the
        // previous pass's ids as this request's routing. Per-pass rows are
        // bounded by `rows_for`, so only a partially-filled tail could ever
        // be read, and a zero weight is what the fold treats as "not routed".
        gpu.memset(indices, 0, elems * 4)?;
        gpu.memset(weights, 0, elems * 4)?;
        Ok(Self {
            indices,
            weights,
            max_rows,
            top_k,
            num_layers,
        })
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// How many of `n` rows this staging can hold. A pass wider than
    /// `max_rows` is truncated rather than refused: telemetry must never
    /// change what a serve can answer, and the drain reports the truncation
    /// so the shortfall is visible instead of being read as "these experts
    /// were not used".
    pub fn rows_for(&self, n: usize) -> usize {
        n.min(self.max_rows)
    }

    fn layer_offset(&self, layer_idx: usize) -> usize {
        layer_idx * self.max_rows * self.top_k * 4
    }

    /// Stage one layer's routing for this pass.
    ///
    /// `indices_dev` / `weights_dev` are the router's `[n * top_k]` outputs,
    /// already written on `stream`. The copy is issued on the same stream so
    /// it is ordered after the top-k kernel, and is capture-legal.
    #[allow(clippy::too_many_arguments)]
    pub fn stage(
        &self,
        gpu: &dyn GpuBackend,
        layer_idx: usize,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        // First staging row these `n` rows belong to. The grouped prefill
        // paths route a whole chunk at once and pass 0; the per-token path
        // (`forward_batched`, taken for prefills of <= 64 tokens) routes one
        // token per iteration and passes its loop index, which is that
        // token's position in the pass.
        row_offset: usize,
        n: usize,
        top_k: usize,
        stream: u64,
    ) -> Result<()> {
        // A model whose top-k or layer count disagrees with the staging it
        // was allocated for would mis-slice every row. Refuse rather than
        // record a plausible-looking wrong answer.
        anyhow::ensure!(
            top_k == self.top_k,
            "expert telemetry staged with top_k={top_k}, allocated for {}",
            self.top_k
        );
        if layer_idx >= self.num_layers {
            anyhow::bail!(
                "expert telemetry staged for layer {layer_idx}, allocated for {} layers",
                self.num_layers
            );
        }
        if row_offset >= self.max_rows {
            // Past the staging width: the drain reports the shortfall.
            return Ok(());
        }
        let rows = n.min(self.max_rows - row_offset);
        if rows == 0 {
            return Ok(());
        }
        let bytes = rows * top_k * 4;
        let off = self.layer_offset(layer_idx) + row_offset * top_k * 4;
        gpu.copy_d2d_async(indices_dev, self.indices.offset(off), bytes, stream)?;
        gpu.copy_d2d_async(weights_dev, self.weights.offset(off), bytes, stream)?;
        Ok(())
    }

    /// Copy this pass's staged rows back to the host.
    ///
    /// Caller must have synchronized `stream` first — every drain site is
    /// one that already syncs to read logits or a sampled token, so this
    /// adds no sync of its own. Returns `(ids, weights)`, each
    /// `[num_layers * rows * top_k]` in layer-major order.
    pub fn drain(&self, gpu: &dyn GpuBackend, rows: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let rows = self.rows_for(rows);
        let per_layer = rows * self.top_k;
        let mut ids = vec![0u32; self.num_layers * per_layer];
        let mut ws = vec![0f32; self.num_layers * per_layer];
        if per_layer == 0 {
            return Ok((ids, ws));
        }
        // Per layer, not one big copy: the staged rows are a prefix of each
        // layer's `max_rows` slot, so the used bytes are not contiguous.
        for layer in 0..self.num_layers {
            let off = self.layer_offset(layer);
            let dst = layer * per_layer;
            let id_bytes = bytemuck_u32_mut(&mut ids[dst..dst + per_layer]);
            gpu.copy_d2h(self.indices.offset(off), id_bytes)?;
            let w_bytes = bytemuck_f32_mut(&mut ws[dst..dst + per_layer]);
            gpu.copy_d2h(self.weights.offset(off), w_bytes)?;
        }
        Ok((ids, ws))
    }
}

impl super::MoeLayer {
    /// Stage this layer's routing for the pass, if telemetry is on and this
    /// MoE is one of the model's own layers.
    ///
    /// Called from the PREFILL paths only. Decode is not staged in v1: its
    /// row→sequence mapping is not knowable inside the layer, because
    /// batched decode calls `FfnComponent::forward` once per sequence with
    /// the offset applied to the input pointer and no row index passed
    /// down. Staging at row 0 from there would have every sequence in a
    /// batch overwrite the previous one's routing, and the result would
    /// look like a plausible answer for whichever sequence happened to run
    /// last. Prefill has one sequence and rows `0..n`, so it is exact.
    ///
    /// A response reports the scope it actually measured (see
    /// `ExpertActivationAcc`), so this limit is visible to callers rather
    /// than being read as "the request used only these experts".
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage_expert_telemetry(
        &self,
        ctx: &crate::layer::ForwardContext<'_>,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        row_offset: usize,
        n: usize,
        top_k: usize,
        stream: u64,
    ) -> Result<()> {
        let (Some(staging), Some(layer_idx)) = (ctx.expert_telemetry, self.site.layer_idx()) else {
            return Ok(());
        };
        staging.stage(
            ctx.gpu,
            layer_idx,
            indices_dev,
            weights_dev,
            row_offset,
            n,
            top_k,
            stream,
        )
    }
}

/// Reinterpret a `u32` slice as bytes for a D2H copy. Both are plain data
/// with no padding and the device wrote little-endian u32s, which is the
/// host layout on every target Atlas builds for.
fn bytemuck_u32_mut(v: &mut [u32]) -> &mut [u8] {
    // SAFETY: u32 has no invalid bit patterns and no padding; the resulting
    // slice covers exactly the same bytes.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn bytemuck_f32_mut(v: &mut [f32]) -> &mut [u8] {
    // SAFETY: as above. Any bit pattern is a valid f32 (NaNs included), and
    // a NaN weight would be a real routing defect worth surfacing, not UB.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
