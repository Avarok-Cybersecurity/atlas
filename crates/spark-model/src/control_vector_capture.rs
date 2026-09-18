// SPDX-License-Identifier: AGPL-3.0-only

//! Deriving a control vector: capturing per-layer mean activations.
//!
//! A control vector is a contrastive MEAN DIFFERENCE, not a trained artifact —
//! no gradients, no optimiser. For each layer you take the mean activation over
//! a "positive" corpus, subtract the mean over a matched "negative" corpus, and
//! normalise. This module is the capture half; `scripts/derive_control_vector.py`
//! does the subtraction and writes the GGUF.
//!
//! # Why derive natively rather than under llama.cpp
//!
//! The published vector for this model was derived from a **Q2_K_XL** quant
//! under llama.cpp. It transfers (measured), but a direction taken from the
//! activations Atlas actually serves has no transfer question at all. It also
//! avoids needing llama.cpp's `l_out` export patch, which exists only because a
//! hyper-connection model has no natural `[n_embd, n_tokens]` layer output for
//! the stock generator to read.
//!
//! # What is accumulated
//!
//! Per layer, the STREAM MEAN — the mean over the `hc_mult` highway rows — summed
//! over every token seen. That choice is not arbitrary: the apply path projects
//! every stream individually, and projection is linear, so a direction derived
//! from the stream mean is the one that composes with it. A direction taken from
//! a single stream would live in a different basis than the one being steered.
//!
//! Accumulation is FP64 on device. A corpus is thousands of tokens whose
//! per-element summands are frequently same-signed, which is exactly where FP32
//! drifts; a stable mean is the entire product of this pass and the cost is
//! nothing beside a forward.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Magic for the accumulator dump: "ACVC" (Atlas Control Vector Capture).
const DUMP_MAGIC: u32 = 0x4356_4341;
const CAPTURE_BLOCK: u32 = 256;

/// Per-layer running sum of stream-mean activations.
pub struct ControlVectorCapture {
    /// `[n_layer, hidden]` FP64 device accumulator.
    acc: DevicePtr,
    n_layer: usize,
    hidden: usize,
    /// Tokens folded into the sum so far. Counted on layer 0 only — every
    /// layer sees the same tokens in a pass, so counting them all would
    /// multiply the divisor by `n_layer`.
    tokens: AtomicU64,
    kernel: KernelHandle,
}

impl ControlVectorCapture {
    pub fn new(gpu: &dyn GpuBackend, n_layer: usize, hidden: usize) -> Result<Self> {
        let bytes = n_layer * hidden * 8;
        let acc = gpu
            .alloc(bytes)
            .context("allocating the control-vector capture accumulator")?;
        gpu.memset(acc, 0, bytes)
            .context("zeroing the capture accumulator")?;
        tracing::info!(
            "control-vector capture ARMED: [{n_layer}, {hidden}] FP64 accumulator \
             ({:.1} MB). This adds a reduction per layer per forward — a derivation \
             mode, not a serving mode.",
            bytes as f64 / 1e6
        );
        Ok(Self {
            acc,
            n_layer,
            hidden,
            tokens: AtomicU64::new(0),
            kernel: gpu.kernel("control_vector", "cvec_capture_accum")?,
        })
    }

    #[inline]
    fn row(&self, layer_idx: usize) -> DevicePtr {
        DevicePtr(self.acc.0 + (layer_idx * self.hidden * 8) as u64)
    }

    /// Fold this layer's activations into the running sum.
    ///
    /// Must be called on UNSTEERED activations — the whole point is to measure
    /// what the model does on its own, so the apply path runs after this.
    pub fn accumulate(
        &self,
        gpu: &dyn GpuBackend,
        highway: DevicePtr,
        layer_idx: usize,
        num_tokens: usize,
        hc_mult: usize,
        stream: u64,
    ) -> Result<()> {
        if layer_idx >= self.n_layer || num_tokens == 0 {
            return Ok(());
        }
        if layer_idx == 0 {
            self.tokens.fetch_add(num_tokens as u64, Ordering::Relaxed);
        }
        let threads = CAPTURE_BLOCK;
        let blocks = (self.hidden as u32).div_ceil(threads).min(256);
        KernelLaunch::new(gpu, self.kernel)
            .grid([blocks, 1, 1])
            .block([threads, 1, 1])
            .arg_ptr(highway)
            .arg_ptr(self.row(layer_idx))
            .arg_u32(self.hidden as u32)
            .arg_u32(hc_mult as u32)
            .arg_u32(num_tokens as u32)
            .launch(stream)
            .with_context(|| format!("capture accumulate at layer {layer_idx}"))
    }

    /// Tokens folded in so far (the divisor for the mean).
    pub fn tokens(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed)
    }

    /// Zero the accumulator. Run between the positive and negative corpus
    /// passes — forgetting this silently averages the two together, which
    /// produces a near-zero difference and looks like "the direction is weak"
    /// rather than like a mistake.
    pub fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.memset(self.acc, 0, self.n_layer * self.hidden * 8)
            .context("resetting the capture accumulator")?;
        self.tokens.store(0, Ordering::Relaxed);
        tracing::info!("control-vector capture: accumulator reset");
        Ok(())
    }

    /// Write the accumulator to `path`.
    ///
    /// Format: 32-byte header (magic, n_layer, hidden, pad, tokens) then
    /// `n_layer * hidden` little-endian f64. Self-describing so the offline
    /// script cannot silently pair dumps of different geometry.
    pub fn dump(&self, gpu: &dyn GpuBackend, path: &Path) -> Result<u64> {
        let n = self.n_layer * self.hidden;
        let mut host = vec![0u8; n * 8];
        gpu.copy_d2h(self.acc, &mut host)
            .context("reading back the capture accumulator")?;
        let tokens = self.tokens();
        anyhow::ensure!(
            tokens > 0,
            "capture dump refused: zero tokens accumulated. Nothing has been \
             prefilled since the last reset, so this file would be all zeros."
        );

        let mut out = Vec::with_capacity(32 + host.len());
        out.extend_from_slice(&DUMP_MAGIC.to_le_bytes());
        out.extend_from_slice(&(self.n_layer as u32).to_le_bytes());
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // pad
        out.extend_from_slice(&tokens.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // reserved
        out.extend_from_slice(&host);
        std::fs::write(path, &out)
            .with_context(|| format!("writing capture dump {}", path.display()))?;
        tracing::info!(
            "control-vector capture: wrote {} ({} layers x {} dims, {tokens} tokens)",
            path.display(),
            self.n_layer,
            self.hidden
        );
        Ok(tokens)
    }
}
