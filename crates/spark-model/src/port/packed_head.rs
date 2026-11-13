// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the output head served from the GGUF file's raw Q6_K
//! super-blocks on targets that do not build the K-quant module (port lane L24,
//! the "packed LM head" of plan G4).
//!
//! # Why
//!
//! Qwen3.8-27B's head is `[248320, 5120]`: 2.54 GB in BF16 and read once per
//! token, against 1.04 GB as the Unsloth file's Q6_K blocks (210 bytes per 256
//! weights). The pre-sprint packed-head path (`model/lm_head_q6k.rs`, written
//! for the DeepSeek-V4.1 GGUF on GB10) resolves `kquant_moe`, a module only
//! the deepseek model directory compiles, so on Ada the head would expand to
//! BF16. This module serves the same blocks through lane L19's `W4Linear`
//! (`port::gguf_q4`, `kernels/ada/common/kquant_mmvq_q4_ada.cu`, the same
//! llama.cpp `Q6_K x q8_1` dot), in chunks of the weight's decode rows.
//! `lm_head_q6k.rs` picks it when, and only when, `kquant_moe` is absent, so
//! the DeepSeek head on GB10 resolves exactly the kernels it did.
//!
//! # Opt-in
//!
//! The head stays packed in the store only when the operator passes
//! `--lm-head-packed` (published here and, as the GGUF loader's head switch,
//! to `spark_runtime::weights::keep_packed_iq4::set_keep_head`), and only on
//! the keep-packed GGUF route (row `gguf_native_iq4`, off by default). Every
//! other consumer of the LM head pointer reads BF16, so [`check_load`] refuses
//! the combinations that would read packed bytes as BF16: a speculative serve
//! (the MTP drafter quantizes its draft head from the BF16 head) and any head
//! requantization (`--lm-head-dtype` other than `bf16`).

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Result, bail};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::dequant_cpu::GgmlType;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::port::w4::{W4Box, W4Linear, W4Scratch};

static REQUESTED: AtomicBool = AtomicBool::new(false);

/// Publish `--lm-head-packed`. Called once by spark-server's `port_apply`
/// before any model loads.
pub fn set_requested(on: bool) {
    REQUESTED.store(on, Ordering::Release);
}

/// The published flag (false until [`set_requested`]).
pub fn requested() -> bool {
    REQUESTED.load(Ordering::Acquire)
}

/// Every name a loader binds the LM head from.
const LM_HEAD_KEYS: [&str; 3] = [
    "lm_head.weight",
    "language_model.lm_head.weight",
    "model.lm_head.weight",
];

/// The factory's check right after the LM head is loaded (`factory/build.rs`,
/// after `load_lm_head`): with `--lm-head-packed`, the store's head must be a
/// kept-packed Q6_K tensor that nothing will read as BF16.
pub fn check_load(store: &WeightStore, config: &ModelConfig, speculative: bool) -> Result<()> {
    check_load_with(requested(), store, config, speculative)
}

/// [`check_load`] with the flag passed in.
pub fn check_load_with(
    requested: bool,
    store: &WeightStore,
    config: &ModelConfig,
    speculative: bool,
) -> Result<()> {
    if !requested {
        return Ok(());
    }
    let head = LM_HEAD_KEYS
        .iter()
        .find_map(|k| store.get(k).ok().map(|t| (*k, t.dtype)));
    let Some((name, dtype)) = head else {
        tracing::warn!(
            "--lm-head-packed: this checkpoint ties its LM head to the embedding; nothing to keep \
             packed, the head is served as before"
        );
        return Ok(());
    };
    if dtype != WeightDtype::Q6K && crate::port::gguf_q4::q4_type_of_dtype(dtype).is_some() {
        // Kept packed as IQ4_XS or Q4_K: the factory installs a packed head
        // only for Q6_K, so every reader would take these blocks as BF16.
        bail!(
            "--lm-head-packed: {name} was kept packed as {dtype:?}, and only a Q6_K head is served \
             packed. Drop --lm-head-packed so the GGUF loader expands the head."
        );
    }
    if dtype != WeightDtype::Q6K {
        tracing::warn!(
            "--lm-head-packed: {name} is {dtype:?}, not a kept-packed Q6_K head (the keep-packed \
             GGUF route, row gguf_native_iq4, keeps only Q6_K heads); the head is served as before"
        );
        return Ok(());
    }
    if speculative {
        bail!(
            "--lm-head-packed: {name} is kept as raw Q6_K blocks, but the MTP drafter builds its \
             draft head from a BF16 LM head and would read the packed bytes as BF16. Serve without \
             --speculative, or drop --lm-head-packed."
        );
    }
    if !config.skip_lm_head_quantization() || config.lm_head_fp8 {
        bail!(
            "--lm-head-packed: {name} is kept as raw Q6_K blocks, and the configured LM head dtype \
             would requantize it from BF16. Pass --lm-head-dtype bf16 (the packed head replaces it)."
        );
    }
    tracing::info!("--lm-head-packed: {name} stays as raw Q6_K blocks for the packed head");
    Ok(())
}

/// The Q6_K head as a `W4Linear` and the workspace its decode rows need.
pub struct PackedHead {
    w: W4Box,
    scratch: DevicePtr,
    scratch_bytes: usize,
    rows: usize,
}

impl std::fmt::Debug for PackedHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let w: &dyn W4Linear = &*self.w;
        f.debug_struct("PackedHead")
            .field("format", &w.format())
            .field("n", &w.n())
            .field("k", &w.k())
            .field("rows", &self.rows)
            .field("scratch_bytes", &self.scratch_bytes)
            .finish()
    }
}

impl PackedHead {
    /// Wrap the store's `[vocab, hidden]` Q6_K blocks (borrowed; the store
    /// keeps them alive) and allocate the decode workspace. Errors when the
    /// build lacks lane L19's Ada modules.
    pub fn new(
        gpu: &dyn GpuBackend,
        blocks: DevicePtr,
        vocab: usize,
        hidden: usize,
    ) -> Result<Self> {
        let w =
            crate::port::gguf_q4::from_device_blocks(gpu, GgmlType::Q6K, blocks, vocab, hidden)?;
        Self::from_linear(gpu, w)
    }

    /// Serve any `W4Linear` head (the constructor above, or a test double).
    pub fn from_linear(gpu: &dyn GpuBackend, w: W4Box) -> Result<Self> {
        let rows = w.max_decode_rows();
        anyhow::ensure!(rows >= 1, "packed head: the weight serves no decode rows");
        let scratch_bytes = w.scratch_bytes(rows);
        let scratch = if scratch_bytes > 0 {
            gpu.alloc(scratch_bytes)?
        } else {
            DevicePtr::NULL
        };
        Ok(Self {
            w,
            scratch,
            scratch_bytes,
            rows,
        })
    }

    /// Device bytes of the head's blocks.
    pub fn device_bytes(&self) -> usize {
        self.w.device_bytes()
    }

    /// Logits for `m` BF16 hidden rows (`[m, hidden]`, contiguous) into
    /// `logits` (`[m, vocab]` BF16), in chunks of the weight's decode rows.
    /// Row r's bits do not depend on the chunk it lands in (the `W4Linear`
    /// contract), so a prompt's rows equal the same rows decoded one at a time.
    pub fn run(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        m: usize,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let (n, k) = (self.w.n(), self.w.k());
        let s = W4Scratch {
            ws: self.scratch,
            ws_bytes: self.scratch_bytes,
            tickets: DevicePtr::NULL,
            xq: DevicePtr::NULL,
            xs: DevicePtr::NULL,
        };
        let mut done = 0usize;
        while done < m {
            let rows = (m - done).min(self.rows);
            let x = hidden.offset(done * k * 2);
            let out = logits.offset(done * n * 2);
            self.w.decode(gpu, x, rows, out, &s, stream)?;
            done += rows;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "packed_head_tests.rs"]
mod tests;
