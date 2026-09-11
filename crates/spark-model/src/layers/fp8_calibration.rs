// SPDX-License-Identifier: AGPL-3.0-only

//! Online FP8 KV cache scale calibration.
//!
//! Tracks running max |K| and max |V| during the first
//! `--fp8-kv-calibration-tokens` tokens of inference to compute per-tensor
//! scales: `scale = amax * headroom / 448.0` (mapping the observed dynamic
//! range onto FP8 E4M3 `[-448, 448]`).
//!
//! # The invariant
//!
//! FP8 KV round-trips (write `fp8 = bf16/scale`, read `bf16 = fp8*scale`) only
//! if the SAME scale quantizes and dequantizes an entry. Paged / multi-query
//! attention reads a sequence's whole history in one pass with ONE
//! `k_scale`/`v_scale`, so a scale that changes under live cache entries
//! dequantizes them through the wrong basis (~6x error → generation garbage:
//! loops, empty completions). That is why the 2026-07-25 hardening froze the
//! scale on the FIRST observe.
//!
//! # Atlas #919
//!
//! Freezing on the first observe made `--fp8-kv-calibration-tokens 256` a lie.
//! Every H100 serve log showed
//! `FP8 KV cache with online calibration (checkpoint ships no k/v scales):
//!  freezing per-tensor scales on the first observed tokens.`
//! and the thing being observed was the readiness probe — so a 24k-context
//! serve ran on scales derived from ~13 tokens.
//!
//! The window is now real: the amax accumulates ACROSS requests until
//! `window_tokens` have been observed, and the batch that reaches the window
//! freezes on the amax of everything seen, itself included. The invariant is
//! preserved by REWRITING the entries written inside the window — the window's
//! BF16 K/V and slot mappings are staged aside and replayed through the
//! existing `reshape_and_cache_fp8` kernel at the frozen scale (see
//! [`staging`] for the full tradeoff). A readiness probe therefore counts
//! toward the window and can never end it on its own.
//!
//! Thread safety: uses `parking_lot::Mutex` for interior mutability. The lock
//! is uncontended (single inference thread) so lock overhead is negligible.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

mod staging;
mod state;

pub use staging::Fp8KvWriteTarget;
use staging::{KvStaging, MAX_STAGED_TOKENS};
use state::{CalibrationState, CalibrationStep, POST_FREEZE_OBSERVE_PERIOD};

/// Scale used for the calibration window's own writes, before the freeze.
///
/// Covers ±896: models with large norm weights (Gemma-4 26B, Mistral) reach
/// |K| ~600, which clips at scale 1.0. Those writes are requantized at the
/// frozen scale when the window closes, so this only has to be SAFE, not tight.
const PROVISIONAL_SCALE: f32 = 2.0;

/// Whether this KV dtype's write path calls [`Fp8KvCalibration::observe`].
///
/// SSOT with `qwen3_attention/decode/write_kv_cache.rs`: only the plain
/// `KvCacheDtype::Fp8` arm observes. BF16 boundary layers
/// (`--kv-high-precision-layers auto` on FP8 KV) never observe; attaching a
/// tracker there leaves `is_calibrating() == true` forever, and
/// `Iterator::find_map` on that first attention layer pins CUDA graphs eager
/// for the life of the process.
pub fn dtype_runs_online_fp8_kv_calibration(kv_dtype: KvCacheDtype) -> bool {
    matches!(kv_dtype, KvCacheDtype::Fp8)
}

/// Lift CUDA-graph suppression once every calibrating layer has frozen.
///
/// `None` = this layer does not calibrate (SSM, BF16 KV, static scales).
/// `Some(false)` = still warming. Vacuously true when no layer calibrates.
/// Must not use `find_map`: a BF16 boundary layer reporting `Some(false)`
/// would shadow later FP8 layers that already froze.
///
/// #919 note: graphs now stay eager for the whole calibration window instead
/// of one batch. That is the cost of calibrating on the requested token count;
/// the window is a few hundred tokens, once per process.
pub fn graphs_ready_after_fp8_kv_cal<I>(states: I) -> bool
where
    I: IntoIterator<Item = Option<bool>>,
{
    states.into_iter().all(|s| s.unwrap_or(true))
}

/// Mutable calibration state protected by Mutex for Send + Sync.
struct CalibrationInner {
    state: CalibrationState,
    staging: KvStaging,
    /// Set when a batch could not be staged, so the freeze log can say so.
    unstaged_tokens: usize,
}

/// Online FP8 KV cache scale calibration tracker for one attention layer.
///
/// Wraps calibration state in a Mutex so it can live inside a `Send + Sync`
/// struct (required by `TransformerLayer` trait).
pub struct Fp8KvCalibration {
    inner: Mutex<CalibrationInner>,
    /// Attention-layer index, for the freeze log line.
    attn_layer_idx: usize,
    /// Tokens the window was asked for, before the `MAX_STAGED_TOKENS` clamp.
    requested_tokens: usize,
    /// GPU buffer for absmax reduction output: `[1]` f32 for K, `[1]` f32 for V.
    /// Layout: `[k_absmax: f32, v_absmax: f32]` = 8 bytes.
    absmax_buf: DevicePtr,
    /// Kernel handle for bf16_absmax reduction.
    absmax_kernel: KernelHandle,
}

// SAFETY: DevicePtr is a raw GPU pointer (u64). It is only accessed from the
// inference thread that owns the CUDA context. The Mutex guards the mutable
// calibration state. All kernel launches are serialized on the CUDA stream.
unsafe impl Send for Fp8KvCalibration {}
unsafe impl Sync for Fp8KvCalibration {}

impl Fp8KvCalibration {
    /// Create a new calibration tracker.
    ///
    /// `attn_layer_idx`: this layer's index, for the freeze log line.
    /// `window_tokens`: `--fp8-kv-calibration-tokens`. The amax accumulates
    ///   over this many observed tokens, across requests, before the scale
    ///   freezes (#919). Clamped to `MAX_STAGED_TOKENS` so the BF16 staging
    ///   cannot reserve gigabytes per layer; 1 reproduces the pre-#919
    ///   freeze-on-first-observe behaviour, and 0 never gets here (the
    ///   attention initializer does not build a calibrator at all).
    /// `headroom`: multiplier on the accumulated amax when freezing
    ///   (`--fp8-kv-headroom`, CLI-validated ≥ 1.0; clamped here as defense in
    ///   depth because a sub-1.0 value guarantees clipping).
    /// `gpu`: GPU backend for allocating the absmax reduction buffer.
    pub fn new(
        attn_layer_idx: usize,
        window_tokens: usize,
        headroom: f32,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let headroom = if headroom >= 1.0 {
            headroom
        } else {
            tracing::warn!("fp8-kv headroom {headroom} < 1.0 guarantees clipping; clamped to 1.0");
            1.0
        };
        let effective = window_tokens.min(MAX_STAGED_TOKENS);
        if effective != window_tokens && attn_layer_idx == 0 {
            tracing::warn!(
                "--fp8-kv-calibration-tokens {window_tokens} exceeds the {MAX_STAGED_TOKENS}-token \
                 staging cap (the window's KV is held in BF16 so it can be requantized at the \
                 freeze); calibrating on {effective} tokens instead."
            );
        }
        let absmax_kernel = gpu.kernel("reshape_and_cache", "bf16_absmax")?;
        // Allocate 8 bytes: [k_absmax: f32, v_absmax: f32]
        let absmax_buf = gpu.alloc(8)?;
        // Initialize to zero
        let zeros = [0u8; 8];
        gpu.copy_h2d(&zeros, absmax_buf)?;

        Ok(Self {
            inner: Mutex::new(CalibrationInner {
                state: CalibrationState::new(effective, headroom, PROVISIONAL_SCALE),
                staging: KvStaging::default(),
                unstaged_tokens: 0,
            }),
            attn_layer_idx,
            requested_tokens: window_tokens,
            absmax_buf,
            absmax_kernel,
        })
    }

    /// Whether calibration is still in warmup phase (scales not yet frozen).
    pub fn is_calibrating(&self) -> bool {
        !self.inner.lock().state.frozen
    }

    /// Get current scales. Returns (k_scale, v_scale).
    ///
    /// Inside the window: [`PROVISIONAL_SCALE`], which every read during the
    /// window also uses — consistent, just coarse. After the freeze: the
    /// data-derived scale the whole window was requantized to (constant
    /// thereafter).
    pub fn scales(&self) -> (f32, f32) {
        let inner = self.inner.lock();
        (inner.state.k_scale, inner.state.v_scale)
    }

    /// Observe K/V projection outputs and update the running max.
    ///
    /// Launches absmax reductions on the K and V buffers, reads them back after
    /// a sync, then either stages this batch (still inside the window) or
    /// freezes and requantizes the staged window. Call this AFTER the K/V
    /// projections and BEFORE writing to the KV cache — the caller's write then
    /// uses [`Self::scales`], which is exactly what this call just decided.
    ///
    /// `k_data`/`v_data`: device BF16 K/V projection outputs.
    /// `num_tokens`/`num_kv_heads`/`head_dim`: this batch's shape.
    /// `target`: where the caller is about to write, so the freeze can replay
    ///   the window into the same pools.
    #[allow(clippy::too_many_arguments)]
    pub fn observe(
        &self,
        gpu: &dyn GpuBackend,
        k_data: DevicePtr,
        v_data: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        stream: u64,
        target: &Fp8KvWriteTarget,
    ) -> Result<()> {
        {
            let inner = self.inner.lock();
            if !inner.state.should_observe(num_tokens as usize) {
                return Ok(());
            }
        }

        let (k_max, v_max) = self.absmax(
            gpu,
            k_data,
            v_data,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
        )?;

        let mut inner = self.inner.lock();
        match inner.state.record(k_max, v_max, num_tokens as usize) {
            CalibrationStep::Stage => {
                let capacity = inner.state.window_tokens;
                let elems = (num_kv_heads * head_dim) as usize;
                let staged = inner.staging.stage(
                    gpu, k_data, v_data, num_tokens, elems, target, capacity, stream,
                )?;
                if !staged {
                    inner.unstaged_tokens += num_tokens as usize;
                }
            }
            CalibrationStep::Freeze {
                k_scale,
                v_scale,
                tokens_seen,
            } => {
                let staged_tokens = inner.staging.used_tokens();
                let inner = &mut *inner;
                inner.staging.replay_and_release(
                    gpu,
                    target,
                    num_kv_heads,
                    head_dim,
                    k_scale,
                    v_scale,
                    stream,
                )?;
                tracing::info!(
                    "FP8 KV scales frozen after {} tokens (requested {}) on attn layer {}: \
                     k_scale={:.6} (amax={:.3}), v_scale={:.6} (amax={:.3}), headroom={:.2}; \
                     requantized {} staged tokens in {} batches, {} unstaged",
                    tokens_seen,
                    self.requested_tokens,
                    self.attn_layer_idx,
                    k_scale,
                    inner.state.k_running_max,
                    v_scale,
                    inner.state.v_running_max,
                    inner.state.headroom,
                    staged_tokens,
                    inner.state.staged_batches,
                    inner.unstaged_tokens,
                );
            }
            CalibrationStep::Frozen => {
                if inner.state.tokens_seen % POST_FREEZE_OBSERVE_PERIOD < num_tokens as usize
                    && ema_recal_enabled()
                {
                    // F5 (2026-05-26): post-freeze EMA recalibration is OPT-IN via
                    // `ATLAS_FP8_KV_EMA_RECAL=1`, default OFF. Moving `k_scale` /
                    // `v_scale` after the freeze makes every already-written cache
                    // entry stale relative to the new scales — attention then reads
                    // the whole history through a shifted quantization basis. The
                    // forensic study of the canonical opencode probe shows
                    // reasoning-channel collapse and drift-to-phantom-path patterns
                    // whose timing matches deep-layer KV read through a rescaled
                    // basis. #919's staging only covers the calibration window, so
                    // this path is still unsafe by construction — it stays off.
                    inner.state.ema_recalibrate(k_max, v_max);
                    tracing::info!(
                        "FP8 KV EMA-recalibrated after {} tokens on attn layer {}: \
                         k_scale={:.6} (amax={:.2}), v_scale={:.6} (amax={:.2})",
                        inner.state.tokens_seen,
                        self.attn_layer_idx,
                        inner.state.k_scale,
                        inner.state.k_running_max,
                        inner.state.v_scale,
                        inner.state.v_running_max,
                    );
                }
            }
        }
        Ok(())
    }

    /// Absmax of the K and V projection outputs, read back to the host.
    #[allow(clippy::too_many_arguments)]
    fn absmax(
        &self,
        gpu: &dyn GpuBackend,
        k_data: DevicePtr,
        v_data: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        stream: u64,
    ) -> Result<(f32, f32)> {
        let n_elems = num_tokens * num_kv_heads * head_dim;
        // Reset the absmax buffer to 0.0 before the reduction (async, to avoid
        // a sync/async conflict on the stream).
        gpu.memset_async(self.absmax_buf, 0, 8, stream)?;
        let k_out = self.absmax_buf;
        super::ops::bf16_absmax(gpu, self.absmax_kernel, k_data, k_out, n_elems, stream)?;
        // V writes to offset 4 = the second f32.
        let v_out = self.absmax_buf.offset(4);
        super::ops::bf16_absmax(gpu, self.absmax_kernel, v_data, v_out, n_elems, stream)?;

        gpu.synchronize(stream)?;
        let mut buf = [0u8; 8];
        gpu.copy_d2h(self.absmax_buf, &mut buf)?;
        Ok((
            f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            f32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        ))
    }
}

fn ema_recal_enabled() -> bool {
    std::env::var("ATLAS_FP8_KV_EMA_RECAL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests;
