// SPDX-License-Identifier: AGPL-3.0-only

//! Online FP8 KV cache scale calibration.
//!
//! Tracks running max |K| and max |V| values during the first N tokens of
//! inference to compute per-tensor scales: `scale = max / 448.0` (mapping the
//! observed dynamic range to FP8 E4M3 [-448, 448]).
//!
//! After the warmup period, scales are frozen and used for all subsequent
//! tokens. During warmup, FP8 KV cache writes use scale=1.0 (uncalibrated).
//! This is safe because typical attention projection outputs are well within
//! [-448, 448] during the first few hundred tokens.
//!
//! Thread safety: uses `parking_lot::Mutex` for interior mutability. The lock
//! is uncontended (single inference thread) so lock overhead is negligible.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// FP8 E4M3 max representable magnitude.
const FP8_E4M3_MAX: f32 = 448.0;

/// Minimum scale to prevent division by zero or denormalized values.
const MIN_SCALE: f32 = 1e-12;

/// Mutable calibration state protected by Mutex for Send + Sync.
struct CalibrationInner {
    /// Running max of |K| values observed so far.
    k_running_max: f32,
    /// Running max of |V| values observed so far.
    v_running_max: f32,
    /// Total tokens processed during calibration.
    tokens_seen: usize,
    /// Whether calibration is complete (scales frozen).
    frozen: bool,
    /// Calibrated k_scale (set once after warmup).
    k_scale: f32,
    /// Calibrated v_scale (set once after warmup).
    v_scale: f32,
}

/// Online FP8 KV cache scale calibration tracker for one attention layer.
///
/// Wraps calibration state in a Mutex so it can live inside a `Send + Sync`
/// struct (required by `TransformerLayer` trait).
pub struct Fp8KvCalibration {
    inner: Mutex<CalibrationInner>,
    /// Number of warmup tokens before freezing scales.
    warmup_tokens: usize,
    /// GPU buffer for absmax reduction output: [1] f32 for K, [1] f32 for V.
    /// Layout: [k_absmax: f32, v_absmax: f32] = 8 bytes.
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
    /// `warmup_tokens`: number of tokens to observe before freezing scales.
    /// `gpu`: GPU backend for allocating the absmax reduction buffer.
    pub fn new(warmup_tokens: usize, gpu: &dyn GpuBackend) -> Result<Self> {
        let absmax_kernel = gpu.kernel("reshape_and_cache", "bf16_absmax")?;
        // Allocate 8 bytes: [k_absmax: f32, v_absmax: f32]
        let absmax_buf = gpu.alloc(8)?;
        // Initialize to zero
        let zeros = [0u8; 8];
        gpu.copy_h2d(&zeros, absmax_buf)?;

        Ok(Self {
            inner: Mutex::new(CalibrationInner {
                k_running_max: 0.0,
                v_running_max: 0.0,
                tokens_seen: 0,
                frozen: false,
                // Start with scale=2.0 (effective range ±896) during warmup.
                // Models with large norm weights (Gemma-4 26B, Mistral) can produce
                // K/V values up to ~600, which exceeds FP8 E4M3 range at scale=1.0 (±448).
                // Scale=2.0 covers ±896 which is safe for all known models.
                k_scale: 2.0,
                v_scale: 2.0,
            }),
            warmup_tokens,
            absmax_buf,
            absmax_kernel,
        })
    }

    /// Whether calibration is still in warmup phase (scales not yet frozen).
    pub fn is_calibrating(&self) -> bool {
        let inner = self.inner.lock();
        !inner.frozen
    }

    /// Get current scales. Returns (k_scale, v_scale).
    ///
    /// During warmup: returns (1.0, 1.0) (uncalibrated).
    /// After warmup: returns calibrated scales.
    pub fn scales(&self) -> (f32, f32) {
        let inner = self.inner.lock();
        (inner.k_scale, inner.v_scale)
    }

    /// Observe K/V projection outputs and update running max.
    ///
    /// Launches absmax reduction kernels on the K and V buffers, then reads
    /// the results back to CPU after a sync. Call this AFTER K/V projections
    /// and BEFORE writing to the KV cache.
    ///
    /// `k_data`: device BF16 buffer of K projection output
    /// `v_data`: device BF16 buffer of V projection output
    /// `num_tokens`: number of tokens in the batch
    /// `num_kv_heads`: number of KV heads
    /// `head_dim`: dimension per head
    pub fn observe(
        &self,
        gpu: &dyn GpuBackend,
        k_data: DevicePtr,
        v_data: DevicePtr,
        num_tokens: u32,
        num_kv_heads: u32,
        head_dim: u32,
        stream: u64,
    ) -> Result<()> {
        // Observe during warmup (always) and periodically after (every 512 tokens)
        // to catch distribution shifts in multi-turn conversations. Without periodic
        // recalibration, FP8 KV scales become stale as context grows, causing recent
        // K values to collapse into fewer E4M3 quantization buckets.
        // Recalibrate every 128 tokens (was 512) to catch distribution shifts
        // in multi-turn conversations where K/V statistics change frequently.
        let should_observe = {
            let inner = self.inner.lock();
            !inner.frozen || (inner.tokens_seen % 128 < num_tokens as usize)
        };
        if !should_observe {
            return Ok(());
        }

        let n_elems = num_tokens * num_kv_heads * head_dim;

        // Reset absmax buffer to 0.0 before reduction (async to avoid sync/async conflict)
        gpu.memset_async(self.absmax_buf, 0, 8, stream)?;

        // Launch absmax for K
        let k_out = self.absmax_buf;
        super::ops::bf16_absmax(gpu, self.absmax_kernel, k_data, k_out, n_elems, stream)?;

        // Launch absmax for V (write to offset 4 = second f32)
        let v_out = self.absmax_buf.offset(4);
        super::ops::bf16_absmax(gpu, self.absmax_kernel, v_data, v_out, n_elems, stream)?;

        // Sync and read back
        gpu.synchronize(stream)?;
        let mut result_buf = [0u8; 8];
        gpu.copy_d2h(self.absmax_buf, &mut result_buf)?;
        let k_max =
            f32::from_le_bytes([result_buf[0], result_buf[1], result_buf[2], result_buf[3]]);
        let v_max =
            f32::from_le_bytes([result_buf[4], result_buf[5], result_buf[6], result_buf[7]]);

        // Update running max and check if warmup is complete
        let mut inner = self.inner.lock();
        inner.k_running_max = inner.k_running_max.max(k_max);
        inner.v_running_max = inner.v_running_max.max(v_max);
        inner.tokens_seen += num_tokens as usize;

        if inner.tokens_seen >= self.warmup_tokens && !inner.frozen {
            // Initial calibration: compute scales from warmup observations.
            inner.k_scale = (inner.k_running_max / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.v_scale = (inner.v_running_max / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.frozen = true;
        } else if inner.frozen && inner.tokens_seen % 128 < num_tokens as usize {
            // Periodic recalibration: blend new observations with existing scales.
            // Adaptive EMA: use α=0.3 when distribution shifts >20%, else α=0.1.
            // This responds faster to multi-turn topic switches without oscillating
            // on stable distributions.
            let new_k = (k_max / FP8_E4M3_MAX).max(MIN_SCALE);
            let new_v = (v_max / FP8_E4M3_MAX).max(MIN_SCALE);
            let k_shift = (new_k - inner.k_scale).abs() / inner.k_scale.max(MIN_SCALE);
            let v_shift = (new_v - inner.v_scale).abs() / inner.v_scale.max(MIN_SCALE);
            let alpha = if k_shift > 0.2 || v_shift > 0.2 {
                0.3
            } else {
                0.1
            };
            inner.k_scale = (1.0 - alpha) * inner.k_scale + alpha * new_k;
            inner.v_scale = (1.0 - alpha) * inner.v_scale + alpha * new_v;
            // Reset running max for next observation window
            inner.k_running_max = k_max;
            inner.v_running_max = v_max;

            tracing::info!(
                "FP8 KV calibrated after {} tokens: k_scale={:.6} (max={:.2}), v_scale={:.6} (max={:.2})",
                inner.tokens_seen,
                inner.k_scale,
                inner.k_running_max,
                inner.v_scale,
                inner.v_running_max,
            );
        }

        Ok(())
    }
}
