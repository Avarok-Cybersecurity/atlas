// SPDX-License-Identifier: AGPL-3.0-only

//! Online FP8 KV cache scale calibration.
//!
//! Tracks running max |K| and max |V| values during the first N tokens of
//! inference to compute per-tensor scales: `scale = max / 448.0` (mapping the
//! observed dynamic range to FP8 E4M3 [-448, 448]).
//!
//! The scale is frozen on the FIRST observation (which runs immediately before
//! the first KV write) and is then constant for the entire lifetime of every
//! cache entry. This is required for correctness: FP8 KV round-trips (write
//! `fp8=bf16/scale`, read `bf16=fp8*scale`) only if the SAME scale quantizes
//! and dequantizes an entry. Freezing after N warmup tokens (the old design)
//! wrote early entries at a placeholder scale, then froze to a different value,
//! silently invalidating all already-written / cached KV — a paged multi-query
//! read spanning the freeze boundary then read history through the wrong scale
//! and generation degenerated. See the freeze block in `observe`.
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
    /// `_warmup_tokens`: retained for API/CLI compat but no longer gates the
    ///   freeze — the scale is now frozen on the FIRST observe (before any KV is
    ///   persisted), so a warmup window would only reintroduce the write/read
    ///   scale mismatch. Any value > 0 simply enables online calibration.
    /// `gpu`: GPU backend for allocating the absmax reduction buffer.
    pub fn new(_warmup_tokens: usize, gpu: &dyn GpuBackend) -> Result<Self> {
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
    /// Before the first observe: the conservative construction default (2.0,
    /// covering ±896) — never used for a persisted write, since observe() runs
    /// before every write and freezes on its first call. After the first
    /// observe: the frozen, data-derived scale (constant thereafter).
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

        if !inner.frozen {
            // HARDENING (2026-07-25): freeze the scale on the FIRST observation,
            // BEFORE any KV is persisted with it. The write path calls observe()
            // immediately before quantizing+writing KV (write_kv_cache.rs:482-485),
            // so the scale frozen HERE is exactly what THIS batch — and every
            // later batch — writes with, and what every read dequantizes with.
            //
            // The previous design wrote the first `warmup_tokens` (256) at a
            // placeholder scale (2.0), then froze to a data-derived value (~0.3)
            // and never re-quantized the already-written entries. A paged /
            // multi-query attention read (chunked prefill, or a prefix-cache
            // resume where seq_len_start>0) covers the whole history in one pass,
            // so once the sequence crossed the freeze boundary it dequantized the
            // pre-freeze KV (and cached/shared prefixes) through the NEW scale —
            // ~6x error → generation garbage (loops/empty). Freezing on the first
            // observe guarantees ONE constant scale for every cache entry's whole
            // lifetime — the exact invariant the EMA-recal guard below documents.
            // `warmup_tokens` no longer gates the freeze (kept for API compat).
            //
            // CALIB_HEADROOM: the first observe sees only the first prefill chunk,
            // so size the scale to cover CALIB_HEADROOM× its observed max, giving
            // headroom for later tokens whose magnitude grows (trades <1 bit of
            // precision for no clipping). Overridable via ATLAS_FP8_KV_HEADROOM.
            let headroom = std::env::var("ATLAS_FP8_KV_HEADROOM")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .filter(|h| *h >= 1.0)
                .unwrap_or(2.0);
            inner.k_scale = (inner.k_running_max * headroom / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.v_scale = (inner.v_running_max * headroom / FP8_E4M3_MAX).max(MIN_SCALE);
            inner.frozen = true;
            tracing::info!(
                "FP8 KV scale frozen on first observe ({} tok, headroom={:.1}): k_scale={:.6} (max={:.2}), v_scale={:.6} (max={:.2}) — constant for all entries",
                inner.tokens_seen,
                headroom,
                inner.k_scale,
                inner.k_running_max,
                inner.v_scale,
                inner.v_running_max,
            );
        } else if inner.frozen
            && inner.tokens_seen % 128 < num_tokens as usize
            && std::env::var("ATLAS_FP8_KV_EMA_RECAL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        {
            // F5 (2026-05-26): Periodic EMA recalibration is now OPT-IN
            // via `ATLAS_FP8_KV_EMA_RECAL=1`, default OFF. Rationale:
            // changing `k_scale` / `v_scale` after `frozen=true` makes
            // previously-written KV cache entries (quantized with the
            // OLD scales) stale relative to the NEW scales — every
            // attention read after a recalibration sees the historical
            // cache through a shifted quantization basis. The comment
            // "responds faster to multi-turn topic switches" was correct
            // about the calibration signal but missed that retroactively
            // rescaling the cache corrupts already-stored multi-turn
            // context. The forensic study of the canonical opencode
            // probe shows reasoning-channel collapse + drift-to-phantom-
            // path patterns whose timing is consistent with deep-layer
            // KV reading through a rescaled basis.
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
