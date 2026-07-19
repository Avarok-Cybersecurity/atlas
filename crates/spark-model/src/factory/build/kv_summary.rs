// SPDX-License-Identifier: AGPL-3.0-only

//! `--high-speed-swap` KV-dtype summary logging, split out of `build.rs`
//! for the ≤500 LoC file-size cap.

use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

// Phase 6.2.c — KV-dtype gating for `--high-speed-swap`.
//
// All quantization variants are now supported via host-side dequant before
// disk-write (the orchestrator's tiled-attention kernel reads BF16):
//   - BF16    : direct stream; predictor anchor (K_lr) computed natively.
//   - FP8     : E4M3 → BF16 (per-tensor calibration scale). Predictor
//               degrades to LRU (BF16-only kernel can't read FP8 layout).
//   - NVFP4   : E2M1 nibble + per-group FP8 scale → BF16. Predictor LRU.
//   - Turbo4  : Lloyd-Max 16-level + per-group FP8 scale + WHT(K/V) on
//               disk. Decode flow's WHT(Q)/iWHT(out) bookends handle the
//               Walsh-Hadamard round-trip transparently. Predictor LRU.
//   - Turbo3  : 3-bit packed (8 vals per 3 bytes), 8-level codebook,
//               per-group FP8 scales, WHT bookended. Predictor LRU.
//   - Turbo8  : FP8 E4M3 + per-group FP8 scales + WHT bookended.
//               Predictor LRU.
fn dtype_label(dt: KvCacheDtype) -> &'static str {
    match dt {
        KvCacheDtype::Bf16
        | KvCacheDtype::Bf16KTurbo4V
        | KvCacheDtype::Bf16KTurbo3V
        | KvCacheDtype::Bf16KTurbo2V => "BF16",
        KvCacheDtype::Fp8
        | KvCacheDtype::Fp8KTurbo4V
        | KvCacheDtype::Fp8KTurbo3V
        | KvCacheDtype::Fp8KTurbo2V => "FP8",
        KvCacheDtype::Nvfp4 => "NVFP4",
        KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V | KvCacheDtype::Turbo2 => "Turbo3",
        KvCacheDtype::Turbo4 | KvCacheDtype::Turbo4KTurbo3V | KvCacheDtype::Turbo4KTurbo8V => {
            "Turbo4"
        }
        KvCacheDtype::Turbo8 => "Turbo8",
    }
}

pub(super) fn log_hss_kv_summary(kv_config: &KvCacheConfig) {
    let mut counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    if kv_config.layer_dtypes.is_empty() {
        *counts.entry(dtype_label(kv_config.dtype)).or_default() += kv_config.num_layers;
    } else {
        for dt in &kv_config.layer_dtypes {
            *counts.entry(dtype_label(*dt)).or_default() += 1;
        }
    }
    let total: usize = counts.values().sum();
    let summary: Vec<String> = counts
        .iter()
        .map(|(name, n)| format!("{n} {name}"))
        .collect();
    tracing::info!(
        "--high-speed-swap KV: {} attn layers ({}); HBM-shrink applies to all \
         (Phase 6.2.c proper — host dequant for FP8/NVFP4/Turbo3/Turbo4/Turbo8; \
         predictor scoring uses LRU for non-BF16 layers)",
        total,
        summary.join(" + ")
    );
}
