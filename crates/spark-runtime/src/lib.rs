// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

pub mod buffers;
#[cfg(feature = "cuda")]
pub mod cublaslt;
// Metal/no-cuda builds get unreachable stubs so spark-model's unconditional
// references to these cuda-only entry points still resolve (compile-only).
#[cfg(not(feature = "cuda"))]
#[path = "cublaslt_metal_stub.rs"]
pub mod cublaslt;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "cuda")]
pub mod cutlass;
#[cfg(not(feature = "cuda"))]
#[path = "cutlass_metal_stub.rs"]
pub mod cutlass;
#[cfg(unix)]
pub mod fast_weights;
#[cfg(feature = "cuda")]
pub mod flashinfer;
#[cfg(not(feature = "cuda"))]
#[path = "flashinfer_metal_stub.rs"]
pub mod flashinfer;
pub mod gpu;
pub mod kernel_args;
pub mod kernel_audit;
pub mod kv_cache;
pub mod kv_dequant;
pub mod kv_spill;
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod prefix_cache;
pub mod radix_tree;
pub mod sampler;
pub mod weights;

/// Last paged-KV block boundary strictly below `total_tokens`.
///
/// A warm multi-turn hit can never match past this point: the chat template's
/// generation-prompt suffix (assistant header, and the empty `<think></think>`
/// block emitted when thinking is disabled) is not reproduced when the next
/// turn re-renders the *completed* assistant message, so the longest common
/// prefix diverges inside the prompt's final block. `RadixTree::walk` then
/// floors `matched_tokens` to this boundary. Placing an SSM snapshot here makes
/// the next turn's restore exact (zero recurrence replay); without it the
/// lookup falls back to the coarse `--ssm-checkpoint-interval` grid.
///
/// Returns `None` when the prompt is too short to have such a boundary.
pub fn ssm_tail_boundary(total_tokens: usize, block_size: usize) -> Option<usize> {
    if block_size == 0 || total_tokens <= block_size {
        return None;
    }
    let boundary = ((total_tokens - 1) / block_size) * block_size;
    (boundary > 0).then_some(boundary)
}

/// OPT-IN switch for the tail checkpoint (`ATLAS_SSM_TAIL_CKPT=1`).
///
/// Default OFF. The 3-traj A/B (2026-07-10, 174 samples/arm) showed it is
/// perf-NEUTRAL: it removes the SSM replay on ~89% of warm turns (mean 254 -> 25
/// tokens), but the prefill-chunk split needed to land a snapshot on
/// `ssm_tail_boundary` costs a median 868 ms extra forward pass for a median of 8
/// trailing tokens, which cancels the ~1374 ms of replay it saves. It becomes a
/// clear win only once the SSM state can be captured MID-CHUNK (in the GDN prefill
/// kernel) instead of via an extra pass. Until then it stays off by default and
/// ungated for accuracy.
pub fn ssm_tail_ckpt_enabled() -> bool {
    matches!(std::env::var("ATLAS_SSM_TAIL_CKPT").as_deref(), Ok("1"))
}

/// Default-ON switch for MID-CHUNK tail SSM capture (opt-out `ATLAS_SSM_TAIL_MIDCHUNK=0`).
///
/// Default ON => mid-chunk capture fires on prefill passes spanning the
/// block-floored matched-prefix boundary. When disabled, the prefill
/// chunk is NOT clamped to `ssm_tail_boundary`; instead each GDN layer's
/// recurrent (h_state) and conv (conv_state) kernels are split at the block-
/// floored matched-prefix boundary and the @tb state is copied into a reserved
/// Marconi snapshot slot in-pass, removing the ~868 ms extra forward pass the
/// clamp-based `ATLAS_SSM_TAIL_CKPT` path costs.
pub fn ssm_tail_midchunk_enabled() -> bool {
    // Default ON (2026-07-19): mid-chunk GDN tail capture eliminates the warm-turn
    // SSM replay (~1.17s component of warm TTFT) by capturing state in-pass at the
    // block-floored matched-prefix boundary. Validated: flag-off byte-identical to
    // the prior baseline; warm-TTFT -9.3% median / -54% max (tail-spike elimination);
    // 20/20 contamination-clean (session-gate prevents cross-request SSM corruption);
    // BFCL e2e 1007/1007. Opt OUT with ATLAS_SSM_TAIL_MIDCHUNK=0.
    !matches!(std::env::var("ATLAS_SSM_TAIL_MIDCHUNK").as_deref(), Ok("0"))
}
