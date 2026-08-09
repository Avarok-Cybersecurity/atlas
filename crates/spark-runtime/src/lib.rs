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
pub mod op_cache;
pub mod pinned_hosts;
pub mod prefix_cache;
pub mod progress;
pub mod radix_tree;
pub mod run_metrics;
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
/// Publish the command line's `--ssm-tail-midchunk`. Call once, at serve time,
/// before any prefill runs.
///
/// `None` means THE FLAG WAS NOT GIVEN, and is not the same as `Some(default)`.
/// Publishing the clap default sealed this cell on every `spark serve`, which
/// made the documented `ATLAS_SSM_TAIL_MIDCHUNK=0` opt-out a silent no-op — an
/// operator could set it, see the flag echoed in the startup log, and get the
/// opposite behaviour with nothing anywhere saying so. A knob that looks like an
/// opt-out and is not costs more than no knob at all, so an absent flag now
/// publishes nothing and leaves the environment fallback below to decide.
pub fn set_ssm_tail_midchunk(on: Option<bool>) {
    if let Some(on) = on {
        let _ = SSM_TAIL_MIDCHUNK.set(on);
    }
}

static SSM_TAIL_MIDCHUNK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn ssm_tail_midchunk_enabled() -> bool {
    // Default ON (2026-07-19): mid-chunk GDN tail capture eliminates the warm-turn
    // SSM replay (~1.17s component of warm TTFT) by capturing state in-pass at the
    // block-floored matched-prefix boundary.
    //
    // ★ `--ssm-tail-midchunk` WINS when it is given, and only then. It used to
    // win unconditionally — serve.rs published the clap default on every boot,
    // sealing this cell before anything asked, so `ATLAS_SSM_TAIL_MIDCHUNK=0`
    // did NOTHING under `spark serve` while still being documented as the
    // opt-out. `set_ssm_tail_midchunk` now takes an `Option` and an absent flag
    // publishes nothing, so the read below is live again for the CLI, for tests
    // and for examples alike.
    //
    // ★ The 2026-07-19 validation did not cover what it appeared to. It read
    // "BFCL e2e 1007/1007" — a COMPLETION count, not an accuracy score — and
    // warm-TTFT, which is a timing signal. Neither can see a wrong recurrent
    // state, and on NVIDIA the captured h_state was in fact never written at
    // all (see `prepare_midchunk_capture`, which now refuses the plan off
    // `atlas_scale`). "flag-off byte-identical" held; it just was not evidence
    // that flag-ON was correct.
    *SSM_TAIL_MIDCHUNK
        .get_or_init(|| !matches!(std::env::var("ATLAS_SSM_TAIL_MIDCHUNK").as_deref(), Ok("0")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_absent_flag_does_not_seal_the_midchunk_cell() {
        // The defect this shape fixes: `set_ssm_tail_midchunk(bool)` was called
        // with the clap default on every `spark serve`, sealing the cell before
        // anything read it — so `ATLAS_SSM_TAIL_MIDCHUNK=0` was documented,
        // echoed back in the startup log, and inert.
        //
        // ★ The cell is process-global with no reset, so this is the only test
        // in this crate that may touch it: a second one would be
        // order-dependent on this.
        for _ in 0..3 {
            set_ssm_tail_midchunk(None);
        }
        set_ssm_tail_midchunk(Some(false));
        assert!(
            !ssm_tail_midchunk_enabled(),
            "an absent flag must leave the cell open for the next writer"
        );
        set_ssm_tail_midchunk(Some(true));
        assert!(!ssm_tail_midchunk_enabled(), "and a SET one is final");
    }

    #[test]
    fn the_tail_boundary_is_the_last_block_strictly_below_the_prompt() {
        // `None` where no such boundary exists, rather than 0 — a snapshot at
        // token 0 is not a cheap restore, it is a full replay wearing one.
        assert_eq!(ssm_tail_boundary(0, 16), None);
        assert_eq!(ssm_tail_boundary(16, 16), None, "not the prompt's own end");
        assert_eq!(ssm_tail_boundary(17, 16), Some(16));
        assert_eq!(ssm_tail_boundary(32, 16), Some(16), "strictly below");
        assert_eq!(ssm_tail_boundary(33, 16), Some(32));
        assert_eq!(ssm_tail_boundary(100, 0), None, "no division by zero");
    }
}
