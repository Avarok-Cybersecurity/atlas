// SPDX-License-Identifier: AGPL-3.0-only

//! Which kernel serves an HDIM>256 prefill, and the BR its grid must be built
//! for. Split out of `prefill_attn_main_a.rs` for the repo's 500-line cap, and
//! it belongs alone regardless: the NAME is chosen in `qwen3_attention::init`
//! while the GRID is built in `prefill_attention`, two different files, and a
//! mismatch between them does not fail — it silently computes the wrong q-tiles.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// The HDIM>256 prefill kernel: its module/entry name and the BR its grid must
/// be built for. ONE reader, because the name is chosen in `qwen3_attention::init`
/// and the grid here, and a mismatch is silent.
///
/// Default is the tensor-core instantiation (`BR=32`). `ATLAS_ATTN_512_TC=0`
/// selects the scalar reference (`BR=16`) — kept reachable because it is the
/// oracle the TC path was validated against (cosine 0.999998, 64.7x faster on
/// S=1024/4q/2kv/causal).
pub fn wide_prefill_kernel(gpu: &dyn GpuBackend) -> (KernelHandle, u32) {
    // ★ RESOLVE WITH A FALLBACK, NOT A FIXED NAME. Only targets that ship the
    // HDIM=512 instantiation have `inferspark_prefill_512tc`; gemma-4-31b, for
    // one, ships only the scalar `inferspark_prefill_512`. Returning the TC name
    // unconditionally leaves those targets with KernelHandle(0), which makes the
    // caller's `hd > 256 && handle != 0` guard go FALSE and quietly routes
    // 512-wide heads into the 64-wide kernel — no error, wrong results. That is
    // the PR #296 failure class (a kernel handle absent, a silent fallback, and
    // both gates green), and this path very nearly reproduced it.
    if std::env::var("ATLAS_ATTN_512_TC").ok().as_deref() != Some("0") {
        let tc =
            crate::layers::try_kernel(gpu, "inferspark_prefill_512tc", "inferspark_prefill_512tc");
        if tc.0 != 0 {
            return (tc, 32);
        }
        tracing::debug!(
            "inferspark_prefill_512tc absent for this target; using the scalar \
             HDIM=512 reference"
        );
    }
    (
        crate::layers::try_kernel(gpu, "inferspark_prefill_512", "inferspark_prefill_512"),
        16,
    )
}
