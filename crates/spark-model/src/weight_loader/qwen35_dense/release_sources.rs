// SPDX-License-Identifier: AGPL-3.0-only

//! The release-on-consume sites of the Qwen3.5-dense loader, and the residency
//! line they are judged on.
//!
//! **WHY (`docs/porting/r9700-residency.md`).** On an AMD Radeon AI PRO R9700
//! (gfx1201, 31.9 GB, SCALE 1.7.1), 2026-09-17, serving
//! `unsloth/Qwen3.8-27B-NVFP4`: all 21.81 GiB of the checkpoint uploads, and
//! the serve then dies in `load_layers` at layer 28 of 64 with `cuMemAlloc_v2
//! failed: status 2, requested 167772160 bytes` against a ledger holding
//! 33.73 GB. **9.94 GiB of the store is dead at that moment.** That checkpoint
//! is `format = mixed-precision`: attention q/k/v/o, the GDN projections, the
//! layer-56..63 MLPs and `lm_head` ship as FP8 E4M3 with a per-CHANNEL `[N,1]`
//! scale, which no `w8a16` kernel can index, so every one of them is
//! dequantised to BF16 and requantised to NVFP4 at load. The E4M3 bytes are
//! read exactly once, by the dequant kernel, and then sit there.
//!
//! `prune_after_load` is the existing answer to this shape and it is four
//! minutes and thirty-six layers too late here: the board is full during the
//! layer loop.
//!
//! **The soundness rule, stated once.** A release site claims a tensor only
//! when `{prefix}.weight` is `FP8E4M3`. That is not a heuristic, it is the
//! proof: `dense_auto` (`weight_map/quant_helpers.rs:284-296`) returns the
//! STORE's pointer unchanged for a BF16 tensor and routes FP8 E4M3 to
//! `dequant_fp8_blockscaled_to_bf16`, which allocates. So an FP8 projection
//! cannot reach a layer except through a fresh allocation, and nothing in the
//! layer aliases the checkpoint's bytes. A BF16 one can and does, which is why
//! [`consumed_fp8_source`] refuses it and the 45 MiB of BF16
//! `in_proj_a`/`in_proj_b` stay resident.
//!
//! The exceptions to that rule are the arms that bind FP8 `.weight`
//! ZERO-COPY, and each call site names its own:
//!
//!   * `rowwise_fp8::load_fp8_per_row` returns `weight: w.ptr`
//!     (`rowwise_fp8.rs:179`), so under `ATLAS_FP8_ROWWISE=1` the GDN
//!     `out_proj` is ALIVE;
//!   * `load_fp8_block_scaled_as_fp8weight` is zero-copy the same way, so the
//!     `ATLAS_DENSE_FP8=1` attention and FFN overlays keep their sources.
//!
//! **Ordering.** `gpu.free` on a buffer a queued kernel still reads is
//! undefined, and `dequant_fp8_blockscaled_to_bf16` deliberately skips its
//! per-call synchronize (`quant_helpers.rs:120-124`: syncing there cost ~104 s
//! of cold-load wall on a 30k-call MoE). `quantize_to_nvfp4` and
//! `transpose_for_gemm_gs` do synchronize, but relying on that would make
//! correctness a property of two helpers nobody edits with this in mind. So
//! [`SourceReleaser::release_projections`] synchronizes the load stream itself
//! before it frees anything. One `cuStreamSynchronize` per site per layer, on
//! a path that already costs minutes, against a class of corruption that
//! surfaces as wrong logits rather than a fault.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore, release_sources_enabled};

/// Free a buffer that MAY be the store's own pointer for `name`, through the
/// store when it is and directly when it is not.
///
/// **WHY (`docs/porting/r9700-residency.md`, "Two latent double-frees this
/// accounting surfaced").** `dense_auto` returns the STORE's pointer unchanged
/// for a BF16 tensor and a fresh allocation for an FP8 one, so a loader that
/// frees its input after a copy is freeing store memory on exactly the
/// checkpoints whose tensors are BF16 — and the store still lists the entry, so
/// teardown frees it a second time. The free itself is RIGHT at every call site
/// below (the bytes really are dead, and keeping them is the duplicate the site
/// exists to avoid); what is wrong is the route.
///
/// This changes no residency at all: the same pointer is freed either way. What
/// it changes is what the store believes afterwards — `contains` and `get` stop
/// claiming a tensor whose memory is gone, `free_matching` and teardown skip it,
/// and a reader that runs too late gets the named "RELEASED on consume" error
/// instead of whatever the allocator handed out next.
///
/// NOT gated on `ATLAS_LOAD_RELEASE_SOURCES`. That knob decides whether to free
/// a LIVE store tensor early; this decides how to record a free that already
/// happens on every target. A correctness fix that only applies where a
/// residency knob is on is not a correctness fix.
pub(super) fn free_maybe_store_owned(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    name: &str,
    ptr: DevicePtr,
) -> Result<()> {
    // Pointer identity, not a dtype guess: a TP shard, a dequant and a concat
    // all reach these sites through the same variable, and only the aliased
    // case may go through the store.
    if store.get(name).is_ok_and(|w| w.ptr == ptr) {
        store.release_tensor(gpu, name)?;
        return Ok(());
    }
    gpu.free(ptr)
}

/// True when `{prefix}.weight` is an FP8 E4M3 tensor, i.e. one that reached
/// its layer through a fresh allocation and whose checkpoint bytes are
/// therefore dead once the consuming kernel has run. See the module docs: this
/// is the soundness rule, not a guess.
pub(super) fn consumed_fp8_source(store: &WeightStore, prefix: &str) -> bool {
    store
        .get(&format!("{prefix}.weight"))
        .is_ok_and(|w| w.dtype == WeightDtype::FP8E4M3)
}

// The companion `weight_scale` / `weight_scale_inv` tensors are DELIBERATELY
// NOT released, though the dequant is their only reader. They are ~3.3 MB
// across this whole 27B checkpoint, and half a dozen predicates in the tree
// key off `store.contains("....weight_scale")` to decide what a checkpoint IS
// (`detect_nvfp4_variant`'s `has_mlp_scale` guard, `proj_is_fp8_any_scale`,
// `rowwise_fp8::proj_is_fp8_per_row`). Keeping them means a release cannot
// change any of those answers, which is worth far more than 3 MB.

/// Releases consumed FP8 checkpoint tensors, and tallies what went.
///
/// Held by `load_layers` across the whole layer loop so the summary line can
/// report one number rather than sixty-four.
pub(super) struct SourceReleaser {
    enabled: bool,
    /// Store bytes freed by `WeightStore::release_tensor`.
    store_bytes: u64,
    store_count: usize,
    /// Derived bytes freed at a site that was leaking them (the attention BF16
    /// dequant intermediate). Counted separately because they were never the
    /// store's, so they do not belong in the store's released total.
    leaked_bytes: u64,
}

impl SourceReleaser {
    pub(super) fn new() -> Self {
        let enabled = release_sources_enabled();
        tracing::info!(
            "ATLAS_LOAD_RELEASE_SOURCES={} — consumed FP8 checkpoint tensors are {} \
             (default on this build: {}). See docs/porting/r9700-residency.md.",
            u8::from(enabled),
            if enabled {
                "freed during the layer loop"
            } else {
                "kept resident for the life of the model"
            },
            if cfg!(atlas_scale) { "on" } else { "off" },
        );
        Self {
            enabled,
            store_bytes: 0,
            store_count: 0,
            leaked_bytes: 0,
        }
    }

    /// Record a derived buffer this loader freed that it previously leaked.
    pub(super) fn note_leaked_free(&mut self, bytes: usize) {
        self.leaked_bytes += bytes as u64;
    }

    /// Release `{prefix}.weight` for each of `prefixes`, skipping any whose
    /// `.weight` is not FP8 E4M3. The scale keys stay (see above).
    ///
    /// Synchronizes `stream` FIRST, once, for the whole batch. A no-op (no
    /// sync, no free) when the policy is off or the list is empty, so an
    /// NVIDIA default build does not even pay the stream sync.
    pub(super) fn release_projections(
        &mut self,
        store: &WeightStore,
        gpu: &dyn GpuBackend,
        stream: u64,
        prefixes: &[String],
    ) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let claimed: Vec<&String> = prefixes
            .iter()
            .filter(|p| consumed_fp8_source(store, p))
            .collect();
        if claimed.is_empty() {
            return Ok(());
        }
        // Everything that reads these bytes was enqueued on this stream.
        gpu.synchronize(stream)?;
        for prefix in claimed {
            let bytes = store.release_tensor(gpu, &format!("{prefix}.weight"))?;
            if bytes > 0 {
                self.store_count += 1;
                self.store_bytes += bytes as u64;
            }
        }
        Ok(())
    }

    /// The line the next R9700 serve is read against.
    ///
    /// `layer-owned` is the ledger's live total minus what the store still
    /// holds, which is exactly "everything a loader allocated on top of the
    /// checkpoint" — the quantity `docs/porting/r9700-residency.md` puts at
    /// 25.25 GiB and the one the two-layout verdict turns on. Omitted rather
    /// than guessed when the backend keeps no ledger.
    pub(super) fn summary(&self, store: &WeightStore, gpu: &dyn GpuBackend) -> String {
        let gb = |b: u64| b as f64 / 1e9;
        let resident = store.resident_bytes() as u64;
        let layer_owned = gpu
            .live_bytes()
            .map(|live| (live as u64).saturating_sub(resident));
        let mut s = format!(
            "weights resident: store {:.2} GB ({:.2} GB released on consume across {} tensors)",
            gb(resident),
            gb(self.store_bytes),
            self.store_count,
        );
        match layer_owned {
            Some(owned) => s.push_str(&format!(
                ", layer-owned {:.2} GB, total {:.2} GB",
                gb(owned),
                gb(resident + owned),
            )),
            None => s.push_str(", layer-owned unknown (this backend keeps no allocation ledger)"),
        }
        if self.leaked_bytes > 0 {
            s.push_str(&format!(
                "; {:.2} GB of dequant intermediates freed that the pre-release loader leaked",
                gb(self.leaked_bytes),
            ));
        }
        s
    }
}

#[cfg(test)]
#[path = "release_sources_tests.rs"]
mod tests;
