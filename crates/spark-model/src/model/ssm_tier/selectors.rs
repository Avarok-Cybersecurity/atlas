// SPDX-License-Identifier: AGPL-3.0-only

//! Env-driven selectors: which store backs the Marconi spill tier and the
//! decode rolling tier.

use anyhow::{Result, bail};

use super::fingerprint::{ModelFingerprint, resolve_decode_ns, resolve_swap_ns};
use super::unified::{TransportSlotArena, build_unified_swap, unified_hot_slots};
use super::{
    ArenaSnapshotStore, FileSnapshotArena, MemBlobStore, PagingSnapshotStore, RdmaSnapshotStore,
    SnapshotBlobStore, UnifiedSnapshotStore, ssm_tier_unified,
};

/// Whether the SSM spill tier is engaged (`ATLAS_SSM_TIER`). Default off ⇒
/// eviction drops exactly as before ⇒ byte-identical to a pre-tier build.
pub(crate) fn ssm_tier_enabled() -> bool {
    std::env::var_os("ATLAS_SSM_TIER").is_some()
}

/// Build the SSM spill-tier store (called only when `ssm_tier_enabled()`).
/// `ATLAS_SSM_RDMA_TIER=host:port` selects the RDMA arena
/// ([`RdmaSnapshotStore`] over a peer blade, `ATLAS_SSM_RDMA_ARENA_SLOTS` slots,
/// default 512); otherwise the host-RAM [`MemBlobStore`]. A connect failure (or
/// a build without RDMA verbs) LOGS and falls back to host-RAM — the tier is
/// optional, never a hard model-init error. With `ATLAS_SSM_RDMA_TIER` unset the
/// result is exactly `MemBlobStore::new(0)` as before ⇒ byte-identical.
///
/// `Err` is reserved for CONFIG errors (a bad `ATLAS_SSM_SWAP_NS` override —
/// PCND fail-fast, resolved BEFORE any connect attempt); connectivity failures
/// keep the non-fatal fallback chain paging → bounded RDMA → host-RAM.
pub(crate) fn build_tier_store(
    fp: ModelFingerprint,
    blob_bytes: usize,
) -> Result<std::sync::Arc<dyn SnapshotBlobStore>> {
    use std::sync::Arc;
    if let Some(peer) = std::env::var("ATLAS_SSM_RDMA_TIER")
        .ok()
        .filter(|s| !s.is_empty())
    {
        let slots: usize = std::env::var("ATLAS_SSM_RDMA_ARENA_SLOTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let arena_bytes = slots as u64 * blob_bytes as u64;
        // WS-A: ATLAS_SSM_SWAP=1 selects PAGING mode — the peer (started with
        // --swap-dir) owns residency and backs the RAM arena with an NVMe swap
        // file, giving infinite depth (never drops) shared across clients. Falls
        // through to the bounded RDMA store / host-RAM on any connect failure.
        if std::env::var("ATLAS_SSM_SWAP").ok().as_deref() == Some("1") {
            // Namespace = ATLAS_SSM_SWAP_NS (explicit u64 override, strict) or
            // the config-derived model fingerprint, so different models sharing
            // one peer can never collide. Resolved BEFORE the connect attempt:
            // a bad override is a config error, not a connectivity error.
            let namespace = resolve_swap_ns(fp)?;
            match spark_storage::RdmaSnapshotArena::connect_paging(&peer, arena_bytes, blob_bytes) {
                Ok(arena) => {
                    tracing::info!(
                        "SSM spill tier = RDMA PAGING peer {peer} ({slots}-slot shared RAM cache × \
                         {blob_bytes} B + NVMe swap = infinite depth; ns={namespace:#x}, model \
                         fingerprint {:#018x})",
                        fp.get(),
                    );
                    return Ok(Arc::new(PagingSnapshotStore::new(
                        Box::new(arena),
                        blob_bytes,
                        namespace,
                    )));
                }
                Err(e) => tracing::warn!(
                    "SSM RDMA paging connect to {peer} failed ({e:#}); trying bounded RDMA"
                ),
            }
        }
        // `connect` errors (and we fall back) both on a real connect failure and
        // in a build without the RDMA verbs (the stub arena always errors).
        match spark_storage::RdmaSnapshotArena::connect(&peer, arena_bytes, blob_bytes) {
            Ok(arena) => {
                if ssm_tier_unified() {
                    // §4 fix (the LIVE bug arm): replace the drop-on-full
                    // fixed-slot allocator with the peer's LRU/never-reject
                    // Residency, client-side, over the same remote arena — an
                    // arena-full PUT now LRU-spills the coldest blob to the
                    // local swap tier instead of silently discarding the spill.
                    let hot = Box::new(TransportSlotArena {
                        transport: Box::new(arena),
                        slot_bytes: blob_bytes,
                        num_slots: slots,
                    });
                    let swap = build_unified_swap(blob_bytes, "marconi-rdma");
                    match UnifiedSnapshotStore::new(hot, swap, blob_bytes) {
                        Ok(s) => {
                            tracing::info!(
                                "SSM spill tier = UNIFIED residency over RDMA peer {peer} \
                                 ({slots} hot slots × {blob_bytes} B, LRU spill, never rejects)"
                            );
                            return Ok(Arc::new(s));
                        }
                        Err(e) => {
                            tracing::warn!(
                                "SSM unified residency init failed ({e:#}); \
                                 falling back to host-RAM"
                            );
                            return Ok(Arc::new(MemBlobStore::new(0)));
                        }
                    }
                }
                tracing::info!(
                    "SSM spill tier = RDMA peer {peer} ({slots} slots × {blob_bytes} B = \
                     {:.2} GiB arena)",
                    arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                return Ok(Arc::new(RdmaSnapshotStore::new(
                    Box::new(arena),
                    blob_bytes,
                    slots,
                )));
            }
            Err(e) => tracing::warn!(
                "SSM RDMA tier connect to {peer} failed ({e:#}); falling back to host-RAM"
            ),
        }
    }
    if ssm_tier_unified() {
        // §4 fix (host-RAM arm): one policy core instead of the FIFO
        // MemBlobStore — a bounded LRU hot arena that spills (never rejects)
        // into the swap tier. NOTE: unlike today's lazily-growing unbounded
        // store, the hot arena is allocated up front (slots × blob_bytes).
        let hot_slots = unified_hot_slots();
        let hot = Box::new(atlas_tier::VecSlotArena::new(blob_bytes, hot_slots));
        let swap = build_unified_swap(blob_bytes, "marconi-host");
        match UnifiedSnapshotStore::new(hot, swap, blob_bytes) {
            Ok(s) => {
                tracing::info!(
                    "SSM spill tier = UNIFIED residency in host RAM ({hot_slots} hot slots × \
                     {blob_bytes} B, LRU spill, never rejects)"
                );
                return Ok(Arc::new(s));
            }
            Err(e) => tracing::warn!(
                "SSM unified residency init failed ({e:#}); falling back to host-RAM store"
            ),
        }
    }
    Ok(Arc::new(MemBlobStore::new(0)))
}

/// Build the **decode rolling-tier** cold store (a SEPARATE instance from the
/// Marconi `build_tier_store`, its own `ATLAS_SSM_DECODE_*` env namespace so keys
/// and budgets never collide). Non-dropping is a HARD requirement: a dropped
/// decode blob is a lost rollback target = corrupt restore (unlike Marconi's
/// miss→recompute). `min_slots` = `(ring_slots − hot_lanes) × max_batch_size` is
/// the worst-case cold residency; the local NVMe arena is sized ≥ that and its
/// undersizing is a preflight ERROR, never a warn.
///
/// Selection (`ATLAS_SSM_DECODE_TIER`):
///   - `nvme` + `ATLAS_SSM_DECODE_NVME_DIR=<dir>` → [`FileSnapshotArena`] behind
///     [`ArenaSnapshotStore`], provably sized ≥ `min_slots`.
///   - `peer`  + `ATLAS_SSM_DECODE_RDMA_TIER=host:port` → the never-dropping
///     [`PagingSnapshotStore`] (peer LRU-spills to its own NVMe), own
///     `ATLAS_SSM_DECODE_NS` namespace fold.
///   - unset / anything else → unbounded host-RAM [`MemBlobStore::new(0)`].
pub(crate) fn build_decode_tier_store(
    fp: ModelFingerprint,
    blob_bytes: usize,
    min_slots: usize,
) -> Result<std::sync::Arc<dyn SnapshotBlobStore>> {
    use std::sync::Arc;
    match std::env::var("ATLAS_SSM_DECODE_TIER").ok().as_deref() {
        Some("nvme") => {
            let dir = std::env::var("ATLAS_SSM_DECODE_NVME_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ATLAS_SSM_DECODE_TIER=nvme requires ATLAS_SSM_DECODE_NVME_DIR=<dir>"
                    )
                })?;
            if ssm_tier_unified() && blob_bytes > 0 && blob_bytes.is_multiple_of(4096) {
                // §4 unification: a RAM hot cache over the lifted O_DIRECT swap
                // file. The uncapped disk tier (max_disk_slots = 0) makes
                // NON-DROPPING hold BY CONSTRUCTION instead of by arena sizing
                // — a decode rollback target can never be refused or dropped.
                std::fs::create_dir_all(&dir)?;
                let path = std::path::Path::new(&dir)
                    .join(format!("atlas-decode-ring.{}.swap", std::process::id()));
                let swap = atlas_tier::DirectSwapFile::create(&path, blob_bytes)?;
                let hot_slots = unified_hot_slots().min(min_slots + 1);
                let hot = Box::new(atlas_tier::VecSlotArena::new(blob_bytes, hot_slots));
                let store = UnifiedSnapshotStore::new(hot, Box::new(swap), blob_bytes)?;
                tracing::info!(
                    "SSM decode cold tier = UNIFIED residency ({hot_slots} hot RAM slots + \
                     O_DIRECT swap in {dir}; non-dropping by construction ≥ min_slots \
                     {min_slots})"
                );
                return Ok(Arc::new(store));
            }
            if ssm_tier_unified() {
                tracing::info!(
                    "SSM decode cold tier: ATLAS_SSM_TIER_UNIFIED set but blob_bytes \
                     {blob_bytes} is not a 4 KiB multiple (O_DIRECT stride); keeping the \
                     sized arena store"
                );
            }
            // Provision to the worst-case cold residency + headroom slot so the
            // non-dropping invariant holds by construction (an undersized arena
            // would return Ok(false) on a live target = corruption).
            let slots = min_slots + 1;
            let capacity = slots as u64 * blob_bytes as u64;
            let arena = FileSnapshotArena::create(&dir, capacity)?;
            tracing::info!(
                "SSM decode cold tier = LOCAL NVMe {dir} ({slots} slots × {blob_bytes} B = \
                 {:.2} GiB, non-dropping ≥ min_slots {min_slots})",
                capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            Ok(Arc::new(ArenaSnapshotStore::new(
                Box::new(arena),
                blob_bytes,
                slots,
            )))
        }
        Some("peer") => {
            let peer = std::env::var("ATLAS_SSM_DECODE_RDMA_TIER")
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ATLAS_SSM_DECODE_TIER=peer requires ATLAS_SSM_DECODE_RDMA_TIER=host:port"
                    )
                })?;
            // Namespace = ATLAS_SSM_DECODE_NS (explicit u64 override, strict)
            // or mix64(mix64(fingerprint, DECODE_DOMAIN), client_salt): the
            // DOMAIN separator keeps decode spills off the same model's Marconi
            // keys (both tiers share ONE peer residency whenever blob_bytes
            // match), the fingerprint keeps them off OTHER models' decode
            // spills, and the per-process client salt keeps them off other
            // same-model CLIENTS' spills (cold keys are slot coordinates —
            // identical across processes without it). NOTE the manager-side
            // cold_key also folds DECODE_DOMAIN over (seq, logical) — that fold
            // stays: removing it while this ns is overridden would re-collide
            // decode with Marconi. Resolved BEFORE the (fatal) connect.
            let namespace = resolve_decode_ns(fp)?;
            // Arena RAM cache size is a hint; the peer pages to its own NVMe so
            // the store never drops regardless of this slot count.
            let slots = (min_slots + 1).max(512);
            let arena_bytes = slots as u64 * blob_bytes as u64;
            let arena =
                spark_storage::RdmaSnapshotArena::connect_paging(&peer, arena_bytes, blob_bytes)?;
            tracing::info!(
                "SSM decode cold tier = RDMA PAGING peer {peer} (non-dropping, ns={namespace:#x}, \
                 model fingerprint {:#018x})",
                fp.get(),
            );
            Ok(Arc::new(PagingSnapshotStore::new(
                Box::new(arena),
                blob_bytes,
                namespace,
            )))
        }
        // Unset = the documented default: unbounded, non-dropping host RAM.
        None => {
            tracing::info!("SSM decode cold tier = host-RAM (unbounded, non-dropping)");
            Ok(Arc::new(MemBlobStore::new(0)))
        }
        // PCND: a typo ("nmve", "peer ", "") must never silently defeat the
        // tiering intent by falling through to unbounded host RAM, where decode
        // spills accumulate until OOM on a long session. Fail fast, name the
        // variable, the bad value, and the accepted values — mirroring the strict
        // `parse_ns` this chunk introduced one match arm away.
        Some(other) => bail!(
            "ATLAS_SSM_DECODE_TIER={other:?} is not recognized (accepted: \"nvme\", \"peer\", or \
             unset for unbounded host-RAM). Refusing to silently fall back to host-RAM."
        ),
    }
}

#[cfg(test)]
#[path = "selectors_tests.rs"]
mod tests;
