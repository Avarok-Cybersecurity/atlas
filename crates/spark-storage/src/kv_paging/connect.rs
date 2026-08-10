// SPDX-License-Identifier: AGPL-3.0-only

//! The KV-paging selection seam: env-driven choice between the raw
//! one-sided `RdmaKvBackend` (flag OFF) and the peer-paging
//! `KvPagingBackend` (`ATLAS_KV_PAGING=1`). Split from `backend.rs` to keep
//! it under the 500-LoC cap.

use std::num::NonZeroU64;

use anyhow::{Result, anyhow, bail};

use super::backend::{KvPagingBackend, KvPagingConnect};
use super::ns;
use crate::backend::StorageBackend;
use crate::group::GroupLayout;
use crate::model_dims::ModelDims;

/// THE selection seam `HighSpeedSwap::new_on_stream` calls when
/// `$ATLAS_KV_PEER` is set. Flag OFF (`ATLAS_KV_PAGING` unset/0) ⇒ the raw
/// one-sided `RdmaKvBackend::connect(peer, layout)` — the identical call and
/// data plane (its handshake is the v2 header with blob == 0),
/// so a regression bisects on this one flag. Flag ON ⇒ resolve the required
/// env (PCND: fail fast), derive the namespace, and connect the paging
/// backend.
pub fn connect_kv_peer_backend(
    peer: &str,
    layout: GroupLayout,
    model: &ModelDims,
    elem_bytes: u32,
    coalesce_blocks: bool,
) -> Result<Box<dyn StorageBackend>> {
    let paging = ns::kv_paging_selected(std::env::var("ATLAS_KV_PAGING").ok().as_deref())?;
    if !paging {
        // DEFAULT: the dumb one-sided path, untouched.
        return Ok(Box::new(crate::rdma_kv_backend::RdmaKvBackend::connect(
            peer, layout,
        )?));
    }
    if !coalesce_blocks {
        bail!(
            "ATLAS_KV_PAGING=1 requires block coalescing (ATLAS_HSS_COALESCE_BLOCKS, the \
             default): the paging record is one whole KV block, and the per-head offload \
             write path cannot be served by a peer-owned block arena"
        );
    }
    let arena_bytes = ns::resolve_arena_bytes_from(
        std::env::var("ATLAS_KV_PAGING_ARENA_GB").ok().as_deref(),
        layout.block_bytes(),
    )?;
    let ns = match std::env::var("ATLAS_KV_PAGING_NS").ok() {
        Some(raw) => {
            let ns = ns::resolve_kv_ns_from(Some(&raw), NonZeroU64::new(1).expect("nonzero"))?;
            tracing::info!(
                "kv-paging: namespace OVERRIDDEN via ATLAS_KV_PAGING_NS={:#018x} — two clients \
                 sharing one explicit ns on one peer WILL cross-serve KV blocks",
                ns.get()
            );
            ns
        }
        None => {
            // The per-model fingerprint is REQUIRED for the derived namespace
            // (PCND: no model-blind default — that is exactly the silent
            // cross-serve the SSM fix made unrepresentable).
            let fp = model.model_fp.ok_or_else(|| {
                anyhow!(
                    "ATLAS_KV_PAGING=1 requires a model fingerprint (ModelDims::model_fp) and \
                     the loader did not derive one — fix the model config, or set \
                     ATLAS_KV_PAGING_NS to an explicit non-zero u64"
                )
            })?;
            let salt =
                ns::resolve_salt_from(std::env::var("ATLAS_KV_PAGING_SALT").ok().as_deref())?
                    .unwrap_or_else(rand::random::<u64>);
            let derived = ns::derive_kv_ns(
                fp.get(),
                &layout,
                elem_bytes,
                model.block_size as u32,
                model.head_dim as u32,
                salt,
            );
            tracing::info!(
                "kv-paging: derived namespace {:#018x} (fp {:#018x}, client salt {salt:#018x} — \
                 pin via ATLAS_KV_PAGING_SALT to reproduce; the salt makes keys CLIENT-PRIVATE: \
                 capacity pooling yes, cross-client warm hits no)",
                derived.get(),
                fp.get(),
            );
            derived
        }
    };
    tracing::info!(
        "kv-paging: connecting {peer} (arena {:.3} GiB, blob {} B); peer must run with \
         --swap-cap-gb-kv 0 (a KV disk cap turns evictions into unrecoverable KV loss)",
        arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        layout.block_bytes(),
    );
    Ok(Box::new(KvPagingBackend::connect(
        peer,
        layout,
        KvPagingConnect { arena_bytes, ns },
    )?))
}
