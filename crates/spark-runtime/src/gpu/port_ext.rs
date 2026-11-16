// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: src/core/decode_graph.cpp:62-80 (`DecodeGraphDefinition::topology_signature`). Changes: translated from C++ to Rust over a backend-neutral `(node type, function)` list instead of a live `cudaGraph_t`, and the node count is folded into the one returned hash rather than returned beside it, by Atlas contributors.

//! Responsibility: the types of the `GpuBackend` extensions
//! and the backend-neutral bodies the trait's defaulted methods delegate to.
//!
//! Four groups, each defaulted so the mock and the Metal backend compile unchanged:
//!
//! * **Graph templates.** `end_capture` instantiates the captured graph and destroys it. The
//!   template methods keep the `cudaGraph_t` alive as a [`GraphTemplate`] so a caller can take
//!   its [`topology_signature`], update an existing exec in place from it
//!   (`graph_exec_update`, the cheap path when only kernel arguments moved) and pre-upload an
//!   exec (`graph_upload`). Backends without graphs refuse the template calls and answer
//!   `Ok(false)` to an update, which callers treat as "instantiate as today".
//! * **Batched async copies.** One call enqueues many host/device copies on a stream
//!   (`cuMemcpyBatchAsync` on CUDA). The default loops the existing single-copy async
//!   primitives, so the bytes moved are identical; only the submission count differs.
//! * **Events.** `event_query` polls without blocking. The default waits for the event and
//!   answers `true`, which is correct on every backend (a backend whose events complete
//!   synchronously never waits).
//! * **Host registration.** `host_register` page-locks caller memory and records it in
//!   [`crate::pinned_hosts`] as TIER-OWNED: tier DMA then never takes the implicit sync that
//!   `copy_h2d_async` adds for transient pinned sources. The default only records, since there
//!   is nothing to lock.
//!
//! Every method is inert until a caller uses it: nothing on any existing path calls these, so
//! a target whose port rows are off behaves exactly as before.

use anyhow::{Result, bail};

use super::{DevicePtr, GpuBackend};

/// A captured CUDA graph (`CUgraph`) that has not been destroyed yet.
///
/// Owned by the caller: release it with `GpuBackend::graph_template_destroy`. It is not
/// `Clone`, so one template cannot be destroyed twice by accident. Unlike [`super::GraphHandle`]
/// (an instantiated exec) it cannot be launched.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GraphTemplate(pub u64);

/// One host/device copy of a batch.
///
/// `host` is a host ADDRESS (not a reference): a batch spans many regions of one page-locked
/// pool, and the caller already guarantees each region's lifetime the way the single-copy async
/// primitives require (valid, and neither read nor rewritten, until the next synchronization
/// point on the stream). Entries with `bytes == 0` are skipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CopyEntry {
    pub dev: DevicePtr,
    pub host: usize,
    pub bytes: usize,
}

/// `CU_GRAPH_NODE_TYPE_KERNEL`: the only node type whose function enters the signature.
pub const NODE_TYPE_KERNEL: u32 = 0;
/// `CU_GRAPH_NODE_TYPE_MEMCPY`.
pub const NODE_TYPE_MEMCPY: u32 = 1;
/// `CU_GRAPH_NODE_TYPE_MEMSET`.
pub const NODE_TYPE_MEMSET: u32 = 2;

const FNV_OFFSET: u64 = 1_469_598_103_934_665_603;
const FNV_PRIME: u64 = 1_099_511_628_211;
const KERNEL_MIX: u64 = 1_000_003;

/// The topology class of a graph: FNV-1a over each node's `(type, kernel function)` in node
/// order, then the node count.
///
/// Two captures with the same signature have the same node types in the same order and launch
/// the same kernels, which is what `cuGraphExecUpdate` needs to update one exec from the other
/// graph (it may still refuse, for example when a memset changes dimensionality; callers treat
/// that as a miss). Launch geometry and kernel arguments are deliberately NOT part of the
/// class: those are exactly what an update may change. The function of a non-kernel node is
/// ignored.
pub fn topology_signature<I>(nodes: I) -> u64
where
    I: IntoIterator<Item = (u32, u64)>,
{
    let mut hash = FNV_OFFSET;
    let mut count: u64 = 0;
    for (ty, func) in nodes {
        let mut payload = u64::from(ty) + 1;
        if ty == NODE_TYPE_KERNEL {
            payload = payload.wrapping_mul(KERNEL_MIX) ^ func;
        }
        hash = (hash ^ payload).wrapping_mul(FNV_PRIME);
        count += 1;
    }
    (hash ^ count).wrapping_mul(FNV_PRIME)
}

/// Reject entries no backend could copy: a null pointer on either side of a non-empty copy, or
/// a range that wraps the address space. Zero-byte entries always pass (they are skipped).
pub fn validate_entries(entries: &[CopyEntry]) -> Result<()> {
    for (i, e) in entries.iter().enumerate() {
        if e.bytes == 0 {
            continue;
        }
        if e.dev.is_null() {
            bail!(
                "copy batch entry {i}: null device pointer for {} bytes",
                e.bytes
            );
        }
        if e.host == 0 {
            bail!(
                "copy batch entry {i}: null host address for {} bytes",
                e.bytes
            );
        }
        let dev_wraps = e.dev.0.checked_add(e.bytes as u64).is_none();
        if dev_wraps || e.host.checked_add(e.bytes).is_none() {
            bail!(
                "copy batch entry {i}: {} bytes at dev {} / host {:#x} wraps the address space",
                e.bytes,
                e.dev,
                e.host
            );
        }
    }
    Ok(())
}

/// Error for a template method on a backend without CUDA graphs.
pub(crate) fn unsupported<T>(what: &str) -> Result<T> {
    bail!("{what}: not supported by this backend")
}

/// Default `copy_batch_d2h_async`: one `copy_d2h_async` per non-empty entry, in entry order.
pub(crate) fn loop_d2h<G>(gpu: &G, entries: &[CopyEntry], stream: u64) -> Result<()>
where
    G: GpuBackend + ?Sized,
{
    validate_entries(entries)?;
    for e in entries.iter().filter(|e| e.bytes > 0) {
        // SAFETY: the caller contract of `GpuBackend::copy_batch_d2h_async`: `host` is valid for
        // `bytes` writable bytes that nothing else reads or writes until the stream syncs, which
        // is the contract `copy_d2h_async` states for its destination slice.
        let dst = unsafe { std::slice::from_raw_parts_mut(e.host as *mut u8, e.bytes) };
        gpu.copy_d2h_async(e.dev, dst, stream)?;
    }
    Ok(())
}

/// Default `copy_batch_h2d_async`: one `copy_h2d_async_retained` per non-empty entry. The
/// retained form is the right one: a batch source is caller-kept memory by contract, so the
/// implicit sync of the transient form would only add stalls.
pub(crate) fn loop_h2d<G>(gpu: &G, entries: &[CopyEntry], stream: u64) -> Result<()>
where
    G: GpuBackend + ?Sized,
{
    validate_entries(entries)?;
    for e in entries.iter().filter(|e| e.bytes > 0) {
        // SAFETY: the caller contract of `GpuBackend::copy_batch_h2d_async`: `host` is valid for
        // `bytes` readable bytes that stay unchanged until the next sync point on `stream`.
        let src = unsafe { std::slice::from_raw_parts(e.host as *const u8, e.bytes) };
        gpu.copy_h2d_async_retained(src, e.dev, stream)?;
    }
    Ok(())
}

/// Default `host_register`: record `[ptr, ptr + bytes)` as tier-owned. Refuses an empty or
/// wrapping range and one that overlaps a live registration, as `cuMemHostRegister` would.
pub(crate) fn record_host_register(ptr: usize, bytes: usize) -> Result<()> {
    check_register_range(ptr, bytes)?;
    crate::pinned_hosts::register_tier(ptr as *const u8, bytes);
    Ok(())
}

/// Default `host_unregister`: forget the region based at `ptr`. Refuses an address that is not
/// the base of a live registration, as `cuMemHostUnregister` would.
pub(crate) fn record_host_unregister(ptr: usize) -> Result<()> {
    if crate::pinned_hosts::region_len(ptr as *const u8).is_none() {
        bail!("host_unregister: {ptr:#x} is not the base of a registered host region");
    }
    crate::pinned_hosts::unregister(ptr as *const u8);
    Ok(())
}

/// The range rules every `host_register` shares (the CUDA backend checks them before the
/// driver call too, so a refused range never reaches `cuMemHostRegister`).
pub(crate) fn check_register_range(ptr: usize, bytes: usize) -> Result<()> {
    if ptr == 0 || bytes == 0 {
        bail!("host_register: empty region ({ptr:#x}, {bytes} bytes)");
    }
    if ptr.checked_add(bytes).is_none() {
        bail!("host_register: {bytes} bytes at {ptr:#x} wraps the address space");
    }
    if crate::pinned_hosts::overlaps(ptr as *const u8, bytes) {
        bail!("host_register: {bytes} bytes at {ptr:#x} overlap a live page-locked region");
    }
    Ok(())
}
