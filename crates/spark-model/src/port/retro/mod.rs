// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: the RetroInfer retrieval-attention port (lanes L33 host side,
//! L34a and L34b kernels): the module tree, and the one place its pieces are
//! named.
//!
//! RetroInfer replaces a full-attention layer's decode attention over a long
//! context with three zones per KV head: an exact zone (the first `sink`
//! tokens, the recent tail, and every token of the few clusters whose
//! centroids score highest: the retrieval zone), an estimation zone (the next
//! clusters, each standing in for its members through its centroid and value
//! sum), and the rest, which is dropped. The index is a segmented spherical
//! k-means over the layer's keys. It is APPROXIMATE attention: the row
//! `retro_attn` is off on every target and is never flipped on by default.
//!
//! Host side (L33): [`config`] (flags, sizes, capacities), [`api`] (flag
//! resolution, validation, registration), [`state`] and [`arena`] (the device
//! buffers the kernels read), [`oracle`] and [`oracle_eval`] (the CPU
//! reference every kernel is tested against), [`shadow`] (measure the zones
//! against dense attention while serving dense), [`update`] (the decode-time
//! index update), [`lifecycle`] (which index belongs to which sequence, and the
//! prefix-keyed index cache), [`verify`] (one selection for the K+1 rows of a
//! speculative verify), [`selector`] (the sparse-seam `KvSelector`) with its
//! state [`runtime`], device operations [`device`] and custom attention
//! [`attend`], and [`metrics`] (`atlas_retro_*`).
//!
//! Kernels (L34a, L34b, by contract): `build` and `select`
//! (k-means build, zone selection), `gather` and `attn` (exact-zone gather,
//! estimation-zone attention), with the types of [`state`] and [`config`].

pub mod api;
pub mod arena;
pub mod attend;
pub mod config;
pub mod device;
pub mod lifecycle;
pub mod metrics;
pub mod oracle;
pub mod oracle_eval;
pub mod runtime;
mod runtime_build;
pub mod selector;
pub mod shadow;
pub mod state;
pub mod update;
pub mod verify;

/// L34b: the merged exact plus estimation attention (`attn::estimation_attention`).
pub mod attn;
/// L34a: segmented spherical k-means build (`build::build_layer_index`).
pub mod build;
/// L34b: the exact-zone gather (`gather::gather_exact`).
pub mod gather;
/// L34a: centroid scoring and zone selection (`select::select_zones`).
pub mod select;

/// L34b: the kernel tests against the L33 oracle.
#[cfg(test)]
mod kernels_tests;

#[cfg(test)]
mod config_tests;
#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod selector_tests;
#[cfg(test)]
mod update_tests;
