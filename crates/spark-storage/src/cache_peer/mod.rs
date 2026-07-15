// SPDX-License-Identifier: AGPL-3.0-only
//
// KV overflow blade — a dumb remote-RAM tier for the high-speed-swap KV cache.
//
// Where `expert_peer` serves READ-ONLY expert weights over one-sided RDMA READ,
// this serves a READ-WRITE slab of RAM: a streaming client OFFLOADS cold K/V
// groups into it with `IBV_WR_RDMA_WRITE` and RESTORES them with
// `IBV_WR_RDMA_READ`, both one-sided, peer CPU idle. It is the "faster than the
// SSD" overflow tier: local pinned RAM → **peer RAM (~12 GB/s over CX7)** →
// local NVMe/USB SSD (~2 GB/s). The peer owns nothing — each group belongs to
// exactly one client sequence; this process is a passive memory blade.
//
// Addressing is the flat group-id space of `GroupLayout`: a group lands at
// `base + group_id * group_stride`, so no per-group bookkeeping on the peer.
//
// Wire protocol (little-endian), connection-oriented:
//   1. client -> [u64 total_bytes]  (num_groups * group_stride it will address)
//   2. peer allocates + registers a RW MR of that size, replies with
//      CacheServerParams [u32 qpn][u32 psn][16 gid][u64 base_addr][u32 rkey]
//   3. client -> VerbsClientParams [u32 qpn][u32 psn][16 gid]
//   4. peer connects its QP, replies [u8 STATUS_OK]
//   5. client does one-sided WRITE/READ; peer idles until the client hangs up,
//      then unregisters + unmaps the blade.
//
// This module is split per the Atlas SDD file-size idiom:
//   `server_impl.rs` — accept loop, first-u64 dispatch, server-side rail
//                      handshake holding the crate's SINGLE `reg_mr_rw` call
//                      site (the access flag stays AT the call site — census-
//                      pinned by `tests/reg_mr_flag_audit.rs`), both data
//                      planes;
//   `registry.rs`    — peer POLICY: the process-global (kind, blob_bytes)
//                      paging-arena registry, the disk-cap carve, and the anon
//                      `Mmap` RAII the RDMA-registered arenas live in.
// The peer's residency IS `atlas_tier::Residency` and its verbs ARE
// `atlas_rdma::verbs`, imported directly — nothing tier- or verbs-shaped is
// hand-rolled here.

// The RW-blade handshake codec lives verbatim in the CUDA-free `atlas-rdma`
// crate; re-exported at its old path so the server below and both RW clients
// are zero-diff. Byte layout golden-pinned in `tests/rdma_wire_golden.rs` —
// it is what the fleet cache-peer binary speaks.
pub use atlas_rdma::wire::CacheServerParams;

// Peer paging policy — verbs-only, like the paging handshake it backs.
#[cfg(atlas_rdma_verbs)]
mod registry;
#[cfg(unix)]
mod server_impl;

#[cfg(unix)]
pub use server_impl::{RdmaConfig, serve};

// `kv_server_params_round_trip` lives WITH the codec in
// `crates/atlas-rdma/tests/wire_roundtrip.rs`; the exact byte layout stays
// pinned by `tests/rdma_wire_golden.rs` here. The `carve_disk_slots`
// precedence pins live in `registry_tests.rs`.
