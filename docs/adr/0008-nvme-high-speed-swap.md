# ADR-0008: NVMe-backed high-speed KV swap

**Status:** Accepted
**Date:** 2026-04-28

## Context

Long-context inference (>32K tokens) blows out the on-device KV cache.
Atlas's paged KV cache reserves a fixed fraction of GPU memory at startup
(`--gpu-memory-utilization`); past that point we either:

- Drop the request (fail-fast on cache overflow).
- Evict cold blocks somewhere we can stream them back.

The "somewhere" choices on GB10:

1. **CPU memory.** Always available. Easy. PCIe-attached host memory has
   bandwidth in the few-tens-of-GB/s range over the LPDDR5X integrated
   memory link. Limited by host RAM (~120 GB minus OS).
2. **NVMe via GPUDirect Storage (GDS / cuFile).** Direct GPU↔NVMe DMA,
   bypassing the host. The "right" answer on hardware that supports it.
3. **NVMe via pinned-host bounce buffer.** GPU↔host DMA, then host↔NVMe
   via standard kernel I/O. Two hops; worse bandwidth than GDS but
   universally supported.

The decision was forced by **GB10 silicon**: cuFile rejects "GPU model
not supported" + no PCIe P2P DMA path exists. GDS is unavailable on this
hardware (see `project_gb10_gds_unsupported`). We can re-evaluate when
hardware that supports GDS lands.

## Decision

Atlas implements **High-Speed Swap (HSS)** in `crates/spark-storage/`
using `io_uring` + a pinned-host bounce buffer:

- Each sequence keeps a fixed number of "hot" KV blocks on-GPU
  (`--high-speed-swap-cache-blocks-per-seq`, default 64 ≈ 1024 tokens at
  block_size=16).
- Cold blocks evict via async `io_uring` writes through the pinned-host
  bounce buffer. The bounce buffer is sized to keep the GPU stream from
  ever waiting on host I/O during steady-state.
- The radix tree tracks `disk_block_id` per block; reads back happen on
  demand when a cold block is referenced again.
- Slide-before-alloc + LIFO recycle (Phase 6.3 fix,
  `project_high_speed_swap_phase62`) prevents head-of-line blocking
  when a sequence's hot window slides.

Single CLI flag: `--high-speed-swap`. Defaults are sized for 64K
context with 8 concurrent sequences.

## Consequences

**Better:**
- Long-context (>32K) is feasible on GB10 without cutting batch size.
  We've live-tested 5×concurrent requests at 65K context across both
  DGX nodes.
- The interface is a single flag; users don't pick file vs CPU vs hybrid.
- The radix tree's existing `disk_block_id` field meant minimal
  scheduler changes to wire the path in.

**Worse:**
- Two-hop path (GPU → pinned host → NVMe) caps bandwidth at ~3 GB/s
  with a gen4 SSD. Below the GPU's effective throughput; we mask it
  with async eviction but a worst-case "everything is cold"
  query stalls visibly.
- The pinned-host bounce buffer is sized at startup. Sizing too small
  → eviction stalls; too large → wasted host RAM. We pick a default
  that's fine for the canonical workload but won't be optimal for all.

**New problems we created:**
- HSS bandwidth is **disk-dominated**. Different NVMe parts give wildly
  different latencies. We document a ≥3 GB/s sequential-write minimum;
  users with consumer SSDs will have a bad time.
- A single-machine `/mnt/fast-nvme/atlas-kv` directory is the canonical
  swap target. Multi-host setups need per-rank distinct directories
  (and a shared filesystem will *not* perform).
- We will revisit when GDS-capable hardware lands. The `cuFile-sys` FFI
  crate is committed but currently dormant for exactly this reason.
