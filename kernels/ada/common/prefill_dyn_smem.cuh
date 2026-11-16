// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill attention tiles in DYNAMIC shared memory, for targets whose ptxas
// refuses them as static `__shared__` (kernels/ada, sm_89).
//
// WHY. The BR=32 / BR=64 prefill kernels (inferspark_prefill.cu,
// inferspark_prefill_fp8kv.cu, prefill_paged_compute.cuh,
// prefill_paged_compute_asym.cuh) declare 66,560 to 90,112 bytes of static
// `__shared__` tiles. ptxas caps static shared memory at 0xc000 (48 KiB) for
// every arch string without an `a` suffix, and sm_89 has no `a` form:
//
//     ptxas error : Entry function 'inferspark_prefill_64' uses too much shared
//                   data (0x16000 bytes, 0xc000 max)
//
// (CUDA 13.3.73; docs/HARDWARE.md, "PTX arch suffix and the static
// shared-memory ceiling", has the same table measured on 13.0.88.) The same
// bytes are legal as DYNAMIC shared memory, which sm_89 lets one block opt into
// up to 99 KiB.
//
// HOW. Each kernel keeps its tiles in one struct whose members are the old
// arrays, same element types, same shapes, same order; the struct is laid over
// `avarok_prefill_smem` and the old names are bound to its members by
// reference, so the kernel bodies are unchanged. Each entry point then EXPORTS
// its own launch size:
//
//     extern "C" __device__ const unsigned int <entry>_dyn_smem_bytes;
//
// and the kernel registry (`avarok_core::registry_dyn_smem`) reads that symbol
// when it resolves `<entry>` and raises every launch of the function to at least
// that many dynamic bytes. The size therefore has one source, the
// `sizeof(struct)` below, and no host constant can drift from it; a kernel that
// exports nothing launches exactly as it always has.
#pragma once

// The per-block shared-memory opt-in on sm_86 and sm_89 (cudaDevAttr
// MaxSharedMemoryPerBlockOptin = 101,376 = 99 KiB). The static_asserts below
// hold every tile set to it, so a tile that grows past what an Ada block can
// hold fails the compile rather than the launch.
#define AVAROK_PREFILL_SMEM_OPTIN_MAX 101376u

extern __shared__ __align__(16) unsigned char avarok_prefill_smem[];

// `<entry>_dyn_smem_bytes`, with `entry` macro-expanded first (KERNEL_NAME,
// PAGED_CONCAT(KERNEL_NAME, _64), AVAROK_PREFILL_ENTRY).
#define AVAROK_DYN_SMEM_SYMBOL_(entry) entry##_dyn_smem_bytes
#define AVAROK_DYN_SMEM_SYMBOL(entry) AVAROK_DYN_SMEM_SYMBOL_(entry)

// Export one entry point's dynamic launch size. File scope, once per entry.
#define AVAROK_EXPORT_DYN_SMEM(entry, Tiles)                                     \
    static_assert(sizeof(Tiles) <= AVAROK_PREFILL_SMEM_OPTIN_MAX,               \
                  "prefill tiles exceed the sm_89 per-block shared opt-in");   \
    extern "C" __device__ const unsigned int AVAROK_DYN_SMEM_SYMBOL(entry) =     \
        (unsigned int)sizeof(Tiles)

// The tile struct, inside a kernel body.
#define AVAROK_PREFILL_TILES(Tiles, var) \
    Tiles& var = *reinterpret_cast<Tiles*>(avarok_prefill_smem)
