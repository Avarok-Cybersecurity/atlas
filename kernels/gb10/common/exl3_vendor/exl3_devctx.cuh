// SPDX-License-Identifier: MIT
//
// Vendored from turboderp's ExLlamaV3 (https://github.com/turboderp-org/exllamav3)
// Copyright (c) 2025 turboderp — MIT license.
// Constants only: the DevCtx host singleton is replaced by the Atlas host side
// (Rust allocates the lock buffer once per device, zeroed once — the kernels'
// barrier/lock protocols self-reset). Snapshot original:
// .research/exllamav3_ref/exl3_devctx.cuh.
//
// Lock buffer layout (int32), total (MAX_TILES_C + 2*MAX_BARRIERS +
// MOE_SCHED_INTS) * 4 = 4,202,760 bytes:
//   [0 .. MAX_TILES_C)                split-k spinlocks (gemm indexes
//                                     locks[slice_m*blocks_n + slice2_n];
//                                     mgemm offsets by blockIdx.z*size_n/128)
//   [MAX_TILES_C .. +2*MAX_BARRIERS)  group_barrier counter/sense pairs
//   [MOE_SCHED_OFFSET .. +66)         MoE scheduler tickets (unused here,
//                                     layout preserved)

#pragma once

// Max allowable output size, in tiles. Used to allocate global lock buffer per device for sync across threadblocks
#define MAX_TILES_C (1024 * 1024)
#define MAX_BARRIERS 1024
#define BARRIER_LOCKS_OFFSET MAX_TILES_C

// MoE expert scheduler state, after the barrier counters: [0] next ticket, [1] retired groups,
// [2 + group] ticket published to group. Self-resetting, zero-initialized with the rest of the buffer
#define MOE_MAX_GROUPS 64
#define MOE_SCHED_OFFSET (MAX_TILES_C + 2 * MAX_BARRIERS)
#define MOE_SCHED_INTS (2 + MOE_MAX_GROUPS)
