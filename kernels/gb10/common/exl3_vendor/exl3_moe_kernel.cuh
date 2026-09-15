// SPDX-License-Identifier: MIT
//
// Vendored from turboderp's ExLlamaV3 (https://github.com/turboderp-org/exllamav3)
// Copyright (c) 2025 turboderp — MIT license.
// Original: exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh, fetched from
// upstream master 2026-09-01 (NOT in the .research/exllamav3_ref snapshot; a
// fetch copy was kept at the build-job tmp dir).
//
// Adaptations (body verbatim otherwise; upstream loop/index/barrier logic
// untouched):
//   * the `__global__` template kernel became `inline __device__ void
//     exl3_moe_kernel_body`: Atlas needs plain extern "C" __global__ entry
//     points selectable by name from the PTX module (see exl3_moe.cu), so
//     __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16) moves
//     to the wrappers
//   * include paths made local (no torch/cublas/stdio; cuda::atomic_ref comes
//     from <cuda/atomic> via ptx.cuh, included by exl3_gemm_inner.cuh)
//   * the runtime-K dispatch switches (the K=0 / t_bits==0 variant for
//     MIXED-K gate/up/down layers) are TRIMMED from upstream's cases 1..8 to
//     cases 2,3,4. The FIXED-K wrapper instances (exl3_moe.cu) cover K in
//     {2,3,4,5,6} — every shipped Qwen3.8-Flash-Next-exl3 branch's routed
//     experts (K=2/3/4/5/6 for 2.05/3.05/4.05/5.05/6.05bpw, each UNIFORM
//     across gate/up/down and across the 512 experts per the header
//     inventory). So a real checkpoint never reaches a k0 instance at K=5/6;
//     the k0 path only exists for a hypothetical mixed-K export, and there it
//     serves K in {2,3,4} only. Each retained switch case instantiates the
//     full pipelined exl3_gemm_kernel_inner at the MoE tile shape in EVERY
//     k0 (cb, N) variant, which is what bounds PTX size / compile time.
//     Measured on GB10 (nvcc --ptx -arch=sm_121f -O3, single job):
//       k0{2,3,4} + fixed{2,3,4}     16 instances  17.7 s   8.86 MB PTX
//       k0{2,3,4} + fixed{2..6}      24 instances  25.6 s  12.24 MB  (SHIPPED)
//       k0{2..6}  + fixed{2..6}      24 instances  53.5 s  14.67 MB  (rejected:
//                                    3.0x compile for a case no checkpoint has)
//     A K outside the switch reaching a k0 wrapper at runtime SILENTLY SKIPS
//     that projection's GEMM (upstream's switch has no default either) — the
//     host must refuse the fused path for a mixed-K layer unless all of
//     K_gate/K_up/K_down are in {2,3,4} (the loader keep-predicate
//     `expert_keep_set` enforces exactly that; uniform K in {2..6} takes the
//     fixed instance). Extend the switches together with new wrapper
//     instances if a mixed-K K=5/6 export ever ships.
//   * DETERMINISTIC EPILOGUE (Atlas addition, default-ON at the host): a 31st
//     argument `float* output_slots`. Upstream's `had_d_out` atomicAdds each
//     expert's weighted row into the token's ONE shared fp32 `output_state`
//     row; because the expert->group assignment below is a dynamic ticket
//     draw and ~6 groups run concurrently, the commit order — and so the
//     non-associative fp32 sum — differs between two identical runs, which is
//     the whole of qwen4_exp's measured temp-0 prefill nondeterminism. With
//     `output_slots` non-null each expert instead PLAIN-STORES its row to its
//     own sorted slot, and the host reduces each token's top_k slots in fixed
//     flat-slot order (`exl3_moe_reduce_slots_f32`). Identical arithmetic,
//     one order. `output_slots == nullptr` restores upstream's arm verbatim.
//   * `(void)` casts for the two kernel args the body never reads
//     (num_experts_per_tok, concurrency) — the Atlas kernel build promotes
//     warnings to errors (--Werror all-warnings)
//
// The scheduler/barrier protocol is self-resetting and lives in the shared
// exl3 locks buffer (see exl3_devctx.cuh layout): group barriers at
// BARRIER_LOCKS_OFFSET, tickets at MOE_SCHED_OFFSET, per-group split-k locks
// at [group_idx * MAX(hidden_dim, intermediate_dim)/128]. All blocks of a
// launch must become co-resident (spin barriers) — see the launch contract in
// exl3_moe.cu.

#pragma once

#include <cuda_fp16.h>

#include "exl3_moe_common.cuh"
#include "exl3_compat.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"

template<int t_bits, int MOE_TILESIZE_N, int cb>
inline __device__
void exl3_moe_kernel_body(EXL3_MOE_KERNEL_ARGS)
{
    (void) num_experts_per_tok;  // shapes derive from expert_count/token_sorted
    (void) concurrency;          // grid geometry carries it (gridDim.z groups)

    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int group_size = gridDim.x;  // SMs per expert, set at launch
    const int num_groups = gridDim.z;
    const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;  // blockDim.x
    const int group_threads = group_size * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    // Buffers for group
    temp_state_g += group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += group_idx * max_tokens_per_expert * intermediate_dim;

    // Barriers for group sync
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Expert scheduler state, self-resetting: [0] next ticket, [1] retired groups, [2 + g] ticket for group g
    int* sched = locks + MOE_SCHED_OFFSET;

    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    // Dynamic expert assignment: active experts are numbered in scan order, and each group processes the active
    // expert matching its current ticket. Initial tickets are the group indices; after finishing an expert, a group
    // draws the next unclaimed ticket, so load balances greedily without assuming uniform cost per expert
    int ticket = group_idx;

    // Loop over experts
    int start = 0;
    int end = 0;
    int expert_idx = 0;
    int expert_idx_assign = 0;
    for (; expert_idx < num_experts; ++expert_idx)
    {
        // Token span for current expert
        start = end;
        end += expert_count[expert_idx];
        int token_count = end - start;

        // Skip if no tokens or too many tokens for fused kernel (batch is handled by reconstruct path outside kernel)
        if (token_count == 0) continue;
        if (token_count > max_tokens_per_expert) continue;

        // Skip if expert is claimed by a different group
        if (expert_idx_assign++ != ticket) continue;

        // EXL3 weights for g, u, d
        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx];
        const half* exp_gate_suh = gate_suh[expert_idx];
        const half* exp_gate_svh = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis = up_trellis[expert_idx];
        const half* exp_up_suh = up_suh[expert_idx];
        const half* exp_up_svh = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx];
        const half* exp_down_suh = down_suh[expert_idx];
        const half* exp_down_svh = down_svh[expert_idx];

        // Gather + input hadamard for g, u. Non-gated mode skips the g staging (and the g GEMM
        // below); the activation synthesizes the gate lane from u
        const bool gated = act_function != MOE_ACT_RELU2_NOGATE;
        auto had_gather_gu_in = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + token_idx * hidden_dim + token_off * 128;
                if (gated)
                    had_hf_r_128_inner<true, false>
                    (
                        in_ptr,
                        temp_state_g + 128 * warp_idx,
                        exp_gate_suh + 128 * token_off,
                        0.088388347648f
                    );
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        };

        had_gather_gu_in();

        // g, u GEMM
        auto gemm_up = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    MIN(size_m, 16),    \
                    hidden_dim,         \
                    intermediate_dim,   \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    MOE_FRAG_STAGES
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, cb, SHAPE_ARGS, false>(ARGS);
                else switch(K)
                {
                    // cases 1, 5-8 removed — see header (Atlas MoE K envelope:
                    // the k0 MIXED-K switch stays at {2,3,4}; K=5/6 serve
                    // through the fixed-K k5/k6 instances)
                    case 2: exl3_gemm_kernel_inner<2, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, cb, SHAPE_ARGS, false>(ARGS); break;
                };
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr += 16 * hidden_dim;
                out_addr += 16 * intermediate_dim;
                size_m -= 16;
            }
        };

        if (gated)
            gemm_up(temp_state_g, temp_intermediate_g, exp_gate_trellis, K_gate);
        gemm_up(temp_state_u, temp_intermediate_u, exp_up_trellis, K_up);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for g, u + activation+gate + input hadamard for d
        auto had_guad = [&]()
        {
            const int warps_per_token = intermediate_dim / 128;
            const int total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner
                (
                    temp_intermediate_g + 128 * warp_idx,
                    temp_intermediate_u + 128 * warp_idx,
                    temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off,
                    exp_up_svh + 128 * token_off,
                    exp_down_suh + 128 * token_off,
                    0.088388347648f,
                    act_limit,
                    act_function
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        };

        had_guad();

        // d GEMM
        auto gemm_down = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int K)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                #define ARGS            \
                    in_addr,            \
                    trellis,            \
                    out_addr,           \
                    MIN(size_m, 16),    \
                    intermediate_dim,   \
                    hidden_dim,         \
                    locks,              \
                    nullptr
                #define SHAPE_ARGS      \
                    MOE_TILESIZE_M,     \
                    MOE_TILESIZE_K,     \
                    MOE_TILESIZE_N,     \
                    MOE_SH_STAGES,      \
                    MOE_FRAG_STAGES
                if constexpr (t_bits)
                    exl3_gemm_kernel_inner<t_bits, false, cb, SHAPE_ARGS, false>(ARGS);
                else switch(K)
                {
                    // cases 1, 5-8 removed — see header (Atlas MoE K envelope:
                    // the k0 MIXED-K switch stays at {2,3,4}; K=5/6 serve
                    // through the fixed-K k5/k6 instances)
                    case 2: exl3_gemm_kernel_inner<2, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 3: exl3_gemm_kernel_inner<3, false, cb, SHAPE_ARGS, false>(ARGS); break;
                    case 4: exl3_gemm_kernel_inner<4, false, cb, SHAPE_ARGS, false>(ARGS); break;
                };
                #undef ARGS
                #undef SHAPE_ARGS

                in_addr += 16 * intermediate_dim;
                out_addr += 16 * hidden_dim;
                size_m -= 16;
            }
        };

        gemm_down(temp_intermediate_g, temp_state_g, exp_down_trellis, K_down);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for d + scatter add.
        //
        // ATLAS DELTA (determinism): with `output_slots` non-null each expert
        // writes its weighted row to its OWN sorted slot (`start + slot_off`,
        // the slot the row already occupies in token_sorted/weight_sorted) by
        // plain store, and the host reduces a token's slots afterwards in a
        // fixed order. Upstream's arm — `output_slots == nullptr` — keeps the
        // atomicAdd into the shared per-token row of `output_state`, whose
        // commit order the ticket scheduler above makes nondeterministic.
        auto had_d_out = [&](float* __restrict__ slots)
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            const half* weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int slot_off = warp_idx / warps_per_token;
                half weight = weights[slot_off];
                int token_off = warp_idx % warps_per_token;
                const half* in_ptr = temp_state_g + 128 * warp_idx;
                const half* svh_ptr = exp_down_svh + 128 * token_off;
                const float r_scale = 0.088388347648f * __half2float(weight);
                if (slots)
                {
                    float* slot_ptr = slots
                        + ((int64_t) (start + slot_off)) * hidden_dim
                        + token_off * 128;
                    had_hf_r_128_d_inner_t<false>(in_ptr, slot_ptr, svh_ptr, r_scale);
                }
                else
                {
                    int token_idx = top_x[slot_off];
                    float* out_ptr = output_state + token_idx * hidden_dim + token_off * 128;
                    had_hf_r_128_d_inner_t<true>(in_ptr, out_ptr, svh_ptr, r_scale);
                }
            }
        };

        had_d_out(output_slots);

        // Draw the next ticket and publish it to the group through the end-of-expert barrier, which also protects
        // the temp buffers for reuse. Grabbed tickets continue from num_groups since 0..num_groups-1 are implicit
        if (block_idx == 0 && threadIdx.x == 0)
            sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);
        group_barrier(group_idx, group_size, barrier_counters_sense);
        ticket = sched[2 + group_idx];
    }

    // Retire group; last group out resets the scheduler for the next launch. The acq_rel increment orders each
    // group's earlier ticket grabs before the last group's reset (plain atomics are relaxed, so without this a
    // straggler's in-flight grab could land after the reset and leak into the next launch)
    if (block_idx == 0 && threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> next_ticket(sched[0]);
        cuda::atomic_ref<int, cuda::thread_scope_device> retired_groups(sched[1]);
        int retired = retired_groups.fetch_add(1, cuda::memory_order_acq_rel);
        if (retired == num_groups - 1)
        {
            next_ticket.store(0, cuda::memory_order_relaxed);
            retired_groups.store(0, cuda::memory_order_relaxed);
        }
    }
}
