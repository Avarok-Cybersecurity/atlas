// SPDX-License-Identifier: AGPL-3.0-only
//
// EXL3 fused MoE prefill: one persistent launch runs gather→Had→gate/up
// trellis GEMM→SiLU·mul→down trellis GEMM→Had→weighted scatter-add for EVERY
// routed expert with 0 < token_count <= max_tokens_per_expert rows, straight
// from packed trellis codes (no reconstruction, no per-expert launches).
// Device code vendored VERBATIM from turboderp's ExLlamaV3 (MIT) — see
// exl3_vendor/exl3_moe_kernel.cuh / exl3_moe_common.cuh for provenance and
// the (documented) deltas. This file only adds extern "C" __global__
// wrappers so the Rust host selects instances by name.
//
// This is a SEPARATE top-level module from exl3_matmul.cu, deliberately:
//   + each common/*.cu is its own nvcc job — the 16 instances here (each
//     embedding the full pipelined exl3_gemm_kernel_inner) don't serialize
//     behind the ~200 gemm/gemv instances of exl3_matmul.cu, and vice versa;
//   − the host must load/attribute-raise two modules for the EXL3 MoE layer
//     (exl3_matmul for the decode-tier mgemm + converters + silu_mul,
//     exl3_moe for the fused prefill kernel).
//   Safe because this kernel keeps NO module-scope __device__ state: barriers,
//   split-k locks and the expert-scheduler tickets all live in the shared
//   per-device locks buffer passed as an argument (unlike mgemm's filtered
//   path, which uses module-scope globals in exl3_matmul's module).
//
// ── Symbol grammar ─────────────────────────────────────────────────────────
// exl3_moe_k{K}_n{N}_cb{CB}
//   K  ∈ {0, 2, 3, 4}: trellis bits/weight for gate, up AND down. K > 0 =
//        compile-time, requires K_gate == K_up == K_down == K. K = 0 =
//        runtime dispatch per projection (mixed-K checkpoints) — SUPPORTS
//        ONLY K_x ∈ {2,3,4}; any other value silently skips that
//        projection's GEMM (vendor header documents the trim), so the host
//        MUST refuse the fused path outside that envelope.
//   N  ∈ {128, 256}: MOE_TILESIZE_N. Upstream selection:
//        N = 256 iff (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0),
//        else N = 128. Upstream index [4*K + 2*cb_idx + N_off] maps here by
//        name: cb_idx 0 → cb1 (MCG), 1 → cb2 (MUL1).
//        (qwen4_exp: hidden 2560, inter 640 → 640 % 256 != 0 → ALWAYS n128;
//        n256 instances exist for other EXL3 checkpoints.)
//   CB ∈ {1, 2}: codebook, 1 = MCG, 2 = MUL1 (qwen4_exp ships MUL1). The
//        3INST codebook (cb=0) has NO fused MoE kernel upstream either —
//        gate/up/down must share one codebook or the fused path is refused.
//   Constraints: hidden_dim % 128 == 0 and intermediate_dim % 128 == 0
//        (Hadamard warps and the locks slicing assume it); dims must also be
//        multiples of the tile (K-dim 32 / N-dim {128,256}) per projection:
//        qwen4_exp gate/up (k=2560, n=640) and down (k=640, n=2560) both
//        satisfy shape n128.
//
// ── Launch contract (host side, mirrors upstream exl3_moe.cu:203-301) ──────
//   block = dim3(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16) = 512 threads.
//   grid  = dim3(group_size, 1, num_groups) where
//     concurrency = temp_state_g.shape[0]; REQUIRE concurrency * 8 <= num_sms
//       (upstream sizes it as exl3_moe_max_concurrency = num_sms /
//        MOE_SMS_PER_EXPERT);
//     num_groups = min(concurrency, MOE_MAX_GROUPS=64);
//     group_size = MOE_SMS_PER_EXPERT = 8;
//     if (num_active > 0)   // count of experts with 0 < count <= 128 rows
//         num_groups = min(num_groups, num_active),
//         group_size = min(num_sms / num_groups, MOE_MAX_SMS_PER_EXPERT=32);
//     (num_active == 0 → do not launch; num_active unknown (-1, the no-sync
//      T*top_k <= 128 shortcut) → keep the defaults.)
//   dynamic smem = SMEM_MAX (92160 B): raise
//     CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES once per function per
//     device first, exactly like the exl3_matmul entries.
//   PLAIN launch (cuLaunchKernel), NOT cooperative — but every block spins on
//     group barriers in the locks buffer, so ALL group_size*num_groups blocks
//     (<= num_sms by construction, 1 block/SM at 90KB smem) must become
//     co-resident. Treat it like the cooperative entries operationally: never
//     under CUDA graph capture on the decode path, and do not launch it
//     concurrently with another exl3 kernel that shares the locks buffer
//     unless ordered on the same stream.
//
// ── Argument list (EXL3_MOE_KERNEL_ARGS, 30 args) ──────────────────────────
//   hidden_state   fp16 (T, hidden_dim)  — token-major activations (RAW, the
//                  kernel applies suh+Hadamard itself while gathering)
//   temp_state_g/u fp16 (C, 128, hidden_dim)         C = concurrency
//   temp_intermediate_g/u fp16 (C, 128, intermediate_dim)
//                  (qwen4_exp, C=6: 2×3.75 MB + 2×0.94 MB ≈ 9.4 MB total;
//                  no zero-init needed, protected by the group barriers;
//                  allocate ONCE at construction — 901 playbook)
//   output_state   fp32 (T, hidden_dim) — MUST be zero-initialized every
//                  call; the kernel atomicAdds weight-scaled expert outputs
//   expert_count   int64 (num_experts + 1) — bincount over LOCAL expert ids
//                  of the sorted assignment; the last (sentinel) bucket
//                  collects EP-remote/invalid slots and is never processed
//   token_sorted   int64 (T * top_k) — original token index per sorted slot
//   weight_sorted  fp16  (T * top_k) — routing weight per sorted slot
//   {gate,up,down}_{trellis,suh,svh}  (num_experts,) device arrays of device
//                  pointers (dense LOCAL tables; remote experts are excluded
//                  via the sentinel bucket, NEVER by null entries at
//                  reachable indices)
//   hidden_dim, intermediate_dim, num_experts (LOCAL count = len(count)-1),
//   num_experts_per_tok, max_tokens_per_expert (= temp rows = 128,
//   TEMP_ROWS_FUSED), concurrency,
//   act_limit      f32, 0.0f = no clamp (qwen4_exp: 0.0f)
//   act_function   0 = SiLU (qwen4_exp), 1 = GELU, 2 = RELU2_NOGATE
//   K_gate/K_up/K_down  bits per projection (k0 instances only; fixed-K
//                  instances ignore mismatches — host must pass equal values)
//   locks          the shared per-device exl3 locks buffer (4,202,760 B,
//                  zeroed once at alloc; barriers/tickets self-reset)
//
// Experts with token_count > max_tokens_per_expert are SKIPPED by the kernel
// before ticket matching (they consume no ticket — num_active must count
// only experts with 0 < count <= 128) and must be served by the overflow
// path (per-expert reconstruct + GEMM).

#include <cstdint>
#include <cuda_fp16.h>

#include "exl3_vendor/exl3_moe_kernel.cuh"

// Forward the named parameters declared by EXL3_MOE_KERNEL_ARGS
#define EXL3_MOE_FWD_ARGS                                                     \
    hidden_state, temp_state_g, temp_state_u,                                 \
    temp_intermediate_g, temp_intermediate_u, output_state,                   \
    gate_trellis, gate_suh, gate_svh,                                         \
    up_trellis, up_suh, up_svh,                                               \
    down_trellis, down_suh, down_svh,                                         \
    expert_count, token_sorted, weight_sorted,                                \
    hidden_dim, intermediate_dim, num_experts, num_experts_per_tok,           \
    max_tokens_per_expert, concurrency, act_limit, act_function,              \
    K_gate, K_up, K_down, locks

#define EXL3_MOE_WRAP(K, N, CB)                                               \
    extern "C" __global__                                                     \
    void __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)      \
    exl3_moe_k##K##_n##N##_cb##CB(EXL3_MOE_KERNEL_ARGS)                       \
    {                                                                         \
        exl3_moe_kernel_body<K, N, CB>(EXL3_MOE_FWD_ARGS);                    \
    }

#define EXL3_MOE_SET(K)                                                       \
    EXL3_MOE_WRAP(K, 128, 1)                                                  \
    EXL3_MOE_WRAP(K, 256, 1)                                                  \
    EXL3_MOE_WRAP(K, 128, 2)                                                  \
    EXL3_MOE_WRAP(K, 256, 2)

EXL3_MOE_SET(0)  // runtime per-projection K dispatch, K_x in {2,3,4} only
EXL3_MOE_SET(2)
EXL3_MOE_SET(3)
EXL3_MOE_SET(4)
