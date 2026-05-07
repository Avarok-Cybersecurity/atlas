// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Nemotron-H MoE Prefill — N-token batched variant.
//
// Generalizes the single-token relu²+down pattern to N tokens by packing
// (token_idx, expert_slot) into blockIdx.y. Each CTA processes one
// (token, expert, projection) triple.
//
// Nemotron-H MoE: 2 projections (up + down) with relu² activation.
// No gate_proj — experts have only up_proj and down_proj.
//
// Pipeline per layer:
//   1. RMS norm (batched, handled externally)
//   2. Gate GEMM: [N, H] × [H, E]^T → [N, E] logits
//   3. Sigmoid routing (batched): per-token top-K selection
//   4. Batched UP GEMV: [N, H] → [N*top_k, inter] via pointer tables
//   5. Shared UP GEMM: [N, H] → [N, shared_inter]
//   6. Batched relu²+down: [N*top_k, inter] → [N*top_k, H]
//   7. Shared relu²+down: [N, shared_inter] → [N, H]
//   8. Weighted sum + shared blend: → [N, H]
//
// Token layout in blockIdx.y for UP/DOWN:
//   y < N*top_k:  routed  (token = y / top_k, slot = y % top_k)
//   y >= N*top_k: shared  (token = y - N*top_k)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_NMP[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// Batched Sigmoid Top-K Routing — N tokens in one launch
// ============================================================================
//
// Grid: (1, num_tokens, 1)  Block: (256, 1, 1)
//
// Each blockIdx.y handles one token's routing independently.
// Inputs: gate_logits [num_tokens, num_experts] BF16
// Outputs: expert_indices [num_tokens * top_k] u32
//          expert_weights [num_tokens * top_k] f32

#define MAX_EXPERTS 512
#define MAX_TOP_K 32  // Must be >= num_experts_per_tok (22 for Super 120B)

extern "C" __global__ void nemotron_moe_topk_sigmoid_batched(
    const __nv_bfloat16* __restrict__ gate_logits,   // [num_tokens, num_experts]
    const float* __restrict__ bias,                   // [num_experts] F32
    unsigned int* __restrict__ expert_indices,        // [num_tokens * top_k]
    float* __restrict__ expert_weights,               // [num_tokens * top_k]
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor,
    unsigned int num_tokens
) {
    const unsigned int token = blockIdx.y;
    if (token >= num_tokens) return;

    const unsigned int tid = threadIdx.x;

    __shared__ float s_sigmoid[MAX_EXPERTS];
    __shared__ float s_selection[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    const __nv_bfloat16* logits = gate_logits + (unsigned long long)token * num_experts;
    unsigned int* out_idx = expert_indices + (unsigned long long)token * top_k;
    float* out_wt = expert_weights + (unsigned long long)token * top_k;

    // Phase 1: Sigmoid + bias
    for (unsigned int i = tid; i < actual_n; i += blockDim.x) {
        float logit = __bfloat162float(logits[i]);
        float sig = 1.0f / (1.0f + __expf(-logit));
        s_sigmoid[i] = sig;
        s_selection[i] = sig + bias[i];
    }
    __syncthreads();

    // Phase 2: Iterative top-K selection
    for (unsigned int k = 0; k < top_k; k++) {
        // Each thread finds its local max
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += blockDim.x) {
            if (s_selection[i] > local_max) {
                local_max = s_selection[i];
                local_idx = i;
            }
        }
        // Warp reduction
        unsigned int warp_id = tid / WARP_SIZE;
        unsigned int lane = tid % WARP_SIZE;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();
        // Final reduction across warps (thread 0)
        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (int w = 1; w < 8; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[k] = best_val;
            s_top_idxs[k] = best_idx;
            s_selection[best_idx] = -1e30f; // mask selected
        }
        __syncthreads();
    }

    // Phase 3: Write outputs with weights from pre-bias sigmoid
    if (tid < top_k) {
        out_idx[tid] = s_top_idxs[tid];
        float w = s_sigmoid[s_top_idxs[tid]];
        // Normalize
        if (normalize) {
            float sum = 0.0f;
            for (unsigned int k = 0; k < top_k; k++) {
                sum += s_sigmoid[s_top_idxs[k]];
            }
            w = (sum > 0.0f) ? (w / sum) : w;
        }
        out_wt[tid] = w * scaling_factor;
    }
}

// ============================================================================
// Batched UP GEMV — N-token variant
// ============================================================================
//
// Grid: (ceil(N_inter/8), num_tokens*(top_k+1), 1)  Block: (128, 1, 1)
//
// blockIdx.y < num_tokens*top_k:  routed expert UP
// blockIdx.y >= num_tokens*top_k: shared expert UP
//
// Routed: reads input[token, K], weights from pointer table via expert_indices
// Shared: reads input[token, K], weights from shared expert pointers

extern "C" __global__ void nemotron_moe_up_prefill(
    const __nv_bfloat16* __restrict__ A,             // [num_tokens, K] BF16
    // Routed expert pointer tables
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ up_out,              // [num_tokens * top_k, N_inter] BF16
    const unsigned int* __restrict__ expert_indices,  // [num_tokens * top_k] u32
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,           // [num_tokens, N_shared_inter] BF16
    unsigned int N,              // N_inter (routed intermediate)
    unsigned int K,              // hidden_size
    unsigned int N_shared,       // shared_expert_intermediate_size
    unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token;
    if (is_shared) {
        token = y - total_routed;
    } else {
        token = y / top_k;
    }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    unsigned int N_out;

    if (is_shared) {
        B_packed = sh_up_packed;
        B_scale = sh_up_scale;
        s2 = sh_up_s2;
        C = sh_up_out + (unsigned long long)token * N_shared;
        N_out = N_shared;
    } else {
        unsigned int expert_slot = y % top_k;
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        C = up_out + (unsigned long long)(token * top_k + expert_slot) * N;
        N_out = N;
        // EP: NULL pointer means remote expert — write zero
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N_out; i += BLOCK_SIZE) {
                C[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    // Standard W4A16 GEMV: 4 outputs per block, warp-shuffle reduction
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_NMP[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

// ============================================================================
// Batched relu²+down GEMV — N-token variant
// ============================================================================
//
// Grid: (ceil(N_out/8), num_tokens*(top_k+1), 1)  Block: (128, 1, 1)
//
// blockIdx.y < num_tokens*top_k:  routed expert relu²+down
// blockIdx.y >= num_tokens*top_k: shared expert relu²+down

extern "C" __global__ void nemotron_moe_relu2_down_prefill(
    const __nv_bfloat16* __restrict__ up_out,        // [num_tokens * top_k, K_routed] BF16
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                    // [num_tokens * top_k, N] routed output
    const unsigned int* __restrict__ expert_indices,   // [num_tokens * top_k]
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_up_in,      // [num_tokens, K_shared] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,          // [num_tokens, N_shared] BF16
    unsigned int N,              // output dim for routed (hidden_size or moe_latent)
    unsigned int K_routed,       // input dim for routed (moe_intermediate_size)
    unsigned int K_shared,       // input dim for shared (shared_expert_intermediate_size)
    unsigned int N_shared,       // output dim for shared (hidden_size)
    unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token;
    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    unsigned int K;
    unsigned int N_out;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        token = y - total_routed;
        B_packed = sh_down_packed;
        B_scale = sh_down_scale;
        s2 = sh_down_s2;
        u_ptr = sh_up_in + (unsigned long long)token * K_shared;
        K = K_shared;
        N_out = N_shared;
    } else {
        token = y / top_k;
        unsigned int expert_slot = y % top_k;
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        u_ptr = up_out + (unsigned long long)(token * top_k + expert_slot) * K_routed;
        K = K_routed;
        N_out = N;
        // EP: NULL pointer means remote expert — write zero
        if (B_packed == 0) {
            __nv_bfloat16* out = C + (unsigned long long)(token * top_k + expert_slot) * N;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N_out; i += BLOCK_SIZE) {
                out[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_NMP[threadIdx.x];

    // Phase 1: Precompute relu²(up) into shared memory
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float u = __bfloat162float(u_ptr[i]);
        float r = fmaxf(u, 0.0f);
        s_act[i] = r * r;
    }
    __syncthreads();

    // Phase 2: GEMV with precomputed activation
    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    // Write output
    __nv_bfloat16* out = is_shared ?
        (sh_down_out + (unsigned long long)token * N_shared) :
        (C + (unsigned long long)(token * top_k + (y % top_k)) * N);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}

// ============================================================================
// Batched Weighted Sum + Shared Blend — N-token variant
// ============================================================================
//
// Grid: (ceil(hidden/256), num_tokens, 1)  Block: (256, 1, 1)
//
// output[t, d] = scale * sum(weights[t*topk+e] * expert_down[(t*topk+e)*H+d])
//                + shared_down[t*H+d]

extern "C" __global__ void nemotron_moe_weighted_sum_prefill(
    __nv_bfloat16* __restrict__ output,               // [num_tokens, hidden]
    const __nv_bfloat16* __restrict__ expert_down,     // [num_tokens * top_k, hidden]
    const float* __restrict__ expert_weights,           // [num_tokens * top_k]
    const __nv_bfloat16* __restrict__ shared_down,     // [num_tokens, hidden]
    unsigned int hidden,
    unsigned int top_k,
    float routed_scaling_factor,
    unsigned int num_tokens
) {
    const unsigned int token = blockIdx.y;
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden || token >= num_tokens) return;

    float routed_sum = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        float w = expert_weights[token * top_k + e];
        routed_sum += w * __bfloat162float(expert_down[(unsigned long long)(token * top_k + e) * hidden + j]);
    }
    float shared_val = __bfloat162float(shared_down[(unsigned long long)token * hidden + j]);
    output[(unsigned long long)token * hidden + j] = __float2bfloat16(routed_scaling_factor * routed_sum + shared_val);
}
