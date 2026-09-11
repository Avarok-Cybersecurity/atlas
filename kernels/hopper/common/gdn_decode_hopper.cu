// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN decode recurrence — Hopper (sm_90a) twins.
//
// SSOT for why this file exists and for every number quoted below:
// `GDN-DECODE-ATTRIBUTION.md` at the repository root (#927/#928), from the
// nsys round-10 trace of 1xH100, Qwen/Qwen3.8-27B-FP8, 48 SSM layers,
// 2026-09-11. The two parents this file twins are, per that trace:
//
//   n=16 step 21.800 ms  gated_delta_rule_decode_f32_strided
//                        48 launches = 2.740 ms (12.6%), 57.1 us/layer
//   C=1  step 18.500 ms  gated_delta_rule_decode_f32
//                        48 launches x 17.8 us = 0.850 ms (4.6%)
//
// Both parents launch grid=(num_v_heads, batch_size), block=(128,1,1) — one
// CTA per (row, head), one thread per state COLUMN. This model has
// num_v_heads = 48 (derived: 3.16 MB of FP32 state per layer per sequence /
// (128*128*4 B)), so at C=1 the grid is 48 CTAs. An H100 SXM5 has 132 SMs:
// 84 of them get no work. 17.8 us to move 6.29 MB (one read + one write of
// 3.15 MB) is 354 GB/s on a 3350 GB/s part.
//
// The contrapositive is measured, and it is why the two twins below differ:
// GB10 has EXACTLY 48 SMs (cudaDeviceProp.multiProcessorCount, dgx2
// 2026-09-11), so on GB10 the parent's 48-CTA grid already fills the machine
// and a narrower tile only costs warps. The underfill is a Hopper-shaped
// defect, not a kernel-shaped one.
//
// WHAT CHANGES
//  1. COLUMN-TILED GRID — the C=1 twin only. grid = (ceil(v_dim/blockDim.x),
//     num_v_heads, batch_size), so the caller picks the tile width: 32 turns
//     48 CTAs into 192 one-warp CTAs that reach all 132 SMs, 128 reproduces
//     the parent's shape. `gdn_hopper_cols_per_cta` in
//     crates/spark-model/src/layers/ops/ssm_gdn_hopper.rs is the SSOT for the
//     choice and narrows only when `num_v_heads * rows < sm_count`. Column i's update touches only column i of the
//     state, so this re-partitions independent work; it does not re-associate
//     anything. Three grid axes rather than a flattened grid.x because
//     `blockIdx.x / tiles` costs a runtime 32-bit division, which ptxas
//     expands to a ~30-instruction MUFU.RCP/I2F/F2I sequence per CTA.
//  2. DEEPER UNROLL — both twins. The parents carry `#pragma unroll 4` over a
//     `j += 4` loop = 16 independent 4-byte loads in flight per thread; 16
//     replications cover k_dim = 128 in two passes of 64. Swept on dgx2
//     (GB10, production flags, fresh state per rep, 3 scored passes of 30):
//     at n=16 the strided twin reads 1.16x / 1.19x / 1.56x / 1.09x of the
//     parent at unroll 4 / 8 / 16 / 32, so 16 is the peak and 32 falls off.
//     ptxas sm_90a reports 32 registers and 0 bytes of spill at 16.
//     Unrolling replicates the loop BODY; it does not reorder the
//     accumulation, so the bit-exactness argument is unaffected.
//     ⚠ The literal is spelled out at each loop because nvcc does not
//     macro-expand a `#pragma unroll` argument — it warns #20169-D and then
//     silently drops the pragma.
//
// WHY THE STRIDED TWIN IS **NOT** COLUMN-TILED. Its parent is compiled with
// SSM_STATE_NORM_ENABLED (defined at kernels/gb10/qwen3.6-27b/nvfp4/
// gated_delta_rule.cu:941, ABOVE the strided kernel and BELOW the C=1 one —
// which is why only one of the two parents carries the clamp). That block
// reduces a Frobenius norm over the WHOLE head, across all four warps of the
// CTA, and rescales the head if it exceeds SSM_STATE_MAX_NORM. Splitting a
// head's 128 columns across several CTAs would make that reduction
// impossible without a grid-wide barrier, and dropping the clamp would be a
// silent numerics change, not an optimisation. So the strided twin keeps one
// CTA per (row, head) at 128 threads — which costs it nothing at n=16, where
// the grid is already 768 CTAs.
//
// WHAT DELIBERATELY DOES NOT CHANGE — state retention. The obvious next lever
// is to keep the [k_dim] column slice live between the two passes and drop
// the re-read (2R+1W -> 1R+1W). It is not here because the tree already
// measured both forms and both lost or tied:
//   * full-width register retention: -11.6% e2e (comment at
//     `gated_delta_rule_decode_f32_strided_norm_half` in the parent file) —
//     ptxas reports 255 registers with 88 B of spill stores for that twin at
//     sm_90a, so it spills;
//   * SMEM staging (`..._strided_norm_smem`, ATLAS_GDN_SMEM_STAGE): dgx2
//     kill-switch A/B at C=128, +0.5% / -0.5%, inside the boot band. The
//     re-read is CTA-local and is served by L2.
// Neither receipt is from an H100, so the lever is un-evidenced here rather
// than refuted; the attribution doc carries the re-probe plan.
//
// NUMERICS — bit-exact against the parents, by construction:
//   * every floating-point expression below is character-identical to its
//     parent's and evaluated in the same order, including the
//     `h0*k0 + h1*k1 + h2*k2 + h3*k3` group shape and the sequential
//     `hk_dot +=` / `q_dot +=` / `norm_acc +=` chains over ascending j;
//   * these kernels compile under `--fmad=false` (kernels/gb10/common/
//     KERNEL.toml and the model's own KERNEL.toml both set it), so there is
//     no FMA-contraction freedom left for a schedule change to exercise;
//   * only the (thread, CTA) -> column mapping changes, and only for the C=1
//     twin. Per column the read addresses, the arithmetic and the reduction
//     order are the parent's.
//
// PRECONDITION: k_dim <= 128 (`smem_k`/`smem_q` are statically sized for it,
// exactly as the parents' are) and k_dim % 4 == 0 (the parents' loop step).
// The strided twin additionally requires blockDim.x == v_dim == 128, the
// parent's own hardcoded shape (`norm_sums[4]`, `tid / 32`). Callers gate on
// this in crates/spark-model/src/layers/ops/ssm_gdn_hopper.rs; the kernels
// re-check k_dim rather than scribble past the staging buffers.

#define GDN_H_MAX_KD 128

// Mirrors the parent file's guard so the clamp compiles identically here.
#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif

// Shared recurrent body. `WITH_STATE_NORM` selects the parent that this
// instantiation twins: false = `gated_delta_rule_decode_f32` (declared above
// the SSM_STATE_NORM_ENABLED define in the parent file, so no clamp), true =
// `gated_delta_rule_decode_f32_strided` (declared below it, so clamp).
//
// `col` is this thread's state column; `H` already points at this
// (row, head) slice. Strides are in ELEMENTS.
template <bool WITH_STATE_NORM>
__device__ __forceinline__ void gdn_decode_hopper_body(
    float* __restrict__ H,
    const float* __restrict__ q_ptr,
    const float* __restrict__ k_ptr,
    const float* __restrict__ v_ptr,
    float g,
    float bt,
    float* __restrict__ out_ptr,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int col,
    float* smem_k,
    float* smem_q
) {
    // k/q are broadcast to every thread of the CTA, so they are staged once.
    // A column-tiled launch re-reads this 1 KB per CTA; the sibling CTAs of
    // one head issue those reads within a few microseconds of each other, so
    // they are L2-hot.
    for (unsigned int i = threadIdx.x; i < k_dim; i += blockDim.x) {
        smem_k[i] = k_ptr[i];
        smem_q[i] = q_ptr[i];
    }
    __syncthreads();

    // AFTER the barrier: a tail CTA on a v_dim that is not a multiple of the
    // tile width still has to reach `__syncthreads()` with the rest. (The
    // parents return BEFORE their barrier, which is only safe because their
    // blockDim.x is always exactly v_dim.)
    if (col >= v_dim) {
        return;
    }

    float v_i = v_ptr[col];
    float hk_dot = 0.0f;
    #pragma unroll 16
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + col];
        float h1 = H[(j + 1) * v_dim + col];
        float h2 = H[(j + 2) * v_dim + col];
        float h3 = H[(j + 3) * v_dim + col];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }

    float v_new_i = (v_i - g * hk_dot) * bt;

    float q_dot = 0.0f;
    float norm_acc = 0.0f;
    #pragma unroll 16
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + col];
        float h1 = H[(j + 1) * v_dim + col];
        float h2 = H[(j + 2) * v_dim + col];
        float h3 = H[(j + 3) * v_dim + col];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + col] = h0;
        H[(j + 1) * v_dim + col] = h1;
        H[(j + 2) * v_dim + col] = h2;
        H[(j + 3) * v_dim + col] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
        if (WITH_STATE_NORM) {
            // Frobenius accumulation from the registers just stored, one add
            // at a time in ascending j — the parent's order exactly.
            norm_acc += h0 * h0;
            norm_acc += h1 * h1;
            norm_acc += h2 * h2;
            norm_acc += h3 * h3;
        }
    }

    if (WITH_STATE_NORM) {
        // Whole-head reduction: requires blockDim.x == v_dim == 128, i.e. the
        // four warps the parent's `norm_sums[4]` and `col / 32` assume.
        float local_sq = norm_acc;
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
        __shared__ float norm_sums[4];
        if (col % 32 == 0) norm_sums[col / 32] = local_sq;
        __syncthreads();
        if (col == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += norm_sums[w];
            norm_sums[0] = total;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                H[j * v_dim + col] *= scale;
            }
        }
    }

    float inv_sqrt_d = rsqrtf((float)k_dim);
    out_ptr[col] = q_dot * inv_sqrt_d;
}

// Strided FP32 GDN decode — Hopper twin of
// `gated_delta_rule_decode_f32_strided`. Identical parameter list, identical
// results, including the SSM_STATE_MAX_NORM clamp.
//
// Launch: grid = (1, num_v_heads, batch_size), block = (128, 1, 1).
extern "C" __global__ void gated_delta_rule_decode_f32_strided_hopper(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int out_stride
) {
    if (k_dim > GDN_H_MAX_KD) return;

    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* k_ptr = key + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* v_ptr = value + (unsigned long long)b * v_stride + vh * v_dim;

    float g_raw = gate[(unsigned long long)b * gb_stride + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[(unsigned long long)b * gb_stride + vh];

    __shared__ float smem_k[GDN_H_MAX_KD];
    __shared__ float smem_q[GDN_H_MAX_KD];

    float* out_ptr = output + (unsigned long long)b * out_stride + vh * v_dim;

    gdn_decode_hopper_body<true>(
        H, q_ptr, k_ptr, v_ptr, g, bt, out_ptr, k_dim, v_dim, col, smem_k, smem_q
    );
}

// Contiguous FP32 GDN decode — Hopper twin of `gated_delta_rule_decode_f32`,
// the C=1 kernel. Its parent sits above the SSM_STATE_NORM_ENABLED define, so
// it carries no clamp and this twin is free to tile columns.
//
// The parent indexes Q/K/V/gate/output inline; those expressions are exactly
// the strided form with qk_stride = num_k_heads*k_dim, v_stride = out_stride
// = num_v_heads*v_dim, gb_stride = num_v_heads, so the body is shared rather
// than copied.
//
// Launch: grid = (ceil(v_dim / block.x), num_v_heads, batch_size).
extern "C" __global__ void gated_delta_rule_decode_f32_hopper(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    if (k_dim > GDN_H_MAX_KD) return;

    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((unsigned long long)(b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (unsigned long long)(b * num_k_heads + kh) * k_dim;
    const float* k_ptr = key + (unsigned long long)(b * num_k_heads + kh) * k_dim;
    const float* v_ptr = value + (unsigned long long)(b * num_v_heads + vh) * v_dim;

    const float g = fminf(fmaxf(gate[b * num_v_heads + vh], 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[b * num_v_heads + vh];

    __shared__ float smem_k[GDN_H_MAX_KD];
    __shared__ float smem_q[GDN_H_MAX_KD];

    float* out_ptr = output + (unsigned long long)(b * num_v_heads + vh) * v_dim;

    gdn_decode_hopper_body<false>(
        H, q_ptr, k_ptr, v_ptr, g, bt, out_ptr, k_dim, v_dim, col, smem_k, smem_q
    );
}
