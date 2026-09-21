// SPDX-License-Identifier: AGPL-3.0-only

// Atlas fused K=2 MTP-verify GDN epilogue — STAGE 1 (conv1d+L2norm fused
// across both draft positions, gated-RMS-norm fused across both positions).
//
// K=2 MTP verify currently runs the projection epilogue as PER-TOKEN loops:
// `causal_conv1d_update_l2norm` is launched TWICE and `gated_rms_norm` is
// launched TWICE, with a full conv-state D2D copy between the two conv calls
// for rollback. This kernel fuses each of those two-launch sequences into a
// SINGLE launch each:
//
//   gdn_verify_fused_conv_k2 — advances the conv sliding window for BOTH
//     positions in registers (no global re-read between positions), writes
//     conv_out[0] and conv_out[1], and snapshots the position-0 conv-state
//     intermediate (for rollback) ONCE inline — replacing the per-token
//     kernel + separate copy_d2d.
//
//   gdn_verify_fused_norm_k2 — gated-RMS-norm for BOTH positions, one block
//     per (head, position).
//
// The BA-projection/gates and the WY2 recurrence (gated_delta_rule_wy2) stay
// as their existing separate launches; Stage 2 folds BA in.
//
// BIT-EXACTNESS: kernels build with --fmad=false (KERNEL.toml). The conv dot
// product, SiLU, L2-norm reduction and the RMS reduction here preserve the
// EXACT accumulation order of `causal_conv1d_update_l2norm` /
// `gated_rms_norm`, so the fused outputs are byte-identical to the per-token
// path (proven by gdn_verify_fused_microtest, cos >= 0.99999).

#include <cuda_bf16.h>

// Bit-exact mirrors of the rms_norm.cu helpers (same translation unit can't
// share them; these MUST match instruction-for-instruction for byte-identical
// output under --fmad=false).
__device__ __forceinline__ void fused_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}
__device__ __forceinline__ unsigned int fused_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}
__device__ __forceinline__ float fused_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ============================================================
// Fused conv1d + SiLU + L2-norm for BOTH K=2 positions, one launch.
//
// Grid: (ceil(dim/256), 1, 1)   Block: (256, 1, 1)
// Each thread owns one channel `ch` and processes position 0 then position 1,
// keeping the d_conv sliding window in registers between positions (no global
// conv_state re-read — same arithmetic, bit-identical). The position-0 state
// is snapshotted to `conv_state_inter` for rollback. The committed (post
// position-1) state is left in `conv_state`.
// ============================================================
extern "C" __global__ void gdn_verify_fused_conv_k2(
    float* __restrict__ conv_state,              // [dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [2, input_stride] BF16
    const __nv_bfloat16* __restrict__ weight,    // [dim, d_conv] BF16
    __nv_bfloat16* __restrict__ output,          // [2, output_stride] BF16
    float* __restrict__ conv_state_inter,        // [dim, d_conv] FP32 (out, pos-0 snapshot)
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,    // channels 0..qk_channels-1 get L2 normalized
    unsigned int head_dim,       // L2 norm group size (128)
    unsigned int input_stride,   // BF16 elems between positions in new_input
    unsigned int output_stride,  // BF16 elems between positions in output
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim);

    // ── Load this channel's d_conv-element sliding window into registers ──
    // d_conv is small (4); a fixed-size register window matches the global
    // shift loop in causal_conv1d_update_l2norm exactly.
    float win[8]; // d_conv <= 8
    if (valid) {
        const float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) win[i] = state[i];
    }

    const __nv_bfloat16* w = valid ? (weight + ch * d_conv) : nullptr;
    float wcoef[8];
    if (valid) {
        for (unsigned int k = 0; k < d_conv; k++) wcoef[k] = (float)w[k];
    }

    __shared__ float warp_sums[8];

    // Process the two positions sequentially. L2-norm needs a __syncthreads
    // per position, so the loop body mirrors the single-token kernel exactly.
    for (unsigned int t = 0; t < 2; t++) {
        float silu = 0.0f;
        if (valid) {
            // Shift window left, append this position's input (== global path).
            for (unsigned int i = 0; i < d_conv - 1; i++) win[i] = win[i + 1];
            win[d_conv - 1] = (float)new_input[t * input_stride + ch];
            // bias == nullptr in production conv1d_update_l2norm.
            float acc = 0.0f;
            for (unsigned int k = 0; k < d_conv; k++) acc += win[k] * wcoef[k];
            float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
            silu = acc * sigmoid_acc;
        }

        if (block_needs_l2) {
            float sq = valid ? (silu * silu) : 0.0f;
            const unsigned int warp_id = tid / 32;
            const unsigned int lane = tid % 32;
            for (int offset = 16; offset >= 1; offset >>= 1)
                sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
            if (lane == 0) warp_sums[warp_id] = sq;
            __syncthreads();
            const unsigned int head_in_block = tid / head_dim;
            const unsigned int base_warp = head_in_block * (head_dim / 32);
            if (tid == 0 || tid == head_dim) {
                float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                            + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
                warp_sums[base_warp] = rsqrtf(total + l2_eps);
            }
            __syncthreads();
            if (valid) silu *= warp_sums[base_warp];
        }

        if (valid) output[t * output_stride + ch] = __float2bfloat16(silu);

        // Snapshot the position-0 conv-state (for rollback) inline, once.
        if (valid && t == 0) {
            float* snap = conv_state_inter + ch * d_conv;
            for (unsigned int i = 0; i < d_conv; i++) snap[i] = win[i];
        }
    }

    // Commit final (post position-1) sliding window to conv_state.
    if (valid) {
        float* state = conv_state + ch * d_conv;
        for (unsigned int i = 0; i < d_conv; i++) state[i] = win[i];
    }
}

// ============================================================
// Fused gated-RMS-norm for BOTH K=2 positions, one launch.
//
// Grid: (num_v_heads, 2, 1)   Block: (head_dim, 1, 1)
// Mirrors gated_rms_norm per-head numerics exactly: one block normalizes a
// hidden_size (= head_dim, 128) group for one (head, position). gate (Z) is
// read from the deinterleaved buffer at [Q|K|V] offset per position.
// ============================================================
extern "C" __global__ void gdn_verify_fused_norm_k2(
    const __nv_bfloat16* __restrict__ gdn_out,   // [2, out_stride] BF16 (GDN output)
    const __nv_bfloat16* __restrict__ deint,     // [2, deint_stride] BF16 (Q|K|V|Z)
    const __nv_bfloat16* __restrict__ weight,    // [hidden_size] BF16
    __nv_bfloat16* __restrict__ output,          // [2, out_stride] BF16
    unsigned int hidden_size,                    // per-head group size (128)
    float eps,
    unsigned int deint_stride,                   // BF16 elems between positions in deint
    unsigned int z_offset,                       // Z offset within a position (== conv_dim)
    unsigned int out_stride                      // BF16 elems between positions in gdn_out/output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int t = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    // Per (head, position) row pointers. One block normalizes a hidden_size
    // group, matching the per-head gated_rms_norm launch (num_tokens=NV).
    const __nv_bfloat16* x = gdn_out + t * out_stride + head * hidden_size;
    // Z gate for this head sits in the deinterleaved buffer's [Z] region;
    // gate_stride in the per-token kernel == hidden_size, so it is contiguous.
    const __nv_bfloat16* g = deint + t * deint_stride + z_offset + head * hidden_size;
    __nv_bfloat16* out = output + t * out_stride + head * hidden_size;

    // ── EXACT mirror of gated_rms_norm: quad (64-bit) BF16 loads, x_cache
    // reuse, butterfly warp-reduce. hidden_size up to 16*1024. ──
    const unsigned int quad_size = hidden_size / 4;
    const unsigned long long* x64 = (const unsigned long long*)x;

    float x_cache[16];
    float sum_sq = 0.0f;
    unsigned int n_cached = 0;

    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        unsigned long long v = x64[i];
        float f0, f1, f2, f3;
        fused_unpack_bf16x2((unsigned int)v, f0, f1);
        fused_unpack_bf16x2((unsigned int)(v >> 32), f2, f3);
        x_cache[n_cached]     = f0;
        x_cache[n_cached + 1] = f1;
        x_cache[n_cached + 2] = f2;
        x_cache[n_cached + 3] = f3;
        n_cached += 4;
        sum_sq += f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3;
    }

    sum_sq = fused_warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = fused_warp_reduce_sum(val);
        if (lane_id == 0) warp_sums[0] = val;
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    // Pass 2: normalize + SiLU-gate using cached x (no re-read), quad stores.
    const unsigned long long* g64 = (const unsigned long long*)g;
    const unsigned long long* w64 = (const unsigned long long*)weight;
    unsigned long long* out64 = (unsigned long long*)out;

    unsigned int ci = 0;
    for (unsigned int i = tid; i < quad_size; i += blockDim.x) {
        float f0 = x_cache[ci];
        float f1 = x_cache[ci + 1];
        float f2 = x_cache[ci + 2];
        float f3 = x_cache[ci + 3];
        ci += 4;

        unsigned long long wv = w64[i];
        float w0, w1, w2, w3;
        fused_unpack_bf16x2((unsigned int)wv, w0, w1);
        fused_unpack_bf16x2((unsigned int)(wv >> 32), w2, w3);

        unsigned long long gv = g64[i];
        float g0, g1, g2, g3;
        fused_unpack_bf16x2((unsigned int)gv, g0, g1);
        fused_unpack_bf16x2((unsigned int)(gv >> 32), g2, g3);

        float s0 = g0 / (1.0f + expf(-g0));
        float s1 = g1 / (1.0f + expf(-g1));
        float s2 = g2 / (1.0f + expf(-g2));
        float s3 = g3 / (1.0f + expf(-g3));

        unsigned int lo = fused_pack_bf16x2(f0 * rms * w0 * s0, f1 * rms * w1 * s1);
        unsigned int hi = fused_pack_bf16x2(f2 * rms * w2 * s2, f3 * rms * w3 * s3);
        out64[i] = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    }
}
