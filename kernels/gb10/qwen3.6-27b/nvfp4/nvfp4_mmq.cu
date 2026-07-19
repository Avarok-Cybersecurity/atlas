// SPDX-License-Identifier: AGPL-3.0-only
//
// ATLAS FFN prefill GEMM via vendored llama.cpp NVFP4 W4A4 MMQ (Blackwell block-scale
// MMA: mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64, e2m1×e2m1, ue4m3 scales).
// De-risk bench (libggml, GB10): 114 TFLOP/s gate/up · 88 down at the 27B FFN shapes — vs
// faith2 int8 44 and the hand-written w4a4_gemm 52 (see MMQ_PORT_HANDOFF.md).
// extern-C entries launched from Rust (ops/nvfp4_mmq.rs). Same conventional 2D tiling as
// the Q4_K wrapper (q4k_mmq.cu): no MoE ids, single channel; prefill shapes have thousands
// of tiles >> 48 SMs so stream-k buys ~nothing. dst is BF16 [M,N] (fused store).
//
// Weight format: llama block_nvfp4 = { uint8 d[4] (UE4M3 per-16 scales); uint8 qs[32]
// (e2m1 nibbles: byte j of sub-block s = val[16s+j] | val[16s+8+j]<<4) } per 64 weights.
// atlas_nvfp4_repack converts the checkpoint layout (packed [N,K/2] low=even/high=odd +
// e4m3 [N,K/16] row-major scales) into block_nvfp4 rows — a pure bit shuffle, zero
// requantization. The per-tensor FP32 scale2 (and the ue4m3-vs-e4m3 decode convention
// factor) is folded downstream by the caller (see ops/nvfp4_mmq.rs).
// Vendored headers in q4k_vendor/ (pristine except quantize_impl.cuh worker extraction).
#include <cuda_bf16.h>
#include "q4k_vendor/mmq.cuh"
#include "q4k_vendor/quantize_impl.cuh"

// Conventional-tiling setup mirroring mul_mat_q's pre-VOLTA path, specialized: no ids,
// nchannels_y=nsamples_y=1 (blockIdx.z==0). Calls the existing __device__ process_tile.
template <int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_nvfp4_tile(
        const char * __restrict__ x, const int * __restrict__ y, __nv_bfloat16 * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int ncols_x,
        const int stride_row_x, const int ncols_y, const int stride_col_dst) {
    constexpr ggml_type type = GGML_TYPE_NVFP4;
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) break;
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int it = blockIdx.x;   // tile over nrows_x (N output features)
    const int jt = blockIdx.y;   // tile over ncols_dst (M tokens)

    // block_fp4_mmq has the same sizeof as block_q8_1_mmq (static_assert in mmq.cuh),
    // so the y tile offset formula is identical to the Q4_K wrapper's.
    const int offset_y   = jt*mmq_x*(int)(sizeof(block_fp4_mmq)/sizeof(int));
    const int offset_dst = jt*mmq_x*stride_col_dst + it*mmq_y;
    const int tile_x_max_i = nrows_x   - it*mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt*mmq_x - 1;
    const int offset_x = it*mmq_y*stride_row_x;
    const int kb0_stop = ncols_x / qk;   // number of K-blocks (qk = 64 for NVFP4)

    mul_mat_q_process_tile<type, mmq_x, need_check, /*fixup=*/false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

// mmq_x=128 entries (need_check = nrows_x not a multiple of mmq_y=128).
extern "C" __global__ void __launch_bounds__(256, 1) atlas_nvfp4_mmq128_nc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    atlas_nvfp4_tile<128, false>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) atlas_nvfp4_mmq128_wc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    atlas_nvfp4_tile<128, true>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}

// Activation quantizer: bf16 [ne1=M rows, ne00=K] -> block_fp4_mmq (e2m1 + ue4m3 group-16
// scales, ±2 exhaustive scale search). One thread per 16-value sub-block.
// grid (ne1, ceil(ne0/(16*128)), 1), block (128). Mirrors llama's host launcher.
extern "C" __global__ void atlas_nvfp4_quantize_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_nvfp4_worker<__nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}

// Weight repack: checkpoint NVFP4 (packed [N, K/2] e2m1 nibbles low=even k / high=odd k,
// scales [N, K/16] E4M3 bytes row-major) -> llama block_nvfp4 rows [N][K/64].
// Pure bit shuffle + scale byte copy: the e2m1 codes and e4m3 scale bytes are reused
// verbatim (both sides are OCP encodings; scale-decode convention handled by the caller's
// scale2 fold). One thread per 64-value output block.
extern "C" __global__ void atlas_nvfp4_repack(
        const uint8_t* __restrict__ packed, const uint8_t* __restrict__ scales,
        block_nvfp4* __restrict__ out, int n_rows, int k) {
    const int64_t nblocks = (int64_t) n_rows * (k / QK_NVFP4);
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;

    const int blocks_per_row = k / QK_NVFP4;
    const int row = (int)(b / blocks_per_row);
    const int kb  = (int)(b % blocks_per_row);

    const uint8_t* prow = packed + (int64_t) row * (k / 2);
    const uint8_t* srow = scales + (int64_t) row * (k / 16);
    block_nvfp4 dst;

#pragma unroll
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {          // 4 sub-blocks of 16
        const int k0 = kb * QK_NVFP4 + s * QK_NVFP4_SUB;
        dst.d[s] = srow[k0 / 16];
#pragma unroll
        for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {             // 8 output bytes
            const int ka = k0 + j;                                // -> low nibble
            const int kb2 = k0 + 8 + j;                           // -> high nibble
            const uint8_t na = (prow[ka >> 1] >> ((ka & 1) * 4)) & 0xF;
            const uint8_t nb = (prow[kb2 >> 1] >> ((kb2 & 1) * 4)) & 0xF;
            dst.qs[s * 8 + j] = (int8_t)(na | (nb << 4));
        }
    }
    out[b] = dst;
}

// SiLU(gate*gs)*(up*us): mirrors moe_silu_mul's math exactly (incl. the swiglu ±10 clamp)
// plus the per-projection scale2 fold — the MMQ GEMM output is missing the checkpoint's
// per-tensor FP32 scale2 (hardware applies only the per-16 e4m3 scales), so it is folded
// here, BEFORE the clamp/nonlinearity, where the values first become "true"-scaled.
// Contained duplicate of moe_silu_mul (5 lines of math) to avoid an ABI change on the
// shared kernel for this flag-gated path.
// In-place ×scale for the down-projection MMQ output (its scale2 has no SiLU-mul to
// ride; the consumer is the residual add). [M, H] bf16, ~0.3ms at M=4096 vs the ~6ms
// the MMQ down GEMM saves.
extern "C" __global__ void atlas_nvfp4_scale_bf16(
    __nv_bfloat16* __restrict__ data, float scale, unsigned int total_elements) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    data[idx] = __float2bfloat16(__bfloat162float(data[idx]) * scale);
}

extern "C" __global__ void atlas_nvfp4_silu_mul_scaled(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output, float gate_scale, float up_scale,
    unsigned int total_elements) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = __bfloat162float(gate[idx]) * gate_scale;
    float u = __bfloat162float(up[idx]) * up_scale;
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    output[idx] = __float2bfloat16(g * sigmoid_g * u);
}

// FUSED SiLU-mul + block_fp4_mmq quantize for the down-MMQ path (ATLAS_FFN_NVFP4_MMQ_DOWN).
// Replaces atlas_nvfp4_silu_mul_scaled + atlas_nvfp4_quantize_bf16: computes
// v = SiLU(clamp(gate·gs)) · clamp(up·us) for a 16-value group and quantizes it straight
// into the y-format the down MMQ consumes — the intermediate [M, inter] bf16 tensor is
// never written or re-read (saves ~2 full activation-tensor round-trips per layer; this
// traffic is exactly why the unfused down arm measured NEUTRAL). Same thread mapping and
// ue4m3 ±2 scale search as quantize_mmq_nvfp4_worker (one thread = one 16-value group);
// the value source is the SiLU-mul instead of a memory load. Grid (M, ceil(kpad/(16·128))),
// block 128. kpad = inter rounded up to 256.
extern "C" __global__ void atlas_nvfp4_silu_mul_quant(
        const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
        void* __restrict__ vy, float gate_scale, float up_scale,
        long ne00 /*inter*/, long ne0 /*kpad*/, int ne1 /*M rows*/) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) return;
    const int64_t i1 = blockIdx.x;
    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) return;

    const int64_t ib = k_block * ne1 + i1;
    block_fp4_mmq* yb = (block_fp4_mmq*) vy + ib;
    const int sub = (int) ((i0_base % QK_K) / QK_NVFP4_SUB);

    const float SWIGLU_LIMIT = 10.0f;
    float vals_raw[QK_NVFP4_SUB];
    float amax_raw = 0.0f;
    const int64_t base_idx = i1 * ne00;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        const int64_t i00 = i0_base + k;
        float v = 0.0f;
        if (i00 < ne00) {
            float g = __bfloat162float(gate[base_idx + i00]) * gate_scale;
            float u = __bfloat162float(up[base_idx + i00]) * up_scale;
            g = fminf(g, SWIGLU_LIMIT);
            u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
            v = g * (1.0f / (1.0f + __expf(-g))) * u;
        }
        vals_raw[k] = v;
        amax_raw = fmaxf(amax_raw, fabsf(v));
    }

    static constexpr int test_offsets[5] = {0, -1, 1, -2, 2};
    const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_raw / 6.0f);
    float best_err = FLT_MAX;
    uint8_t fp8_code = 0;
    float subblock_scale = 0.0f;
#pragma unroll
    for (int i = 0; i < 5; i++) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7e) continue;
        const uint8_t code = (uint8_t) test_code;
        const float test_scale = ggml_cuda_ue4m3_to_fp32(code);
        const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
        float cur_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals_raw[k];
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
            const float err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * test_scale;
            cur_err = fmaf(err_diff, err_diff, cur_err);
        }
        if (cur_err < best_err) {
            best_err = cur_err;
            fp8_code = code;
            subblock_scale = test_scale;
        }
    }

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    uint32_t q0 = 0, q1 = 0;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 0], inv_scale) << (8 * k);
        q0 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 8], inv_scale) << (8 * k + 4);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 4], inv_scale) << (8 * k);
        q1 |= (uint32_t) ggml_cuda_float_to_fp4_e2m1(vals_raw[k + 12], inv_scale) << (8 * k + 4);
    }
    uint32_t* yqs = reinterpret_cast<uint32_t*>(yb->qs);
    yqs[2 * sub + 0] = q0;
    yqs[2 * sub + 1] = q1;
    reinterpret_cast<uint8_t*>(yb->d4)[sub] = fp8_code;
#else
    NO_DEVICE_CODE;
#endif
}
