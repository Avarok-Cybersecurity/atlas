// Standalone timing harness for qwen3.8-flash-next's `hc_pre` collapse.
//
// ONE question: is `qhc_collapse` superlinear in T (rows)? The batched verify
// runs it at T = R = sum(ks) instead of T = K, and the step profile said the
// forward is 93% of the cost and gets WORSE per row (1 seq / 3 rows ~57 ms vs
// 2 seqs / 6 rows ~203 ms).
//
// Why a microtest and not ncu: ncu's kernel replay snapshots written buffers
// per pass and has hard-rebooted this box against a large rank. This allocates
// a few hundred MB and answers the question directly.
//
// Shapes are the real ones: H=2560, hc=4, rank=320 (from the serve log
// "hc 4 streams x rank 320" and config hidden_size=2560).
//
// The kernel body below is COPIED from
// kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu so the timing
// reflects the shipped code; values are garbage, only the shape and the memory
// traffic matter.

#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

#define QHC_MAX_MULT 8

__device__ __forceinline__ float qhc_silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float qhc_sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

extern "C" __global__ void hc_pre_copy(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_norm_w,
    const __nv_bfloat16* __restrict__ down_w,
    const __nv_bfloat16* __restrict__ up_w,
    const __nv_bfloat16* __restrict__ inject_w,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ inj_out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int rank,
    const float norm_eps
) {
    extern __shared__ float smem[];
    const unsigned int t   = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int H   = hidden_size;
    const unsigned int hc  = hc_mult;
    const unsigned int hc_dim = hc * H;
    const float inv_hc = 1.0f / (float)hc;
    const unsigned int lane  = tid & 31u;
    const unsigned int warp  = tid >> 5;
    const unsigned int warps = blockDim.x >> 5;

    float* smem_normed = smem;              // [hc_dim]
    float* smem_low    = smem + hc_dim;     // [rank]

    // ── RMS over the whole stream vector, then normalize ──
    const float* sx = streams + (size_t)t * hc_dim;
    float ss = 0.0f;
    for (unsigned int i = tid; i < hc_dim; i += blockDim.x) {
        const float v = sx[i];
        ss += v * v;
    }
    __shared__ float red[32];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (unsigned int w = 0; w < warps; ++w) tot += red[w];
        red[0] = rsqrtf(tot / (float)hc_dim + norm_eps);
    }
    __syncthreads();
    const float inv_rms = red[0];
    for (unsigned int i = tid; i < hc_dim; i += blockDim.x) {
        smem_normed[i] = sx[i] * inv_rms * (float)hc_norm_w[i];
    }
    __syncthreads();

    // ── down projection: warp per rank row ──
    for (unsigned int r = warp; r < rank; r += warps) {
        const __nv_bfloat16* row = down_w + (size_t)r * hc_dim;
        float acc = 0.0f;
        for (unsigned int i = lane; i < hc_dim; i += 32) acc += (float)row[i] * smem_normed[i];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0) smem_low[r] = qhc_silu(acc * inv_hc);
    }
    __syncthreads();

    // ── up + gate + mean over streams — THE SUSPECT ──
    // Every block walks the WHOLE up_w ([hc*H, rank]) for its own token.
    __nv_bfloat16* y = y_out + (size_t)t * H;
    for (unsigned int d = tid; d < H; d += blockDim.x) {
        float mixed = 0.0f;
        for (unsigned int s2 = 0; s2 < hc; ++s2) {
            const unsigned int i = s2 * H + d;
            const __nv_bfloat16* urow = up_w + (size_t)i * rank;
            float acc = 0.0f;
            for (unsigned int r = 0; r < rank; ++r) acc += (float)urow[r] * smem_low[r];
            mixed += qhc_sigmoid(acc) * smem_normed[i];
        }
        y[d] = __float2bfloat16(mixed * inv_hc);
    }

    if (inject_w != nullptr) {
        __syncthreads();
        for (unsigned int s2 = warp; s2 < hc; s2 += warps) {
            const __nv_bfloat16* row = inject_w + (size_t)s2 * hc_dim;
            float acc = 0.0f;
            for (unsigned int i = lane; i < hc_dim; i += 32) acc += (float)row[i] * smem_normed[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) inj_out[(size_t)t * hc + s2] = 2.0f * qhc_sigmoid(acc * inv_hc);
        }
    }
}

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)

int main() {
    const unsigned int H = 2560, hc = 4, rank = 320;
    const unsigned int hc_dim = hc * H;
    const unsigned int T_MAX = 16;

    float *streams, *inj_out;
    __nv_bfloat16 *hc_norm_w, *down_w, *up_w, *inject_w, *y_out;
    CK(cudaMalloc(&streams,   (size_t)T_MAX * hc_dim * sizeof(float)));
    CK(cudaMalloc(&inj_out,   (size_t)T_MAX * hc * sizeof(float)));
    CK(cudaMalloc(&hc_norm_w, (size_t)hc_dim * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&down_w,    (size_t)rank * hc_dim * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&up_w,      (size_t)hc_dim * rank * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&inject_w,  (size_t)hc * hc_dim * sizeof(__nv_bfloat16)));
    CK(cudaMalloc(&y_out,     (size_t)T_MAX * H * sizeof(__nv_bfloat16)));
    CK(cudaMemset(streams, 0x3c, (size_t)T_MAX * hc_dim * sizeof(float)));
    CK(cudaMemset(down_w,  0x3c, (size_t)rank * hc_dim * sizeof(__nv_bfloat16)));
    CK(cudaMemset(up_w,    0x3c, (size_t)hc_dim * rank * sizeof(__nv_bfloat16)));
    CK(cudaMemset(inject_w,0x3c, (size_t)hc * hc_dim * sizeof(__nv_bfloat16)));
    CK(cudaMemset(hc_norm_w,0x3c,(size_t)hc_dim * sizeof(__nv_bfloat16)));

    const size_t smem = ((size_t)hc_dim + rank) * sizeof(float);
    CK(cudaFuncSetAttribute(hc_pre_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    const double up_bytes = (double)hc_dim * rank * 2.0;  // up_w, per BLOCK
    printf("H=%u hc=%u rank=%u   up_w=%.1f MB   smem=%.1f KB\n",
           H, hc, rank, up_bytes / 1e6, smem / 1024.0);
    printf(" T   ms/call    us/row   vs T=1/row   implied up_w GB/s\n");

    double per_row_t1 = 0.0;
    for (unsigned int T : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u}) {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        for (int i = 0; i < 20; ++i)   // warm
            hc_pre_copy<<<T, 256, smem>>>(streams, hc_norm_w, down_w, up_w, inject_w,
                                          y_out, inj_out, H, hc, rank, 1e-6f);
        CK(cudaDeviceSynchronize());
        const int ITERS = 200;
        CK(cudaEventRecord(a));
        for (int i = 0; i < ITERS; ++i)
            hc_pre_copy<<<T, 256, smem>>>(streams, hc_norm_w, down_w, up_w, inject_w,
                                          y_out, inj_out, H, hc, rank, 1e-6f);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0.0f; CK(cudaEventElapsedTime(&ms, a, b));
        const double per_call = ms / ITERS;
        const double per_row  = per_call * 1000.0 / T;
        if (T == 1) per_row_t1 = per_row;
        printf("%3u  %8.4f  %8.2f   %8.2fx   %10.1f\n",
               T, per_call, per_row, per_row / per_row_t1,
               (up_bytes * T) / (per_call * 1e-3) / 1e9);
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
    return 0;
}
