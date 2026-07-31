// SPDX-License-Identifier: AGPL-3.0-only
// SCALE Phase-0 probe A: BF16 m16n8k16 tensor-core MMA + warp-shuffle reduction.
// Goal: does SCALE accept Atlas's two most pervasive NVIDIA constructs for
// gfx1151, and what does `--ptx` emit (text vs binary)? Compile-only signal;
// numeric correctness is validated once a GPU runtime is available.

#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

// Mirrors the reduction pattern in Atlas attention kernels: 32-lane
// __shfl_xor_sync tree reduction with a hardcoded full mask.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// Mirrors dense_gemm_tc.cu: a single m16n8k16 BF16 MMA via inline PTX.
__global__ void bf16_mma_shfl(const __nv_bfloat16* __restrict__ A,
                              const __nv_bfloat16* __restrict__ B,
                              float* __restrict__ C) {
    const unsigned tid = threadIdx.x;
    const unsigned lane_id = tid % 32;
    const unsigned warp_id = tid / 32;

    unsigned a0, a1, a2, a3, b0, b1;
    const unsigned* Ap = reinterpret_cast<const unsigned*>(A);
    const unsigned* Bp = reinterpret_cast<const unsigned*>(B);
    a0 = Ap[lane_id * 4 + 0];
    a1 = Ap[lane_id * 4 + 1];
    a2 = Ap[lane_id * 4 + 2];
    a3 = Ap[lane_id * 4 + 3];
    b0 = Bp[lane_id * 2 + 0];
    b1 = Bp[lane_id * 2 + 1];

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));

    float s = warp_reduce_sum(acc[0] + acc[1] + acc[2] + acc[3]);
    if (lane_id == 0) C[warp_id] = s;
}

int main() {
    printf("bf16_mma_shfl_probe: host stub (compile-only Phase-0 probe)\n");
    return 0;
}
