// SPDX-License-Identifier: AGPL-3.0-only
// SCALE Phase-0 probe B (highest risk): FP8 E4M3 pack/unpack +
// cvt.rn.satfinite.e4m3x2.f32 + m16n8k32.e4m3 tensor-core MMA + cp.async
// double-buffer. These are exactly the Atlas FP8/NVFP4 hotspots Spectral
// flagged (MMA→MFMA not yet auto-permuted; AMD e4m3 bit-layout may differ on
// older parts). Compile-only signal first; the numeric round-trip vs a CPU
// reference runs once a gfx1151 runtime exists, to answer "does gfx1151 e4m3
// match NVIDIA bit-for-bit".

#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>

// BF16x?/F32 → E4M3 pack via the NVIDIA PTX cvt Atlas uses in w4a16_gemm.cu.
__device__ __forceinline__ unsigned f32x4_to_e4m3x4(float f0, float f1,
                                                     float f2, float f3) {
    unsigned short h0, h1;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(h0) : "f"(f1), "f"(f0));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(h1) : "f"(f3), "f"(f2));
    return ((unsigned)h1 << 16) | (unsigned)h0;
}

// E4M3 → f32 via the storage-cast intrinsic Atlas uses in
// paged_decode_attn_fp8.cu / moe_fp8_grouped_gemm.cu.
__device__ __forceinline__ float e4m3_to_f32(unsigned char b) {
    __nv_fp8_storage_t s = (__nv_fp8_storage_t)b;
    return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3));
}

__global__ void e4m3_mma_cpasync(const float* __restrict__ Af,
                                  const unsigned char* __restrict__ Bq,
                                  float* __restrict__ C) {
    __shared__ unsigned char smem[2][256];
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid % 32;

    // cp.async.cg double-buffered global→shared (Atlas attention prefill /
    // w4a16 pipeline pattern).
    unsigned dst = (unsigned)__cvta_generic_to_shared(&smem[0][tid * 4]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(Bq + tid * 4));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    unsigned a0 = f32x4_to_e4m3x4(Af[lane*4+0], Af[lane*4+1],
                                  Af[lane*4+2], Af[lane*4+3]);
    unsigned a1 = f32x4_to_e4m3x4(Af[lane*4+4], Af[lane*4+5],
                                  Af[lane*4+6], Af[lane*4+7]);
    unsigned a2 = a0, a3 = a1;
    unsigned b0 = *reinterpret_cast<const unsigned*>(&smem[0][lane*4]);
    unsigned b1 = b0;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));

    // Touch the unpack path so the intrinsic is exercised too.
    float dq = e4m3_to_f32(smem[0][lane]);
    if (lane == 0) C[0] = acc[0] + acc[1] + acc[2] + acc[3] + dq;
}

int main() {
    printf("e4m3_mma_cpasync_probe: host stub (compile-only Phase-0 probe)\n");
    return 0;
}
