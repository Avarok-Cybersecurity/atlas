// SPDX-License-Identifier: AGPL-3.0-only
// Isolates the FP8 E4M3 tensor-core MMA from the cvt-pack helper. Inputs are
// pre-packed uint32 (4x e4m3 lanes) read from global memory — NO
// cvt.rn.satfinite.e4m3x2.f32, NO fp8 intrinsics. Answers the decisive
// question: can SCALE 1.7.0 codegen `mma.sync m16n8k32 .e4m3.e4m3.f32` for
// gfx1151 at all (vs the FP8-pack cvt, which we can do in software)?

#include <cstdint>
#include <cstdio>

__global__ void e4m3_mma_only(const unsigned* __restrict__ Aq,
                              const unsigned* __restrict__ Bq,
                              float* __restrict__ C) {
    const unsigned lane = threadIdx.x % 32;
    unsigned a0 = Aq[lane * 4 + 0];
    unsigned a1 = Aq[lane * 4 + 1];
    unsigned a2 = Aq[lane * 4 + 2];
    unsigned a3 = Aq[lane * 4 + 3];
    unsigned b0 = Bq[lane * 2 + 0];
    unsigned b1 = Bq[lane * 2 + 1];

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));

    if (lane == 0) C[0] = acc[0] + acc[1] + acc[2] + acc[3];
}

int main() {
    printf("e4m3_mma_only_probe: host stub (compile-only Phase-0 probe)\n");
    return 0;
}
