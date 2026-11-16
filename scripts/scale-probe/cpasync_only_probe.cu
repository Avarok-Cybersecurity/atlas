// SPDX-License-Identifier: AGPL-3.0-only
// Isolates the cp.async double-buffer pattern (Atlas attention-prefill /
// w4a16 pipeline). SCALE treats cp.async as sync on AMD and batches it;
// this confirms the PTX is accepted for gfx1151.

#include <cstdint>
#include <cstdio>

__global__ void cpasync_only(const unsigned char* __restrict__ G,
                             float* __restrict__ C) {
    __shared__ unsigned char smem[2][256];
    const unsigned tid = threadIdx.x;

    unsigned d0 = (unsigned)__cvta_generic_to_shared(&smem[0][tid * 4]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d0), "l"(G + tid * 4));
    unsigned d1 = (unsigned)__cvta_generic_to_shared(&smem[1][tid * 4]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(d1), "l"(G + 1024 + tid * 4));
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    if (tid == 0) C[0] = (float)(smem[0][0] + smem[1][0]);
}

int main() {
    printf("cpasync_only_probe: host stub (compile-only Phase-0 probe)\n");
    return 0;
}
