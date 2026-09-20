// SPDX-License-Identifier: AGPL-3.0-only
//
// Batched per-expert uint8 matrix transpose for MoE weight relayout.
//
// Transposes one [rows, cols] uint8 matrix per expert from per-expert source
// pointers (`src_ptrs[expert]`) into per-expert destination pointers
// (`dst_ptrs[expert]`). Used to build a per-prefill `[K/2, N]` packed
// scratch for the coalesced down_proj GEMM kernel without keeping a
// persistent transposed copy. Decode keeps using the untransposed source.
//
// Both src and dst pointers may be NULL — for an EP-remote expert, the
// loader inserts NULL on this rank so the kernel exits early at block
// level. The corresponding scratch slot stays uninitialised and the
// downstream GEMM kernel skips that expert via its own NULL guard.
//
// Same 32×32 shared-memory tile pattern as transpose_u8.cu, with one
// extra grid.z dimension iterating experts.

#define TILE 32

extern "C" __global__ void moe_transpose_u8_batched(
    const unsigned long long* __restrict__ src_ptrs,
    const unsigned long long* __restrict__ dst_ptrs,
    unsigned int rows,
    unsigned int cols
) {
    const unsigned int expert = blockIdx.z;
    const unsigned char* src = (const unsigned char*)src_ptrs[expert];
    unsigned char* dst = (unsigned char*)dst_ptrs[expert];
    if (src == 0 || dst == 0) return;

    __shared__ unsigned char tile[TILE][TILE + 1];  // +1 avoids bank conflicts

    const unsigned int ix = blockIdx.x * TILE + threadIdx.x;
    const unsigned int iy_base = blockIdx.y * TILE + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int r = iy_base + j;
        if (r < rows && ix < cols) {
            tile[threadIdx.y + j][threadIdx.x] = src[r * cols + ix];
        }
    }

    __syncthreads();

    const unsigned int ox = blockIdx.y * TILE + threadIdx.x;
    const unsigned int oy_base = blockIdx.x * TILE + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int c = oy_base + j;
        if (c < cols && ox < rows) {
            dst[c * rows + ox] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
