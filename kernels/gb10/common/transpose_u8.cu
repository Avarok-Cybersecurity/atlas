// SPDX-License-Identifier: AGPL-3.0-only

// Atlas uint8 matrix transpose kernel.
//
// Transposes a [rows, cols] uint8 matrix to [cols, rows]:
//   out[c * rows + r] = in[r * cols + c]
//
// Uses 32x32 shared memory tiles with +1 padding to avoid bank conflicts.
// Both reads and writes are coalesced within warps.
//
// Grid: (ceil(cols/32), ceil(rows/32))  Block: (32, 8)

#define TILE 32

extern "C" __global__ void transpose_u8(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    unsigned int rows,
    unsigned int cols
) {
    __shared__ unsigned char tile[TILE][TILE + 1]; // +1 avoids bank conflicts

    // Input tile coordinates
    unsigned int ix = blockIdx.x * TILE + threadIdx.x;
    unsigned int iy_base = blockIdx.y * TILE + threadIdx.y;

    // Load tile: each thread loads 4 rows (8 threads × 4 = 32 rows)
    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int r = iy_base + j;
        if (r < rows && ix < cols) {
            tile[threadIdx.y + j][threadIdx.x] = in[r * cols + ix];
        }
    }

    __syncthreads();

    // Output tile coordinates (swapped block indices)
    unsigned int ox = blockIdx.y * TILE + threadIdx.x;
    unsigned int oy_base = blockIdx.x * TILE + threadIdx.y;

    // Write transposed tile: coalesced writes
    #pragma unroll
    for (int j = 0; j < TILE; j += 8) {
        unsigned int c = oy_base + j;
        if (c < cols && ox < rows) {
            out[c * rows + ox] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
