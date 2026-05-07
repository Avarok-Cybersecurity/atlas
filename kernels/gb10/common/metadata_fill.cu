// SPDX-License-Identifier: AGPL-3.0-only

// Small metadata preparation kernels for chunked paged prefill.

extern "C" __global__ void fill_slots_from_block_table(
    long long* __restrict__ slots,
    const unsigned int* __restrict__ block_table,
    const unsigned int start_pos,
    const unsigned int count,
    const unsigned int block_size
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const unsigned int pos = start_pos + idx;
        const unsigned int logical_block = pos / block_size;
        const unsigned int block_offset = pos % block_size;
        const unsigned int physical_block = block_table[logical_block];
        slots[idx] = (long long)((unsigned long long)physical_block * block_size + block_offset);
    }
}
