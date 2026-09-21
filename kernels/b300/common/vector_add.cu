// SPDX-License-Identifier: AGPL-3.0-only

// Atlas test kernel: vector add (C = A + B)
// Used to verify the Rust → nvcc → CUDA → Python round-trip.

extern "C" __global__ void vector_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
