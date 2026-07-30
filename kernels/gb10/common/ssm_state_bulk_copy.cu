// SPDX-License-Identifier: AGPL-3.0-only

// Batched SSM state copy: every SSM layer in ONE launch.
//
// Replaces the per-layer `cuMemcpyDtoDAsync` loops on the DFlash/MTP
// checkpoint path (`async_chkpt.rs`). Measured on
// `~/dflash-logs/nsys-moe-cap4-0931`, MoE cap4, 573 steps:
//
//   stream 13 (default)    30.1 h_state + 30.1 conv per step  (pre-verify seed)
//   stream 54 (secondary)  43.9 h_state + 43.9 conv per step  (verify commit)
//
// i.e. ~148 separate D2D calls per step at ~8.78 us of driver time each,
// plus 30 serialized 2.1 MB copies that run at ~123 GB/s against a
// ~273 GB/s ceiling because each one is its own copy-engine op.
//
// The SSM pools are one allocation PER LAYER with a fixed
// `base[layer] + offset` slot layout (`ssm_pool.rs`), so the per-layer
// base pointers are constant for the model's lifetime. They live in
// device memory and are uploaded once at pool construction; a single
// launch then covers every layer, and the caller passes only the two
// byte offsets that vary per call.
//
// Grid: (blocks_per_layer, n_layers, 1)  Block: (256, 1, 1)
//
// `n_u4` = bytes / 16. Every SSM state buffer is cudaMalloc'd (256-byte
// aligned) and every size is a multiple of 16 (h_state = nv*vd*kd*4,
// conv_state = conv_dim*d_conv*4), so the uint4 path is always valid.
// The caller falls back to the memcpy loop if that ever stops holding.
//
// Bit-identical to the loop it replaces: same bytes, same direction, same
// stream. Kill-switch on the host side: ATLAS_SSM_BULK_COPY=0.

extern "C" __global__ void ssm_state_bulk_copy(
    const unsigned long long* __restrict__ src_bases,
    const unsigned long long* __restrict__ dst_bases,
    unsigned long long src_off,
    unsigned long long dst_off,
    unsigned int n_u4
) {
    const unsigned int layer = blockIdx.y;

    const uint4* __restrict__ src =
        (const uint4*)(src_bases[layer] + src_off);
    uint4* __restrict__ dst = (uint4*)(dst_bases[layer] + dst_off);

    const unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_u4;
         i += stride) {
        dst[i] = src[i];
    }
}
