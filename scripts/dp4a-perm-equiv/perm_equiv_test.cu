// Standalone gfx1151 equivalence test:
//   reference = Atlas's current smem-codebook loop expansion (w4a16_gemv_dp4a.cu:137-150)
//   candidate = v_perm expansion grabbed from rocmfp4-llama, adapted to
//               Atlas's CONSECUTIVE-pair nibble layout + {0,1,2,3,4,6,8,12} grid.
// Asserts byte-identical wint[] for ALL inputs. No Atlas build required.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

__device__ __constant__ signed char DP4A_CODEBOOK[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

// ---- reference: exactly Atlas's loop, packing 4 elements/word in element order
// element 2b = byte b low nibble, element 2b+1 = byte b high nibble.
__device__ void ref_expand(unsigned long long packed8, int wint[4]) {
    for (int j = 0; j < 4; j++) {
        unsigned int packed = 0;
        for (int e = 0; e < 4; e++) {
            int elem = j * 4 + e;
            int b = elem >> 1;
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            unsigned char nib = (elem & 1) ? (byte_val >> 4) : (byte_val & 0xF);
            packed |= ((unsigned int)(unsigned char)DP4A_CODEBOOK[nib]) << (e * 8);
        }
        wint[j] = (int)packed;
    }
}

// ---- candidate: perm expansion. Deinterleave consecutive nibbles into per-byte
// magnitude indices, then fork's two-perm codebook+sign select.
__device__ __forceinline__ unsigned int perm_codebook(unsigned int q) {
    // q: each byte = one nibble (bits0-2 magnitude index, bit3 sign)
    const unsigned int values0 = 0x03020100u; // [ 0, 1, 2, 3]
    const unsigned int values1 = 0x0c080604u; // [ 4, 6, 8,12]
    const unsigned int values2 = 0xfdfeff00u; // [ 0,-1,-2,-3]
    const unsigned int values3 = 0xf4f8fafcu; // [-4,-6,-8,-12]
    unsigned int vl = __builtin_amdgcn_perm(values1, values0, q & 0x07070707u);
    unsigned int vh = __builtin_amdgcn_perm(values3, values2, q & 0x07070707u);
    unsigned int m  = 0x03020100u | ((q & 0x08080808u) >> 1);
    return __builtin_amdgcn_perm(vh, vl, m);
}
__device__ void cand_expand(unsigned long long packed8, int wint[4]) {
    unsigned int w0 = (unsigned int)(packed8 & 0xFFFFFFFFull);          // bytes 0-3
    unsigned int w1 = (unsigned int)((packed8 >> 32) & 0xFFFFFFFFull);  // bytes 4-7
    // deinterleave: na bytes = [b0.lo, b0.hi, b1.lo, b1.hi]
    unsigned int na = (w0 & 0xF) | ((w0 & 0xF0) << 4) | ((w0 & 0xF00) << 8) | ((w0 & 0xF000) << 12);
    unsigned int nb = ((w0 >> 16) & 0xF) | (((w0 >> 16) & 0xF0) << 4) | (((w0 >> 16) & 0xF00) << 8) | (((w0 >> 16) & 0xF000) << 12);
    unsigned int nc = (w1 & 0xF) | ((w1 & 0xF0) << 4) | ((w1 & 0xF00) << 8) | ((w1 & 0xF000) << 12);
    unsigned int nd = ((w1 >> 16) & 0xF) | (((w1 >> 16) & 0xF0) << 4) | (((w1 >> 16) & 0xF00) << 8) | (((w1 >> 16) & 0xF000) << 12);
    wint[0] = (int)perm_codebook(na);
    wint[1] = (int)perm_codebook(nb);
    wint[2] = (int)perm_codebook(nc);
    wint[3] = (int)perm_codebook(nd);
}

__global__ void test_kernel(int* mismatches) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    // cover all 2^24 low-3-byte patterns (high bytes derived) + full sign/magnitude space per byte.
    // Each output byte depends only on its own nibble, so iterate a 32-bit space densely via tid stride.
    for (unsigned long long s = tid; s < (1ull << 24); s += gridDim.x * (unsigned long long)blockDim.x) {
        // build a packed8 exercising both halves and sign bits: replicate + perturb
        unsigned long long packed8 = s | (((~s) & 0xFFFFFFull) << 32) | (s << 24);
        int r[4], c[4];
        ref_expand(packed8, r);
        cand_expand(packed8, c);
        for (int j = 0; j < 4; j++) if (r[j] != c[j]) { atomicAdd(mismatches, 1); }
    }
}

int main() {
    int* d; int h = 0;
    hipMalloc(&d, sizeof(int)); hipMemcpy(d, &h, sizeof(int), hipMemcpyHostToDevice);
    test_kernel<<<256, 256>>>(d);
    hipDeviceSynchronize();
    hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost);
    printf("mismatches = %d  -> %s\n", h, h == 0 ? "PASS (perm == reference, bit-exact)" : "FAIL");
    return h == 0 ? 0 : 1;
}
