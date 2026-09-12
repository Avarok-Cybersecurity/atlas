// SPDX-License-Identifier: AGPL-3.0-only
//
// Gate self-test, positive half: a kernel that compiles for every SM Atlas
// targets. If THIS fails, the gate's toolchain discovery or flag handling is
// broken and no result it prints means anything.

extern "C" __global__ void atlas_gate_selftest_good(const float *in, float *out,
                                                    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * 2.0f + 1.0f;
  }
}
