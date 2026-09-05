// SPDX-License-Identifier: AGPL-3.0-only
//
// Gate self-test fixture for the LEDGER, not for an ISA gap: one translation
// unit whose ptxas rejection names TWO entry functions.
//
// ptxas reports one `error   :` line per rejected entry function. The ledger
// used to keep only the first, so a file with several rejected entries was
// counted once and the Failures table under-reported the work: 22 files
// standing in for 42 rejected entries. This is the smallest thing that
// reproduces that shape, so `errors`, `error_count` and
// `summary.rejected_entries` have something to be checked against.
//
// The rejection is a static `__shared__` array over the per-entry shared
// limit. It is a LIMIT, not an instruction, so unlike the per-arch negative
// fixtures next to this one it is rejected on every architecture Atlas
// targets and there is no arch table to keep current. Dynamic shared memory
// would not do -- `extern __shared__` is sized at launch and ptxas never sees
// a size to reject.
//
// The size is 0x40004, NOT the 48 KiB (0xc000) that the older ptxas message
// quotes: on these architectures the static limit ptxas enforces is the whole
// opt-in per-block allowance, and 49156 bytes assembles fine for sm_90a
// (measured). 256 KiB + 4 clears every limit below.
//
// The store through a `volatile` pointer is load-bearing: without a side
// effect the front end drops the array and ptxas has nothing to reject.
//
// MEASURED with CUDA 13.0.88 (nvcc/ptxas V13.0.88, aarch64), 2026-09-05.
// `nvcc --ptx` PASSES for all three -- the front end does not check the
// limit -- and `ptxas` emits exactly two `error   :` lines:
//
//   -arch=sm_90a    FAIL  "uses too much shared data (0x40004 bytes, 0x38c00 max)"
//   -arch=sm_100a   FAIL  (same, 0x38c00 max)
//   -arch=sm_120a   FAIL  (same, 0x18c00 max)
//
// The third entry point is valid and must stay valid: a file in which
// EVERYTHING fails would not distinguish "one error per entry function" from
// "one error per file", which is the whole point of the fixture.

// 0x40004 -- four bytes over 256 KiB, past every per-block shared limit above.
#define ATLAS_GATE_OVERSIZED_SHARED 262148

extern "C" __global__ void atlas_gate_two_entries_bad_a(char *out) {
  __shared__ char buf[ATLAS_GATE_OVERSIZED_SHARED];
  volatile char *p = buf;
  p[threadIdx.x] = (char)(threadIdx.x + 1);
  __syncthreads();
  out[blockIdx.x] = p[threadIdx.x];
}

extern "C" __global__ void atlas_gate_two_entries_bad_b(char *out) {
  __shared__ char buf[ATLAS_GATE_OVERSIZED_SHARED];
  volatile char *p = buf;
  p[threadIdx.x] = (char)(threadIdx.x + 2);
  __syncthreads();
  out[blockIdx.x] = p[threadIdx.x];
}

extern "C" __global__ void atlas_gate_two_entries_good(const float *in,
                                                       float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * 3.0f - 1.0f;
  }
}
