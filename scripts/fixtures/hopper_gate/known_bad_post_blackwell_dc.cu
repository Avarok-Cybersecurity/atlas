// SPDX-License-Identifier: AGPL-3.0-only
//
// Gate self-test, negative half for sm_100a: a kernel that MUST NOT compile
// for Blackwell DATACENTER.
//
// `mma.sync ... .kind::mxf4nvf4.block_scale` is the WARP-LEVEL block-scaled
// NVFP4 MMA. It exists on consumer/GB10 Blackwell (sm_120a / sm_121a) and it
// does NOT exist on sm_100a, where block-scaled MMA is issued through the
// tensor-memory path (`tcgen05.mma`) instead. So the two Blackwell
// architectures are not ordered here: neither is a superset of the other, and
// a fixture that fails on one passes on the other.
//
// That is exactly what makes it the right negative for sm_100a. The gate's
// original fixture, `known_bad_post_hopper.cu`, uses `redux.sync.max.abs.f32`,
// which sm_100a is the ONE architecture to support — so at sm_100a it is a
// valid instruction and the self-test's failure path never executes.
//
// Written as inline PTX for the same reason as its sibling: nvcc passes inline
// asm through unexamined, so the rejection comes from ptxas — the stage a
// front-end-only fixture (`#if __CUDA_ARCH__ ... #error`) would never reach,
// and the one that decides whether a kernel can actually be assembled for a
// device.
//
// MEASURED with CUDA 13.0.88 (nvcc/ptxas V13.0.88, aarch64) on 2026-09-05:
//
//   -arch=sm_90a   FAIL  "Instruction 'mma with block scale' not supported"
//   -arch=sm_100a  FAIL  "Instruction 'mma with block scale' not supported on
//                         .target 'sm_100a'"          ← the case this exists for
//   -arch=sm_120a  PASS
//   -arch=sm_121a  PASS
//   -arch=sm_121f  PASS
//
// So this fixture is a valid negative for sm_90a and sm_100a, and INVALID for
// the sm_120/sm_121 family — the mirror image of `known_bad_post_hopper.cu`,
// which is valid everywhere EXCEPT sm_100a. Between them the gate has a
// working negative for every architecture Atlas currently targets; the gate
// picks per arch and refuses to run where it has none.
//
// The operand shape is the one `kernels/gb10/qwen3.6-35b-a3b/nvfp4/
// moe_w4a16_grouped_gemm.cu` issues, reduced to a single instruction: four f32
// accumulators, an 8-register A fragment, a 2-register B fragment, and the two
// scale operands with their (byte-id, thread-id) selectors.
//
// A checker that has never failed has never been tested. If this file starts
// PASSING for sm_100a, the gate is not checking what it claims to.

extern "C" __global__ void atlas_gate_selftest_bad_sm100(const unsigned int *in,
                                                         float *out) {
  unsigned int a0 = in[0], a1 = in[1], a2 = in[2], a3 = in[3];
  unsigned int b0 = in[4], b1 = in[5];
  unsigned int sfa = in[6], sfb = in[7];
  unsigned short bid_a = 0, tid_a = 0, bid_b = 0, tid_b = 0;
  float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},"
      "{%4,%5,%6,%7},"
      "{%8,%9},"
      "{%10,%11,%12,%13},"
      "{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(acc0), "=f"(acc1), "=f"(acc2), "=f"(acc3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(acc0),
        "f"(acc1), "f"(acc2), "f"(acc3), "r"(sfa), "h"(bid_a), "h"(tid_a),
        "r"(sfb), "h"(bid_b), "h"(tid_b));
  out[0] = acc0 + acc1 + acc2 + acc3;
}
