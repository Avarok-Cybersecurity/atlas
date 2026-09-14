// SPDX-License-Identifier: AGPL-3.0-only
//
// Gate self-test, negative half: a kernel that MUST NOT compile for Hopper.
//
// `redux.sync.max.abs.f32` is the floating-point warp reduction added with
// Blackwell datacenter (sm_100a). Hopper's `redux.sync` handles integers only.
//
// Written as inline PTX on purpose: nvcc passes inline asm through to the PTX
// unexamined, so the rejection comes from ptxas — the stage a front-end-only
// check (a `#if __CUDA_ARCH__ ... #error` fixture) would never reach, and the
// one that decides whether a kernel can actually be assembled for a device.
//
// MEASURED with CUDA 13.0.88 (nvcc/ptxas V13.0.88, aarch64), 2026-09-04:
//
//   -arch=sm_90a   FAIL  "Instruction 'redux.f32' not supported on .target 'sm_90a'"
//   -arch=sm_100a  PASS
//   -arch=sm_120a  FAIL  (consumer Blackwell has no redux.f32 either)
//
// So this fixture is a valid negative for every arch EXCEPT sm_100a. The gate
// asserts the failure rather than assuming it, and if someone points the gate
// at sm_100a the self-test reports `bad_failed: false` and the gate refuses to
// publish results — which is the correct outcome: at that point the fixture,
// not the tree, is what is broken.
//
// A checker that has never failed has never been tested. If this file starts
// PASSING for sm_90a, the gate is not checking what it claims to.

extern "C" __global__ void atlas_gate_selftest_bad(float *inout) {
  float v = inout[0];
  float r;
  asm volatile("redux.sync.max.abs.f32 %0, %1, 0xffffffff;" : "=f"(r) : "f"(v));
  inout[0] = r;
}
