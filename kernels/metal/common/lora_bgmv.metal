// SPDX-License-Identifier: AGPL-3.0-only
//
// STUB. Metal per-request LoRA routing (bgmv) is not implemented; the metal
// build is broken on the current Linux dev box and cannot be typechecked here.
// The CUDA gb10 kernel (kernels/gb10/common/lora_bgmv.cu) is the target that
// matters for M2. Provided so the metal target's kernel set is not surprised
// by a missing stem; a real port mirrors the two gb10 kernels (shrink then
// expand+fold), byte-identical to N sequential single-adapter LoRA deltas.
#include <metal_stdlib>
using namespace metal;

kernel void lora_bgmv_shrink_stub() {}
kernel void lora_bgmv_expand_fold_stub() {}
