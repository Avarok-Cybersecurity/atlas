// SPDX-License-Identifier: AGPL-3.0-only
//
// Qwen3.6-27B DFlash drafter: head_dim=128.
// The common inferspark_prefill_paged_indirect defaults to HDIM=256;
// that causes corrupted attn_out — the kernel reads 256 elements per
// KV head when only 128 are valid, clobbering smem tiles with adjacent-
// head data and producing wrong attention scores and output values.
// This override defines HDIM=128 before the common macro + compute
// headers are included, so every compile-time tile/loop constant is
// correct for the 128-element head the drafter actually has.
#define HDIM 128
#include "../../common/inferspark_prefill_paged_indirect.cu"
