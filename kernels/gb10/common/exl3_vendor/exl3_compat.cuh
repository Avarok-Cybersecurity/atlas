// SPDX-License-Identifier: MIT
//
// Vendored from turboderp's ExLlamaV3 (https://github.com/turboderp-org/exllamav3)
// Copyright (c) 2025 turboderp — MIT license.
// Freestanding subset of upstream util.h + util.cuh: only the pieces the EXL3
// matmul device code needs (no torch, no cublas, no host error macros).
// Snapshot originals: .research/exllamav3_ref/util.h, util.cuh.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// util.h
#define CEIL_DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// util.cuh
typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    // upstream writes `__device__ half4() = default;` — the annotation is
    // ignored on an explicitly-defaulted ctor and nvcc warns (20012-D), which
    // the strict Atlas kernel build promotes to an error
    half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};
