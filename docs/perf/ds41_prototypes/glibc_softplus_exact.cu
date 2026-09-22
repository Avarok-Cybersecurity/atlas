// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
// Build: nvcc -O3 -fmad=false -arch=sm_121a (the -fmad=false matters: nvcc contracts a*b+c by default and glibc does not, except where the FMA pattern below says so).
// Result on GB10, 2026-09-19: expf 0, log1pf 0, sqrt(softplus) 0 mismatches over all 2^32 inputs against glibc 2.39 aarch64.
// glibc 2.39 aarch64 expf / log1pf ported to CUDA with the FMA pattern of the installed libm (objdump of __expf_finite, __log1pf),
// checked against the host libm over all 2^32 inputs: expf alone, log1pf alone, and the routing score sqrtf(softplus(x)).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <thread>
#include <cuda_runtime.h>
__device__ __constant__ uint64_t EXP2F_TAB[32] = {
0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51, 0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d, 0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13, 0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540 };
__device__ __forceinline__ float g_expf(float x) {
  const double N = 32.0;
  const double InvLn2N = 0x1.71547652b82fep+0 * N;
  const double C0 = 0x1.c6af84b912394p-5 / N / N / N, C1 = 0x1.ebfce50fac4f3p-3 / N / N, C2 = 0x1.62e42ff0c52d6p-1 / N;
  uint32_t abstop = (__float_as_uint(x) >> 20) & 0x7ff;
  if (abstop >= ((__float_as_uint(88.0f) >> 20) & 0x7ff)) {
    if (__float_as_uint(x) == __float_as_uint(-INFINITY)) return 0.0f;
    if (abstop >= ((__float_as_uint(INFINITY) >> 20) & 0x7ff)) return x + x;
    if (x > 0x1.62e42ep6f) return INFINITY;          // __math_oflowf
    if (x < -0x1.9fe368p6f) return 0.0f;             // __math_uflowf
  }
  double xd = (double)x;
  double z = InvLn2N * xd;
  double kd = rint(z);                    // frinta = round half away: use round()
  kd = round(z);
  long long ki = (long long)kd;           // fcvtas
  double r = z - kd;
  uint64_t t = EXP2F_TAB[ki & 31];
  t += (uint64_t)ki << (52 - 5);
  double s = __longlong_as_double((long long)t);
  double zz = fma(C0, r, C1);
  double r2 = r * r;
  double y = fma(C2, r, 1.0);
  y = fma(zz, r2, y);
  y = y * s;
  return (float)y;
}
__device__ __forceinline__ float g_log1pf(float x) {
  const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f, two25 = 3.355443200e+07f;
  const float Lp1 = 6.6666668653e-01f, Lp2 = 4.0000000596e-01f, Lp3 = 2.8571429849e-01f, Lp4 = 2.2222198546e-01f, Lp5 = 1.8183572590e-01f, Lp6 = 1.5313838422e-01f, Lp7 = 1.4798198640e-01f;
  float hfsq, f, c = 0.0f, s, z, R, u; int32_t k, hx, hu, ax;
  hx = (int32_t)__float_as_uint(x); ax = hx & 0x7fffffff; k = 1;
  if (hx < 0x3ed413d7) {
    if (ax >= 0x3f800000) { if (x == -1.0f) return -two25 / 0.0f; else return (x - x) / (x - x); }
    if (ax < 0x31000000) { if (ax < 0x24800000) return x; else return __fmaf_rn(-(x * x), 0.5f, x); }  // x - x*x*0.5 (fmsub)
    if (hx > 0 || hx <= (int32_t)0xbe95f61f) { k = 0; f = x; hu = 1; }
  }
  if (hx >= 0x7f800000) return x + x;
  if (k != 0) {
    if (hx < 0x5a000000) { u = 1.0f + x; hu = (int32_t)__float_as_uint(u); k = (hu >> 23) - 127; c = (k > 0) ? 1.0f - (u - x) : x - (u - 1.0f); c /= u; }
    else { u = x; hu = (int32_t)__float_as_uint(u); k = (hu >> 23) - 127; c = 0; }
    hu &= 0x007fffff;
    if (hu < 0x3504f7) { u = __uint_as_float((uint32_t)(hu | 0x3f800000)); }
    else { k += 1; u = __uint_as_float((uint32_t)(hu | 0x3f000000)); hu = (0x00800000 - hu) >> 2; }
    f = u - 1.0f;
  }
  hfsq = (0.5f * f) * f;
  float kf = (float)k;
  if (hu == 0) {
    if (f == 0.0f) { if (k == 0) return 0.0f; else { c = __fmaf_rn(kf, ln2_lo, c); return __fmaf_rn(kf, ln2_hi, c); } }
    R = __fmaf_rn(-0.66666666666666666f, f, 1.0f) * hfsq;          // hfsq*(1 - 0.667f*f), fmsub then fmul
    if (k == 0) return f - R;
    else return __fmaf_rn(kf, ln2_hi, -((R - __fmaf_rn(kf, ln2_lo, c)) - f));
  }
  s = f / (2.0f + f);
  z = s * s;
  float p = __fmaf_rn(z, Lp7, Lp6);
  p = __fmaf_rn(p, z, Lp5); p = __fmaf_rn(p, z, Lp4); p = __fmaf_rn(p, z, Lp3); p = __fmaf_rn(p, z, Lp2); p = __fmaf_rn(p, z, Lp1);
  float hr = __fmaf_rn(p, z, hfsq);           // hfsq + R with R = z*p, fused
  if (k == 0) return f - (hfsq - s * hr);
  else return __fmaf_rn(kf, ln2_hi, -((hfsq - (s * hr + __fmaf_rn(kf, ln2_lo, c))) - f));
}
__global__ void k_exp(uint32_t base, float* o, uint32_t n) { uint32_t i = base + blockIdx.x * blockDim.x + threadIdx.x; if (i - base < n) o[i - base] = g_expf(__uint_as_float(i)); }
__global__ void k_l1p(uint32_t base, float* o, uint32_t n) { uint32_t i = base + blockIdx.x * blockDim.x + threadIdx.x; if (i - base < n) o[i - base] = g_log1pf(__uint_as_float(i)); }
__global__ void k_score(uint32_t base, float* o, uint32_t n) { uint32_t i = base + blockIdx.x * blockDim.x + threadIdx.x; if (i - base < n) { float x = __uint_as_float(i); o[i - base] = sqrtf(x > 20.0f ? x : g_log1pf(g_expf(x))); } }
static float h_score(float x) { return sqrtf(x > 20.0f ? x : log1pf(expf(x))); }
int main() {
  const uint32_t CH = 1u << 26; float* d; cudaMalloc(&d, CH * 4); std::vector<float> h(CH);
  for (int which = 0; which < 3; which++) {
    uint64_t mism = 0; int shown = 0;
    for (uint64_t base = 0; base < (1ull << 32); base += CH) {
      if (which == 0) k_exp<<<CH / 256, 256>>>((uint32_t)base, d, CH); else if (which == 1) k_l1p<<<CH / 256, 256>>>((uint32_t)base, d, CH); else k_score<<<CH / 256, 256>>>((uint32_t)base, d, CH);
      cudaMemcpy(h.data(), d, CH * 4, cudaMemcpyDeviceToHost);
      const int T = 20; std::vector<uint64_t> m(T, 0); std::vector<std::thread> th;
      for (int t = 0; t < T; t++) th.emplace_back([&, t] { for (uint32_t i = t; i < CH; i += T) { uint32_t bits = (uint32_t)base + i; float x; memcpy(&x, &bits, 4);
        float r = which == 0 ? expf(x) : which == 1 ? log1pf(x) : h_score(x); uint32_t a, b; memcpy(&a, &r, 4); memcpy(&b, &h[i], 4);
        if (a != b && !(std::isnan(r) && std::isnan(h[i]))) m[t]++; } });
      for (auto& x : th) x.join(); for (int t = 0; t < T; t++) mism += m[t];
      if (mism && !shown) { for (uint32_t i = 0; i < CH; i++) { uint32_t bits = (uint32_t)base + i; float x; memcpy(&x, &bits, 4); float r = which == 0 ? expf(x) : which == 1 ? log1pf(x) : h_score(x); uint32_t a, b; memcpy(&a, &r, 4); memcpy(&b, &h[i], 4); if (a != b && !(std::isnan(r) && std::isnan(h[i]))) { printf("  first mismatch: x=%.9g (0x%08x) host=%.9g dev=%.9g\n", x, bits, r, h[i]); shown = 1; break; } } }
    }
    printf("%s: mismatches over all 2^32 inputs = %llu\n", which == 0 ? "expf" : which == 1 ? "log1pf" : "sqrt(softplus)", (unsigned long long)mism);
  }
  return 0;
}
