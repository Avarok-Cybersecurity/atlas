// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
// Build: nvcc -O3 -fmad=false -arch=sm_121a. Result on GB10, 2026-09-19: 20,000 random tokens with forced ties, 0 pick / weight-bit / plan / miss-flag mismatches against the host route_from_logits.
// S2 prototype: the whole single-token selection on the device, bit-identical to the host `route_from_logits`:
// score_e = sqrt(softplus(logit_e)) (glibc-exact), rank by score+bias descending with index ascending on ties (stable sort),
// top-6, weights = score / (sum in pick order + 1e-20) * route_scale, then the plan order (ascending expert id) with weights.
// Test: random logits/bias (+ forced ties), host replica in C with glibc, compare picks, weights bits, plan order.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#define NR 384
#define TOPK 6
__device__ __constant__ uint64_t EXP2F_TAB[32] = {
0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51, 0x3fef72b83c7d517b, 0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1,
0x3fef06fe0a31b715, 0x3feef1a7373aa9cb, 0x3feedea64c123422, 0x3feece086061892d, 0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429, 0x3feea47eb03a5585,
0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13, 0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d,
0x3feee89f995ad3ad, 0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f, 0x3fefa4afa2a490da, 0x3fefd0765b6e4540 };
__device__ __forceinline__ float g_expf(float x) {
  const double N = 32.0, InvLn2N = 0x1.71547652b82fep+0 * N, C0 = 0x1.c6af84b912394p-5 / N / N / N, C1 = 0x1.ebfce50fac4f3p-3 / N / N, C2 = 0x1.62e42ff0c52d6p-1 / N;
  uint32_t abstop = (__float_as_uint(x) >> 20) & 0x7ff;
  if (abstop >= ((__float_as_uint(88.0f) >> 20) & 0x7ff)) { if (__float_as_uint(x) == __float_as_uint(-INFINITY)) return 0.0f; if (abstop >= ((__float_as_uint(INFINITY) >> 20) & 0x7ff)) return x + x; if (x > 0x1.62e42ep6f) return INFINITY; if (x < -0x1.9fe368p6f) return 0.0f; }
  double z = __dmul_rn(InvLn2N, (double)x); double kd = round(z); long long ki = (long long)kd; double r = __dsub_rn(z, kd);
  uint64_t t = EXP2F_TAB[ki & 31]; t += (uint64_t)ki << 47; double s = __longlong_as_double((long long)t);
  double zz = fma(C0, r, C1); double r2 = __dmul_rn(r, r); double y = fma(C2, r, 1.0); y = fma(zz, r2, y); y = __dmul_rn(y, s); return (float)y;
}
__device__ __forceinline__ float g_log1pf(float x) {
  const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f, two25 = 3.355443200e+07f;
  const float Lp1 = 6.6666668653e-01f, Lp2 = 4.0000000596e-01f, Lp3 = 2.8571429849e-01f, Lp4 = 2.2222198546e-01f, Lp5 = 1.8183572590e-01f, Lp6 = 1.5313838422e-01f, Lp7 = 1.4798198640e-01f;
  float hfsq, f, c = 0.0f, s, z, R, u; int32_t k, hx, hu, ax;
  hx = (int32_t)__float_as_uint(x); ax = hx & 0x7fffffff; k = 1;
  if (hx < 0x3ed413d7) {
    if (ax >= 0x3f800000) { if (x == -1.0f) return -two25 / 0.0f; else return (x - x) / (x - x); }
    if (ax < 0x31000000) { if (ax < 0x24800000) return x; else return __fmaf_rn(-__fmul_rn(x, x), 0.5f, x); }
    if (hx > 0 || hx <= (int32_t)0xbe95f61f) { k = 0; f = x; hu = 1; }
  }
  if (hx >= 0x7f800000) return x + x;
  if (k != 0) {
    if (hx < 0x5a000000) { u = __fadd_rn(1.0f, x); hu = (int32_t)__float_as_uint(u); k = (hu >> 23) - 127; c = (k > 0) ? __fsub_rn(1.0f, __fsub_rn(u, x)) : __fsub_rn(x, __fsub_rn(u, 1.0f)); c = __fdiv_rn(c, u); }
    else { u = x; hu = (int32_t)__float_as_uint(u); k = (hu >> 23) - 127; c = 0; }
    hu &= 0x007fffff;
    if (hu < 0x3504f7) { u = __uint_as_float((uint32_t)(hu | 0x3f800000)); } else { k += 1; u = __uint_as_float((uint32_t)(hu | 0x3f000000)); hu = (0x00800000 - hu) >> 2; }
    f = __fsub_rn(u, 1.0f);
  }
  hfsq = __fmul_rn(__fmul_rn(0.5f, f), f); float kf = (float)k;
  if (hu == 0) {
    if (f == 0.0f) { if (k == 0) return 0.0f; else { c = __fmaf_rn(kf, ln2_lo, c); return __fmaf_rn(kf, ln2_hi, c); } }
    R = __fmul_rn(__fmaf_rn(-0.66666666666666666f, f, 1.0f), hfsq);
    if (k == 0) return __fsub_rn(f, R); else return __fmaf_rn(kf, ln2_hi, -__fsub_rn(__fsub_rn(R, __fmaf_rn(kf, ln2_lo, c)), f));
  }
  s = __fdiv_rn(f, __fadd_rn(2.0f, f)); z = __fmul_rn(s, s);
  float p = __fmaf_rn(z, Lp7, Lp6); p = __fmaf_rn(p, z, Lp5); p = __fmaf_rn(p, z, Lp4); p = __fmaf_rn(p, z, Lp3); p = __fmaf_rn(p, z, Lp2); p = __fmaf_rn(p, z, Lp1);
  float hr = __fmaf_rn(p, z, hfsq);
  if (k == 0) return __fsub_rn(f, __fsub_rn(hfsq, __fmul_rn(s, hr)));
  else return __fmaf_rn(kf, ln2_hi, -__fsub_rn(__fsub_rn(hfsq, __fadd_rn(__fmul_rn(s, hr), __fmaf_rn(kf, ln2_lo, c))), f));
}
__device__ __forceinline__ float score_of(float logit) { return sqrtf(logit > 20.0f ? logit : g_log1pf(g_expf(logit))); }
// one block of 384 threads per token: scores, then a rank count per expert (strict total order: key desc, index asc), picks by rank,
// weights in pick order, then the plan (ascending id) with slot lookup (slot_of == 0xFFFFFFFF -> miss flag).
__global__ void route_select_dev(const float* __restrict__ logits, const float* __restrict__ bias, float route_scale, int norm_topk,
                                 const uint32_t* __restrict__ slot_of, int* __restrict__ picks, float* __restrict__ weights, int* __restrict__ plan_ids, float* __restrict__ plan_w, uint32_t* __restrict__ plan_slot, int* __restrict__ miss_flag) {
  __shared__ float sc[NR]; __shared__ float key[NR]; __shared__ int pick[TOPK]; __shared__ float wsum[1];
  int e = threadIdx.x;
  float s = score_of(logits[e]); sc[e] = s; key[e] = __fadd_rn(s, bias[e]);
  __syncthreads();
  float mk = key[e]; int rank = 0;
  for (int j = 0; j < NR; j++) { float kj = key[j]; if (kj > mk || (kj == mk && j < e)) rank++; }
  if (rank < TOPK) pick[rank] = e;
  __syncthreads();
  if (e == 0) {
    float sum = 0.0f;
    for (int i = 0; i < TOPK; i++) sum = __fadd_rn(sum, sc[pick[i]]);
    wsum[0] = __fadd_rn(sum, 1e-20f);
  }
  __syncthreads();
  if (e < TOPK) {
    float w = sc[pick[e]];
    if (norm_topk) w = __fdiv_rn(w, wsum[0]);
    w = __fmul_rn(w, route_scale);
    picks[e] = pick[e]; weights[e] = w;
    // plan position = number of picked ids smaller than mine (ids are distinct)
    int pos = 0; for (int i = 0; i < TOPK; i++) if (pick[i] < pick[e]) pos++;
    plan_ids[pos] = pick[e]; plan_w[pos] = w; uint32_t sl = slot_of[pick[e]]; plan_slot[pos] = sl;
    if (sl == 0xFFFFFFFFu) atomicOr(miss_flag, 1);
  }
}
// host replica of route_from_logits (Rust): stable sort_by descending (score+bias), picks, weights
static void host_route(const float* logits, const float* bias, float route_scale, bool norm, int* picks, float* weights) {
  float sc[NR]; for (int e = 0; e < NR; e++) { float x = logits[e]; sc[e] = sqrtf(x > 20.0f ? x : log1pf(expf(x))); }
  std::vector<int> order(NR); for (int i = 0; i < NR; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return (sc[b] + bias[b]) < (sc[a] + bias[a]); }); // partial_cmp(b,a): a before b if key[a] > key[b]
  float wt[TOPK]; float sum = 0.0f; for (int i = 0; i < TOPK; i++) { picks[i] = order[i]; wt[i] = sc[order[i]]; }
  if (norm) { for (int i = 0; i < TOPK; i++) sum += wt[i]; sum += 1e-20f; for (int i = 0; i < TOPK; i++) wt[i] /= sum; }
  for (int i = 0; i < TOPK; i++) { wt[i] *= route_scale; weights[i] = wt[i]; }
}
int main() {
  std::mt19937 rng(7); std::normal_distribution<float> nd(0.0f, 3.0f); std::uniform_real_distribution<float> ud(-0.5f, 0.5f);
  const int T = 20000; std::vector<float> L(T * NR), B(NR); for (auto& b : B) b = ud(rng);
  for (int t = 0; t < T; t++) for (int e = 0; e < NR; e++) L[t * NR + e] = nd(rng) * (e % 7 == 0 ? 3.0f : 1.0f);
  // forced ties: copy expert 5's logit into 6 sometimes (bias differs, so ties need equal keys: also copy bias)
  for (int t = 0; t < T; t += 10) { L[t * NR + 6] = L[t * NR + 5]; }
  B[6] = B[5];
  float *dL, *dB, *dW, *dPW; int *dP, *dPI, *dM; uint32_t *dS, *dPS;
  cudaMalloc(&dL, T * NR * 4); cudaMalloc(&dB, NR * 4); cudaMalloc(&dW, TOPK * 4); cudaMalloc(&dP, TOPK * 4); cudaMalloc(&dPI, TOPK * 4); cudaMalloc(&dPW, TOPK * 4); cudaMalloc(&dPS, TOPK * 4); cudaMalloc(&dM, 4); cudaMalloc(&dS, NR * 4);
  std::vector<uint32_t> slot(NR); for (int e = 0; e < NR; e++) slot[e] = (e % 5 == 0) ? 0xFFFFFFFFu : e * 3;
  cudaMemcpy(dL, L.data(), T * NR * 4, cudaMemcpyHostToDevice); cudaMemcpy(dB, B.data(), NR * 4, cudaMemcpyHostToDevice); cudaMemcpy(dS, slot.data(), NR * 4, cudaMemcpyHostToDevice);
  int bad_pick = 0, bad_w = 0, bad_plan = 0, bad_miss = 0;
  for (int t = 0; t < T; t++) {
    cudaMemset(dM, 0, 4);
    route_select_dev<<<1, NR>>>(dL + t * NR, dB, 2.5f, 1, dS, dP, dW, dPI, dPW, dPS, dM);
    int hp[TOPK], dp[TOPK], dpi[TOPK], dm; float hw[TOPK], dw[TOPK], dpw[TOPK]; uint32_t dps[TOPK];
    cudaMemcpy(dp, dP, TOPK * 4, cudaMemcpyDeviceToHost); cudaMemcpy(dw, dW, TOPK * 4, cudaMemcpyDeviceToHost); cudaMemcpy(dpi, dPI, TOPK * 4, cudaMemcpyDeviceToHost); cudaMemcpy(dpw, dPW, TOPK * 4, cudaMemcpyDeviceToHost); cudaMemcpy(dps, dPS, TOPK * 4, cudaMemcpyDeviceToHost); cudaMemcpy(&dm, dM, 4, cudaMemcpyDeviceToHost);
    host_route(&L[t * NR], B.data(), 2.5f, true, hp, hw);
    if (memcmp(hp, dp, sizeof hp)) bad_pick++;
    if (memcmp(hw, dw, sizeof hw)) bad_w++;
    // host plan: sort picks ascending with weights
    std::vector<std::pair<int, float>> pl; for (int i = 0; i < TOPK; i++) pl.push_back({hp[i], hw[i]}); std::sort(pl.begin(), pl.end());
    bool miss = false; for (int i = 0; i < TOPK; i++) { if (pl[i].first != dpi[i] || memcmp(&pl[i].second, &dpw[i], 4) || dps[i] != slot[pl[i].first]) bad_plan++; if (slot[pl[i].first] == 0xFFFFFFFFu) miss = true; }
    if ((dm != 0) != miss) bad_miss++;
  }
  printf("%d tokens: pick mismatches %d, weight-bit mismatches %d, plan mismatches %d, miss-flag mismatches %d\n", T, bad_pick, bad_w, bad_plan, bad_miss);
  return 0;
}
