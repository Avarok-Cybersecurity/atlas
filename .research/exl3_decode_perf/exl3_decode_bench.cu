// EXL3 decode-shape microbenchmark for Atlas on GB10.
// Compiles kernels/gb10/common/exl3_matmul.cu (Atlas's exact extern "C" wrappers over the
// vendored ExLlamaV3 kernels) and times every decode-path projection of Qwen3.8-Flash-Next
// 4.05bpw at m=1 under (a) Atlas's current grid choice and (b) alternatives.
//
// DRAM realism: each shape owns NCOPIES distinct weight buffers, cycled per launch, so the
// working set exceeds L2. Results are HYPOTHESES (isolated-kernel numbers), per the repo's
// measurement discipline; only an e2e A/B decides ship/no-ship.

#include "exl3_matmul.cu"   // brings in every wrapper + converters (-I kernels/gb10/common)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#define CK(x) do { cudaError_t _ck = (x); if (_ck != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_ck), __FILE__, __LINE__); exit(1);} } while(0)

static int g_sms = 0;
static size_t g_l2 = 0;

// ---------------------------------------------------------------- helpers
static const int TILESIZE_K_T[5] = {0, 16, 32, 32, 16};
static const int TILESIZE_N_T[5] = {0, 128, 128, 256, 512};
static const int BLOCKDIM_T[5]   = {0, 256, 512, 512, 256};

// Port of Atlas select_exl3_gemm_shape (Blackwell branch)
static int select_shape(int k, int n, int kb, bool multi, int bszm_in, int bszm_out) {
    bool mod256 = n % 256 == 0, mod512 = n % 512 == 0;
    k *= bszm_in; n *= bszm_out;
    if ((kb == 4 || kb == 2) && !multi && k <= 2048) return 1;
    if (kb >= 7) { if (mod256 && n <= 8192) return k > 32768 ? 3 : 2; if (mod512 && n > 32768) return 4; return 2; }
    if (mod256 && n <= 4096) return (k > 8192 && kb >= 3) ? 3 : 2;
    if (mod512 && n > 16384) return 4;
    return mod256 ? 3 : 2;
}
static bool shape_compat(int s, int k, int n) { return k % TILESIZE_K_T[s] == 0 && n % TILESIZE_N_T[s] == 0; }
// Port of Atlas mgemm_grid::grid
static bool mgemm_grid(int tiles, int slots, int sms, int* per_slot_out, int* conc_out) {
    int group = slots;
    if (tiles == 0 || slots == 0) return false;
    int per_slot = tiles;
    if (per_slot > sms / group) per_slot = std::max(sms / group, 1);
    if (per_slot <= sms && tiles / per_slot > 48) per_slot = std::min(sms, per_slot * 2);
    *per_slot_out = per_slot; *conc_out = std::max(std::min(sms / per_slot, slots), 1);
    return true;
}

typedef void (*gemm_fn)(EXL3_GEMM_ARGS);
typedef void (*mgemm_fn)(EXL3_MGEMM_ARGS);

#define GEMM_SEL(K, S) \
    if (kb == K && shape == S) return c_fp32 ? (void*) exl3_gemm_k##K##_cb2_sh##S##_f32 : (void*) exl3_gemm_k##K##_cb2_sh##S##_f16;
static void* gemm_kernel(int kb, int shape, bool c_fp32) {
    GEMM_SEL(2,1) GEMM_SEL(4,1)
    GEMM_SEL(2,2) GEMM_SEL(2,3) GEMM_SEL(2,4)
    GEMM_SEL(3,2) GEMM_SEL(3,3) GEMM_SEL(3,4)
    GEMM_SEL(4,2) GEMM_SEL(4,3) GEMM_SEL(4,4)
    GEMM_SEL(5,2) GEMM_SEL(5,3) GEMM_SEL(5,4)
    GEMM_SEL(6,2) GEMM_SEL(6,3) GEMM_SEL(6,4)
    GEMM_SEL(8,2) GEMM_SEL(8,3) GEMM_SEL(8,4)
    return nullptr;
}
#define MGEMM_SEL(K, S) \
    if (kb == K && shape == S) return c_fp32 ? (void*) exl3_mgemm_k##K##_cb2_sh##S##_f32 : (void*) exl3_mgemm_k##K##_cb2_sh##S##_f16;
static void* mgemm_kernel(int kb, int shape, bool c_fp32) {
    MGEMM_SEL(2,2) MGEMM_SEL(2,3) MGEMM_SEL(2,4)
    MGEMM_SEL(3,2) MGEMM_SEL(3,3) MGEMM_SEL(3,4)
    MGEMM_SEL(4,2) MGEMM_SEL(4,3) MGEMM_SEL(4,4)
    MGEMM_SEL(5,2) MGEMM_SEL(5,3) MGEMM_SEL(5,4)
    MGEMM_SEL(6,2) MGEMM_SEL(6,3) MGEMM_SEL(6,4)
    return nullptr;
}
#define GEMV_SEL(K, MM, CFG) \
    if (kb == K && mmode == MM && cfg == CFG) return c_fp32 ? (void*) exl3_gemv_k##K##_cb2_m##MM##_cfg##CFG##_f32 : (void*) exl3_gemv_k##K##_cb2_m##MM##_cfg##CFG##_f16;
static void* gemv_kernel(int kb, int mmode, int cfg, bool c_fp32) {
    GEMV_SEL(2,0,0) GEMV_SEL(2,0,1) GEMV_SEL(2,1,0) GEMV_SEL(2,1,1)
    GEMV_SEL(3,0,0) GEMV_SEL(3,0,1) GEMV_SEL(3,1,0) GEMV_SEL(3,1,1)
    GEMV_SEL(4,0,0) GEMV_SEL(4,0,1) GEMV_SEL(4,1,0) GEMV_SEL(4,1,1)
    return nullptr;
}

static void set_smem(void* k) {
    static std::vector<void*> done;
    if (std::find(done.begin(), done.end(), k) != done.end()) return;
    CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
    done.push_back(k);
}
static int occupancy(void* k, int block, size_t smem) {
    int b = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, k, block, smem)); return b;
}

__global__ void fill_rand_u16(uint16_t* p, size_t n, uint32_t seed) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t) i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = (uint16_t) x;
    }
}
__global__ void fill_half(half* p, size_t n, float scale, uint32_t seed) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t) i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        float f = ((x & 0xffff) / 65535.0f - 0.5f) * 2.0f * scale;
        p[i] = __float2half(f);
    }
}
__global__ void fill_sign(half* p, size_t n, uint32_t seed) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t) i * 2654435761u ^ seed; x ^= x >> 13;
        p[i] = __float2half((x & 1) ? 1.0f : -1.0f);
    }
}
__global__ void noop_kernel(int* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1; }
__global__ void coop_noop_kernel(int* p) {
    auto g = cg::this_grid();
    if (threadIdx.x == 0) atomicAdd(p, 1);
    g.sync();
}
__global__ void read_bw_kernel(const uint4* __restrict__ p, size_t n, uint4* out) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x;
    uint4 acc = {0,0,0,0};
    for (; i < n; i += (size_t) gridDim.x * blockDim.x) { uint4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w; }
    if (acc.x == 0xdeadbeef && acc.y == 1) out[0] = acc;
}

template <typename F>
static float time_us(F&& launch, int warm, int iters) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < warm; ++i) launch(i);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch(warm + i);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaGetLastError());
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.0f / iters;
}

// ---------------------------------------------------------------- dense shape bench
struct Dense {
    const char* name; int k, n, kb, count; // count = launches per token
};

static int* g_locks = nullptr;

static void bench_dense(const Dense& d, int ncopies, bool sweep) {
    size_t wbytes = (size_t) d.k * d.n * d.kb / 8;
    std::vector<uint16_t*> W(ncopies); std::vector<half*> SUH(ncopies), SVH(ncopies);
    for (int c = 0; c < ncopies; ++c) {
        CK(cudaMalloc(&W[c], wbytes));
        fill_rand_u16<<<1024, 256>>>(W[c], wbytes / 2, 17 + c);
        CK(cudaMalloc(&SUH[c], d.k * 2)); fill_sign<<<64,256>>>(SUH[c], d.k, 3 + c);
        CK(cudaMalloc(&SVH[c], d.n * 2)); fill_sign<<<64,256>>>(SVH[c], d.n, 5 + c);
    }
    half *A, *Ahad; void* C;
    CK(cudaMalloc(&A, d.k * 2)); fill_half<<<64,256>>>(A, d.k, 1.0f, 99);
    CK(cudaMalloc(&Ahad, d.k * 2));
    CK(cudaMalloc(&C, (size_t) d.n * 4));
    CK(cudaDeviceSynchronize());

    int m = 1;
    auto run_gemm = [&](void* kern, int shape, int grid, int i) {
        int c = i % ncopies;
        const half* Ap = A; const uint16_t* Bp = W[c]; void* Cp = C; int mm = m, kk = d.k, nn = d.n;
        int* locks = g_locks; const half* suh = SUH[c]; half* ah = Ahad; const half* svh = SVH[c];
        void* args[] = { &Ap, &Bp, &Cp, &mm, &kk, &nn, &locks, &suh, &ah, &svh };
        cudaError_t e = cudaLaunchCooperativeKernel(kern, dim3(grid), dim3(BLOCKDIM_T[shape]), args, SMEM_MAX, 0);
        if (e != cudaSuccess) { fprintf(stderr, "coop launch failed grid=%d: %s\n", grid, cudaGetErrorString(e)); exit(1); }
    };
    auto run_gemv = [&](void* kern, int cfg, int grid, int i) {
        int c = i % ncopies;
        const half* Ap = A; const uint16_t* Bp = W[c]; void* Cp = C; int mm = m, kk = d.k, nn = d.n;
        int* locks = g_locks; const half* suh = SUH[c]; half* ah = Ahad; const half* svh = SVH[c];
        void* args[] = { &Ap, &Bp, &Cp, &mm, &kk, &nn, &locks, &suh, &ah, &svh };
        int block = cfg == 0 ? 512 : 256;
        cudaError_t e = cudaLaunchCooperativeKernel(kern, dim3(grid), dim3(block), args, 0, 0);
        if (e != cudaSuccess) { fprintf(stderr, "gemv coop launch failed grid=%d: %s\n", grid, cudaGetErrorString(e)); exit(1); }
    };

    // Atlas's choice: shape from heuristic w/ fallback; grid = min(tiles, sms); f32 C (dense arm uses fp32 C at m<=8)
    int h = select_shape(d.k, d.n, d.kb, false, 1, 1);
    int shape = 0;
    for (int s : {h, 2, 3, 4, 1}) {
        bool avail = (s == 1) ? (d.kb == 2 || d.kb == 4) : (s >= 2 && s <= 4);
        if (avail && shape_compat(s, d.k, d.n)) { shape = s; break; }
    }
    int tiles = (d.k / TILESIZE_K_T[shape]) * (d.n / TILESIZE_N_T[shape]);
    int atlas_grid = std::max(std::min(tiles, g_sms), 1);
    void* kern = gemm_kernel(d.kb, shape, true);
    set_smem(kern);
    int occ = occupancy(kern, BLOCKDIM_T[shape], SMEM_MAX);
    float us = time_us([&](int i) { run_gemm(kern, shape, atlas_grid, i); }, 5, 40);
    double gbs = wbytes / (us * 1e-6) / 1e9;
    printf("DENSE %-14s k=%5d n=%6d K=%d  ATLAS gemm sh%d grid=%d blk=%d occ=%d : %8.1f us  %6.1f GB/s  x%d/token = %7.2f ms\n",
           d.name, d.k, d.n, d.kb, shape, atlas_grid, BLOCKDIM_T[shape], occ, us, gbs, d.count, us * d.count / 1000.0);

    // Atlas GEMV tier when K in 2..4 (heuristic gemv_cfg_blackwell w/ occ=1)
    if (d.kb >= 2 && d.kb <= 4 && d.k % 128 == 0 && d.n % 128 == 0) {
        int coresident = g_sms; // Atlas assumes occ 1
        int cfg;
        if (d.kb == 2) cfg = d.n <= 8192 ? 0 : 1;
        else if (d.n / 32 <= coresident) cfg = 0;
        else if (d.k <= 2048 && d.n <= 8192) cfg = 0;
        else if (d.kb == 3) cfg = -1;
        else if (d.n >= 8192 && d.k <= 4096) cfg = 1;
        else cfg = -1;
        if (cfg >= 0 && d.k % (cfg == 0 ? 256 : 128) != 0) { printf("      %-14s                         ATLAS gemv cfg%d would be selected but k=%d %% %d != 0 -> SKIPPED in bench (possible OOB)\n", "", cfg, d.k, cfg == 0 ? 256 : 128); cfg = -1; }
        if (cfg >= 0) {
            void* gk = gemv_kernel(d.kb, 0, cfg, true);
            int cols = cfg == 0 ? 32 : 64;
            int grid = std::min(d.n / cols, coresident);
            float gus = time_us([&](int i) { run_gemv(gk, cfg, grid, i); }, 5, 40);
            printf("      %-14s                         ATLAS gemv cfg%d grid=%d (occ-assumed 1)   : %8.1f us  %6.1f GB/s\n",
                   "", cfg, grid, gus, wbytes / (gus * 1e-6) / 1e9);
        } else {
            printf("      %-14s                         ATLAS gemv heuristic DECLINES -> gemm\n", "");
        }
    }

    if (!sweep) goto cleanup;
    // Sweep: every compatible shape x grid in {4,8,...,sms} + true-occupancy gemv grids
    {
        float best = 1e30f; std::string best_desc;
        for (int s = 1; s <= 4; ++s) {
            bool avail = (s == 1) ? (d.kb == 2 || d.kb == 4) : true;
            if (!avail || !shape_compat(s, d.k, d.n)) continue;
            void* kk = gemm_kernel(d.kb, s, true); if (!kk) continue;
            set_smem(kk);
            int t = (d.k / TILESIZE_K_T[s]) * (d.n / TILESIZE_N_T[s]);
            for (int g = 4; g <= g_sms; g += 4) {
                if (g > t) break;
                float u = time_us([&](int i) { run_gemm(kk, s, g, i); }, 3, 20);
                if (u < best) { best = u; best_desc = "gemm sh" + std::to_string(s) + " grid=" + std::to_string(g); }
            }
        }
        if (d.kb >= 2 && d.kb <= 4) {
            for (int cfg = 0; cfg <= 1; ++cfg) {
                if (d.k % (cfg == 0 ? 256 : 128) != 0) { printf("      %-14s                         gemv cfg%d SKIPPED: k=%d not a multiple of %d (k-split x 16)\n", "", cfg, d.k, cfg == 0 ? 256 : 128); continue; }
                void* gk = gemv_kernel(d.kb, 0, cfg, true);
                int block = cfg == 0 ? 512 : 256, cols = cfg == 0 ? 32 : 64;
                int occv = occupancy(gk, block, 0);
                int maxgrid = std::min(d.n / cols, occv * g_sms);
                for (int g = std::min(8, maxgrid); g <= maxgrid; g = std::min(g * 2, maxgrid)) {
                    float u = time_us([&](int i) { run_gemv(gk, cfg, g, i); }, 3, 20);
                    if (u < best) { best = u; best_desc = "gemv cfg" + std::to_string(cfg) + " grid=" + std::to_string(g) + " (occ " + std::to_string(occv) + ")"; }
                    if (g == maxgrid) break;
                }
            }
        }
        printf("      %-14s                         BEST  %-34s : %8.1f us  %6.1f GB/s  x%d/token = %7.2f ms   (%.2fx vs Atlas)\n",
               "", best_desc.c_str(), best, wbytes / (best * 1e-6) / 1e9, d.count, best * d.count / 1000.0, us / best);
    }
cleanup:
    for (int c = 0; c < ncopies; ++c) { cudaFree(W[c]); cudaFree(SUH[c]); cudaFree(SVH[c]); }
    cudaFree(A); cudaFree(Ahad); cudaFree(C);
}

// ---------------------------------------------------------------- MoE mgemm bench (decode tier, S = T*top_k slots)
static void bench_moe(int hidden, int inter, int kb, int num_experts, int top_k, int T, int layers, bool sweep) {
    // Three projections: gate/up hidden->inter (f16 C), down inter->hidden (f32 C, weighted grouped reduction)
    struct Proj { int k, n; bool fp32; const char* name; } projs[3] = {
        { hidden, inter, false, "gate" }, { hidden, inter, false, "up" }, { inter, hidden, true, "down" } };
    int S = T * top_k;
    std::vector<uint16_t*> Wp[3]; std::vector<half*> SUp[3], SVp[3];
    const uint16_t** Blist[3]; const half** SUlist[3]; const half** SVlist[3];
    for (int p = 0; p < 3; ++p) {
        size_t wbytes = (size_t) projs[p].k * projs[p].n * kb / 8;
        Wp[p].resize(num_experts); SUp[p].resize(num_experts); SVp[p].resize(num_experts);
        std::vector<const uint16_t*> hb(num_experts); std::vector<const half*> hsu(num_experts), hsv(num_experts);
        for (int e = 0; e < num_experts; ++e) {
            CK(cudaMalloc(&Wp[p][e], wbytes)); fill_rand_u16<<<256,256>>>(Wp[p][e], wbytes / 2, 1000 + e * 3 + p);
            CK(cudaMalloc(&SUp[p][e], projs[p].k * 2)); fill_sign<<<16,256>>>(SUp[p][e], projs[p].k, 7 + e);
            CK(cudaMalloc(&SVp[p][e], projs[p].n * 2)); fill_sign<<<16,256>>>(SVp[p][e], projs[p].n, 11 + e);
            hb[e] = Wp[p][e]; hsu[e] = SUp[p][e]; hsv[e] = SVp[p][e];
        }
        CK(cudaMalloc(&Blist[p], num_experts * sizeof(void*))); CK(cudaMemcpy((void*) Blist[p], hb.data(), num_experts * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&SUlist[p], num_experts * sizeof(void*))); CK(cudaMemcpy((void*) SUlist[p], hsu.data(), num_experts * sizeof(void*), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&SVlist[p], num_experts * sizeof(void*))); CK(cudaMemcpy((void*) SVlist[p], hsv.data(), num_experts * sizeof(void*), cudaMemcpyHostToDevice));
    }
    // routing: NIDX distinct random index sets, cycled per launch (distinct experts per "layer")
    const int NIDX = 64;
    std::vector<int64_t*> idx(NIDX); std::vector<half*> wts(NIDX);
    std::mt19937 rng(42);
    for (int i = 0; i < NIDX; ++i) {
        std::vector<int64_t> h(S); std::vector<half> w(S);
        for (int t = 0; t < T; ++t) {
            std::vector<int> perm(num_experts); for (int e = 0; e < num_experts; ++e) perm[e] = e;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int j = 0; j < top_k; ++j) { h[t * top_k + j] = perm[j]; w[t * top_k + j] = __float2half(1.0f / top_k); }
        }
        CK(cudaMalloc(&idx[i], S * 8)); CK(cudaMemcpy(idx[i], h.data(), S * 8, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&wts[i], S * 2)); CK(cudaMemcpy(wts[i], w.data(), S * 2, cudaMemcpyHostToDevice));
    }
    half *A, *Ahad, *Cg, *Cu, *Inter; float* Cd;
    CK(cudaMalloc(&A, (size_t) S * hidden * 2)); fill_half<<<256,256>>>(A, (size_t) S * hidden, 1.0f, 5);
    CK(cudaMalloc(&Ahad, (size_t) S * hidden * 2));
    CK(cudaMalloc(&Cg, (size_t) S * inter * 2)); CK(cudaMalloc(&Cu, (size_t) S * inter * 2)); CK(cudaMalloc(&Inter, (size_t) S * inter * 2));
    CK(cudaMalloc(&Cd, (size_t) S * hidden * 4));
    CK(cudaDeviceSynchronize());

    auto launch_mgemm = [&](int p, int shape, int per_slot, int conc, int i, half* Ain, void* Cout, bool weighted) {
        void* kern = mgemm_kernel(kb, shape, projs[p].fp32); set_smem(kern);
        const half* Ap = Ain; const uint16_t** Bl = Blist[p]; void* Cp = Cout; int mm = 1, kk = projs[p].k, nn = projs[p].n;
        int* locks = g_locks; const half** sul = SUlist[p]; half* ah = Ahad; const half** svl = SVlist[p];
        int64_t* bi = idx[i % NIDX]; half* bw = weighted ? wts[i % NIDX] : nullptr;
        int bin = S, bout = S, mini = -1, maxi = -1, nt = weighted ? T : 1; const int* snl = nullptr; void** cl = nullptr;
        void* args[] = { &Ap, &Bl, &Cp, &mm, &kk, &nn, &locks, &sul, &ah, &svl, &bi, &bw, &bin, &bout, &mini, &maxi, &nt, &snl, &cl };
        cudaError_t e = cudaLaunchCooperativeKernel(kern, dim3(per_slot, 1, conc), dim3(BLOCKDIM_T[shape]), args, SMEM_MAX, 0);
        if (e != cudaSuccess) { fprintf(stderr, "mgemm launch failed (%d,%d): %s\n", per_slot, conc, cudaGetErrorString(e)); exit(1); }
    };
    auto silu = [&](int) {
        size_t n2 = (size_t) S * inter / 2; unsigned grid = std::min<size_t>((n2 + 255) / 256, 4096);
        exl3_silu_mul_f16<<<grid, 256>>>(Cg, Cu, Inter, 0.0f, (uint64_t) n2);
    };

    // Atlas config
    int shape_gu = select_shape(hidden, inter, kb, true, S, S); if (!shape_compat(shape_gu, hidden, inter)) shape_gu = 2;
    int shape_d  = select_shape(inter, hidden, kb, true, S, S); if (!shape_compat(shape_d, inter, hidden)) shape_d = 2;
    int tiles_gu = (hidden / TILESIZE_K_T[shape_gu]) * (inter / TILESIZE_N_T[shape_gu]);
    int tiles_d  = (inter / TILESIZE_K_T[shape_d]) * (hidden / TILESIZE_N_T[shape_d]);
    int ps_gu, c_gu, ps_d, c_d; mgemm_grid(tiles_gu, S, g_sms, &ps_gu, &c_gu); mgemm_grid(tiles_d, S, g_sms, &ps_d, &c_d);
    size_t bytes_tok = (size_t) S * (2 * (size_t) hidden * inter + (size_t) inter * hidden) * kb / 8;

    float u_gate = time_us([&](int i) { launch_mgemm(0, shape_gu, ps_gu, c_gu, i, A, Cg, false); }, 5, 40);
    float u_up   = time_us([&](int i) { launch_mgemm(1, shape_gu, ps_gu, c_gu, i, A, Cu, false); }, 5, 40);
    float u_silu = time_us(silu, 5, 40);
    float u_down = time_us([&](int i) { launch_mgemm(2, shape_d, ps_d, c_d, i, Inter, Cd, true); }, 5, 40);
    float u_chain = time_us([&](int i) {
        launch_mgemm(0, shape_gu, ps_gu, c_gu, i, A, Cg, false);
        launch_mgemm(1, shape_gu, ps_gu, c_gu, i, A, Cu, false);
        silu(i);
        launch_mgemm(2, shape_d, ps_d, c_d, i, Inter, Cd, true);
    }, 5, 40);
    printf("MOE T=%d S=%d K=%d  ATLAS gate/up sh%d grid=(%d,%d) down sh%d grid=(%d,%d): gate %.1f up %.1f silu %.1f down %.1f us | chain %.1f us  %.1f GB/s(weights) x%d layers = %.2f ms/token\n",
           T, S, kb, shape_gu, ps_gu, c_gu, shape_d, ps_d, c_d, u_gate, u_up, u_silu, u_down, u_chain,
           bytes_tok / (u_chain * 1e-6) / 1e9, layers, u_chain * layers / 1000.0);

    if (sweep) {
        // grid sweep at fixed shape: per_slot in {1,2,4,6,8,12,16,24,48}, conc = min(sms/per_slot, S)
        for (int p = 0; p < 3; p += 2) {
            int shape = p == 0 ? shape_gu : shape_d; int tiles = p == 0 ? tiles_gu : tiles_d;
            float best = 1e30f; int bps = 0, bc = 0, bsh = shape;
            for (int sh = 2; sh <= 4; ++sh) {
                if (!shape_compat(sh, projs[p].k, projs[p].n)) continue;
                int tl = (projs[p].k / TILESIZE_K_T[sh]) * (projs[p].n / TILESIZE_N_T[sh]);
                for (int ps : {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}) {
                    if (ps > tl || ps > g_sms) continue;
                    int conc = std::max(std::min(g_sms / ps, S), 1);
                    float u = time_us([&](int i) { launch_mgemm(p, sh, ps, conc, i, p == 0 ? A : Inter, p == 0 ? (void*) Cg : (void*) Cd, p == 2); }, 3, 20);
                    if (u < best) { best = u; bps = ps; bc = conc; bsh = sh; }
                }
            }
            (void) tiles;
            printf("      %s BEST sh%d grid=(%d,%d): %.1f us  (%.2fx vs Atlas)\n", projs[p].name, bsh, bps, bc, best, (p == 0 ? u_gate : u_down) / best);
        }
    }
    // cleanup omitted for brevity (process exits)
}

int main(int argc, char** argv) {
    bool sweep = argc > 1 && (std::string(argv[1]) == "sweep" || std::string(argv[1]) == "moe");
    bool moe_only = argc > 1 && std::string(argv[1]) == "moe";
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    g_sms = prop.multiProcessorCount; g_l2 = prop.l2CacheSize;
    int clk = 0; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    int memclk = 0; cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
    int bus = 0; cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("DEVICE %s cc=%d.%d SMs=%d L2=%zu MB smem/SM=%zu KB clock=%d MHz memclk=%d MHz bus=%d bit coop=%d\n",
           prop.name, prop.major, prop.minor, g_sms, g_l2 >> 20, prop.sharedMemPerMultiprocessor >> 10, clk / 1000, memclk / 1000, bus, prop.cooperativeLaunch);

    // Peak read bandwidth (1 GiB streaming read)
    {
        size_t bytes = 1ull << 30; uint4* p; uint4* o; CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&o, 16)); CK(cudaMemset(p, 1, bytes));
        float us = time_us([&](int) { read_bw_kernel<<<g_sms * 8, 256>>>(p, bytes / 16, o); }, 2, 10);
        printf("PEAK streaming read: %.1f GB/s (%.1f us per GiB)\n", bytes / (us * 1e-6) / 1e9, us);
        cudaFree(p); cudaFree(o);
    }
    // Launch overhead floor: dependent chains of trivial kernels
    {
        int* p; CK(cudaMalloc(&p, 4)); CK(cudaMemset(p, 0, 4));
        float u1 = time_us([&](int) { noop_kernel<<<1, 32>>>(p); }, 50, 500);
        float u2 = time_us([&](int) { noop_kernel<<<48, 256>>>(p); }, 50, 500);
        int* pp = p; void* args[] = { &pp };
        float u3 = time_us([&](int) { CK(cudaLaunchCooperativeKernel((void*) coop_noop_kernel, dim3(48), dim3(256), args, 0, 0)); }, 50, 500);
        CK(cudaFuncSetAttribute((void*) coop_noop_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
        float u4 = time_us([&](int) { CK(cudaLaunchCooperativeKernel((void*) coop_noop_kernel, dim3(48), dim3(512), args, SMEM_MAX, 0)); }, 50, 500);
        printf("LAUNCH floor: plain<1,32> %.2f us | plain<48,256> %.2f us | coop<48,256> grid.sync %.2f us | coop<48,512,90KB smem> %.2f us  (per launch, back-to-back)\n", u1, u2, u3, u4);
        cudaFree(p);
    }
    CK(cudaMalloc(&g_locks, 4 * (1024 * 1024 + 2 * 1024 + 66))); CK(cudaMemset(g_locks, 0, 4 * (1024 * 1024 + 2 * 1024 + 66)));

    // Qwen3.8-Flash-Next 4.05bpw decode shapes (m=1). K=6 dense, K=4 experts, K=6 lm_head.
    // Count = launches per token (layers x projections). Attention q has the output gate (2x).
    Dense dense[] = {
        { "gdn.in_proj_qkv", 2560, 10240, 6, 36 },
        { "gdn.in_proj_z",   2560,  6144, 6, 36 },
        { "gdn.out_proj",    6144,  2560, 6, 36 },
        { "attn.q_proj",     2560, 12288, 6, 12 },
        { "attn.k_proj",     2560,   512, 6, 12 },
        { "attn.v_proj",     2560,   512, 6, 12 },
        { "attn.o_proj",     6144,  2560, 6, 12 },
        { "lm_head",         2560, 248320, 6, 1 },
        // K=4 variants of the GDN shapes: what the 4.00bpw Qwen3.8-27B-style export / a K=4 dense tier would pay
        { "gdn.in_proj_qkv@K4", 2560, 10240, 4, 36 },
        { "gdn.out_proj@K4",    6144,  2560, 4, 36 },
        // shared expert at K=4 (currently served NVFP4 in Atlas; upstream-native comparison)
        { "shared.gate@K4",  2560,   640, 4, 48 },
        { "shared.down@K4",   640,  2560, 4, 48 },
    };
    for (auto& d : dense) {
        if (moe_only) break;
        int copies = d.n >= 100000 ? 2 : (d.k * d.n * d.kb / 8 > 8 << 20 ? 24 : 48);
        bench_dense(d, copies, sweep);
    }
    // Routed experts: 512 experts, top-10, T=1 and T=3 (MTP verify width)
    bench_moe(2560, 640, 4, 512, 10, 1, 48, sweep);
    bench_moe(2560, 640, 4, 512, 10, 3, 48, sweep);
    return 0;
}
