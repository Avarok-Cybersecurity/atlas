# FP8 MMA pattern extracted from `w4a16_gemm.cu`

Source: `/workspace/atlas-mtp/kernels/gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu`
Target kernels: `fp8_gemm_t` (L371-480, BF16×FP8) and `fp8_fp8_gemm_t` (L560-667, FP8×FP8).

## 1. MMA PTX (L312, L436, L622)

```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
    {acc0..3}, {a0..3}, {b0,b1}, {acc0..3};
```

32 K elements per warp tile, fp32 accumulator. Confirmed on SM121/GB10.

## 2. Tile / grid

L15-23, L558: `M_TILE=64`, `N_TILE_LG=128`, `K_STEP_T=32`, `A_FP8_STRIDE=32`.
Block `(128,1,1)` = 4 warps × M=16 each. Grid `(ceil(N/128), ceil(M/64))`.
Per warp: `nt=0..15` over 16 sub-N tiles (L432, L618). `acc[16][4]` fp32 (L578).

## 3. Smem (L575-576)

```c
__shared__ unsigned char smem_Af[2][M_TILE][A_FP8_STRIDE];   // 2×64×32 = 4 KB
__shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];    // 2×128×32 = 8 KB
```

A row-major (M,K); B transposed (N,K) — K=32 slice is contiguous → one 16-B
cp.async per half-row. Double-buffered. No PAD needed (32 B rows bank-aligned).

## 4. cp.async (L152-165, L586-607)

`cp.async.ca.shared.global [dst],[src],16,%2;` + commit + `wait_group 0`.
Prolog loads buf 0; main loop overlaps `LOAD(nxt)` ‖ `COMPUTE(cur)` (L633-650).
A: 128 threads × 16 B = full 64×32 tile, 1 cp.async/thread. B: each thread does
2 cp.async (`[0..15]`,`[16..31]`) → 128×32 tile.

## 5. Register fragments (L613-621)

m16n8k32 wants A = 4×u32 (lane-mapped 16×32), B = 2×u32 (8×32), each u32 = 4 FP8:

```c
unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8;     // group_id = lane>>2
unsigned int a0 = *(uint*)&smem_Af[buf][fr0][4 * tid];          // tid = lane&3
unsigned int a1 = *(uint*)&smem_Af[buf][fr1][4 * tid];
unsigned int a2 = *(uint*)&smem_Af[buf][fr0][16 + 4 * tid];
unsigned int a3 = *(uint*)&smem_Af[buf][fr1][16 + 4 * tid];
// B per nt:
unsigned int b0 = *(uint*)&smem_Bf[buf][nc][4 * tid];
unsigned int b1 = *(uint*)&smem_Bf[buf][nc][16 + 4 * tid];
```

## 6. K-loop (L633-650)

```c
FF_LOADS(0, 0); cp_async_commit(); cp_async_wait_all(); __syncthreads();
int cur = 0;
for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
    int nxt = 1 - cur;
    FF_LOADS(nxt, k_base); cp_async_commit();
    FF_COMPUTE(cur, cur);            // 16 MMAs covering N_TILE_LG×K_STEP_T
    cp_async_wait_all(); __syncthreads();
    cur = nxt;
}
FF_COMPUTE(cur, cur);
```

`FF_COMPUTE` (L610-631): builds 4 A regs once, then 16 MMAs over N (2 B regs each),
all into `acc[nt][0..3]`.

## 7. Scale handling — CRITICAL

**Neither `fp8_gemm_t` nor `fp8_fp8_gemm_t` take a scale arg.** Signatures (L371-375,
L560-564) are `(A, B_fp8, C, M, N, K)` — **scale pre-baked into weights**:
`w4a16_gemm_t` (L210) and `predequant_nvfp4_to_fp8` (L490-521) absorb `scale2` into B
at FP4→FP8 conversion time. So these kernels are **not** DeepGEMM-compatible as-is;
they assume a single global scale folded into B at quant time.

## 8. Extension cost to per-(N/128,K/128) FP32 block scales

Required:
1. Add args `const float* A_scale` (`[M, K/128]`), `const float* B_scale` (`[N/128, K/128]`).
2. **Per-K-block accumulator**: K_STEP_T=32, one 128-K block = 4 outer iters. Use
   `acc_blk[16][4]` for 4 iters, then `acc[nt][i] += acc_blk[nt][i] * as * bs; acc_blk=0`.
3. Prefetch scales (tiny: 8 floats per warp per block).
4. Block-boundary counter.

Footprint vs `fp8_fp8_gemm_t` (108 lines): args/smem +6, scale prefetch +8, fold +20.
**Estimate: ~35-40 extra LOC**, no algorithmic restructure. M_TILE/N_TILE already
match DeepGEMM (128 N, 64 M); K_STEP_T=32 → clean 4-iter rollup per 128-K block.

## 9. Scaffold — `fp8_fp8_gemm_t_blockscaled`

```c
extern "C" __global__ void fp8_fp8_gemm_t_blockscaled(
    const unsigned char* __restrict__ A_fp8,   // [M, K]
    const unsigned char* __restrict__ B_fp8,   // [N, K]
    const float* __restrict__ A_scale,         // [M, K/128]   row-major
    const float* __restrict__ B_scale,         // [N/128, K/128]
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K)
{
    // ── same setup: cta_m/cta_n/warp/lane/tid, smem_Af[2], smem_Bf[2],
    //    acc[16][4] init to 0 ───────────────────────────────────────
    float acc_blk[16][4] = {0};        // NEW: per-128K-block accumulator
    const int ITERS_PER_BLOCK = 128 / K_STEP_T;     // = 4

    // ── prolog: FF_LOADS(0, 0); commit/wait/sync ───────────────────
    int cur = 0, iter = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FF_LOADS(nxt, k_base); cp_async_commit();
        FF_COMPUTE_INTO(acc_blk, cur);        // same MMAs but → acc_blk
        cp_async_wait_all(); __syncthreads();
        cur = nxt; iter++;
        if (iter == ITERS_PER_BLOCK) {
            // NEW: fold per-block fp32 scales
            unsigned int k128 = (k_base - K_STEP_T) / 128;
            float bs = B_scale[(cta_n/128) * (K/128) + k128];
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                // a_scale lookup uses (cta_m+warp_m_offset+group_id, k128) etc.
                float as0 = A_scale[(cta_m+warp_m_offset+group_id)*(K/128)+k128];
                float as1 = A_scale[(cta_m+warp_m_offset+group_id+8)*(K/128)+k128];
                acc[nt][0] += acc_blk[nt][0] * as0 * bs;  acc_blk[nt][0] = 0;
                acc[nt][1] += acc_blk[nt][1] * as0 * bs;  acc_blk[nt][1] = 0;
                acc[nt][2] += acc_blk[nt][2] * as1 * bs;  acc_blk[nt][2] = 0;
                acc[nt][3] += acc_blk[nt][3] * as1 * bs;  acc_blk[nt][3] = 0;
            }
            iter = 0;
        }
    }
    FF_COMPUTE_INTO(acc_blk, cur);
    // ── final fold for last block, then BF16 writeback (unchanged) ─
}
```

`FF_COMPUTE_INTO(dst, b_buf)` is `FF_COMPUTE` from L610-631 with `acc` replaced by
`dst`. K must be multiple of 128 (already true for Qwen3.6 hidden=2048 and expert
intermediate=768 padded — verify pad path).
