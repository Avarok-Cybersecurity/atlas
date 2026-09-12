// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN decode recurrence — Hopper (sm_90a) BATCHED-STRIDED twin.
//
// SSOT for why this file exists and for every number quoted below:
// `GDN-DECODE-ATTRIBUTION.md` at the repository root, section "Round 17 — the
// n>=4 strided twin" (#927). Not to be confused with `gdn_decode_hopper.cu`,
// which twins the SAME parent for a different reason and is DEFAULT OFF: that
// file re-partitions COLUMNS to fill a 132-SM device at n=1, and H100 round 12
// measured it 0.83x at contiguous n=1 and a null (+0.19%) at n=16. This file
// re-partitions nothing. It changes how many times the state is READ.
//
// ── THE COST, measured ────────────────────────────────────────────────────
// nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8,
// round 13 cell V, median n=16 decode step 19.887 ms busy
// (`h100-r13-attribution.md` SS C.2/C.3): the strided GDN decode recurrence is
// 48 graph nodes, **2 748.9 us = 13.82% of the step**, 57.27 us per launch.
// Geometry nk=16 nv=48 kd=vd=128, so per launch the live state is
// `n * nv * kd * vd * 4 B` = 50.33 MB and the COMPULSORY traffic (read it
// once, write it once) is 100.66 MB — 1 758 GB/s, 52.5% of this part's
// 3 350 GB/s.
//
// WHY THAT 52.5% IS NOT SLACK, and what this kernel actually attacks. The
// parent walks the state TWICE: once to form `hk_dot = (H^T k)`, once to apply
// `H <- g*H + k (x) v_new` while forming `q_dot = (H_new^T q)`. Its ISSUED
// traffic is 2R+1W = 150.99 MB, i.e. **2 637 GB/s = 78.7% of HBM** if the
// second read misses. Whether it misses is the load-bearing uncertainty and it
// is an H100-shaped question: the re-read is CTA-local, which is why the same
// experiment was a null on GB10 (the in-tree receipt at
// `gated_delta_rule_decode_f32_strided_norm_smem` in the parent file — dgx2
// kill-switch A/B at C=128, +0.5%/-0.5%), but at n=16 on THIS part the live
// state is 50.33 MB against an H100's 50 MB of L2 and all 768 CTAs are
// resident at once, so the pass-1 stream evicts itself before pass 2 asks for
// it. The honest band, all of it arithmetic on the measured 57.27 us:
//
//   second read served by L2   | parent HBM | this kernel | saving/step (48)
//   ---------------------------|------------|-------------|-----------------
//   not at all                 | 150.99 MB  |    42.95 us | -687 us (-3.5%)
//   half                       | 125.83 MB  |    48.68 us | -412 us (-2.1%)
//   entirely                   | 100.66 MB  |    57.27 us | instruction-side
//
// (`units` = multiples of the 50.33 MB state; the parent is 3.00 and the
// compulsory floor is 2.00. The 2.25 is derived below.) The round-13 lever
// table's "945 us/step at an 80%-of-HBM target" is the ceiling of that band,
// not a prediction. Round 17's nsys is the receipt that picks a point in it;
// this file ships behind a default that says so.
//
// ── WHAT CHANGES — one thing ──────────────────────────────────────────────
// THE STATE IS READ ONCE, for 96 of its 128 rows. Each (sequence, head) tile
// is 128 rows x 128 columns of f32 = 64 KB, and this kernel splits its rows
// three ways:
//
//   rows [0, 72)     staged in SHARED MEMORY on pass 1, read back from SRAM
//                    on pass 2.                            1 global read
//   rows [72, 96)    RETAINED IN REGISTERS across the two passes.
//                                                          1 global read
//   rows [96, 128)   re-read from global on pass 2, exactly as the parent
//                    does for all 128.                     2 global reads
//
// Total 2.25 reads-equivalents + 1 write against the parent's 2 + 1 — 25% less
// state traffic. The 32 re-read rows are 16 KB per tile = 12.6 MB across the
// n=16 launch, comfortably inside 50 MB of L2 and re-read within a few
// microseconds, so their second read is the one that plausibly hits; the 96
// rows this kernel keeps are the ones that plausibly do not. That is the
// argument for putting the boundary here rather than at 0 or 128.
//
// The staged half is fetched with `float4` (LDG.128, 512 B per warp
// instruction against the parent's 128 B) because a flat 36 KB copy is
// vectorisable where a column walk is not — the columns of one row are
// contiguous, the rows of one column are 512 B apart. That is a second-order,
// instruction-side effect; the traffic reduction is the first-order one.
//
// ── WHY 72/24/32 — the ptxas sweep, sm_90a, CUDA 13.0, `--fmad=false` ──────
// The binding constraint is OCCUPANCY, and the parent sets the bar: at n=16
// its grid is `nv * n` = 768 CTAs of 128 threads on 132 SMs = 5.82 CTAs/SM =
// 23.3 warps/SM in ONE wave (its own limit is 12 CTAs/SM at 40 registers and
// 1 040 B of smem — the grid, not the kernel, is what caps it). A twin that
// halves the traffic and also halves the CTAs/SM has made a trade, not a fix,
// which is exactly how the n=1 twin became a 0.83x. So the shipped point is
// the one that keeps SIX CTAs resident — 792 slots for 768 CTAs, still one
// wave — at the lowest traffic that fits:
//
//   smem rows / reg rows | registers | smem B | CTAs/SM | traffic units
//   ---------------------|-----------|--------|---------|--------------
//   (parent)             |        40 |  1 040 | 5.82(*) | 3.000
//   72 / 16              |        66 | 37 904 |       6 | 2.313
//   **72 / 24**          |    **80** | **37 904** | **6** | **2.250**
//   72 / 32              |        96 | 37 904 |       5 | 2.188
//   80 / 24              |        72 | 42 000 |       5 | 2.188
//   80 / 48 (no re-read) |       128 | 42 000 |       4 | 2.000
//   64 / 64 (no re-read) |   255 +68 B spill | 33 808 |  2 | 2.000
//
//   (*) grid-limited, not resource-limited. CTAs/SM is
//   `min(65536 / (regs * 128), 233472 / smem)` on a GH100 SM.
//
// The last row is the one the tree had already measured from the other
// direction: full-width register retention spills (the in-tree note at
// `gated_delta_rule_decode_f32_strided_norm_half`, -11.6% e2e), and this
// sweep reproduces it — 255 registers and 68 B of spill stores, i.e. the
// retained columns become LOCAL memory, which is the traffic the lever was
// removing. All-shared is the other end: 128 rows x 128 columns x 4 B = 64 KB
// per CTA is past the 48 KB static limit AND caps residency at 3 CTAs/SM.
//
// ── NUMERICS: BIT-IDENTICAL, and this is the whole risk ───────────────────
//
// The parent's per-element reduction over `kd` is a SERIAL CHAIN owned by one
// thread. For state column `i`:
//
//     acc = 0;
//     for (j = 0; j < 128; j += 4)
//         acc = acc + (((h[j]*k[j] + h[j+1]*k[j+1]) + h[j+2]*k[j+2])
//                                                   + h[j+3]*k[j+3]);
//
// f32 addition is not associative, so ANY re-partition of `j` across threads —
// a warp-shuffle butterfly, a split into partial sums, a reduction tree of any
// shape — changes the answer. It follows that the ONLY bit-identical partition
// is the parent's own: one thread owns one column and walks every `j` itself.
// That is why this file keeps `block = (v_dim,1,1)` with `tid` = column, why
// it does NOT tile columns the way `gdn_decode_hopper.cu` does, and why there
// is no shuffle anywhere in the two dot products. The only cross-thread
// reduction here is the Frobenius clamp, which the parent also does across the
// whole head and which is reproduced shuffle for shuffle.
//
// Given that mapping, bit-identity is a STORAGE argument and nothing else:
//   * every float pass 2 consumes is the EXACT float pass 1 loaded from `H` at
//     the same index — an f32 round trip through shared memory or a register
//     is the identity, and each thread touches only its own `tid` column of
//     `smem_h`, so no other thread can observe or perturb it;
//   * `j` is visited in ascending order in both passes, in groups of four,
//     with the expression text copied character for character from
//     `gated_delta_rule_decode_f32_strided` — including the group shape
//     `h0*k0 + h1*k1 + h2*k2 + h3*k3` and the sequential `hk_dot +=`,
//     `q_dot +=` and `norm_acc +=` chains. The three row segments are three
//     spellings of the SAME loop body, split only by where `h0..h3` come
//     from; splitting a loop at a constant bound does not re-bracket the
//     accumulator, because the accumulator is carried across the split;
//   * the WRITE set and the write ADDRESSES are the parent's, element for
//     element, so the state a later kernel reads is laid out identically;
//   * this file compiles under `--fmad=false` (`kernels/gb10/common/
//     KERNEL.toml`, and the model's own KERNEL.toml sets it too), so there is
//     no FMA-contraction freedom left for a schedule change to exercise.
// `native_gdn_decode_hopper_microtest` asserts `state_diff == 0` and
// `out_diff == 0` against the parent on 6 legs (n in {1,4,16} x hs in
// {0.05, 20}), not a tolerance, and carries a KNOWN_BAD control that proves
// the comparison can fail.
//
// ── PRECONDITIONS ─────────────────────────────────────────────────────────
// `k_dim == v_dim == 128` and `blockDim.x == v_dim`: `smem_h` and the
// retention array are statically sized for it, and the state-norm clamp
// reduces across exactly the four warps the parent's `norm_sums[4]` and
// `tid / 32` assume. The host gates on this in
// `crates/spark-model/src/layers/ops/ssm_gdn_strided_hopper.rs`; the kernel
// re-checks rather than scribble past its staging buffer.
//
// ⚠️ It also declines below a WIDTH THRESHOLD, on the host and not here: one
// CTA per (sequence, head) means the grid is `nv * n`, and at six resident
// CTAs per SM this kernel wants at least one full wave of them. The rule, its
// threshold and its unit tests are `ops::gdn_decode_strided_hopper_accept`.

#define GDN_STR_MAX_KD 128
// Rows of the state tile staged in SHARED memory, then rows RETAINED in
// registers; whatever is left is re-read from global on pass 2, as the parent
// does for every row. See "WHY 72/24/32" above — these two literals are the
// whole occupancy/spill/traffic trade and are not tunable from the host.
#define GDN_STR_SMEM_ROWS 72
#define GDN_STR_REG_ROWS 24
#define GDN_STR_REREAD_FROM (GDN_STR_SMEM_ROWS + GDN_STR_REG_ROWS)
// `float4`s in the staged segment: rows x columns / 4.
#define GDN_STR_STAGE_V4 (GDN_STR_SMEM_ROWS * GDN_STR_MAX_KD / 4)

// Mirrors the parent file's guard so the clamp compiles identically here. The
// strided parent is declared BELOW `SSM_STATE_NORM_ENABLED` in
// `kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu`, so it carries the
// clamp and so must this twin.
#ifndef SSM_STATE_NORM_ENABLED
#define SSM_STATE_NORM_ENABLED
#define SSM_STATE_MAX_NORM 1000.0f
#endif

// Strided FP32 GDN decode, one state read for 96 of 128 rows — Hopper twin of
// `gated_delta_rule_decode_f32_strided`. Identical parameter list, identical
// bytes out, including the SSM_STATE_MAX_NORM clamp.
//
// `__launch_bounds__(128, 6)` is the occupancy CONTRACT the sweep above picked,
// not a hint: it makes ptxas fail loudly (as spill) if a future toolkit cannot
// hold the shipped split in 85 registers, instead of silently dropping the
// kernel to five resident CTAs and two waves.
//
// Launch: grid = (num_v_heads, batch_size), block = (128, 1, 1) — the parent's.
extern "C" __global__ __launch_bounds__(128, 6) void gated_delta_rule_decode_f32_strided_hopper_smem(
    float* __restrict__ h_state,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int out_stride
) {
    // The staging buffer is statically sized; a caller that ignores the
    // contract gets nothing done here, not a stomped shared-memory window.
    if (k_dim != GDN_STR_MAX_KD || v_dim != GDN_STR_MAX_KD) return;

    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= v_dim) return;

    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    float* H = h_state + ((b * num_v_heads + vh) * k_dim * v_dim);
    const float* q_ptr = query + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* k_ptr = key + (unsigned long long)b * qk_stride + kh * k_dim;
    const float* v_ptr = value + (unsigned long long)b * v_stride + vh * v_dim;

    float g_raw = gate[(unsigned long long)b * gb_stride + vh];
    const float g = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
    const float bt = beta[(unsigned long long)b * gb_stride + vh];

    __shared__ float smem_k[GDN_STR_MAX_KD];
    __shared__ float smem_q[GDN_STR_MAX_KD];
    // Rows [0, GDN_STR_SMEM_ROWS) of this tile, row-major exactly as in `H`.
    // Indexed `[row][tid]`, i.e. stride-1 across the threads of a warp, so
    // every column read below is bank-conflict-free.
    __shared__ __align__(16) float smem_h[GDN_STR_SMEM_ROWS][GDN_STR_MAX_KD];

    if (tid < k_dim) {
        smem_k[tid] = k_ptr[tid];
        smem_q[tid] = q_ptr[tid];
    }

    // ── Stage rows [0, GDN_STR_SMEM_ROWS), vectorised ─────────────────────
    // A flat copy of a contiguous window: `float4` per thread, each warp
    // instruction moving 512 B. `H` is `h_state` plus a multiple of
    // `k_dim * v_dim` floats, so both sides are 16 B aligned.
    {
        const float4* src = reinterpret_cast<const float4*>(H);
        float4* dst = reinterpret_cast<float4*>(&smem_h[0][0]);
        #pragma unroll 4
        for (unsigned int i = tid; i < GDN_STR_STAGE_V4; i += GDN_STR_MAX_KD) {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    float v_i = v_ptr[tid];

    // ── Pass 1: hk_dot = (H^T k)[tid], ascending j, the parent's chain ────
    float hk_dot = 0.0f;
    // ★ EVERY hreg index below is a compile-time constant. A runtime index
    //   puts the array in LOCAL memory, which converts the retention into
    //   spill traffic — the failure the full-width variant measured at
    //   -11.6% e2e and this file's sweep reproduces at 68 B of spill stores.
    float hreg[GDN_STR_REG_ROWS];
    #pragma unroll 4
    for (unsigned int j = 0; j < GDN_STR_SMEM_ROWS; j += 4) {
        float h0 = smem_h[j + 0][tid];
        float h1 = smem_h[j + 1][tid];
        float h2 = smem_h[j + 2][tid];
        float h3 = smem_h[j + 3][tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }
    #pragma unroll
    for (unsigned int j = 0; j < GDN_STR_REG_ROWS; j += 4) {
        const unsigned int r = j + GDN_STR_SMEM_ROWS;
        hreg[j + 0] = H[(r + 0) * v_dim + tid];
        hreg[j + 1] = H[(r + 1) * v_dim + tid];
        hreg[j + 2] = H[(r + 2) * v_dim + tid];
        hreg[j + 3] = H[(r + 3) * v_dim + tid];
        hk_dot += hreg[j + 0] * smem_k[r] + hreg[j + 1] * smem_k[r+1]
                + hreg[j + 2] * smem_k[r+2] + hreg[j + 3] * smem_k[r+3];
    }
    #pragma unroll 4
    for (unsigned int j = GDN_STR_REREAD_FROM; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        hk_dot += h0 * smem_k[j] + h1 * smem_k[j+1] + h2 * smem_k[j+2] + h3 * smem_k[j+3];
    }

    float v_new_i = (v_i - g * hk_dot) * bt;

    // ── Pass 2: state update + q_dot, ascending j, same three segments ────
    float q_dot = 0.0f;
#ifdef SSM_STATE_NORM_ENABLED
    float norm_acc = 0.0f;
#endif
    #pragma unroll 4
    for (unsigned int j = 0; j < GDN_STR_SMEM_ROWS; j += 4) {
        float h0 = smem_h[j + 0][tid];
        float h1 = smem_h[j + 1][tid];
        float h2 = smem_h[j + 2][tid];
        float h3 = smem_h[j + 3][tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
        // Frobenius accumulation from the registers we just stored, one add at
        // a time in ascending j — the parent's order exactly.
        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }
    #pragma unroll
    for (unsigned int j = 0; j < GDN_STR_REG_ROWS; j += 4) {
        const unsigned int r = j + GDN_STR_SMEM_ROWS;
        float h0 = hreg[j + 0];
        float h1 = hreg[j + 1];
        float h2 = hreg[j + 2];
        float h3 = hreg[j + 3];
        h0 = g * h0 + smem_k[r]     * v_new_i;
        h1 = g * h1 + smem_k[r + 1] * v_new_i;
        h2 = g * h2 + smem_k[r + 2] * v_new_i;
        h3 = g * h3 + smem_k[r + 3] * v_new_i;
        H[(r + 0) * v_dim + tid] = h0;
        H[(r + 1) * v_dim + tid] = h1;
        H[(r + 2) * v_dim + tid] = h2;
        H[(r + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[r] + h1 * smem_q[r+1] + h2 * smem_q[r+2] + h3 * smem_q[r+3];
#ifdef SSM_STATE_NORM_ENABLED
        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }
    #pragma unroll 4
    for (unsigned int j = GDN_STR_REREAD_FROM; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        h0 = g * h0 + smem_k[j]     * v_new_i;
        h1 = g * h1 + smem_k[j + 1] * v_new_i;
        h2 = g * h2 + smem_k[j + 2] * v_new_i;
        h3 = g * h3 + smem_k[j + 3] * v_new_i;
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
        q_dot += h0 * smem_q[j] + h1 * smem_q[j+1] + h2 * smem_q[j+2] + h3 * smem_q[j+3];
#ifdef SSM_STATE_NORM_ENABLED
        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
#endif
    }

    #ifdef SSM_STATE_NORM_ENABLED
    {
        // Whole-head Frobenius clamp — the parent's reduction, shuffle for
        // shuffle, over the four warps `norm_sums[4]` and `tid / 32` assume.
        // The rescale re-reads `H` exactly as the parent does: an f32 store
        // followed by a load is the identity, so keeping the parent's shape
        // costs nothing on the path that never fires and keeps the clamped
        // path bit-identical too.
        float local_sq = norm_acc;
        for (int offset = 16; offset >= 1; offset >>= 1)
            local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
        __shared__ float norm_sums[4];
        if (tid % 32 == 0) norm_sums[tid / 32] = local_sq;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int w = 0; w < 4; w++) total += norm_sums[w];
            norm_sums[0] = total;
        }
        __syncthreads();
        float head_norm_sq = norm_sums[0];
        if (head_norm_sq > SSM_STATE_MAX_NORM * SSM_STATE_MAX_NORM) {
            float scale = SSM_STATE_MAX_NORM * rsqrtf(head_norm_sq);
            for (unsigned int j = 0; j < k_dim; j++) {
                H[j * v_dim + tid] *= scale;
            }
        }
    }
    #endif

    float inv_sqrt_d = rsqrtf((float)k_dim);
    output[(unsigned long long)b * out_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
}
