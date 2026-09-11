// SPDX-License-Identifier: AGPL-3.0-only

//! Split-K W8A16 decode GEMV — `w8a16_gemv_splitk.cu` (#928).
//!
//! SSOT for the split plan. `w8a16_gemv.cu` puts `N_PER_BLOCK=4` outputs in a
//! 256-thread CTA, so the CTA count of an M=1 decode GEMV is `ceil(N/4)` and
//! nothing else — K does not enter it. On a 132-SM H100 that makes effective
//! bandwidth a function of N alone (nsys, 1xH100, Qwen/Qwen3.8-27B-FP8,
//! 2026-09-11 round 7, C=1 steady-state decode step 21.891 ms):
//!
//! | kernel                | shape            | grid | us/launch | GB/s  |
//! |-----------------------|------------------|------|-----------|-------|
//! | `w8a16_gemv_dual`     | N=17408x2 K=5120 | 4352 |      90.1 | 1,979 |
//! | `w8a16_gemv`          | N=16384  K=5120  | 4096 |         - | 1,852 |
//! | `w8a16_gemv_silu_input` | N=5120 K=17408 | 1280 |     103.9 |   858 |
//! | `w8a16_gemv` (k/v)    | N=1024   K=5120  |  256 |         - |   861 |
//!
//! ~8 of these CTAs co-reside per SM, so grid 4352 is ~4.1 full waves (a ~2%
//! tail), grid 1280 is ~1.2 waves — one full wave plus a 224-CTA tail that
//! leaves 83% of the machine idle for a whole second wave — and grid 256 never
//! fills the machine at all. Splitting K across CTAs is the lever that moves
//! the grid without touching the per-lane work.
//!
//! ptxas pins that 8: `nvcc -cubin -Xptxas -v -arch=sm_90a --fmad=false`
//! (CUDA 13.0, 2026-09-11) reports 32 registers and 1,056 B smem for BOTH
//! `w8a16_gemv` and `w8a16_gemv_splitk`, and 32 x 256 x 8 = 65,536 is exactly
//! the SM register file — so the split buys CTAs without costing occupancy.
//! (`w8a16_gemv_silu_input`, by contrast, needs 53 registers and fits only 4
//! CTAs/SM.)
//!
//! The plan below targets [`SPLITK_TARGET_BLOCKS`] CTAs, which is where the
//! measured shapes stop losing to wave quantisation, and caps the split at
//! [`SPLITK_MAX`] so the partial buffer and the combine stay trivial. On the
//! Qwen3.8-27B shapes it resolves to: FFN down (N=5120, K=17408) -> 4 splits,
//! grid 5,120, 5 chunk iterations per lane (the same per-lane profile as the
//! 1,979 GB/s `w8a16_gemv_dual`); attention k/v (N=1024, K=5120) -> 5 splits,
//! grid 1,280; gate/up (N=17408) -> 1 split, i.e. the already-fast shapes are
//! left exactly as they are.
//!
//! NUMERICS: `splits == 1` is bit-identical to `w8a16_gemv` (the kernel header
//! carries the argument). `splits > 1` reassociates the final combine of at
//! most [`SPLITK_MAX`] FP32 addends and nothing else, so the dispatch keeps it
//! behind `ATLAS_FFN_DOWN_SPLITK` (presence, default OFF). Oracle:
//! `examples/native_fp8_ffn_down_gemv_microtest`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// Outputs per CTA — `w8a16_gemv.cu`'s `N_PER_BLOCK`.
pub const GEMV_OUTS_PER_BLOCK: u32 = 4;
/// Lanes per output — `w8a16_gemv.cu`'s `threads_per_out` (256 / 4).
pub const GEMV_LANES_PER_OUT: u32 = 64;
/// K values one lane consumes per loop iteration (one `uint4` of FP8 bytes).
pub const GEMV_K_PER_CHUNK: u32 = 16;
/// Upper bound on splits. Caps the partial buffer at `SPLITK_MAX * N` FP32 and
/// the combine at `SPLITK_MAX` addends, which bounds the reassociation the
/// lever admits.
pub const SPLITK_MAX: u32 = 8;
/// CTA count the plan aims for. The round-7 table above reads ~1,850-1,980
/// GB/s from grid 4,096-4,352 and ~860 from grid 256-1,280, so this is the
/// measured knee, not a guess.
pub const SPLITK_TARGET_BLOCKS: u32 = 4096;

/// How `w8a16_gemv_splitk` is launched for one shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitKPlan {
    /// `gridDim.z`. 1 means "run the plain `w8a16_gemv` instead".
    pub splits: u32,
    /// Chunk-loop iterations one lane walks inside a split. Each iteration is
    /// `GEMV_LANES_PER_OUT` chunks of `GEMV_K_PER_CHUNK` K values, so a split
    /// owns a 64-chunk-aligned run and the lane -> chunk map is the scalar
    /// kernel's within it.
    pub iters_per_split: u32,
}

/// Total chunk-loop iterations one lane walks over the whole K.
fn total_iters(k: u32) -> u32 {
    div_ceil(k / GEMV_K_PER_CHUNK, GEMV_LANES_PER_OUT)
}

/// Pick a split for `[N, K]`. Never returns an empty split: `splits` is
/// recomputed from `iters_per_split` so `splits * iters_per_split` cannot
/// overshoot the chunk count by a whole split.
pub fn splitk_plan(n: u32, k: u32) -> SplitKPlan {
    let iters = total_iters(k);
    let base_blocks = div_ceil(n, GEMV_OUTS_PER_BLOCK);
    if base_blocks == 0 || iters <= 1 {
        return SplitKPlan {
            splits: 1,
            iters_per_split: iters.max(1),
        };
    }
    let want = div_ceil(SPLITK_TARGET_BLOCKS, base_blocks)
        .clamp(1, SPLITK_MAX)
        .min(iters);
    let iters_per_split = div_ceil(iters, want);
    SplitKPlan {
        splits: div_ceil(iters, iters_per_split),
        iters_per_split,
    }
}

/// Bytes the caller must reserve for the `[SPLITK_MAX, N]` FP32 partials.
/// Sized for the cap, not for a shape's plan, so one allocation serves every
/// projection a layer runs.
pub fn splitk_partial_bytes(n: u32) -> usize {
    SPLITK_MAX as usize * n as usize * std::mem::size_of::<f32>()
}

/// Stage 1: per-split partial sums into `partials` `[splits, N]` FP32.
///
/// Kernel: `w8a16_gemv_splitk(A, B, block_scale, partials, N, K, iters_per_split)`
/// Grid: (ceil(N/4), 1, splits)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_splitk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    partials: DevicePtr,
    n: u32,
    k: u32,
    plan: SplitKPlan,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, GEMV_OUTS_PER_BLOCK), 1, plan.splits])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(partials)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(plan.iters_per_split)
        .launch(stream)
}

/// Stage 2: combine the partials in increasing split order, round once to BF16.
///
/// Kernel: `w8a16_gemv_splitk_reduce(partials, C, N, splits)`
/// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)
pub fn w8a16_gemv_splitk_reduce(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    partials: DevicePtr,
    output: DevicePtr,
    n: u32,
    splits: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(partials)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(splits)
        .launch(stream)
}

/// Handles + scratch a call site needs to be allowed to take the split-K arm.
#[derive(Clone, Copy)]
pub struct SplitKGemv {
    pub splitk: KernelHandle,
    pub reduce: KernelHandle,
    pub partials: DevicePtr,
}

impl SplitKGemv {
    /// True when the arm is actually launchable: both entry points resolved on
    /// this target AND the partial scratch exists. A shadow that lacks the
    /// kernels leaves the handles at 0, which must fall back rather than
    /// launch nothing.
    pub fn armed(&self) -> bool {
        self.splitk.0 != 0 && self.reduce.0 != 0 && self.partials != DevicePtr::NULL
    }
}

/// The M=1 decode GEMV, split-K when the lever is on AND the shape asks for it.
///
/// `enabled` is `ModelLevers::ffn_down_splitk` (`ATLAS_FFN_DOWN_SPLITK`,
/// presence, default OFF). Everything else — a missing kernel, missing
/// scratch, or a shape whose plan comes back at 1 split — falls through to the
/// plain `w8a16_gemv`, which is the bit-exact reference the batch oracles
/// compare against.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_decode_gemv(
    gpu: &dyn GpuBackend,
    gemv: KernelHandle,
    split: SplitKGemv,
    enabled: bool,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let plan = splitk_plan(n, k);
    if enabled && split.armed() && plan.splits > 1 {
        w8a16_gemv_splitk(
            gpu,
            split.splitk,
            input,
            weight,
            block_scale,
            split.partials,
            n,
            k,
            plan,
            stream,
        )?;
        return w8a16_gemv_splitk_reduce(
            gpu,
            split.reduce,
            split.partials,
            output,
            n,
            plan.splits,
            stream,
        );
    }
    super::w8a16_gemv(gpu, gemv, input, weight, block_scale, output, n, k, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.8-27B dense FFN + attention decode shapes (hidden 5120,
    /// intermediate 17408, 4 kv heads x head_dim 256 = 1024).
    const DOWN: (u32, u32) = (5120, 17408);
    const GATE_UP: (u32, u32) = (17408, 5120);
    const KV: (u32, u32) = (1024, 5120);
    const O_PROJ: (u32, u32) = (5120, 6144);

    fn grid_z(n: u32, k: u32) -> u32 {
        div_ceil(n, GEMV_OUTS_PER_BLOCK) * splitk_plan(n, k).splits
    }

    #[test]
    fn down_projection_splits_to_the_target_grid() {
        let plan = splitk_plan(DOWN.0, DOWN.1);
        // 17408/16 = 1088 chunks / 64 lanes = 17 iterations, split 4 ways.
        assert_eq!(
            plan,
            SplitKPlan {
                splits: 4,
                iters_per_split: 5
            }
        );
        assert_eq!(grid_z(DOWN.0, DOWN.1), 5120);
        assert!(grid_z(DOWN.0, DOWN.1) >= SPLITK_TARGET_BLOCKS);
    }

    #[test]
    fn kv_projection_splits_to_its_chunk_count() {
        let plan = splitk_plan(KV.0, KV.1);
        // 5120/16 = 320 chunks / 64 lanes = 5 iterations; one per split.
        assert_eq!(
            plan,
            SplitKPlan {
                splits: 5,
                iters_per_split: 1
            }
        );
        assert_eq!(grid_z(KV.0, KV.1), 1280);
    }

    #[test]
    fn already_fast_shapes_are_left_alone() {
        // grid 4352 measures 1,979 GB/s — splitting it would only add a
        // combine pass to a shape that is already at the knee.
        assert_eq!(splitk_plan(GATE_UP.0, GATE_UP.1).splits, 1);
    }

    #[test]
    fn o_projection_takes_a_modest_split() {
        let plan = splitk_plan(O_PROJ.0, O_PROJ.1);
        // 6144/16 = 384 chunks / 64 = 6 iterations, 2 per split.
        assert_eq!(
            plan,
            SplitKPlan {
                splits: 3,
                iters_per_split: 2
            }
        );
    }

    #[test]
    fn no_plan_ever_leaves_an_empty_split() {
        for n in [4_u32, 256, 1024, 5120, 16384, 17408] {
            for k in [512_u32, 2048, 5120, 6144, 17408, 32768] {
                let plan = splitk_plan(n, k);
                let iters = total_iters(k);
                assert!(plan.splits >= 1 && plan.splits <= SPLITK_MAX, "{n}x{k}");
                assert!(plan.iters_per_split >= 1, "{n}x{k}");
                // Every split has work: dropping the last one would not cover K.
                assert!(
                    (plan.splits - 1) * plan.iters_per_split < iters,
                    "{n}x{k} leaves split {} empty",
                    plan.splits - 1
                );
                // And together they cover all of K.
                assert!(plan.splits * plan.iters_per_split >= iters, "{n}x{k}");
            }
        }
    }

    #[test]
    fn degenerate_shapes_stay_on_the_plain_kernel() {
        // K < one full lane sweep (64 chunks = 1024 K values) has nothing to
        // split; so does N=0, which never launches.
        assert_eq!(splitk_plan(5120, 1024).splits, 1);
        assert_eq!(splitk_plan(0, 17408).splits, 1);
    }

    #[test]
    fn partial_bytes_cover_the_cap_not_the_plan() {
        assert_eq!(splitk_partial_bytes(5120), 8 * 5120 * 4);
        let plan = splitk_plan(DOWN.0, DOWN.1);
        assert!(splitk_partial_bytes(DOWN.0) >= (plan.splits * DOWN.0) as usize * 4);
    }

    #[test]
    fn a_missing_kernel_or_scratch_disarms_the_split() {
        let full = SplitKGemv {
            splitk: KernelHandle(7),
            reduce: KernelHandle(9),
            partials: DevicePtr(0x1000),
        };
        assert!(full.armed());
        assert!(
            !SplitKGemv {
                splitk: KernelHandle(0),
                ..full
            }
            .armed()
        );
        assert!(
            !SplitKGemv {
                reduce: KernelHandle(0),
                ..full
            }
            .armed()
        );
        assert!(
            !SplitKGemv {
                partials: DevicePtr::NULL,
                ..full
            }
            .armed()
        );
    }
}
