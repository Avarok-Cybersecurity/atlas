// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 vocab projection: which kernel serves `M` rows of the LM head.
//!
//! Split out of `impl_a3.rs` when the batched arm pushed that file past the
//! 500-line cap. The decision is self-contained -- pick a kernel by row count
//! -- so it moves whole rather than being shaved out of the file it grew.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use crate::layers::ops;
use crate::weight_map::Fp8DenseWeight;

impl super::TransformerModel {
    /// Project `num_tokens` hidden rows through the FP8 LM head into `logits`.
    ///
    /// Every arm is bit-identical to the others: `dense_gemv_fp8w_batchm` uses
    /// the same per-element chain and the same pre-reduction scale as the M=1
    /// `dense_gemv_fp8w`, so the only thing that changes with M is how many
    /// times the ~1.27 GB head is read.
    pub(super) fn fp8_vocab_project(
        &self,
        fp8: &Fp8DenseWeight,
        hidden: DevicePtr,
        logits: DevicePtr,
        num_tokens: u32,
        v: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        // One pass over the head for up to 8 tokens. The loop below reads
        // the ENTIRE ~1.27 GB FP8 head once PER TOKEN, which is why the
        // FP8 head measured slower than BF16 at M>2 despite carrying half
        // the bytes. `dense_gemv_fp8w_batchm` is bit-identical to that
        // loop (same per-element chain, same k order, scale applied
        // per-thread before the reduction), so this is bandwidth only.
        //
        // It also covers M=2, replacing `dense_gemv_fp8w_batch2`: that
        // kernel's `mac4` contracts four products before accumulating,
        // which is NOT the M=1 chain its header claims parity with.
        if (2..=8).contains(&num_tokens) && self.dense_gemv_fp8w_batchm_kernel.0 != 0 {
            ops::dense_gemv_fp8w_batchm(
                self.gpu.as_ref(),
                self.dense_gemv_fp8w_batchm_kernel,
                hidden,
                fp8,
                logits,
                num_tokens,
                v,
                h,
                stream,
            )?;
        } else if num_tokens == 2 && self.dense_gemv_fp8w_batch2_kernel.0 != 0 {
            ops::dense_gemv_fp8w_batch2(
                self.gpu.as_ref(),
                self.dense_gemv_fp8w_batch2_kernel,
                hidden,
                fp8,
                logits,
                v,
                h,
                stream,
            )?;
        } else if num_tokens > 8 && self.dense_gemv_fp8w_batchm_kernel.0 != 0 {
            // Above the kernel's M bound, CHUNK rather than fall all the
            // way back to one row at a time. A cross-sequence verify at
            // C=8 presents 8*(gamma+1) = 64 rows; the per-token loop reads
            // the whole 1.27 GB head 64 times, chunks of 8 read it 8.
            // Each chunk is bit-identical to the rows it replaces, so the
            // result is unchanged either way.
            let mut done = 0u32;
            while done < num_tokens {
                let m = (num_tokens - done).min(8);
                ops::dense_gemv_fp8w_batchm(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_batchm_kernel,
                    hidden.offset(done as usize * h as usize * bf16),
                    fp8,
                    logits.offset(done as usize * v as usize * bf16),
                    m,
                    v,
                    h,
                    stream,
                )?;
                done += m;
            }
        } else {
            for i in 0..num_tokens as usize {
                ops::dense_gemv_fp8w(
                    self.gpu.as_ref(),
                    self.dense_gemv_fp8w_kernel,
                    hidden.offset(i * h as usize * bf16),
                    fp8,
                    logits.offset(i * v as usize * bf16),
                    v,
                    h,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
