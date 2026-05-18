// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 3b: paged metadata uploads (block_table delta + seq_len) and
//! `fill_slots_from_block_table` kernel for chunked prefill.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::SequenceState;

impl TransformerModel {
    pub(in crate::model) fn prefill_b_upload_paged(
        &self,
        seq: &mut SequenceState,
        total: usize,
        proc_start: usize,
        proc_count: usize,
        meta_base: DevicePtr,
        slot_offset: usize,
        kv_cache: &PagedKvCache,
        stream: u64,
    ) -> Result<()> {
        let bs = kv_cache.block_size();
        let current_blocks = seq.block_table.len();
        let upload_start = self
            .ensure_chunked_prefill_meta(seq, total, bs)?
            .uploaded_blocks;
        // Phase 6.3: when HSS sliding has occurred, the rolling window
        // can't be uploaded by absolute index without re-mapping. The
        // orchestrator path bypasses the production paged kernel that
        // reads this metadata, so skip the upload entirely in HSS mode.
        if upload_start < current_blocks && seq.hss_window_start() == 0 {
            let new_blocks = &seq.block_table[upload_start..];
            let bt_bytes = unsafe {
                std::slice::from_raw_parts(
                    new_blocks.as_ptr() as *const u8,
                    std::mem::size_of_val(new_blocks),
                )
            };
            let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
            self.gpu.copy_h2d_async(
                bt_bytes,
                block_table_base.offset(upload_start * std::mem::size_of::<u32>()),
                stream,
            )?;
            seq.chunked_prefill_meta.as_mut().unwrap().uploaded_blocks = current_blocks;
        }

        let seq_len_val = (proc_start + proc_count) as u32;
        let seq_len_bytes = unsafe {
            std::slice::from_raw_parts(
                &seq_len_val as *const u32 as *const u8,
                std::mem::size_of::<u32>(),
            )
        };
        let seq_len_base = seq.chunked_prefill_meta.as_ref().unwrap().seq_len;
        self.gpu
            .copy_h2d_async(seq_len_bytes, seq_len_base, stream)?;

        let block_table_base = seq.chunked_prefill_meta.as_ref().unwrap().block_table;
        ops::fill_slots_from_block_table(
            self.gpu.as_ref(),
            self.fill_slots_kernel,
            meta_base.offset(slot_offset),
            block_table_base,
            proc_start as u32,
            proc_count as u32,
            bs as u32,
            stream,
        )?;

        Ok(())
    }
}
