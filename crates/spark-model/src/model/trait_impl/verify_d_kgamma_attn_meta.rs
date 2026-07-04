// SPDX-License-Identifier: AGPL-3.0-only

//! Attention-metadata upload for the K=γ (DFlash) verify path. Extracted
//! from `verify_d.rs` to keep that file under the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::types::TransformerModel;
use crate::layer::AttnMetadataDev;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Uploads positions/slots/block-table metadata for K=γ verify and
    /// returns the resulting `AttnMetadataDev`.
    ///
    /// Two layouts depending on whether prefill-mode attention is active:
    ///   decode:   K-row format (num_seqs=K, K copies of block_table)
    ///   prefill:  1-row format (num_seqs=1, single block_table row)
    /// Positions and slots are identical in both layouts.
    pub(super) fn build_kgamma_attn_metadata(
        &self,
        k: usize,
        seq: &SequenceState,
        bs: usize,
        use_prefill_attn: bool,
        stream: u64,
    ) -> Result<AttnMetadataDev> {
        let meta_base = self.buffers.scratch().offset(32768);
        let max_blocks = self.max_blocks_per_seq;

        // positions[K×4] — same for both layouts
        let positions: Vec<u32> = (0..k).map(|t| (seq.seq_len + t) as u32).collect();
        let pos_bytes =
            unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, k * 4) };
        self.gpu.copy_h2d_async(pos_bytes, meta_base, stream)?;

        // slots[K×8] — same for both layouts
        let mut slots = vec![0i64; k];
        for t in 0..k {
            let pos = seq.seq_len + t;
            let block_idx = pos / bs;
            let block_offset = pos % bs;
            let physical_block = seq.physical_block_for(block_idx).unwrap_or(0);
            slots[t] = (physical_block as i64) * (bs as i64) + (block_offset as i64);
        }
        let slot_bytes = unsafe { std::slice::from_raw_parts(slots.as_ptr() as *const u8, k * 8) };
        self.gpu
            .copy_h2d_async(slot_bytes, meta_base.offset(256), stream)?;

        if use_prefill_attn {
            // Prefill layout: single seq_len value, single block_table row.
            // prefill_attention_paged computes kv_len from seq_len_start + num_tokens
            // and reads block_table as a flat 1D array — no seq_idx stride.
            let seq_len_val = [(seq.seq_len + k) as u32];
            let sl_bytes =
                unsafe { std::slice::from_raw_parts(seq_len_val.as_ptr() as *const u8, 4) };
            self.gpu
                .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

            let mb = max_blocks as usize;
            let mut bt_buf = vec![0i32; mb];
            for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                bt_buf[j] = block as i32;
            }
            let bt_bytes =
                unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, mb * 4) };
            self.gpu
                .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

            Ok(AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(256),
                seq_len: meta_base.offset(512),
                block_table: meta_base.offset(768),
                max_blocks_per_seq: seq.block_table.len() as u32,
                num_seqs: 1,
            })
        } else {
            // Decode layout: K rows of seq_lens and block_table.
            let seq_lens: Vec<i32> = (0..k).map(|t| (seq.seq_len + t + 1) as i32).collect();
            let sl_bytes =
                unsafe { std::slice::from_raw_parts(seq_lens.as_ptr() as *const u8, k * 4) };
            self.gpu
                .copy_h2d_async(sl_bytes, meta_base.offset(512), stream)?;

            let mb = max_blocks as usize;
            let needed = k * mb;
            let mut bt_buf = vec![0i32; needed];
            for row in 0..k {
                for (j, &block) in seq.block_table.iter().enumerate().take(mb) {
                    bt_buf[row * mb + j] = block as i32;
                }
            }
            let bt_bytes =
                unsafe { std::slice::from_raw_parts(bt_buf.as_ptr() as *const u8, needed * 4) };
            self.gpu
                .copy_h2d_async(bt_bytes, meta_base.offset(768), stream)?;

            Ok(AttnMetadataDev {
                positions: meta_base,
                positions_h: meta_base,
                positions_w: meta_base,
                slot: meta_base.offset(256),
                seq_len: meta_base.offset(512),
                block_table: meta_base.offset(768),
                max_blocks_per_seq: max_blocks,
                num_seqs: k as u32,
            })
        }
    }
}
