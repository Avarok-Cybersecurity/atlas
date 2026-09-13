// SPDX-License-Identifier: AGPL-3.0-only

//! `QsaIndexer::decode_select`, split out of `qsa.rs` to keep it under the
//! 500-line cap.

use super::*;

impl QsaIndexer {
    pub fn decode_select(
        &self,
        st: &mut QsaSeqState,
        normed: DevicePtr,
        pos: usize,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        block_table_dev: DevicePtr,
        block_size: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<QsaSelection>> {
        anyhow::ensure!(
            pos == st.ingested,
            "QSA: decode at pos {pos} but {} tokens ingested — the indexer \
             cache lost sync (prefix-cache skip or a rewound sequence)",
            st.ingested
        );
        anyhow::ensure!(
            pos < self.max_tokens,
            "QSA: pos {pos} >= ATLAS_QSA_MAX_TOKENS"
        );

        let hd = self.hd as usize;
        let qkw = self.qk_width();
        // qk GEMV for this token; row 0 of the scratch.
        ops::cublas_bf16_proj_dense(
            normed,
            self.qk_proj_w,
            self.qk_scratch,
            1,
            qkw as u32,
            self.hidden,
            stream,
        )
        .context("QSA qk projection (decode)")?;
        gpu.copy_d2d_async(
            self.qk_scratch.offset(self.n_heads as usize * hd * 2),
            st.raw_keys.offset(pos * hd * 2),
            hd * 2,
            stream,
        )?;
        st.ingested = pos + 1;
        self.pool_new_blocks(st, gpu, stream)?;

        let visible = pos + 1;
        let complete = visible / self.ratio as usize;
        if complete <= self.block_topk as usize {
            return Ok(None); // provably all-visible: dense path is exact
        }

        // q prep + block scores.
        ops::qsa_qprep(
            gpu,
            self.k_qprep_k,
            self.qk_scratch,
            self.q_norm_w,
            self.q_post,
            self.n_heads,
            self.hd,
            self.rot,
            pos as u32,
            self.theta,
            self.eps,
            stream,
        )?;
        ops::qsa_score(
            gpu,
            self.k_score_k,
            self.q_post,
            st.block_keys,
            self.scores_dev,
            complete as u32,
            self.n_heads,
            self.hd,
            stream,
        )?;

        // Top-k -> ascending block ids -> the expanded token-index array,
        // plus `seq_len_dev` and the identity table. On device by default and
        // transfer-free; see `qsa_decode.rs`.
        let n_sel = self.decode_build_sel(
            &mut st.table_len,
            complete,
            visible,
            block_size,
            gpu,
            stream,
        )?;
        ops::qsa_gather(
            gpu,
            self.k_gather_k,
            k_pool,
            v_pool,
            block_table_dev,
            self.sel_dev,
            self.k_scratch,
            self.v_scratch,
            n_sel,
            block_size,
            self.nkv_attn,
            self.hd_attn,
            stream,
        )?;

        // `table_dev` / `seq_len_dev` — the scratch-as-paged-cache view — are
        // written by the selection tail above.
        let pages = (n_sel as usize).div_ceil(block_size as usize);

        Ok(Some(QsaSelection {
            k_scratch: self.k_scratch,
            v_scratch: self.v_scratch,
            table_dev: self.table_dev,
            seq_len_dev: self.seq_len_dev,
            n_sel,
            max_blocks: pages as u32,
        }))
    }
}
