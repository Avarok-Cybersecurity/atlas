// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::types::TransformerModel;

impl TransformerModel {
    pub(super) fn restore_mtp_streams(&self, stream: u64) -> Result<()> {
        if self.mtp_streams_save.is_null() || self.config.hc_mult == 0 {
            return Ok(());
        }
        let bytes = self.config.hc_mult * self.config.hidden_size * 4;
        self.gpu.copy_d2d_async(
            self.mtp_streams_save,
            self.buffers.hc_streams(),
            bytes,
            stream,
        )
    }
}
