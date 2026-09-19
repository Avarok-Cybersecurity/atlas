// SPDX-License-Identifier: AGPL-3.0-only

//! Intra-block AttnRes stream (completed blocks + running partial).

use super::super::attnres::attnres_mix;

/// Matches HF `KimiDecoderLayer._forward_attn_residual`: mix the incoming
/// prefix with already-archived blocks, then at `layer_idx % block_size == 0`
/// archive that incoming prefix and reset the intra-block sum. Layer 0
/// therefore archives the embedding as its own source, not `embed + mixer`.
#[derive(Clone, Debug)]
pub struct AttnResStream {
    pub completed: Vec<Vec<f32>>,
    pub partial: Vec<f32>,
    block_size: usize,
}

impl AttnResStream {
    pub fn new(hidden: usize, block_size: usize) -> Self {
        Self {
            completed: Vec::new(),
            partial: vec![0.0; hidden],
            block_size: block_size.max(1),
        }
    }

    fn sources(&self) -> Vec<Vec<f32>> {
        // sources[0] is the skip (current prefix). Mix=0 must return this,
        // not the first archived block.
        let mut s = vec![self.partial.clone()];
        s.extend(self.completed.iter().cloned());
        s
    }

    pub fn mix(&self, query: &[f32], norm_w: &[f32], eps: f32, mix: f32) -> Vec<f32> {
        attnres_mix(&self.sources(), query, norm_w, eps, mix)
    }

    pub(super) fn add(&mut self, delta: &[f32]) {
        for (p, d) in self.partial.iter_mut().zip(delta) {
            *p += *d;
        }
    }

    pub(super) fn archive_incoming_at_block_start(&mut self, layer_idx: usize) {
        if layer_idx.is_multiple_of(self.block_size) {
            self.completed.push(self.partial.clone());
            self.partial.fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attnres_archives_incoming_at_block_start() {
        let mut s = AttnResStream::new(2, 4);
        s.partial = vec![1.0, 2.0];
        s.archive_incoming_at_block_start(0);
        assert_eq!(s.completed, vec![vec![1.0, 2.0]]);
        assert_eq!(s.partial, vec![0.0, 0.0]);
        s.add(&[0.5, 0.25]);
        assert_eq!(s.partial, vec![0.5, 0.25]);
        assert_eq!(
            s.completed[0],
            vec![1.0, 2.0],
            "archive is embed, not embed+mixer"
        );
        s.archive_incoming_at_block_start(1);
        assert_eq!(s.completed.len(), 1, "non-boundary layer must not archive");
        s.archive_incoming_at_block_start(4);
        assert_eq!(s.completed.len(), 2);
        assert_eq!(s.completed[1], vec![0.5, 0.25]);
        assert_eq!(s.partial, vec![0.0, 0.0]);
    }
}
