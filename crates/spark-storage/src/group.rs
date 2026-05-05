// SPDX-License-Identifier: AGPL-3.0-only
//
// Group identity for the high-speed-swap layer.
//
// A *group* is the unit of NVMe ↔ HBM movement: one (layer, block, kv_head)
// tuple's K-stripe or V-stripe. Each group is `block_size × head_dim ×
// elem_bytes` bytes contiguous on disk, sized to round up to the device's
// optimal I/O block (typically 4 KiB).
//
// The bijection (layer, block, kv_head) ⇆ group_id is computed deterministically
// from the dimensions; we never store the inverse mapping. Group IDs are dense
// 64-bit integers.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GroupId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    K = 0,
    V = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupKey {
    pub layer: u32,
    pub block: u32,
    pub kv_head: u16,
    pub kv_kind: u8, // KvKind as u8 (no Hash on enum w/o derive on payload)
}

impl GroupKey {
    pub fn new(layer: u32, block: u32, kv_head: u16, kv_kind: KvKind) -> Self {
        Self {
            layer,
            block,
            kv_head,
            kv_kind: kv_kind as u8,
        }
    }
    pub fn kind(self) -> KvKind {
        match self.kv_kind {
            0 => KvKind::K,
            _ => KvKind::V,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GroupLayout {
    pub num_layers: u32,
    pub num_blocks: u32,
    pub num_kv_heads: u16,
    /// `block_size × head_dim × elem_bytes`, rounded up to fs_block_size.
    pub group_stride: u64,
    /// Filesystem block size (4 KiB on most NVMe). Group stride is a multiple.
    pub fs_block_size: u64,
}

impl GroupLayout {
    pub fn new(
        num_layers: u32,
        num_blocks: u32,
        num_kv_heads: u16,
        block_size: u32,
        head_dim: u32,
        elem_bytes: u32,
        fs_block_size: u64,
    ) -> Self {
        let raw = (block_size as u64) * (head_dim as u64) * (elem_bytes as u64);
        let group_stride = raw.div_ceil(fs_block_size) * fs_block_size;
        Self {
            num_layers,
            num_blocks,
            num_kv_heads,
            group_stride,
            fs_block_size,
        }
    }

    /// Bytes occupied by one full layer in its file (K + V across all blocks).
    pub fn bytes_per_layer(&self) -> u64 {
        2 * (self.num_blocks as u64) * (self.num_kv_heads as u64) * self.group_stride
    }

    /// File offset for `key` within its layer's file.
    pub fn file_offset(&self, key: GroupKey) -> u64 {
        debug_assert!(key.block < self.num_blocks);
        debug_assert!(key.kv_head < self.num_kv_heads);
        let kv_stride = (self.num_kv_heads as u64) * self.group_stride;
        (key.block as u64) * (2 * kv_stride)
            + (key.kv_kind as u64) * kv_stride
            + (key.kv_head as u64) * self.group_stride
    }

    /// Dense `GroupId` for `key`.
    pub fn group_id(&self, key: GroupKey) -> GroupId {
        let per_layer = 2 * (self.num_blocks as u64) * (self.num_kv_heads as u64);
        let per_block = 2 * (self.num_kv_heads as u64);
        GroupId(
            (key.layer as u64) * per_layer
                + (key.block as u64) * per_block
                + (key.kv_kind as u64) * (self.num_kv_heads as u64)
                + (key.kv_head as u64),
        )
    }

    /// Number of bytes a single group occupies on disk (== group_stride).
    pub fn group_bytes(&self) -> u64 {
        self.group_stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_and_id_are_consistent() {
        let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
        // BF16: 16 * 128 * 2 = 4096 bytes — already aligned.
        assert_eq!(l.group_stride, 4096);
        let k0 = GroupKey::new(0, 0, 0, KvKind::K);
        assert_eq!(l.file_offset(k0), 0);
        let k1 = GroupKey::new(0, 0, 0, KvKind::V);
        assert_eq!(l.file_offset(k1), (8u64) * 4096);
        let k2 = GroupKey::new(0, 1, 0, KvKind::K);
        assert_eq!(l.file_offset(k2), 2 * 8 * 4096);
        let k3 = GroupKey::new(0, 0, 7, KvKind::V);
        assert_eq!(l.file_offset(k3), 8 * 4096 + 7 * 4096);
    }

    #[test]
    fn rounds_up_to_fs_block() {
        // Hypothetical odd shape: block_size=16, head_dim=96, BF16 → 3 KiB raw.
        let l = GroupLayout::new(1, 1, 1, 16, 96, 2, 4096);
        assert_eq!(l.group_stride, 4096);
    }

    #[test]
    fn bytes_per_layer_correct() {
        let l = GroupLayout::new(1, 4, 2, 16, 128, 2, 4096);
        // 4 blocks * 2 kv_heads * 4096 bytes * 2 (K+V) = 65536
        assert_eq!(l.bytes_per_layer(), 65536);
    }
}
