// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//! The staged miss path's direct reads: the ring geometry that puts every
//! segment at the residue of its file offset mod 512, the aligned windows,
//! and the read plan of one miss. Split from `expert_arena.rs` (500-LoC cap).

use super::expert_stream::{DirectSeg, SlotLayout};

/// The direct miss path reads 512 B windows: a segment's ring region is its
/// bytes rounded up to a block plus this slack (a head block for the
/// rounding-down of its start, and the shift that puts the segment at the
/// same residue mod 512 as its file offset, plus a tail block).
pub(super) const DIRECT_BLOCK: usize = 512;
pub(super) const DIRECT_SLACK: usize = 3 * DIRECT_BLOCK;

/// Ring bytes of one slot: the three segments' padded regions, in slot order
/// (the buffered path writes the slot image contiguously from the slot's
/// start, which the padding never shortens).
pub(super) fn ring_slot_bytes(l: &SlotLayout) -> usize {
    direct_region(l, 3)
}

/// Ring offset of segment `which` (0 gate, 1 up, 2 down; 3 = the end).
pub(super) fn direct_region(l: &SlotLayout, which: usize) -> usize {
    let pad = |x: usize| x.next_multiple_of(DIRECT_BLOCK) + DIRECT_SLACK;
    [l.gate_bytes, l.up_bytes, l.down_bytes]
        .iter()
        .take(which)
        .map(|&b| pad(b))
        .sum()
}

/// The `O_DIRECT` windows covering file bytes `a0..a0 + len` in about
/// `parts` pieces: 512 B aligned, disjoint, in order, the first rounding
/// `a0` down and the last rounding the end up.
pub(super) fn direct_windows(a0: u64, len: u64, parts: usize) -> Vec<(u64, u64)> {
    let b = DIRECT_BLOCK as u64;
    let end = a0 + len;
    let (lo, hi) = (a0 / b * b, end.div_ceil(b) * b);
    let first = a0.div_ceil(b);
    let inner = hi.saturating_sub(first * b) / b;
    let parts = parts.clamp(1, inner.max(1) as usize) as u64;
    let step = inner.div_ceil(parts).max(1);
    let mut out = Vec::with_capacity(parts as usize + 1);
    let mut x = lo;
    let mut y = (first + step) * b;
    while x < hi {
        let y2 = y.min(hi);
        out.push((x, y2));
        x = y2;
        y += step * b;
    }
    out
}

/// One read of the staged miss path: slot index, then either a range of the
/// slot image through `read_expert_range` or a direct window of a shard;
/// `ring_off` is the byte offset of the destination inside the ring half.
pub(super) enum StagedRead {
    Buffered {
        slot: u32,
        key: (u32, u32),
        off: usize,
        len: usize,
        ring_off: usize,
    },
    Direct {
        slot: u32,
        shard: usize,
        file_off: u64,
        len: usize,
        need: usize,
        ring_off: usize,
    },
}

impl StagedRead {
    pub(super) fn slot(&self) -> u32 {
        match self {
            StagedRead::Buffered { slot, .. } | StagedRead::Direct { slot, .. } => *slot,
        }
    }
}

/// The direct reads of one miss whose ring slot starts at `slot_ring_off`:
/// each segment lands in its padded region at the residue of its file
/// offset mod 512, so every window is aligned at both ends; returns the
/// reads and, per segment, the ring offset its bytes start at.
pub(super) fn direct_plan(
    slot: u32,
    segs: &[DirectSeg],
    layout: &SlotLayout,
    slot_ring_off: usize,
    parts: usize,
) -> (Vec<StagedRead>, Vec<usize>) {
    let total: usize = segs.iter().map(|s| s.len).sum::<usize>().max(1);
    let mut reads = Vec::new();
    let mut bases = Vec::with_capacity(segs.len());
    for (which, seg) in segs.iter().enumerate() {
        let base = slot_ring_off
            + direct_region(layout, which)
            + DIRECT_BLOCK
            + (seg.file_off as usize % DIRECT_BLOCK);
        bases.push(base);
        let np = (parts * seg.len).div_ceil(total).max(1);
        let end = seg.file_off + seg.len as u64;
        for (x, y) in direct_windows(seg.file_off, seg.len as u64, np) {
            // the bytes of this window that belong to the segment
            let need = (y.min(end) - x) as usize;
            reads.push(StagedRead::Direct {
                slot,
                shard: seg.shard,
                file_off: x,
                len: (y - x) as usize,
                need,
                ring_off: (base as i64 + (x as i64 - seg.file_off as i64)) as usize,
            });
        }
    }
    (reads, bases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_aligned_disjoint_and_cover_the_range() {
        for (a0, len, parts) in [
            (1952u64, 3_870_720u64, 12usize),
            (4979605408 + 2048, 5_068_800, 4),
            (511, 1, 16),
            (512, 512, 3),
            (100, 100_000, 1),
        ] {
            let w = direct_windows(a0, len, parts);
            assert!(w[0].0 <= a0 && w[0].0 % 512 == 0 && a0 - w[0].0 < 512);
            let end = a0 + len;
            let last = w[w.len() - 1].1;
            assert!(last >= end && last % 512 == 0 && last - end < 512);
            for (i, &(x, y)) in w.iter().enumerate() {
                assert!(
                    x < y && x % 512 == 0 && y % 512 == 0,
                    "{a0} {len} {parts}: {w:?}"
                );
                if i > 0 {
                    assert_eq!(w[i - 1].1, x);
                }
            }
            assert!(w.len() <= parts + 1);
        }
    }

    #[test]
    fn a_plan_keeps_every_window_inside_its_region_at_the_file_residue() {
        let l = SlotLayout::new(3_870_720, 3_870_720, 5_068_800);
        let segs = [
            DirectSeg {
                shard: 1,
                file_off: 6931093408 + 7 * 3_870_720,
                slot_off: l.gate_off,
                len: l.gate_bytes,
            },
            DirectSeg {
                shard: 1,
                file_off: 8425273248 + 7 * 3_870_720,
                slot_off: l.up_off,
                len: l.up_bytes,
            },
            DirectSeg {
                shard: 1,
                file_off: 4979605408 + 7 * 5_068_800,
                slot_off: l.down_off,
                len: l.down_bytes,
            },
        ];
        let slot_ring = 3 * ring_slot_bytes(&l);
        let (reads, bases) = direct_plan(9, &segs, &l, slot_ring, 12);
        assert_eq!(bases.len(), 3);
        let mut covered = 0usize;
        for r in &reads {
            let StagedRead::Direct {
                slot,
                file_off,
                len,
                need,
                ring_off,
                ..
            } = r
            else {
                panic!()
            };
            assert_eq!(*slot, 9);
            assert_eq!(
                *ring_off % 512,
                (*file_off % 512) as usize,
                "ring address == file offset mod 512"
            );
            assert!(*need <= *len);
            covered += *need;
            // inside the slot's ring bytes
            assert!(*ring_off >= slot_ring && ring_off + len <= slot_ring + ring_slot_bytes(&l));
        }
        // every window's owed bytes: the slot image plus the head bytes
        // in front of each segment (the file has them; only a tail past the
        // end of the file may be short)
        let heads: usize = segs.iter().map(|s| (s.file_off % 512) as usize).sum();
        assert_eq!(covered, l.bytes + heads);
        for (which, (seg, &base)) in segs.iter().zip(&bases).enumerate() {
            assert_eq!(base % 512, (seg.file_off % 512) as usize);
            let region = slot_ring + direct_region(&l, which);
            assert!(
                base >= region + 512
                    && base + seg.len + 512 <= slot_ring + direct_region(&l, which + 1)
            );
        }
    }
}
