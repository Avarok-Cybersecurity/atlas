// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/cuda/qlinear.zig:250-268 (GemvPart, GemvManyPart, Seg and Segs: the lane GEMV's by-value
// launch descriptors). Copyright 2026 TensorFold contributors. Licensed under the Apache License,
// Version 2.0; code written before TensorFold 0.6.0 is also available under the MIT License. See
// TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig extern structs to explicit little-endian byte images; a Seg
// carries its weight's own ticket pointer instead of a ticket offset.

//! Responsibility: the byte images of the lane GEMV's by-value kernel parameters
//! (`kernels/ada/common/affine4_lane_ada.cu`: `Part` 48 bytes, `Segs` 264, `ManyParts` 232), laid out
//! exactly like the C++ structs on a 64-bit target (pointers 8-aligned, ints 4 bytes, tail padding to 8).

use spark_runtime::gpu::DevicePtr;

use super::weight::{AffineWeight, MAX_SEGS};

/// Little-endian byte image of a by-value kernel parameter struct, aligned like the C++ struct.
#[derive(Default)]
pub struct ParamBytes(pub Vec<u8>);

impl ParamBytes {
    pub fn ptr(&mut self, p: DevicePtr) -> &mut Self {
        self.0.extend_from_slice(&p.0.to_le_bytes());
        self
    }
    pub fn int(&mut self, v: usize) -> &mut Self {
        self.0.extend_from_slice(&(v as i32).to_le_bytes());
        self
    }
    /// Pad to a multiple of 8 (every struct here holds pointers).
    pub fn align8(&mut self) -> &mut Self {
        while !self.0.len().is_multiple_of(8) {
            self.0.push(0);
        }
        self
    }
}

/// `avarok_affine_lane::Part` (48 bytes): w, scales, biases, out, then n, npad, sk, tiles.
pub fn part_bytes(b: &mut ParamBytes, w: &AffineWeight, out: DevicePtr) {
    b.ptr(w.words)
        .ptr(w.scales)
        .ptr(w.biases)
        .ptr(out)
        .int(w.n)
        .int(w.npad)
        .int(w.sk())
        .int(w.tiles());
}

/// One `(weight, output)` member of a multi-projection launch.
pub type Member<'a> = (&'a AffineWeight, DevicePtr);

/// `avarok_affine_lane::Segs` (264 bytes) for `_seg`: per member its Part, its own tickets, its first
/// block and its partial slot offset (in units of 16 x 64 floats). Returns the bytes, the total block
/// count and the partial slots used.
pub fn segs_bytes(members: &[Member<'_>]) -> (Vec<u8>, usize, usize) {
    let mut b = ParamBytes::default();
    let (mut blocks, mut work) = (0usize, 0usize);
    for &(w, out) in members {
        part_bytes(&mut b, w, out);
        b.ptr(w.tickets).int(blocks).int(work);
        let sk = w.sk();
        blocks += w.tiles() * sk;
        if sk > 1 {
            work += w.tiles() * sk;
        }
    }
    b.0.resize(64 * MAX_SEGS, 0);
    b.int(members.len()).align8();
    (b.0, blocks, work)
}

/// `avarok_affine_lane::ManyParts` (232 bytes) for `_many`: per member its Part, its first block and its
/// persistent block count. Returns the bytes and the total block count.
pub fn many_bytes(members: &[Member<'_>], blocks_each: usize) -> (Vec<u8>, usize) {
    let mut b = ParamBytes::default();
    let mut blocks = 0usize;
    for &(w, out) in members {
        let local = w.tiles().min(blocks_each).max(1);
        part_bytes(&mut b, w, out);
        b.int(blocks).int(local);
        blocks += local;
    }
    b.0.resize(56 * MAX_SEGS, 0);
    b.int(members.len()).align8();
    (b.0, blocks)
}
