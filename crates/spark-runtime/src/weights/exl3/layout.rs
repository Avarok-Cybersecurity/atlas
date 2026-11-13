// SPDX-License-Identifier: AGPL-3.0-only
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/src/core/exl3_layout.zig:17-192 and zig/src/core/exl3_prepare.zig:5-21 (Apache-2.0,
// ashhart/TensorFold, commit 3885dce; Copyright 2026 TensorFold contributors). Changes: translated from
// Zig to Rust; errors are a Rust enum; the expert layout owns its vectors; `strips` works on int16
// elements instead of bytes. See EXL3-ATTRIBUTION.md.

//! Responsibility: EXL3 group geometry as the pack stores it: tile counts and words a tile from the
//! trellis shape, a tensor-parallel rank's slice of a group as byte spans (whole 128-value Hadamard
//! blocks only), the stacked or ragged layout of a projection's routed experts, and the "strips"
//! re-layout (each 128-column block's tiles in K order) a decode kernel may prefer to the stored order.

use std::fmt;

use super::dequant::HADAMARD_DIM;

/// A geometry or layout the EXL3 format does not allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutError {
    /// The trellis shape is not `[K/16, N/16, words]` with 16..=128 words, a multiple of 8.
    BadTrellis,
    /// A tensor-parallel split that does not fall on whole 128-value blocks.
    SplitNotOnBlocks,
    /// An expert layout of no experts.
    NoExperts,
    /// Experts of one projection with different K or N.
    ExpertShapesDiffer,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BadTrellis => {
                "EXL3 trellis shape is not [K/16, N/16, 16 * bits] with 1 to 8 bits"
            }
            Self::SplitNotOnBlocks => "EXL3 split does not fall on whole 128-value Hadamard blocks",
            Self::NoExperts => "EXL3 expert layout has no experts",
            Self::ExpertShapesDiffer => "EXL3 experts of one projection differ in K or N",
        })
    }
}

impl std::error::Error for LayoutError {}

/// How a rank holds a group under tensor parallelism (every boundary a multiple of 128).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Split {
    /// All of it, on every rank.
    Whole,
    /// This rank's output columns: N tiles and `svh`.
    Col,
    /// This rank's input rows: K tiles and `suh` (a partial sum the ranks add).
    Row,
}

/// One EXL3 matrix `y = x @ W`, `W [K, N]`: trellis int16 `[K/16, N/16, words]`, `suh` fp16 `[K]`,
/// `svh` fp16 `[N]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    /// Tiles along K.
    pub k_tiles: u32,
    /// Tiles along N.
    pub n_tiles: u32,
    /// int16 words a 16x16 tile: `16 * bits` (the trellis's last dimension).
    pub words: u32,
}

impl Group {
    /// From the trellis entry's shape; a tile must hold whole K2 bytes and 1 to 8 bits a value.
    pub fn from_shape(shape: &[usize]) -> Result<Self, LayoutError> {
        let [kt, nt, w] = shape else {
            return Err(LayoutError::BadTrellis);
        };
        let max = (u32::MAX / 16) as usize;
        if *kt > max || *nt > max || *kt == 0 || *nt == 0 {
            return Err(LayoutError::BadTrellis);
        }
        if *w < 16 || !w.is_multiple_of(8) || *w > 128 {
            return Err(LayoutError::BadTrellis);
        }
        Ok(Self {
            k_tiles: *kt as u32,
            n_tiles: *nt as u32,
            words: *w as u32,
        })
    }

    /// Inputs.
    pub fn k(self) -> u32 {
        16 * self.k_tiles
    }

    /// Outputs.
    pub fn n(self) -> u32 {
        16 * self.n_tiles
    }

    /// Half bits a value (the kernels' K2): 4 for 2 bits, 16 for 8 bits.
    pub fn k2(self) -> u32 {
        self.words / 8
    }

    /// Bits a value (exact in halves).
    pub fn bits(self) -> f32 {
        self.words as f32 / 16.0
    }

    /// Bytes a tile.
    pub fn tile_bytes(self) -> u64 {
        u64::from(self.words) * 2
    }

    /// Bytes of the whole trellis.
    pub fn trellis_bytes(self) -> u64 {
        u64::from(self.k_tiles) * u64::from(self.n_tiles) * self.tile_bytes()
    }

    /// The rank's part of the group; a split that does not fall on 128-value blocks is refused.
    pub fn part(self, split: Split, rank: u32, world: u32) -> Result<Part, LayoutError> {
        if world == 0 || rank >= world || world > u32::MAX / HADAMARD_DIM as u32 {
            return Err(LayoutError::SplitNotOnBlocks);
        }
        let mut p = Part {
            group: self,
            trellis: Span::run(0, self.trellis_bytes()),
            suh: Span::run(0, 2 * u64::from(self.k())),
            svh: Span::run(0, 2 * u64::from(self.n())),
        };
        let block = u64::from(world) * HADAMARD_DIM as u64;
        let rank = u64::from(rank);
        match split {
            Split::Whole => {}
            Split::Row => {
                if !u64::from(self.k()).is_multiple_of(block) {
                    return Err(LayoutError::SplitNotOnBlocks);
                }
                let per = u64::from(self.k_tiles) / u64::from(world);
                p.group.k_tiles = per as u32;
                let row = u64::from(self.n_tiles) * self.tile_bytes();
                p.trellis = Span::run(rank * per * row, per * row);
                p.suh = Span::run(2 * 16 * rank * per, 2 * 16 * per);
            }
            Split::Col => {
                if !u64::from(self.n()).is_multiple_of(block) {
                    return Err(LayoutError::SplitNotOnBlocks);
                }
                let per = u64::from(self.n_tiles) / u64::from(world);
                p.group.n_tiles = per as u32;
                // One K-tile row at a time: this rank's N tiles are a run inside each row.
                p.trellis = Span {
                    offset: rank * per * self.tile_bytes(),
                    len: per * self.tile_bytes(),
                    stride: u64::from(self.n_tiles) * self.tile_bytes(),
                    count: u64::from(self.k_tiles),
                };
                p.svh = Span::run(2 * 16 * rank * per, 2 * 16 * per);
            }
        }
        Ok(p)
    }
}

/// Bytes of a stored tensor to read: `count` runs of `len` bytes, `stride` apart, from `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// First byte of the first run.
    pub offset: u64,
    /// Bytes a run.
    pub len: u64,
    /// Bytes between run starts (0 for a single run).
    pub stride: u64,
    /// Runs.
    pub count: u64,
}

impl Span {
    /// One contiguous run.
    pub fn run(offset: u64, len: u64) -> Self {
        Self {
            offset,
            len,
            stride: 0,
            count: 1,
        }
    }

    /// Bytes in all runs.
    pub fn bytes(self) -> u64 {
        self.len * self.count
    }

    /// The `i`th run's start.
    pub fn at(self, i: u64) -> u64 {
        self.offset + i * self.stride
    }
}

/// A rank's slice of a group: its geometry and where its trellis, `suh` and `svh` bytes sit in the
/// stored tensors (fp16 scales; packed sign words are expanded before slicing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    /// The slice's geometry.
    pub group: Group,
    /// Its trellis bytes.
    pub trellis: Span,
    /// Its `suh` bytes.
    pub suh: Span,
    /// Its `svh` bytes.
    pub svh: Span,
}

impl Part {
    /// Bytes of the slice.
    pub fn bytes(self) -> u64 {
        self.trellis.bytes() + self.suh.bytes() + self.svh.bytes()
    }
}

/// Whether a projection's experts share one width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertLayoutKind {
    /// One width for every expert.
    Stack,
    /// A width per expert, trellises back to back.
    Ragged,
}

/// Routed experts of one projection on a rank: one width (stack) or a width per expert (ragged).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    /// Stack or ragged.
    pub kind: ExpertLayoutKind,
    /// Tiles along K (shared).
    pub k_tiles: u32,
    /// Tiles along N (shared).
    pub n_tiles: u32,
    /// Each expert's int16 words a tile.
    pub words: Vec<u16>,
    /// Each expert's first int16 element in the projection's trellis buffer (back to back).
    pub offsets: Vec<u64>,
    /// int16 elements of every expert's trellis.
    pub elements: u64,
}

impl ExpertLayout {
    /// The layout of the rank's expert parts; parts must share K and N.
    pub fn of(parts: &[Part]) -> Result<Self, LayoutError> {
        let first = parts.first().ok_or(LayoutError::NoExperts)?;
        let (kt, nt) = (first.group.k_tiles, first.group.n_tiles);
        let mut words = Vec::with_capacity(parts.len());
        let mut offsets = Vec::with_capacity(parts.len());
        let mut at = 0u64;
        let mut same = true;
        for p in parts {
            if p.group.k_tiles != kt || p.group.n_tiles != nt {
                return Err(LayoutError::ExpertShapesDiffer);
            }
            words.push(u16::try_from(p.group.words).map_err(|_| LayoutError::BadTrellis)?);
            offsets.push(at);
            let added = (u64::from(kt) * u64::from(nt))
                .checked_mul(u64::from(p.group.words))
                .ok_or(LayoutError::BadTrellis)?;
            at = at.checked_add(added).ok_or(LayoutError::BadTrellis)?;
            if at > u64::MAX / 2 {
                return Err(LayoutError::BadTrellis);
            }
            same &= p.group.words == first.group.words;
        }
        Ok(Self {
            kind: if same {
                ExpertLayoutKind::Stack
            } else {
                ExpertLayoutKind::Ragged
            },
            k_tiles: kt,
            n_tiles: nt,
            words,
            offsets,
            elements: at,
        })
    }

    /// Experts.
    pub fn count(&self) -> usize {
        self.words.len()
    }

    /// Bytes of every expert's trellis.
    pub fn trellis_bytes(&self) -> u64 {
        2 * self.elements
    }

    /// Byte offset of expert `e`'s trellis in the projection buffer (16-byte aligned: a tile is 16 K2
    /// bytes).
    pub fn byte_offset(&self, e: usize) -> u64 {
        2 * self.offsets[e]
    }

    /// Experts at each K2 (index = K2, 2..=16).
    pub fn histogram(&self) -> [u32; 17] {
        let mut h = [0u32; 17];
        for &w in &self.words {
            h[usize::from(w / 8)] += 1;
        }
        h
    }

    /// Distinct widths (one grouped launch each).
    pub fn widths(&self) -> u32 {
        self.histogram().iter().filter(|&&c| c > 0).count() as u32
    }
}

/// Stored trellis int16 `[kt, nt, words]` -> strips int16 `[nt/8, kt, 8, words]`: each 128-column
/// block's tiles in K order, so a warp walking K reads contiguous words (`exl3_prepare.zig:5-21`).
pub fn strips(
    trellis: &[i16],
    kt: usize,
    nt: usize,
    words: usize,
) -> Result<Vec<i16>, LayoutError> {
    if kt == 0 || nt == 0 || !nt.is_multiple_of(8) || words == 0 || !words.is_multiple_of(2) {
        return Err(LayoutError::BadTrellis);
    }
    let count = kt
        .checked_mul(nt)
        .and_then(|t| t.checked_mul(words))
        .ok_or(LayoutError::BadTrellis)?;
    if trellis.len() != count {
        return Err(LayoutError::BadTrellis);
    }
    let run = 8 * words;
    let mut out = vec![0i16; count];
    for (i, dst) in out.chunks_exact_mut(run).enumerate() {
        let (nb, k) = (i / kt, i % kt);
        let src = (k * nt + nb * 8) * words;
        dst.copy_from_slice(&trellis[src..src + run]);
    }
    Ok(out)
}
