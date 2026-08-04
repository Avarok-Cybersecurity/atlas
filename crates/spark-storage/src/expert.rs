// SPDX-License-Identifier: AGPL-3.0-only
//
// Expert identity + on-disk record geometry for the MoE expert streamer.
//
// This is the expert-streaming analogue of `group.rs`: where a *group* is the
// unit of NVMe <-> HBM movement for the KV cache (one `(layer, block, kv_head)`
// K/V stripe), an *expert record* is the unit of movement for MoE weights — one
// `(moe_layer, expert)` tuple's full set of gate/up/down projections, stored
// contiguously and rounded up to the device's optimal I/O block (4 KiB).
//
// The engine that moves these records is the exact one that already ships for
// KV (`backend::{IoUringBackend, PosixBackend}` driven off a `Layout`): the
// backend only ever calls `fd(layer)`, `offset(key)` and `*_bytes()`. So the
// streamer reuses that machinery unchanged by presenting expert geometry
// through the same trio of accessors.
//
// Two facts make expert geometry simpler than KV geometry:
//   * There is no K/V duplication and no per-head striping — one record per
//     expert, period.
//   * On every Atlas MoE checkpoint the expert dims are uniform across MoE
//     layers, so `record_stride` is a single constant for the whole model.
//
// The bijection `(layer, expert) <-> record` is computed deterministically from
// the dims; nothing stores the inverse.

/// Dense 64-bit expert-record id (analogue of `group::GroupId`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExpertRecordId(pub u64);

/// Identifies one expert's weight record: `(moe_layer, expert)`.
///
/// `layer` is a *dense MoE-layer index* (0..num_moe_layers), not the model's
/// absolute layer index — dense attention layers carry no experts and are
/// skipped when the index is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

/// The three projections that make up one routed expert, in a fixed order.
///
/// Order is load-bearing: it is the order sub-buffers are laid out inside a
/// record and the order the streamer patches pointer tables in. Never reorder
/// without bumping [`ExpertRecordHeader::VERSION`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proj {
    Gate = 0,
    Up = 1,
    Down = 2,
}

impl Proj {
    pub const ALL: [Proj; 3] = [Proj::Gate, Proj::Up, Proj::Down];
}

/// Byte geometry of one NVFP4 projection sub-buffer inside an expert record.
///
/// NVFP4 (W4A4, group_size 16) stores each projection as two device buffers:
///   * `packed`  — E2M1 nibbles, 2 values/byte, `[K/2, N]` in prefill-resident
///     (transposed) layout, so `packed_bytes = N * K / 2`.
///   * `scale`   — per-group FP8-E4M3 block scales, `[K/16, N]`, so
///     `scale_bytes = N * K / group_size`.
///
/// The two per-projection scalars (`weight_scale_2`, `input_scale`) are carried
/// in the record header, which is why they are absent here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjBytes {
    pub packed_bytes: u64,
    pub scale_bytes: u64,
}

impl ProjBytes {
    /// `n` = output rows, `k` = contraction dim, `group_size` = NVFP4 block.
    pub fn nvfp4(n: u64, k: u64, group_size: u64) -> Self {
        Self {
            packed_bytes: n * k / 2,
            scale_bytes: n * k / group_size,
        }
    }
}

/// Where every sub-buffer of one expert record sits, relative to the record's
/// base. Shared by the offline builder (to place bytes) and the streamer (to
/// compute the device pointers it patches into the `ExpertPtrTable`).
///
/// Layout within a record (all offsets are relative to the record base and are
/// aligned to `sub_align`, which must satisfy the fused MoE kernels' pointer
/// alignment requirement):
///
/// ```text
///   [ header (ExpertRecordHeader::BYTES) ]
///   [ gate.packed ][ gate.scale ]
///   [ up.packed   ][ up.scale   ]
///   [ down.packed ][ down.scale ]
///   [ pad to record_stride ]
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertRecordSpec {
    pub inter: u64,
    pub hidden: u64,
    pub group_size: u64,
    pub sub_align: u64,
    /// `[packed_off, scale_off]` for gate, up, down — relative to record base.
    offsets: [(u64, u64); 3],
    bytes: [ProjBytes; 3],
    /// Total raw bytes (header + all sub-buffers), before rounding to a device
    /// I/O block. `ExpertLayout` rounds this up to `record_stride`.
    raw_bytes: u64,
}

/// Rounds `off` up to the next multiple of `align` (align must be a power of 2).
#[inline]
fn align_up(off: u64, align: u64) -> u64 {
    (off + align - 1) & !(align - 1)
}

impl ExpertRecordSpec {
    /// Build the canonical record layout for a model whose experts have the
    /// given `inter`(mediate) and `hidden` dims. `sub_align` is the alignment
    /// applied to every sub-buffer (256 is a safe default for CUTLASS/MMQ).
    pub fn new(inter: u64, hidden: u64, group_size: u64, sub_align: u64) -> Self {
        assert!(
            sub_align.is_power_of_two(),
            "sub_align must be a power of two"
        );
        // gate/up: N=inter, K=hidden ; down: N=hidden, K=inter.
        // packed = N*K/2 and scale = N*K/group_size are symmetric in (N,K),
        // so all three projections have identical byte sizes — but we keep them
        // per-projection so a future non-square expert stays correct.
        let bytes = [
            ProjBytes::nvfp4(inter, hidden, group_size), // gate
            ProjBytes::nvfp4(inter, hidden, group_size), // up
            ProjBytes::nvfp4(hidden, inter, group_size), // down
        ];
        let mut cursor = align_up(ExpertRecordHeader::BYTES, sub_align);
        let mut offsets = [(0u64, 0u64); 3];
        for i in 0..3 {
            let packed_off = cursor;
            cursor = align_up(packed_off + bytes[i].packed_bytes, sub_align);
            let scale_off = cursor;
            cursor = align_up(scale_off + bytes[i].scale_bytes, sub_align);
            offsets[i] = (packed_off, scale_off);
        }
        Self {
            inter,
            hidden,
            group_size,
            sub_align,
            offsets,
            bytes,
            raw_bytes: cursor,
        }
    }

    pub fn proj_bytes(&self, p: Proj) -> ProjBytes {
        self.bytes[p as usize]
    }

    /// Offset of a projection's packed-weight sub-buffer within the record.
    pub fn packed_off(&self, p: Proj) -> u64 {
        self.offsets[p as usize].0
    }

    /// Offset of a projection's block-scale sub-buffer within the record.
    pub fn scale_off(&self, p: Proj) -> u64 {
        self.offsets[p as usize].1
    }

    /// Total raw record bytes (header + sub-buffers), before I/O-block rounding.
    pub fn raw_bytes(&self) -> u64 {
        self.raw_bytes
    }

    /// Sum of all six sub-buffer payloads (excludes header + alignment padding).
    pub fn payload_bytes(&self) -> u64 {
        self.bytes
            .iter()
            .map(|b| b.packed_bytes + b.scale_bytes)
            .sum()
    }
}

/// Fixed-size, versioned header written at the front of every expert record.
///
/// Invariant D of the streaming-experts plan: *disk format = resident format*.
/// Nothing is transformed at fetch time, so the format must be self-describing
/// and versioned — there is no runtime enforcement otherwise. The header
/// carries exactly the per-expert data that is *not* recomputable from the
/// model dims: the two NVFP4 scalars per projection (`weight_scale_2` and
/// `input_scale`), plus enough identity/shape to detect a mismatched file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertRecordHeader {
    pub layer: u32,
    pub expert: u32,
    pub inter: u32,
    pub hidden: u32,
    pub group_size: u32,
    /// Per-projection `weight_scale_2` (per-tensor FP32 scale), gate/up/down.
    pub scale2: [f32; 3],
    /// Per-projection `input_scale` (activation scale). `None` = weight-only
    /// W4A16 path with no activation scale (the streamer patches a NULL device
    /// pointer). Presence is stored out of band in a flags byte, so equality is
    /// exact (no NaN sentinel to break `PartialEq`).
    pub input_scale: [Option<f32>; 3],
}

impl ExpertRecordHeader {
    pub const MAGIC: u32 = 0x5850_5254; // "XPRT"
    pub const VERSION: u32 = 1;
    /// Reserved on-disk header size. Generous vs. the packed fields so the
    /// format can grow without moving sub-buffer offsets.
    pub const BYTES: u64 = 256;

    /// Serialize into the fixed 256-byte on-disk header block. Layout:
    ///   u32 magic, u32 version, u32 layer, u32 expert,
    ///   u32 inter, u32 hidden, u32 group_size, u32 input_scale_flags,
    ///   `f32 scale2[3]`, `f32 input_scale[3]` (0.0 where absent), zero pad to 256.
    /// `input_scale_flags` bit `i` set => projection `i` has an activation scale.
    pub fn to_bytes(&self) -> [u8; Self::BYTES as usize] {
        let mut out = [0u8; Self::BYTES as usize];
        let mut w = |off: usize, v: u32| out[off..off + 4].copy_from_slice(&v.to_le_bytes());
        w(0, Self::MAGIC);
        w(4, Self::VERSION);
        w(8, self.layer);
        w(12, self.expert);
        w(16, self.inter);
        w(20, self.hidden);
        w(24, self.group_size);
        let mut flags = 0u32;
        for (i, s) in self.input_scale.iter().enumerate() {
            if s.is_some() {
                flags |= 1 << i;
            }
        }
        w(28, flags);
        for (i, s) in self.scale2.iter().enumerate() {
            out[32 + i * 4..36 + i * 4].copy_from_slice(&s.to_le_bytes());
        }
        for (i, s) in self.input_scale.iter().enumerate() {
            let v = s.unwrap_or(0.0);
            out[44 + i * 4..48 + i * 4].copy_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Parse a header block, validating magic + version. Returns `None` on any
    /// mismatch (wrong file, wrong version) — never panics on bad input.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::BYTES as usize {
            return None;
        }
        let r = |off: usize| -> u32 {
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };
        let rf = |off: usize| -> f32 {
            f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };
        if r(0) != Self::MAGIC || r(4) != Self::VERSION {
            return None;
        }
        let flags = r(28);
        let iscale = |i: usize, off: usize| -> Option<f32> {
            if flags & (1 << i) != 0 {
                Some(rf(off))
            } else {
                None
            }
        };
        Some(Self {
            layer: r(8),
            expert: r(12),
            inter: r(16),
            hidden: r(20),
            group_size: r(24),
            scale2: [rf(32), rf(36), rf(40)],
            input_scale: [iscale(0, 44), iscale(1, 48), iscale(2, 52)],
        })
    }
}

/// Deterministic file geometry for a directory of per-MoE-layer expert files.
///
/// One file per MoE layer (`experts_{layer:05}.xpr`), each holding
/// `num_experts` fixed-stride records back to back. `record_stride` is the raw
/// record size rounded up to `fs_block_size` (O_DIRECT requires the read
/// offset and length to be block-aligned). Mirrors `group::GroupLayout`'s
/// `fd`/`offset`/`*_bytes` surface so the KV backends drive it verbatim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertLayout {
    pub num_layers: u32,
    pub num_experts: u32,
    /// Fixed per-expert record stride on disk (a multiple of `fs_block_size`).
    pub record_stride: u64,
    pub fs_block_size: u64,
}

impl ExpertLayout {
    /// Build a layout from a record spec. `record_stride` is `spec.raw_bytes()`
    /// rounded up to `fs_block_size`.
    pub fn from_spec(
        num_layers: u32,
        num_experts: u32,
        spec: &ExpertRecordSpec,
        fs_block_size: u64,
    ) -> Self {
        let record_stride = spec.raw_bytes().div_ceil(fs_block_size) * fs_block_size;
        Self {
            num_layers,
            num_experts,
            record_stride,
            fs_block_size,
        }
    }

    /// Bytes occupied by one MoE layer's file (all experts, back to back).
    pub fn bytes_per_layer(&self) -> u64 {
        (self.num_experts as u64) * self.record_stride
    }

    /// File offset of `key`'s record within its layer file.
    pub fn file_offset(&self, key: ExpertKey) -> u64 {
        debug_assert!(key.expert < self.num_experts);
        (key.expert as u64) * self.record_stride
    }

    /// Dense record id across the whole model (layer-major).
    pub fn record_id(&self, key: ExpertKey) -> ExpertRecordId {
        ExpertRecordId((key.layer as u64) * (self.num_experts as u64) + (key.expert as u64))
    }

    /// Bytes of one record on disk (== `record_stride`); the fixed read size.
    pub fn record_bytes(&self) -> u64 {
        self.record_stride
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Qwen3.5-35B-A3B dims: inter=512, hidden=2048, group_size=16.
    const A3B_INTER: u64 = 512;
    const A3B_HIDDEN: u64 = 2048;
    const GS: u64 = 16;

    #[test]
    fn a3b_per_expert_payload_matches_formula() {
        // Plan formula: per-expert payload = 3 * inter * hidden * 9/16.
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let expected = 3 * A3B_INTER * A3B_HIDDEN * 9 / 16;
        assert_eq!(spec.payload_bytes(), expected);
        assert_eq!(expected, 1_769_472); // 1.6875 MiB, from recon.
    }

    #[test]
    fn projection_bytes_split_8_to_1() {
        // packed is 8/9 of payload, scale is 1/9 (0.5 byte/elem vs 1 byte/16).
        let pb = ProjBytes::nvfp4(A3B_INTER, A3B_HIDDEN, GS);
        assert_eq!(pb.packed_bytes, A3B_INTER * A3B_HIDDEN / 2);
        assert_eq!(pb.scale_bytes, A3B_INTER * A3B_HIDDEN / 16);
        assert_eq!(pb.packed_bytes, 8 * pb.scale_bytes);
    }

    #[test]
    fn sub_buffers_are_aligned_and_non_overlapping() {
        let align = 256;
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, align);
        // Header first, everything after it aligned and monotonic.
        let mut prev_end = ExpertRecordHeader::BYTES;
        for p in Proj::ALL {
            let po = spec.packed_off(p);
            let so = spec.scale_off(p);
            let pb = spec.proj_bytes(p);
            assert_eq!(po % align, 0, "packed off aligned");
            assert_eq!(so % align, 0, "scale off aligned");
            assert!(po >= prev_end, "packed does not overlap previous");
            assert!(so >= po + pb.packed_bytes, "scale does not overlap packed");
            prev_end = so + pb.scale_bytes;
        }
        assert!(spec.raw_bytes() >= prev_end);
    }

    #[test]
    fn layout_offsets_are_record_strided() {
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let layout = ExpertLayout::from_spec(40, 256, &spec, 4096);
        assert_eq!(layout.record_stride % 4096, 0, "O_DIRECT alignment");
        assert!(layout.record_stride >= spec.raw_bytes());
        assert_eq!(layout.file_offset(ExpertKey::new(3, 0)), 0);
        assert_eq!(
            layout.file_offset(ExpertKey::new(3, 5)),
            5 * layout.record_stride
        );
        assert_eq!(layout.bytes_per_layer(), 256 * layout.record_stride);
    }

    #[test]
    fn record_id_is_dense_layer_major() {
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let layout = ExpertLayout::from_spec(40, 256, &spec, 4096);
        assert_eq!(layout.record_id(ExpertKey::new(0, 0)).0, 0);
        assert_eq!(layout.record_id(ExpertKey::new(0, 255)).0, 255);
        assert_eq!(layout.record_id(ExpertKey::new(1, 0)).0, 256);
    }

    #[test]
    fn header_round_trips() {
        let h = ExpertRecordHeader {
            layer: 7,
            expert: 42,
            inter: A3B_INTER as u32,
            hidden: A3B_HIDDEN as u32,
            group_size: GS as u32,
            scale2: [0.5, 0.25, 1.5],
            input_scale: [Some(2.0), None, Some(3.0)],
        };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), ExpertRecordHeader::BYTES as usize);
        let back = ExpertRecordHeader::from_bytes(&bytes).expect("valid header");
        // Exact struct equality now holds (no NaN sentinel).
        assert_eq!(back, h);
        assert_eq!(back.scale2, [0.5, 0.25, 1.5]);
        assert_eq!(back.input_scale, [Some(2.0), None, Some(3.0)]);
    }

    #[test]
    fn header_rejects_bad_magic_and_version() {
        let mut bytes = ExpertRecordHeader {
            layer: 0,
            expert: 0,
            inter: 1,
            hidden: 1,
            group_size: GS as u32,
            scale2: [1.0; 3],
            input_scale: [Some(1.0); 3],
        }
        .to_bytes();
        // Corrupt the magic.
        bytes[0] ^= 0xFF;
        assert!(ExpertRecordHeader::from_bytes(&bytes).is_none());
        // Too-short buffer.
        assert!(ExpertRecordHeader::from_bytes(&bytes[..10]).is_none());
    }

    #[test]
    fn a3b_record_stride_is_4k_aligned_and_reasonable() {
        // The full-model on-disk size should land near the recon's ~200 GB for
        // 397B and a small multiple of payload for a3b. Here just sanity-check
        // a3b: stride within one 4K block of the raw size.
        let spec = ExpertRecordSpec::new(A3B_INTER, A3B_HIDDEN, GS, 256);
        let layout = ExpertLayout::from_spec(40, 256, &spec, 4096);
        assert!(layout.record_stride - spec.raw_bytes() < 4096);
    }
}
