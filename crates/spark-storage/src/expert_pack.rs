// SPDX-License-Identifier: AGPL-3.0-only
//
// On-disk expert-record (de)serialization + the directory manifest.
//
// Two layers:
//   * Pure format functions (`pack_record` / `unpack_record`) that assemble and
//     parse one fixed-stride record in memory. No I/O, no CUDA, no safetensors —
//     unit-testable with synthetic bytes.
//   * A portable file writer/reader that lays those records into one file per
//     MoE layer, plus an `ExpertIndex` manifest (JSON) describing the geometry
//     so the streamer can reconstruct `ExpertRecordSpec` / `ExpertLayout` and
//     open the files without re-deriving anything from the checkpoint.
//
// The offline builder (checkpoint -> resident records) is the sole writer; the
// runtime streamer is a reader. This module is the contract between them.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::expert::ExpertKey;
use crate::expert::{ExpertLayout, ExpertRecordHeader, ExpertRecordSpec, Proj};

/// Borrowed packed+scale bytes for one projection, as they will sit on disk
/// (prefill-resident / transposed layout).
#[derive(Clone, Copy, Debug)]
pub struct ProjData<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [u8],
}

/// Borrowed view of one projection's sub-buffers inside a parsed record.
#[derive(Clone, Copy, Debug)]
pub struct ProjView<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [u8],
}

/// Assemble one complete `stride`-byte record: header at offset 0, each
/// projection's packed+scale bytes placed at the spec's sub-offsets, zero
/// padding everywhere else. Returns exactly `stride` bytes.
///
/// Errors (never panics) if any projection's byte lengths disagree with the
/// spec, or if the assembled payload would not fit in `stride` — those are
/// builder bugs we want surfaced loudly, not silently truncated records.
pub fn pack_record(
    spec: &ExpertRecordSpec,
    stride: u64,
    header: &ExpertRecordHeader,
    projs: &[ProjData; 3],
) -> Result<Vec<u8>> {
    let stride = stride as usize;
    if (spec.raw_bytes() as usize) > stride {
        bail!(
            "record stride {} smaller than raw record bytes {}",
            stride,
            spec.raw_bytes()
        );
    }
    let mut buf = vec![0u8; stride];
    let hdr = header.to_bytes();
    buf[..hdr.len()].copy_from_slice(&hdr);

    for p in Proj::ALL {
        let pb = spec.proj_bytes(p);
        let d = &projs[p as usize];
        if d.packed.len() as u64 != pb.packed_bytes {
            bail!(
                "{:?} packed len {} != expected {}",
                p,
                d.packed.len(),
                pb.packed_bytes
            );
        }
        if d.scale.len() as u64 != pb.scale_bytes {
            bail!(
                "{:?} scale len {} != expected {}",
                p,
                d.scale.len(),
                pb.scale_bytes
            );
        }
        let po = spec.packed_off(p) as usize;
        let so = spec.scale_off(p) as usize;
        buf[po..po + d.packed.len()].copy_from_slice(d.packed);
        buf[so..so + d.scale.len()].copy_from_slice(d.scale);
    }
    Ok(buf)
}

/// Parse a record `buf` (>= `spec.raw_bytes()`), returning the header and
/// borrowed views of each projection's sub-buffers. Validates the header magic
/// and version; returns an error on any mismatch.
pub fn unpack_record<'a>(
    spec: &ExpertRecordSpec,
    buf: &'a [u8],
) -> Result<(ExpertRecordHeader, [ProjView<'a>; 3])> {
    if (buf.len() as u64) < spec.raw_bytes() {
        bail!(
            "record buffer {} smaller than raw record bytes {}",
            buf.len(),
            spec.raw_bytes()
        );
    }
    let header = ExpertRecordHeader::from_bytes(buf)
        .context("record header magic/version mismatch (wrong file or format version?)")?;
    let mut views = [ProjView {
        packed: &[],
        scale: &[],
    }; 3];
    for p in Proj::ALL {
        let pb = spec.proj_bytes(p);
        let po = spec.packed_off(p) as usize;
        let so = spec.scale_off(p) as usize;
        views[p as usize] = ProjView {
            packed: &buf[po..po + pb.packed_bytes as usize],
            scale: &buf[so..so + pb.scale_bytes as usize],
        };
    }
    Ok((header, views))
}

/// Directory manifest describing a built expert store. Serialized as
/// `manifest.json` next to the per-layer `.xpr` files. This is the streamer's
/// entry point — everything it needs to reconstruct geometry and open files.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExpertIndex {
    /// Format version; must equal [`ExpertRecordHeader::VERSION`].
    pub version: u32,
    pub num_moe_layers: u32,
    pub num_experts: u32,
    pub inter: u64,
    pub hidden: u64,
    pub group_size: u64,
    pub sub_align: u64,
    pub fs_block_size: u64,
    pub record_stride: u64,
    pub record_raw_bytes: u64,
    /// `printf`-style template for per-layer file names, e.g. `experts_{:05}.xpr`.
    pub file_template: String,
    /// Dense MoE-layer index -> absolute model layer index. Lets the runtime map
    /// a model layer back to its expert file (dense attention layers are absent).
    pub moe_layer_to_model_layer: Vec<u32>,
}

impl ExpertIndex {
    pub const FILE_TEMPLATE: &'static str = "experts_{:05}.xpr";
    pub const MANIFEST_NAME: &'static str = "manifest.json";

    pub fn new(
        inter: u64,
        hidden: u64,
        group_size: u64,
        sub_align: u64,
        fs_block_size: u64,
        moe_layer_to_model_layer: Vec<u32>,
        num_experts: u32,
    ) -> Self {
        let spec = ExpertRecordSpec::new(inter, hidden, group_size, sub_align);
        let layout = ExpertLayout::from_spec(
            moe_layer_to_model_layer.len() as u32,
            num_experts,
            &spec,
            fs_block_size,
        );
        Self {
            version: ExpertRecordHeader::VERSION,
            num_moe_layers: moe_layer_to_model_layer.len() as u32,
            num_experts,
            inter,
            hidden,
            group_size,
            sub_align,
            fs_block_size,
            record_stride: layout.record_stride,
            record_raw_bytes: spec.raw_bytes(),
            file_template: Self::FILE_TEMPLATE.to_string(),
            moe_layer_to_model_layer,
        }
    }

    /// Load just the manifest (`manifest.json`) from a store dir — geometry
    /// only, no file handles. Lets the streamer size its arena before opening a
    /// tier. Validates the format version.
    #[cfg(unix)]
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let p = dir.join(Self::MANIFEST_NAME);
        let json = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
        let index: ExpertIndex =
            serde_json::from_str(&json).with_context(|| format!("parse {}", p.display()))?;
        if index.version != ExpertRecordHeader::VERSION {
            bail!(
                "manifest version {} != supported {}",
                index.version,
                ExpertRecordHeader::VERSION
            );
        }
        Ok(index)
    }

    pub fn spec(&self) -> ExpertRecordSpec {
        ExpertRecordSpec::new(self.inter, self.hidden, self.group_size, self.sub_align)
    }

    pub fn layout(&self) -> ExpertLayout {
        ExpertLayout::from_spec(
            self.num_moe_layers,
            self.num_experts,
            &self.spec(),
            self.fs_block_size,
        )
    }

    /// Per-layer file name for a dense MoE-layer index.
    pub fn file_name(&self, moe_layer: u32) -> String {
        // Only `{:05}` is supported; kept simple + explicit rather than a format
        // mini-language. Bump this if `file_template` ever needs to vary.
        format!("experts_{moe_layer:05}.xpr")
    }

    /// Total on-disk bytes across all layer files.
    pub fn total_bytes(&self) -> u64 {
        (self.num_moe_layers as u64) * self.layout().bytes_per_layer()
    }
}

pub use fs_impl::{ExpertFileReader, ExpertFileWriter};

#[path = "expert_pack_fs.rs"]
mod fs_impl;

#[cfg(test)]
#[path = "expert_pack_tests.rs"]
mod tests;
