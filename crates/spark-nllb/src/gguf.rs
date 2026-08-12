// SPDX-License-Identifier: AGPL-3.0-only

//! Minimal standalone GGUF reader for the NLLB CPU runtime.
//!
//! `spark-nllb` is deliberately dependency-light, so rather than reach into
//! `spark-runtime`'s (crate-private) GGUF parser this module carries its own
//! ~compact reader. It only handles what the shipped NLLB-200 GGUF needs:
//! parse the container header + metadata KV block + tensor directory, then read
//! **F16 → f32** and **F32** tensors (the file is F16/F32 only — no K-quant
//! dequant path is required). Tensor `dims` are returned in GGUF (ggml) order,
//! i.e. fastest-varying first — the reverse of the PyTorch/HF shape.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use half::f16;

/// "GGUF" read as a little-endian `u32`.
const GGUF_MAGIC: u32 = 0x4655_4747;
/// Fallback alignment when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: usize = 32;

/// One tensor read out of a GGUF file, materialised as row-major `f32`.
#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    /// GGUF (ggml) dim order — fastest-varying first (reverse of torch shape).
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

/// A byte cursor over the mmapped file with little-endian primitive reads.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(
            self.pos + n <= self.buf.len(),
            "gguf: unexpected EOF (need {n} bytes at {})",
            self.pos
        );
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let b = self.take(len)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
}

/// Byte width of a scalar GGUF metadata value type (`None` for string/array).
fn scalar_size(vtype: u32) -> Option<usize> {
    match vtype {
        0 | 1 | 7 => Some(1), // u8, i8, bool
        2 | 3 => Some(2),     // u16, i16
        4..=6 => Some(4),     // u32, i32, f32
        10..=12 => Some(8),   // u64, i64, f64
        _ => None,
    }
}

/// Consume one metadata value of `vtype`, returning its `u64` widening for the
/// unsigned-32 case (used only to capture `general.alignment`); other types are
/// skipped and yield `None`.
fn skip_value(c: &mut Cursor, vtype: u32) -> Result<Option<u64>> {
    match vtype {
        4 => Ok(Some(c.u32()? as u64)), // u32 — may be general.alignment
        8 => {
            c.string()?;
            Ok(None)
        }
        9 => {
            let elem_type = c.u32()?;
            let count = c.u64()?;
            for _ in 0..count {
                skip_value(c, elem_type)?;
            }
            Ok(None)
        }
        v => {
            let sz = scalar_size(v).with_context(|| format!("gguf: unknown value type {v}"))?;
            c.take(sz)?;
            Ok(None)
        }
    }
}

fn align_up(pos: usize, align: usize) -> usize {
    pos.div_ceil(align) * align
}

/// A tensor directory entry (name, ggml dims, type id, data offset).
struct Entry {
    name: String,
    dims: Vec<usize>,
    type_id: u32,
    offset: usize,
}

/// Read every F16/F32 tensor from the GGUF at `path` into host `f32`.
pub fn read_gguf_f32(path: &Path) -> Result<Vec<GgufTensor>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading GGUF file {}", path.display()))?;
    let mut c = Cursor::new(&bytes);

    let magic = c.u32().context("gguf magic")?;
    ensure!(
        magic == GGUF_MAGIC,
        "not a GGUF file: magic 0x{magic:08x} != 0x{GGUF_MAGIC:08x}"
    );
    let version = c.u32().context("gguf version")?;
    ensure!(
        version == 2 || version == 3,
        "unsupported GGUF version {version} (only 2/3 supported)"
    );
    let tensor_count = c.u64().context("tensor_count")? as usize;
    let kv_count = c.u64().context("kv_count")? as usize;

    // Metadata KV block: skip everything except `general.alignment`.
    let mut alignment = DEFAULT_ALIGNMENT;
    for i in 0..kv_count {
        let key = c.string().with_context(|| format!("metadata key #{i}"))?;
        let vtype = c
            .u32()
            .with_context(|| format!("metadata type for {key:?}"))?;
        let val = skip_value(&mut c, vtype).with_context(|| format!("metadata value {key:?}"))?;
        if key == "general.alignment" {
            if let Some(a) = val {
                ensure!(a > 0, "general.alignment must be > 0");
                alignment = a as usize;
            }
        }
    }

    // Tensor directory.
    let mut entries = Vec::with_capacity(tensor_count);
    for i in 0..tensor_count {
        let name = c.string().with_context(|| format!("tensor name #{i}"))?;
        let ndim = c.u32().with_context(|| format!("ndim for {name:?}"))? as usize;
        ensure!(ndim <= 8, "tensor {name:?} has implausible ndim {ndim}");
        let mut dims = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            dims.push(c.u64()? as usize);
        }
        let type_id = c.u32().with_context(|| format!("ggml_type for {name:?}"))?;
        let offset = c.u64().with_context(|| format!("offset for {name:?}"))? as usize;
        entries.push(Entry {
            name,
            dims,
            type_id,
            offset,
        });
    }

    let data_offset = align_up(c.pos, alignment);
    ensure!(
        data_offset <= bytes.len(),
        "tensor-data section starts past EOF ({data_offset} > {})",
        bytes.len()
    );

    let mut out = Vec::with_capacity(entries.len());
    for e in entries {
        let n: usize = e.dims.iter().product();
        let start = data_offset + e.offset;
        let nbytes = n * type_bytes(e.type_id)?;
        ensure!(
            start + nbytes <= bytes.len(),
            "tensor {:?} data runs past EOF ({} > {})",
            e.name,
            start + nbytes,
            bytes.len()
        );
        let raw = &bytes[start..start + nbytes];
        let data = match e.type_id {
            0 => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            1 => raw
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            other => bail!(
                "tensor {:?}: unsupported ggml type id {other} (F16/F32 only)",
                e.name
            ),
        };
        out.push(GgufTensor {
            name: e.name,
            dims: e.dims,
            data,
        });
    }
    Ok(out)
}

/// On-disk element width for the two element types this reader accepts.
fn type_bytes(type_id: u32) -> Result<usize> {
    match type_id {
        0 => Ok(4),
        1 => Ok(2),
        other => bail!("unsupported ggml type id {other}"),
    }
}

#[cfg(test)]
#[path = "gguf_tests.rs"]
mod tests;
