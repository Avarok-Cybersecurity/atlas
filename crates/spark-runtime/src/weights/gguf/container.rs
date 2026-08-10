// SPDX-License-Identifier: AGPL-3.0-only
//! Zero-copy GGUF v3 container parser.
//!
//! Parses the header, metadata key/value block, and tensor directory of a GGUF
//! file directly over an mmapped `&[u8]` (the caller owns the mapping). It does
//! NOT touch the tensor data section — it only computes where that section
//! begins (`data_offset`) and, per tensor, the relative byte `offset` into it.
//!
//! Container layout (little-endian, GGUF v2/v3):
//!   header:  u32 magic ("GGUF"), u32 version, u64 tensor_count, u64 kv_count
//!   metadata:  kv_count × { string key, u32 value_type, value }
//!   tensors:   tensor_count × { string name, u32 ndim, u64 dims[ndim],
//!                               u32 ggml_type, u64 offset }
//!   pad to `general.alignment` (default 32)
//!   tensor data (not parsed here)
//!
//! This is a private parser module that deliberately exposes a complete GGUF
//! accessor surface (typed metadata getters, tensor directory lookups); not
//! every accessor is wired into the loader yet, so `dead_code` is allowed here.
#![allow(dead_code)]

use anyhow::{Context, Result, bail, ensure};

mod reader;
use reader::{Reader, align_up, read_value};

/// "GGUF" read as a little-endian u32.
const GGUF_MAGIC: u32 = 0x4655_4747;
/// Fallback alignment when `general.alignment` is absent.
const DEFAULT_ALIGNMENT: u64 = 32;

// ---------------------------------------------------------------------------
// ggml type ids
// ---------------------------------------------------------------------------

/// A ggml tensor element type, identified by its on-disk type id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    /// PrismML-private ternary (id 41).
    Q1_0,
    /// PrismML-private ternary (id 42). Group size (128 vs 64) is NOT encoded
    /// in the type id — see [`Q2Group`].
    Q2_0,
}

/// Distinguishes the two PrismML `Q2_0` (id 42) block layouts, which share a
/// type id but differ in group size. Not derivable from the type id alone; the
/// loader selects it from file metadata / fork provenance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Q2Group {
    /// Group-128, fp16 scale at front, 34-byte block (2.125 bpw) — the shipped
    /// Ternary-Bonsai file.
    G128,
    /// Fork-master variant: group-64, 18-byte block (2.25 bpw).
    G64,
}

impl GgmlType {
    /// Decode an on-disk ggml type id.
    pub fn from_id(id: u32) -> Result<Self> {
        use GgmlType::*;
        Ok(match id {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            16 => Iq2Xxs,
            17 => Iq2Xs,
            18 => Iq3Xxs,
            19 => Iq1S,
            20 => Iq4Nl,
            21 => Iq3S,
            22 => Iq2S,
            23 => Iq4Xs,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            29 => Iq1M,
            30 => Bf16,
            34 => Tq1_0,
            35 => Tq2_0,
            41 => Q1_0,
            42 => Q2_0,
            other => bail!("unsupported ggml type id {other}"),
        })
    }

    /// The on-disk type id.
    pub fn id(self) -> u32 {
        use GgmlType::*;
        match self {
            F32 => 0,
            F16 => 1,
            Q4_0 => 2,
            Q4_1 => 3,
            Q5_0 => 6,
            Q5_1 => 7,
            Q8_0 => 8,
            Q8_1 => 9,
            Q2_K => 10,
            Q3_K => 11,
            Q4_K => 12,
            Q5_K => 13,
            Q6_K => 14,
            Q8_K => 15,
            Iq2Xxs => 16,
            Iq2Xs => 17,
            Iq3Xxs => 18,
            Iq1S => 19,
            Iq4Nl => 20,
            Iq3S => 21,
            Iq2S => 22,
            Iq4Xs => 23,
            I8 => 24,
            I16 => 25,
            I32 => 26,
            I64 => 27,
            F64 => 28,
            Iq1M => 29,
            Bf16 => 30,
            Tq1_0 => 34,
            Tq2_0 => 35,
            Q1_0 => 41,
            Q2_0 => 42,
        }
    }

    /// `(elements_per_block, bytes_per_block)`. For `Q2_0` the group must be
    /// supplied since two layouts share the id. Errors for types whose block
    /// layout is not needed by the loader (IQ family, `Q1_0`).
    pub(crate) fn block_layout(self, q2: Q2Group) -> Result<(usize, usize)> {
        use GgmlType::*;
        Ok(match self {
            F32 => (1, 4),
            F16 | Bf16 => (1, 2),
            F64 => (1, 8),
            I8 => (1, 1),
            I16 => (1, 2),
            I32 => (1, 4),
            I64 => (1, 8),
            Q4_0 => (32, 18),
            Q4_1 => (32, 20),
            Q5_0 => (32, 22),
            Q5_1 => (32, 24),
            Q8_0 => (32, 34),
            Q8_1 => (32, 36),
            Q2_K => (256, 84),
            Q3_K => (256, 110),
            Q4_K => (256, 144),
            Q5_K => (256, 176),
            Q6_K => (256, 210),
            Q8_K => (256, 292),
            Tq1_0 => (256, 54),
            Tq2_0 => (256, 66),
            Q2_0 => match q2 {
                Q2Group::G128 => (128, 34),
                Q2Group::G64 => (64, 18),
            },
            other => bail!("block layout not defined for {other:?}"),
        })
    }
}

// ---------------------------------------------------------------------------
// metadata values
// ---------------------------------------------------------------------------

/// A GGUF metadata value (all 13 value types, including nested arrays).
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    /// Widen any (non-negative) integer / bool to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            MetaValue::U8(v) => *v as u64,
            MetaValue::U16(v) => *v as u64,
            MetaValue::U32(v) => *v as u64,
            MetaValue::U64(v) => *v,
            MetaValue::I8(v) if *v >= 0 => *v as u64,
            MetaValue::I16(v) if *v >= 0 => *v as u64,
            MetaValue::I32(v) if *v >= 0 => *v as u64,
            MetaValue::I64(v) if *v >= 0 => *v as u64,
            MetaValue::Bool(v) => *v as u64,
            _ => return None,
        })
    }

    /// Widen any integer / bool to `i64`.
    pub fn as_i64(&self) -> Option<i64> {
        Some(match self {
            MetaValue::U8(v) => *v as i64,
            MetaValue::U16(v) => *v as i64,
            MetaValue::U32(v) => *v as i64,
            MetaValue::U64(v) => return i64::try_from(*v).ok(),
            MetaValue::I8(v) => *v as i64,
            MetaValue::I16(v) => *v as i64,
            MetaValue::I32(v) => *v as i64,
            MetaValue::I64(v) => *v,
            MetaValue::Bool(v) => *v as i64,
            _ => return None,
        })
    }

    /// Read a float (widening integers too).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::F32(v) => Some(*v as f64),
            MetaValue::F64(v) => Some(*v),
            _ => self.as_i64().map(|x| x as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// tensor directory + parsed file
// ---------------------------------------------------------------------------

/// One entry from the tensor directory. `offset` is relative to the tensor-data
/// section (i.e. to [`GgufFile::data_offset`]). `dims` are in GGUF (ggml) order
/// — fastest-varying first, i.e. the reverse of the HuggingFace/torch shape.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    /// Product of `dims` (1 for a 0-d scalar).
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

/// A fully-parsed GGUF container header (no tensor data read).
#[derive(Debug, Clone)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<(String, MetaValue)>,
    pub tensors: Vec<TensorInfo>,
    /// Alignment (`general.alignment`, default 32).
    pub alignment: u64,
    /// Absolute byte offset (from start of file) where the tensor-data section
    /// begins. A tensor's absolute offset is `data_offset + tensor.offset`.
    pub data_offset: usize,
}

impl GgufFile {
    /// Parse the header/metadata/tensor-directory of a GGUF v2/v3 buffer.
    pub fn parse(buf: &[u8]) -> Result<GgufFile> {
        let mut r = Reader::new(buf);

        let magic = r.u32().context("reading magic")?;
        ensure!(
            magic == GGUF_MAGIC,
            "not a GGUF file: magic 0x{magic:08x} != 0x{GGUF_MAGIC:08x}"
        );
        let version = r.u32().context("reading version")?;
        ensure!(
            version == 2 || version == 3,
            "unsupported GGUF version {version} (only 2/3 supported)"
        );

        let tensor_count = r.u64().context("reading tensor_count")?;
        let kv_count = r.u64().context("reading kv_count")?;

        let mut metadata = Vec::with_capacity(kv_count.min(1 << 16) as usize);
        for i in 0..kv_count {
            let key = r.string().with_context(|| format!("metadata key #{i}"))?;
            let vtype = r
                .u32()
                .with_context(|| format!("metadata type for {key:?}"))?;
            let value =
                read_value(&mut r, vtype).with_context(|| format!("metadata value for {key:?}"))?;
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(1 << 20) as usize);
        for i in 0..tensor_count {
            let name = r.string().with_context(|| format!("tensor name #{i}"))?;
            let ndim = r.u32().with_context(|| format!("ndim for {name:?}"))?;
            ensure!(ndim <= 8, "tensor {name:?} has implausible ndim {ndim}");
            let mut dims = Vec::with_capacity(ndim as usize);
            for _ in 0..ndim {
                dims.push(r.u64()? as usize);
            }
            let type_id = r.u32().with_context(|| format!("ggml_type for {name:?}"))?;
            let ggml_type =
                GgmlType::from_id(type_id).with_context(|| format!("tensor {name:?}"))?;
            let offset = r.u64().with_context(|| format!("offset for {name:?}"))?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        ensure!(alignment > 0, "general.alignment must be > 0");

        let data_offset = align_up(r.pos, alignment as usize);
        ensure!(
            data_offset <= buf.len(),
            "tensor-data section starts past EOF ({data_offset} > {})",
            buf.len()
        );

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
        })
    }

    /// Raw metadata lookup.
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key)
            .and_then(MetaValue::as_u64)
            .and_then(|v| u32::try_from(v).ok())
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(MetaValue::as_u64)
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(MetaValue::as_f64)
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(MetaValue::as_bool)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetaValue::as_str)
    }

    pub fn get_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        self.get(key)?
            .as_array()?
            .iter()
            .map(|e| e.as_u64().and_then(|v| u32::try_from(v).ok()))
            .collect()
    }

    /// Length of an array-typed metadata value (e.g. `tokenizer.ggml.tokens`).
    pub fn arr_len(&self, key: &str) -> Option<usize> {
        self.get(key)?.as_array().map(|a| a.len())
    }

    /// Tensor directory lookup by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Absolute byte offset of a tensor's data within the file.
    pub fn tensor_abs_offset(&self, t: &TensorInfo) -> usize {
        self.data_offset + t.offset as usize
    }

    /// On-disk byte size of a tensor's data (for slicing / bounds checks).
    pub(crate) fn tensor_byte_size(&self, t: &TensorInfo, q2: Q2Group) -> Result<usize> {
        let (block_elems, block_bytes) = t.ggml_type.block_layout(q2)?;
        let n = t.num_elements();
        ensure!(
            n.is_multiple_of(block_elems),
            "tensor {:?} element count {n} not a multiple of block size {block_elems}",
            t.name
        );
        Ok(n / block_elems * block_bytes)
    }
}

#[cfg(test)]
mod tests;
