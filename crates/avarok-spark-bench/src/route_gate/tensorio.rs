// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: read and write safetensors for the routing gate. The reader memory-maps the file (the ARC release
//! is 1.29 GB and the study needs 80 MB of `stage_one.*` and 3 MB of `signals.*`), validates the header through the
//! `safetensors` crate and widens every numeric dtype to f64 on request (BF16 is the top half of an f32, so exact).

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::{Metadata, TensorView};
use safetensors::{Dtype, SafeTensors};

use super::error::{GResult, GateError, key_err};

/// A tensor widened to f64: the shape and the flat row-major data.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    /// Builds a tensor, checking that the data fills the shape.
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Self {
        debug_assert_eq!(shape.iter().product::<usize>(), data.len());
        Self { shape, data }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

fn st_err(path: &Path, e: impl std::fmt::Display) -> GateError {
    GateError::Safetensors(format!("{}: {e}", path.display()))
}

/// Dtypes this reader accepts (the same set as the old Python reader).
fn supported(d: Dtype) -> bool {
    matches!(
        d,
        Dtype::F64
            | Dtype::F32
            | Dtype::F16
            | Dtype::BF16
            | Dtype::I64
            | Dtype::I32
            | Dtype::I16
            | Dtype::I8
            | Dtype::U64
            | Dtype::U32
            | Dtype::U16
            | Dtype::U8
            | Dtype::BOOL
    )
}

/// An open safetensors file.
pub struct TensorFile {
    path: PathBuf,
    mmap: Mmap,
    meta: Metadata,
    base: usize,
}

impl TensorFile {
    /// Opens and validates `path`: offsets inside the buffer, sizes matching shapes, no overlap, supported dtypes.
    pub fn open(path: impl AsRef<Path>) -> GResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        // SAFETY: the file is opened read-only and only read through shared slices; the study never modifies a
        // release file while it is mapped.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| st_err(&path, e))?;
        // `read_metadata` validates the header: offsets inside the buffer, sizes matching shapes, no gaps or overlap.
        let (n, meta) = SafeTensors::read_metadata(&mmap).map_err(|e| st_err(&path, e))?;
        for (name, info) in meta.tensors() {
            if !supported(info.dtype) {
                return Err(GateError::Safetensors(format!(
                    "{name}: unsupported dtype {:?}",
                    info.dtype
                )));
            }
        }
        Ok(Self {
            path,
            mmap,
            meta,
            base: 8 + n,
        })
    }

    /// The `__metadata__` strings of the header.
    pub fn metadata(&self) -> HashMap<String, String> {
        self.meta.metadata().clone().unwrap_or_default()
    }

    /// Sorted tensor names.
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.meta.tensors().keys().cloned().collect();
        v.sort();
        v
    }

    /// True when the file holds a tensor called `name`.
    pub fn has(&self, name: &str) -> bool {
        self.meta.info(name).is_some()
    }

    fn info(&self, name: &str) -> GResult<&safetensors::tensor::TensorInfo> {
        self.meta.info(name).ok_or_else(|| {
            GateError::Safetensors(format!("{}: no tensor named {name:?}", self.path.display()))
        })
    }

    /// Shape of tensor `name`.
    pub fn shape(&self, name: &str) -> GResult<Vec<usize>> {
        Ok(self.info(name)?.shape.clone())
    }

    /// Dtype of tensor `name` as the safetensors header spells it (`"BF16"`, `"F64"`, ...).
    pub fn dtype(&self, name: &str) -> GResult<String> {
        Ok(format!("{}", self.info(name)?.dtype))
    }

    /// Raw little-endian bytes of tensor `name`.
    pub fn raw(&self, name: &str) -> GResult<&[u8]> {
        let info = self.info(name)?;
        let (s, e) = info.data_offsets;
        Ok(&self.mmap[self.base + s..self.base + e])
    }

    /// One tensor widened to f64 (BF16 and F16 are exact in f64; integers convert by value).
    pub fn read(&self, name: &str) -> GResult<Tensor> {
        let info = self.info(name)?;
        let bytes = self.raw(name)?;
        Ok(Tensor {
            shape: info.shape.clone(),
            data: widen(info.dtype, bytes),
        })
    }

    /// One float tensor as f32 (F64 narrows, BF16 and F16 widen).
    pub fn read_f32(&self, name: &str) -> GResult<(Vec<usize>, Vec<f32>)> {
        let t = self.read(name)?;
        Ok((t.shape, t.data.iter().map(|&v| v as f32).collect()))
    }

    /// The BF16 bit patterns of a BF16 tensor (kept as `u16` so a layer's weights stay half the size of f32).
    pub fn read_bf16_bits(&self, name: &str) -> GResult<(Vec<usize>, Vec<u16>)> {
        let info = self.info(name)?;
        if info.dtype != Dtype::BF16 {
            return Err(GateError::Safetensors(format!(
                "{name}: expected BF16, found {}",
                info.dtype
            )));
        }
        let bits = self
            .raw(name)?
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok((info.shape.clone(), bits))
    }

    /// Every tensor whose name starts with `prefix`; the prefix is removed from the key when `strip` is true.
    pub fn read_prefix(&self, prefix: &str, strip: bool) -> GResult<BTreeMap<String, Tensor>> {
        let mut out = BTreeMap::new();
        for name in self.names() {
            if let Some(rest) = name.strip_prefix(prefix) {
                let key = if strip {
                    rest.to_string()
                } else {
                    name.clone()
                };
                out.insert(key, self.read(&name)?);
            }
        }
        Ok(out)
    }
}

fn widen(dtype: Dtype, bytes: &[u8]) -> Vec<f64> {
    fn each<const W: usize>(bytes: &[u8], f: impl Fn([u8; W]) -> f64) -> Vec<f64> {
        bytes
            .chunks_exact(W)
            .map(|c| {
                let mut a = [0u8; W];
                a.copy_from_slice(c);
                f(a)
            })
            .collect()
    }
    match dtype {
        Dtype::F64 => each::<8>(bytes, f64::from_le_bytes),
        Dtype::F32 => each::<4>(bytes, |a| f64::from(f32::from_le_bytes(a))),
        Dtype::F16 => each::<2>(bytes, |a| f64::from(half::f16::from_le_bytes(a).to_f32())),
        Dtype::BF16 => each::<2>(bytes, |a| f64::from(half::bf16::from_le_bytes(a).to_f32())),
        Dtype::I64 => each::<8>(bytes, |a| i64::from_le_bytes(a) as f64),
        Dtype::I32 => each::<4>(bytes, |a| f64::from(i32::from_le_bytes(a))),
        Dtype::I16 => each::<2>(bytes, |a| f64::from(i16::from_le_bytes(a))),
        Dtype::I8 => bytes.iter().map(|&b| f64::from(b as i8)).collect(),
        Dtype::U64 => each::<8>(bytes, |a| u64::from_le_bytes(a) as f64),
        Dtype::U32 => each::<4>(bytes, |a| f64::from(u32::from_le_bytes(a))),
        Dtype::U16 => each::<2>(bytes, |a| f64::from(u16::from_le_bytes(a))),
        _ => bytes.iter().map(|&b| f64::from(b)).collect(),
    }
}

/// Data to store with [`write_safetensors`].
#[derive(Clone, Debug)]
pub enum OutData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    /// f32 values stored as BF16 with round-to-nearest-even.
    Bf16(Vec<f32>),
}

/// One tensor to write.
#[derive(Clone, Debug)]
pub struct OutTensor {
    pub shape: Vec<usize>,
    pub data: OutData,
}

impl OutTensor {
    pub fn f64(shape: &[usize], data: Vec<f64>) -> Self {
        Self {
            shape: shape.to_vec(),
            data: OutData::F64(data),
        }
    }
    pub fn f32(shape: &[usize], data: Vec<f32>) -> Self {
        Self {
            shape: shape.to_vec(),
            data: OutData::F32(data),
        }
    }
    pub fn i64(shape: &[usize], data: Vec<i64>) -> Self {
        Self {
            shape: shape.to_vec(),
            data: OutData::I64(data),
        }
    }
    pub fn u8(shape: &[usize], data: Vec<u8>) -> Self {
        Self {
            shape: shape.to_vec(),
            data: OutData::U8(data),
        }
    }
    pub fn bf16(shape: &[usize], data: Vec<f32>) -> Self {
        Self {
            shape: shape.to_vec(),
            data: OutData::Bf16(data),
        }
    }

    fn bytes(&self) -> (Dtype, Vec<u8>) {
        match &self.data {
            OutData::F64(v) => (Dtype::F64, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            OutData::F32(v) => (Dtype::F32, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            OutData::I64(v) => (Dtype::I64, v.iter().flat_map(|x| x.to_le_bytes()).collect()),
            OutData::U8(v) => (Dtype::U8, v.clone()),
            OutData::Bf16(v) => (
                Dtype::BF16,
                v.iter()
                    .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
                    .collect(),
            ),
        }
    }
}

/// Writes `tensors` (any order) and the string `metadata` to `path`.
pub fn write_safetensors(
    path: impl AsRef<Path>,
    tensors: &[(String, OutTensor)],
    metadata: &BTreeMap<String, String>,
) -> GResult<()> {
    let path = path.as_ref();
    let blobs: Vec<(Dtype, Vec<u8>)> = tensors.iter().map(|(_, t)| t.bytes()).collect();
    let mut views = Vec::with_capacity(tensors.len());
    for ((name, t), (dtype, bytes)) in tensors.iter().zip(&blobs) {
        let view = TensorView::new(*dtype, t.shape.clone(), bytes)
            .map_err(|e| st_err(path, format!("{name}: {e}")))?;
        views.push((name.clone(), view));
    }
    let meta: HashMap<String, String> = metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let meta = if meta.is_empty() { None } else { Some(meta) };
    safetensors::serialize_to_file(views, meta, path).map_err(|e| st_err(path, e))
}

/// Looks `name` up in a tensor map, as the head loaders do.
pub fn need<'a>(map: &'a BTreeMap<String, Tensor>, name: &str) -> GResult<&'a Tensor> {
    map.get(name)
        .map_or_else(|| key_err(format!("tensor {name:?} is missing")), Ok)
}
