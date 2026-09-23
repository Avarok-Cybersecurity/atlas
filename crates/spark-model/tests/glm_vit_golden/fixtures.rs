// SPDX-License-Identifier: AGPL-3.0-only

//! Fixture plumbing for `glm_vit_golden`: a float32 `.npy` reader, a
//! safetensors uploader, and the error statistics the gate reads. Split out
//! only for the 500-LoC file cap; nothing here is GLM-specific except the
//! 347-tensor assertion.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

// ── minimal .npy reader (C-order float32 only) ───────────────────────────────

pub fn read_npy_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() > 10 && &bytes[0..6] == b"\x93NUMPY",
        "{}: not a .npy",
        path.display()
    );
    let header_len = match bytes[6] {
        1 => u16::from_le_bytes([bytes[8], bytes[9]]) as usize + 10,
        _ => u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize + 12,
    };
    let header = std::str::from_utf8(&bytes[(if bytes[6] == 1 { 10 } else { 12 })..header_len])?;
    ensure!(
        header.contains("'<f4'") || header.contains("\"<f4\""),
        "{}: golden must be little-endian float32, header was {header}",
        path.display()
    );
    ensure!(
        header.contains("False"),
        "{}: Fortran-order golden",
        path.display()
    );
    let shape: Vec<usize> = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .context("no shape in npy header")?
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect();
    let data: Vec<f32> = bytes[header_len..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ensure!(
        shape.iter().product::<usize>() == data.len(),
        "{}: shape {shape:?} does not match {} elements",
        path.display(),
        data.len()
    );
    Ok((shape, data))
}

// ── safetensors → device ─────────────────────────────────────────────────────

/// Upload the 347 BF16 `model.visual.*` tensors, returning a name → pointer
/// map keyed WITHOUT the prefix.
pub fn upload_tower(path: &Path, gpu: &dyn GpuBackend) -> Result<HashMap<String, DevicePtr>> {
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let mut header = vec![0u8; header_len];
    f.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    let obj = header
        .as_object()
        .context("safetensors header is not an object")?;

    let mut out = HashMap::new();
    let mut buf = Vec::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("");
        ensure!(dtype == "BF16", "{name}: expected BF16, got {dtype}");
        let off = meta
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .context("offsets")?;
        let (start, end) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
        let n = (end - start) as usize;
        buf.resize(n, 0u8);
        f.seek(SeekFrom::Start(8 + header_len as u64 + start))?;
        f.read_exact(&mut buf)?;
        let ptr = gpu.alloc(n.max(1))?;
        gpu.copy_h2d(&buf, ptr)?;
        let key = name
            .strip_prefix("model.visual.")
            .unwrap_or(name)
            .to_string();
        out.insert(key, ptr);
    }
    ensure!(
        out.len() == 347,
        "expected 347 vision tensors, uploaded {}",
        out.len()
    );
    Ok(out)
}

pub fn ptr(w: &HashMap<String, DevicePtr>, name: &str) -> Result<DevicePtr> {
    w.get(name)
        .copied()
        .with_context(|| format!("missing tensor {name}"))
}

// ── comparison ───────────────────────────────────────────────────────────────

pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

pub struct Stats {
    pub max_abs: f64,
    pub mean_abs: f64,
    pub min_cos: f64,
    pub mean_cos: f64,
    /// Mean over rows of `||a - b|| / ||b||` — the scale-aware error, which is
    /// what the paired control is compared on.
    pub mean_rel: f64,
}

/// Per-ROW cosine (a token's embedding is the unit the LM consumes), plus the
/// elementwise error budget.
pub fn compare(actual: &[f32], expect: &[f32], rows: usize, cols: usize) -> Result<Stats> {
    ensure!(
        actual.len() >= rows * cols && expect.len() == rows * cols,
        "compare: {} actual / {} expected for {rows}x{cols}",
        actual.len(),
        expect.len()
    );
    let (mut max_abs, mut sum_abs, mut min_cos) = (0.0f64, 0.0f64, 1.0f64);
    let (mut sum_cos, mut sum_rel, mut counted) = (0.0f64, 0.0f64, 0usize);
    for r in 0..rows {
        let (mut dot, mut na, mut nb, mut diff) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for c in 0..cols {
            let (a, b) = (actual[r * cols + c] as f64, expect[r * cols + c] as f64);
            ensure!(a.is_finite(), "row {r} col {c} is not finite: {a}");
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            sum_abs += d;
            diff += d * d;
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        if na > 0.0 && nb > 0.0 {
            let cos = dot / (na.sqrt() * nb.sqrt());
            min_cos = min_cos.min(cos);
            sum_cos += cos;
            sum_rel += diff.sqrt() / nb.sqrt();
            counted += 1;
        }
    }
    let n = counted.max(1) as f64;
    Ok(Stats {
        max_abs,
        mean_abs: sum_abs / (rows * cols) as f64,
        min_cos,
        mean_cos: sum_cos / n,
        mean_rel: sum_rel / n,
    })
}
