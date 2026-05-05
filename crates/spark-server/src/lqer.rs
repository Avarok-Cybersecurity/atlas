// SPDX-License-Identifier: AGPL-3.0-only

//! LQER — Low-Rank Quantization Error Reconstruction (C.2, 2026-04-25).
//!
//! Reference: Zhang et al., ICML 2024. Per quantized linear weight
//! matrix W (FP8 / NVFP4), compute SVD of the dequantization error:
//!
//! ```text
//! E = W_bf16 - dequant(Q(W))
//! E ≈ U_k Σ_k V_k^T   (rank-k truncated SVD)
//! ```
//!
//! Then at inference, output ← Q(W) · x + (U_k Σ_k V_k^T) · x. The
//! second term is a small BF16 GEMM applied in parallel; rank=10%
//! closes >50% of the quantization gap, rank=30% fully closes it.
//! Atlas-kernels already has BF16 GEMM, so this is loader + dispatch
//! work, not a new kernel.
//!
//! ## Scope
//!
//! This module ships the **per-layer correction descriptor** —
//! the offline tool computes SVD and produces a serialized
//! `LqerCorrection` per quantized layer; the runtime loader reads
//! these and the dispatcher applies the small BF16 GEMM in parallel.
//!
//! Production wiring requires:
//!   1. Offline calibration script (`scripts/lqer_compute.py`):
//!      - For each quantized linear, compute E = W_bf16 - dequant(Q)
//!      - Truncated SVD at rank-k (k = 10% of min(rows, cols))
//!      - Save (U_k * sqrt(Σ_k)) [rows × k] and (sqrt(Σ_k) * V_k^T)
//!        [k × cols] as BF16 tensors
//!   2. Loader to read these alongside the quantized weights
//!   3. Dispatch fusion: alongside the FP8/NVFP4 GEMM, run the
//!      BF16 GEMM for the correction and sum

use std::path::Path;

/// Per-layer LQER correction. Stored as two BF16 matrices:
///   - left:  [rows × rank]
///   - right: [rank × cols]
///
/// Output of the correction is `left @ right @ x` (or equivalently
/// the rank-k approximation of the quantization error matrix
/// applied to activations x).
#[derive(Debug, Clone)]
pub struct LqerCorrection {
    pub layer_name: String,
    pub rank: usize,
    pub rows: usize,
    pub cols: usize,
    /// BF16 bytes for the `left` matrix [rows × rank].
    pub left_bf16: Vec<u8>,
    /// BF16 bytes for the `right` matrix [rank × cols].
    pub right_bf16: Vec<u8>,
}

impl LqerCorrection {
    /// Approximate memory footprint of this correction in bytes.
    /// Used to budget how many layers can carry corrections within
    /// a memory ceiling.
    pub fn memory_bytes(&self) -> usize {
        self.left_bf16.len() + self.right_bf16.len()
    }

    /// Sanity-check the descriptor's matrix sizes match the rank.
    /// Returns `Err(reason)` if the shape is impossible.
    pub fn validate(&self) -> Result<(), String> {
        let expected_left = self.rows * self.rank * 2;
        if self.left_bf16.len() != expected_left {
            return Err(format!(
                "left matrix size mismatch: expected {expected_left}B, got {}B",
                self.left_bf16.len()
            ));
        }
        let expected_right = self.rank * self.cols * 2;
        if self.right_bf16.len() != expected_right {
            return Err(format!(
                "right matrix size mismatch: expected {expected_right}B, got {}B",
                self.right_bf16.len()
            ));
        }
        Ok(())
    }
}

/// Set of LQER corrections keyed by layer name. Loaded from a
/// directory of `.bin` files at server startup.
#[derive(Debug, Clone, Default)]
pub struct LqerCorrectionSet {
    by_layer: std::collections::HashMap<String, LqerCorrection>,
}

impl LqerCorrectionSet {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, c: LqerCorrection) {
        self.by_layer.insert(c.layer_name.clone(), c);
    }

    pub fn get(&self, layer: &str) -> Option<&LqerCorrection> {
        self.by_layer.get(layer)
    }

    pub fn len(&self) -> usize {
        self.by_layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_layer.is_empty()
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.by_layer.values().map(|c| c.memory_bytes()).sum()
    }
}

/// Suggest a rank given the linear's dimensions. Per LQER paper:
/// rank = ceil(0.10 * min(rows, cols)) for the "minimal recovery"
/// preset, 0.30 for "full recovery". Rank is bounded by the
/// dimensions to avoid degenerate cases.
pub fn suggest_rank(rows: usize, cols: usize, fraction: f32) -> usize {
    let bound = rows.min(cols);
    let r = ((bound as f32) * fraction).ceil() as usize;
    r.max(1).min(bound)
}

/// On-disk format for an LQER correction file (`.lqer` suffix).
///
/// ```text
/// offset  size  field
/// ────────────────────────────────────
/// 0       8     magic = b"ATLASLQE"
/// 8       4     format version (u32 little-endian) — currently 1
/// 12      4     rank          (u32 LE)
/// 16      4     rows          (u32 LE)
/// 20      4     cols          (u32 LE)
/// 24      4     name_len      (u32 LE)
/// 28      N     layer_name    (UTF-8, length = name_len)
/// 28+N    PAD   zero-padding to 8-byte alignment
/// ────────────────────────────────────
/// next    rows × rank × 2     left  BF16 matrix (column-major)
/// next    rank × cols × 2     right BF16 matrix (row-major)
/// ```
///
/// Multi-byte integers are little-endian. Tensor data is raw BF16
/// (each element 2 bytes, the bit pattern of the truncated FP32
/// representation), no compression. Files are produced by the
/// offline calibration script (TBD `scripts/lqer_compute.py`)
/// alongside the regular weight checkpoint.
const LQER_MAGIC: &[u8; 8] = b"ATLASLQE";
const LQER_VERSION: u32 = 1;

/// Errors from LQER file I/O. Each variant names the failed
/// invariant precisely so operators can diagnose corrupt artefacts.
#[derive(Debug)]
pub enum LqerLoadError {
    Io(std::io::Error),
    BadMagic,
    UnsupportedVersion(u32),
    Truncated { expected: usize, got: usize },
    InvalidName(std::string::FromUtf8Error),
    ShapeMismatch(String),
}

impl std::fmt::Display for LqerLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::BadMagic => write!(f, "bad magic — expected ATLASLQE"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::Truncated { expected, got } => {
                write!(f, "truncated: expected {expected} bytes, got {got}")
            }
            Self::InvalidName(e) => write!(f, "invalid utf-8 in layer name: {e}"),
            Self::ShapeMismatch(s) => write!(f, "shape mismatch: {s}"),
        }
    }
}

impl std::error::Error for LqerLoadError {}

impl From<std::io::Error> for LqerLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Parse a single `.lqer` file's bytes into an `LqerCorrection`.
/// Pure / no I/O — exposed for testing the format independently of
/// the filesystem.
pub fn parse_lqer_bytes(buf: &[u8]) -> Result<LqerCorrection, LqerLoadError> {
    if buf.len() < 28 {
        return Err(LqerLoadError::Truncated {
            expected: 28,
            got: buf.len(),
        });
    }
    if &buf[0..8] != LQER_MAGIC {
        return Err(LqerLoadError::BadMagic);
    }
    let version = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if version != LQER_VERSION {
        return Err(LqerLoadError::UnsupportedVersion(version));
    }
    let rank = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
    let rows = u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize;
    let cols = u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize;
    let name_len = u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;

    let name_end = 28 + name_len;
    if buf.len() < name_end {
        return Err(LqerLoadError::Truncated {
            expected: name_end,
            got: buf.len(),
        });
    }
    let layer_name = std::str::from_utf8(&buf[28..name_end])
        .map_err(|_| {
            LqerLoadError::InvalidName(String::from_utf8(buf[28..name_end].to_vec()).unwrap_err())
        })?
        .to_string();

    // Pad name region up to 8-byte alignment.
    let aligned = (name_end + 7) & !7;
    if buf.len() < aligned {
        return Err(LqerLoadError::Truncated {
            expected: aligned,
            got: buf.len(),
        });
    }

    // rows × rank × 2 (BF16) for left
    let left_bytes = rows
        .checked_mul(rank)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| LqerLoadError::ShapeMismatch("rows × rank × 2 overflowed".into()))?;
    let left_end = aligned + left_bytes;
    if buf.len() < left_end {
        return Err(LqerLoadError::Truncated {
            expected: left_end,
            got: buf.len(),
        });
    }
    let left_bf16 = buf[aligned..left_end].to_vec();

    let right_bytes = rank
        .checked_mul(cols)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| LqerLoadError::ShapeMismatch("rank × cols × 2 overflowed".into()))?;
    let right_end = left_end + right_bytes;
    if buf.len() < right_end {
        return Err(LqerLoadError::Truncated {
            expected: right_end,
            got: buf.len(),
        });
    }
    let right_bf16 = buf[left_end..right_end].to_vec();

    let c = LqerCorrection {
        layer_name,
        rank,
        rows,
        cols,
        left_bf16,
        right_bf16,
    };
    c.validate().map_err(LqerLoadError::ShapeMismatch)?;
    Ok(c)
}

/// Serialise an `LqerCorrection` to the `.lqer` on-disk format.
/// Used by the offline calibration tool and round-trip tests.
pub fn write_lqer_bytes(c: &LqerCorrection) -> Result<Vec<u8>, LqerLoadError> {
    c.validate().map_err(LqerLoadError::ShapeMismatch)?;
    let name_bytes = c.layer_name.as_bytes();
    let header_end = 28 + name_bytes.len();
    let aligned = (header_end + 7) & !7;
    let total = aligned + c.left_bf16.len() + c.right_bf16.len();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(LQER_MAGIC);
    buf.extend_from_slice(&LQER_VERSION.to_le_bytes());
    buf.extend_from_slice(&(c.rank as u32).to_le_bytes());
    buf.extend_from_slice(&(c.rows as u32).to_le_bytes());
    buf.extend_from_slice(&(c.cols as u32).to_le_bytes());
    buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    while buf.len() < aligned {
        buf.push(0);
    }
    buf.extend_from_slice(&c.left_bf16);
    buf.extend_from_slice(&c.right_bf16);
    Ok(buf)
}

/// Load every `.lqer` file in `dir` into an `LqerCorrectionSet`.
/// Returns an empty set when the directory doesn't exist (graceful
/// degrade — production servers without LQER-compiled corrections
/// keep their existing behaviour). Files that fail to parse are
/// logged and skipped — a single bad file doesn't kill the whole
/// load.
pub fn load_from_dir(dir: &Path) -> LqerCorrectionSet {
    let mut set = LqerCorrectionSet::empty();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return set,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("lqer") {
            continue;
        }
        match std::fs::read(&path)
            .map_err(LqerLoadError::from)
            .and_then(|b| parse_lqer_bytes(&b))
        {
            Ok(c) => set.insert(c),
            Err(e) => {
                tracing::warn!("Skipping malformed LQER file {}: {}", path.display(), e);
            }
        }
    }
    if !set.is_empty() {
        tracing::info!(
            count = set.len(),
            mem_mb = set.total_memory_bytes() / 1_048_576,
            "Loaded LQER corrections from {}",
            dir.display()
        );
    }
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy(rows: usize, cols: usize, rank: usize) -> LqerCorrection {
        LqerCorrection {
            layer_name: "test".into(),
            rank,
            rows,
            cols,
            left_bf16: vec![0u8; rows * rank * 2],
            right_bf16: vec![0u8; rank * cols * 2],
        }
    }

    #[test]
    fn validate_passes_on_correct_shape() {
        let c = dummy(64, 128, 8);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn validate_fails_on_wrong_left_shape() {
        let mut c = dummy(64, 128, 8);
        c.left_bf16.truncate(10);
        let err = c.validate().unwrap_err();
        assert!(err.contains("left matrix size mismatch"));
    }

    #[test]
    fn memory_bytes_is_sum_of_matrices() {
        let c = dummy(64, 128, 8);
        // 64*8*2 + 8*128*2 = 1024 + 2048 = 3072
        assert_eq!(c.memory_bytes(), 3072);
    }

    #[test]
    fn correction_set_round_trip() {
        let mut s = LqerCorrectionSet::empty();
        assert!(s.is_empty());
        let mut c = dummy(64, 128, 8);
        c.layer_name = "l0".into();
        s.insert(c);
        assert_eq!(s.len(), 1);
        assert!(s.get("l0").is_some());
        assert!(s.get("l1").is_none());
    }

    #[test]
    fn suggest_rank_at_various_fractions() {
        // 0.10 of min(2048, 6144) = 0.10 * 2048 = 204.8 → 205
        assert_eq!(suggest_rank(2048, 6144, 0.10), 205);
        // 0.30 of 2048 = 614.4 → 615
        assert_eq!(suggest_rank(2048, 6144, 0.30), 615);
        // Always ≥ 1
        assert_eq!(suggest_rank(100, 100, 0.0), 1);
        // Never exceed min dim
        assert_eq!(suggest_rank(100, 100, 1.5), 100);
    }

    fn make_correction(name: &str, rows: usize, cols: usize, rank: usize) -> LqerCorrection {
        let left_bf16: Vec<u8> = (0..rows * rank * 2).map(|i| (i & 0xFF) as u8).collect();
        let right_bf16: Vec<u8> = (0..rank * cols * 2)
            .map(|i| ((i ^ 0xA5) & 0xFF) as u8)
            .collect();
        LqerCorrection {
            layer_name: name.to_string(),
            rank,
            rows,
            cols,
            left_bf16,
            right_bf16,
        }
    }

    #[test]
    fn write_then_parse_round_trips() {
        let original = make_correction("model.layers.0.mlp.experts.5.down_proj", 64, 128, 8);
        let bytes = write_lqer_bytes(&original).expect("serialise");
        let parsed = parse_lqer_bytes(&bytes).expect("parse");
        assert_eq!(parsed.layer_name, original.layer_name);
        assert_eq!(parsed.rank, original.rank);
        assert_eq!(parsed.rows, original.rows);
        assert_eq!(parsed.cols, original.cols);
        assert_eq!(parsed.left_bf16, original.left_bf16);
        assert_eq!(parsed.right_bf16, original.right_bf16);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        let mut bytes = write_lqer_bytes(&make_correction("x", 8, 8, 2)).unwrap();
        bytes[0] = b'X';
        let err = parse_lqer_bytes(&bytes).unwrap_err();
        assert!(matches!(err, LqerLoadError::BadMagic));
    }

    #[test]
    fn parse_rejects_unsupported_version() {
        let mut bytes = write_lqer_bytes(&make_correction("x", 8, 8, 2)).unwrap();
        // Bump version field at offset 8
        bytes[8] = 99;
        let err = parse_lqer_bytes(&bytes).unwrap_err();
        assert!(matches!(err, LqerLoadError::UnsupportedVersion(_)));
    }

    #[test]
    fn parse_rejects_truncated() {
        let bytes = write_lqer_bytes(&make_correction("x", 8, 8, 2)).unwrap();
        let truncated = &bytes[..bytes.len() - 4];
        let err = parse_lqer_bytes(truncated).unwrap_err();
        assert!(matches!(err, LqerLoadError::Truncated { .. }));
    }

    #[test]
    fn load_from_missing_dir_returns_empty() {
        let set = load_from_dir(Path::new("/nonexistent/path/should/not/exist"));
        assert!(set.is_empty());
    }

    #[test]
    fn load_from_dir_reads_multiple_lqer_files() {
        let tmp = std::env::temp_dir().join(format!(
            "atlas_lqer_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp).unwrap();
        // Two valid corrections, one malformed file (should be skipped)
        let a = make_correction("a", 16, 16, 4);
        let b = make_correction("b", 32, 16, 4);
        std::fs::write(tmp.join("a.lqer"), write_lqer_bytes(&a).unwrap()).unwrap();
        std::fs::write(tmp.join("b.lqer"), write_lqer_bytes(&b).unwrap()).unwrap();
        std::fs::write(tmp.join("garbage.lqer"), b"not a real file").unwrap();
        // Non-.lqer file should be ignored entirely
        std::fs::write(tmp.join("readme.txt"), b"hello").unwrap();
        let set = load_from_dir(&tmp);
        assert_eq!(set.len(), 2, "two valid files load, garbage skipped");
        assert!(set.get("a").is_some());
        assert!(set.get("b").is_some());
        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn write_layer_name_with_dots_round_trips() {
        // Real layer names contain dots and slashes.
        let c = make_correction("model.layers.10.mlp.experts.0.down_proj.weight", 32, 64, 4);
        let bytes = write_lqer_bytes(&c).unwrap();
        let parsed = parse_lqer_bytes(&bytes).unwrap();
        assert_eq!(parsed.layer_name, c.layer_name);
    }
}
