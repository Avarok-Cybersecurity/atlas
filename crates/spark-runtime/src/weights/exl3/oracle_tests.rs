// SPDX-License-Identifier: AGPL-3.0-only
// After TensorFold zig/src/core/exl3_format.zig:189-338 (the oracle tests and the layout of
// zig/tests/fixtures/exl3_oracle.bin; Apache-2.0, ashhart/TensorFold, commit 3885dce). Changes:
// translated to Rust; the fixture is read from outside the repository (it is not vendored) and the
// test skips when it is absent; added the real-checkpoint check. See EXL3-ATTRIBUTION.md.

//! Responsibility: the EXL3 reader against data it did not produce.
//!
//! 1. TensorFold's numpy oracle (`zig/tests/fixtures/exl3_oracle.bin`, 371,478 bytes at commit 3885dce):
//!    tile states for every codebook-and-width record (27 records of 3x2 tiles), codebook spot values,
//!    and one real 128x128 block of `UnstableLlama/Qwen3.5-2B-exl3-8.00bpw` (mcg, 8 bits) with its
//!    expected `W_q` bits and FP64 dequantization. The file is read from `ATLAS_EXL3_ORACLE`, else from
//!    `$HOME/research/TensorFold/zig/tests/fixtures/exl3_oracle.bin`; the test prints SKIP and passes
//!    when neither exists (CI has no clone). Reading the file is data only; no upstream code runs.
//! 2. A real EXL3 checkpoint (`ATLAS_EXL3_MODEL_DIR`, for example a `turboderp/Qwen3.8-27B-exl3`
//!    branch): `#[ignore]`d, run by the lane's pass snippet once the download exists.

use std::path::PathBuf;

use super::dequant;
use super::format::{self, Codebook, Width};
use super::load::Exl3Scan;

fn oracle_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ATLAS_EXL3_ORACLE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home).join("research/TensorFold/zig/tests/fixtures/exl3_oracle.bin");
    p.is_file().then_some(p)
}

struct Reader<'a> {
    data: &'a [u8],
}

impl Reader<'_> {
    fn u16(&self, at: usize) -> u16 {
        u16::from_le_bytes(self.data[at..at + 2].try_into().unwrap())
    }
    fn u32(&self, at: usize) -> u32 {
        u32::from_le_bytes(self.data[at..at + 4].try_into().unwrap())
    }
    fn u64(&self, at: usize) -> u64 {
        u64::from_le_bytes(self.data[at..at + 8].try_into().unwrap())
    }
    fn i16s(&self, at: usize, count: usize) -> Vec<i16> {
        (0..count).map(|i| self.u16(at + 2 * i) as i16).collect()
    }
    fn u16s(&self, at: usize, count: usize) -> Vec<u16> {
        (0..count).map(|i| self.u16(at + 2 * i)).collect()
    }
}

/// TensorFold's enum order: `inst3`, `mcg`, `mul1`.
fn codebook_of(index: u16) -> Codebook {
    Codebook::ALL[usize::from(index)]
}

#[test]
fn tensorfold_oracle_if_present() {
    let Some(path) = oracle_path() else {
        eprintln!("SKIP tensorfold_oracle_if_present: set ATLAS_EXL3_ORACLE to exl3_oracle.bin");
        return;
    };
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let r = Reader { data: &data };
    assert_eq!(
        &data[..8],
        b"EXL3ORAC",
        "{} is not the EXL3 oracle",
        path.display()
    );
    let count = r.u64(8) as usize;
    let mut at = 16;
    let mut tiles_checked = 0;
    for _ in 0..count {
        let w = Width::from_halves(r.u16(at)).expect("record width");
        let (kt, nt) = (r.u32(at + 4) as usize, r.u32(at + 8) as usize);
        let words = w.tile_words();
        // Every tile's trellis words first (k-major), then every tile's 256 u32 states.
        let trellis_base = at + 12;
        let states_base = trellis_base + kt * nt * words * 2;
        for ti in 0..kt * nt {
            let tile = r.i16s(trellis_base + ti * words * 2, words);
            let states = format::tile_states(w, &tile).unwrap();
            for (p, &state) in states.iter().enumerate() {
                let want = r.u32(states_base + ti * 1024 + p * 4);
                assert_eq!(
                    u32::from(state),
                    want,
                    "halves {} tile {ti} value {p}",
                    w.halves()
                );
            }
            tiles_checked += 1;
        }
        at += 12 + kt * nt * (words * 2 + 1024);
    }
    assert_eq!(tiles_checked, 27 * 6);

    let spots = r.u32(at) as usize;
    at += 4;
    for _ in 0..spots {
        let cb = codebook_of(r.u16(at));
        let (state, want) = (r.u16(at + 2), r.u16(at + 4));
        at += 6;
        assert_eq!(
            format::codebook_value(cb, state),
            want,
            "{} state {state}",
            cb.name()
        );
    }

    // The real mcg 8.00 bpw block.
    let w = Width::from_halves(r.u16(at)).expect("block width");
    let (kt, nt) = (r.u32(at + 2) as usize, r.u32(at + 6) as usize);
    let trellis_len = r.u32(at + 10) as usize;
    at += 14;
    assert_eq!((kt, nt), (8, 8));
    let trellis = r.i16s(at, trellis_len / 2);
    at += trellis_len;
    let mut vector = || {
        let len = r.u32(at) as usize;
        let v = r.u16s(at + 4, len / 2);
        at += 4 + len;
        v
    };
    let suh = vector();
    let svh = vector();
    let wq_want = vector();
    let dq_len = r.u32(at) as usize;
    let dq_want: Vec<f64> = (0..dq_len / 8)
        .map(|i| f64::from_bits(r.u64(at + 4 + 8 * i)))
        .collect();
    let wq = format::unpack(w, Codebook::Mcg, &trellis, kt, nt).unwrap();
    assert_eq!(wq, wq_want, "W_q of the real block, bit for bit");
    let dq = dequant::dequantize(w, Codebook::Mcg, &trellis, kt, nt, &suh, &svh).unwrap();
    assert_eq!(dq.len(), dq_want.len());
    let mag = dq_want.iter().fold(0f64, |m, v| m.max(v.abs()));
    let diff = dq
        .iter()
        .zip(dq_want.iter())
        .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
    // The FP64 reference may sum the Hadamard products in another order.
    assert!(
        diff <= 1e-9 * (1.0 + mag),
        "max |diff| {diff} over max |w| {mag}"
    );
    eprintln!(
        "tensorfold_oracle_if_present: {tiles_checked} tiles, {spots} codebook spots, real block max diff {diff:e} ({})",
        path.display()
    );
}

#[test]
#[ignore = "needs ATLAS_EXL3_MODEL_DIR (a downloaded EXL3 checkpoint); run by bench/pass.d/L20a.sh"]
fn real_checkpoint_scans_and_decodes() {
    let dir = std::env::var("ATLAS_EXL3_MODEL_DIR").expect("ATLAS_EXL3_MODEL_DIR");
    let dir = PathBuf::from(dir);
    if let Ok(Some(cfg)) = super::read_config(&dir) {
        eprintln!("config: {cfg:?}");
    }
    let scan = Exl3Scan::open(&dir).unwrap();
    let s = scan.summary();
    eprintln!(
        "groups {} (3inst {}, mcg {}, mul1 {}), fp16 scales {}, sign scales {}, bias {}, \
         average {:.3} trellis bits a weight, {} plain tensors, {} refused",
        s.groups,
        s.inst3,
        s.mcg,
        s.mul1,
        s.fp16_scales,
        s.sign_scales,
        s.bias,
        s.average_bits(),
        scan.plain_tensors().len(),
        scan.refused().len()
    );
    for (h, c) in s.by_halves.iter().enumerate().filter(|(_, c)| **c > 0) {
        eprintln!("  {:.1} bits: {c} groups", h as f64 / 2.0);
    }
    assert!(s.groups > 0, "no EXL3 group in {}", dir.display());
    assert!(scan.refused().is_empty(), "refused: {:?}", scan.refused());
    // The smallest group in full, and block (0, 0) of the first group, against the factorized matvec.
    let smallest = (0..scan.groups().len())
        .min_by_key(|&i| scan.groups()[i].k * scan.groups()[i].n)
        .unwrap();
    for gi in [0, smallest] {
        let t = scan.load(gi).unwrap();
        let block = t.dequantize_block(0, 0).unwrap();
        assert!(
            block.iter().all(|v| v.is_finite()),
            "{}: non-finite weight",
            t.prefix
        );
        assert!(
            block.iter().any(|&v| v != 0.0),
            "{}: an all-zero block",
            t.prefix
        );
        for r in [0usize, 1, 63, 127] {
            let mut x = vec![0f64; t.k];
            x[r] = 1.0;
            let y = t.matvec_f64(&x).unwrap();
            for c in 0..128 {
                let bias = t.bias.as_ref().map_or(0.0, |b| super::fp16::to_f64(b[c]));
                let want = block[r * 128 + c] + bias;
                assert!(
                    (y[c] - want).abs() <= 1e-9 * (1.0 + want.abs()),
                    "{} row {r} col {c}: matvec {} vs block {want}",
                    t.prefix,
                    y[c]
                );
            }
        }
        let rms = (block.iter().map(|v| v * v).sum::<f64>() / block.len() as f64).sqrt();
        eprintln!(
            "  {}: [{}, {}] {} bits {}, block (0,0) rms {rms:.4e}",
            t.prefix,
            t.k,
            t.n,
            t.width.bits(),
            t.codebook.name()
        );
    }
}
