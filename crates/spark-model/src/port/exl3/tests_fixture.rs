// SPDX-License-Identifier: AGPL-3.0-only
// Fixture layout after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/exl3_format.zig:189-338 (the reader of zig/tests/fixtures/exl3_oracle.bin, a numpy oracle
// of every codebook and width plus one public-checkpoint block). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes: no code copied; the record walk is rewritten in Rust and the file is read, never vendored.

//! Responsibility: the INDEPENDENT oracle of the EXL3 format decoder: TensorFold's numpy-generated
//! fixture (`exl3_oracle.bin`: 162 tiles of trellis words with their expected 16-bit states over every
//! width and codebook, codebook spot values, and one real MCG 8.00 bpw 128 x 128 block from a public
//! checkpoint with its decoded `Wq` bits and its FP64 dequant). The file is data, read in place: point
//! `ATLAS_EXL3_ORACLE` at it (for example `<TensorFold checkout>/zig/tests/fixtures/exl3_oracle.bin`)
//! and these tests check the scalar decoder, the lane mirror and both dequant paths against it.
//! Without the variable they return early (the mock-green CPU tests above still run).

use super::format::Exl3Codebook;
use super::reference::{codebook_value, lane_states, tile_states, tile_u32, unpack};
use super::reference_math::{bf16_to_f32, dequant_bf16_mirror, dequant_f64};

const ENV: &str = "ATLAS_EXL3_ORACLE";

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.b[self.at], self.b[self.at + 1]]);
        self.at += 2;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.at..self.at + 4].try_into().unwrap());
        self.at += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.at..self.at + 8].try_into().unwrap());
        self.at += 8;
        v
    }
    fn u16s(&mut self, bytes: usize) -> Vec<u16> {
        let v = self.b[self.at..self.at + bytes]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        self.at += bytes;
        v
    }
}

fn codebook(id: u16) -> Exl3Codebook {
    match id {
        0 => Exl3Codebook::Inst3,
        1 => Exl3Codebook::Mcg,
        2 => Exl3Codebook::Mul1,
        other => panic!("fixture codebook {other}"),
    }
}

fn load() -> Option<Vec<u8>> {
    let path = std::env::var_os(ENV)?;
    Some(std::fs::read(&path).unwrap_or_else(|e| panic!("{ENV}={path:?}: {e}")))
}

/// Walk the tile records and the codebook spots; returns the offset of the real block.
fn check_records(b: &[u8]) -> usize {
    assert_eq!(&b[..8], b"EXL3ORAC", "fixture magic");
    let mut c = Cursor { b, at: 8 };
    let count = c.u64();
    let mut tiles = 0usize;
    for _ in 0..count {
        let start = c.at;
        let halves = c.u16();
        let cb = codebook(u16::from(b[c.at]));
        c.at = start + 4;
        let (kt, nt) = (c.u32() as usize, c.u32() as usize);
        let words = 8 * halves as usize;
        let trellis_at = c.at;
        let states_at = trellis_at + kt * nt * words * 2;
        for ti in 0..kt * nt {
            let mut t = Cursor {
                b,
                at: trellis_at + ti * words * 2,
            };
            let tile = t.u16s(words * 2);
            let got = tile_states(halves, &tile);
            let mut s = Cursor {
                b,
                at: states_at + ti * 256 * 4,
            };
            for (p, &g) in got.iter().enumerate() {
                assert_eq!(
                    u32::from(g),
                    s.u32(),
                    "{halves} half-bits {cb:?}, tile {ti}, value {p}"
                );
            }
            let w32 = tile_u32(&tile);
            for lane in 0..32 {
                assert_eq!(
                    &lane_states(halves, &w32, lane)[..],
                    &got[8 * lane..8 * lane + 8]
                );
            }
            tiles += 1;
        }
        c.at = states_at + kt * nt * 256 * 4;
    }
    assert_eq!(tiles, 27 * 6, "the fixture's tile count");
    let spots = c.u32();
    for _ in 0..spots {
        let cb = codebook(c.u16());
        let state = c.u16();
        let want = c.u16();
        assert_eq!(codebook_value(cb, state), want, "{cb:?} state {state}");
    }
    c.at
}

#[test]
fn the_decoder_reproduces_tensorfolds_numpy_oracle() {
    let Some(b) = load() else {
        eprintln!("{ENV} not set: the independent EXL3 fixture check is skipped");
        return;
    };
    check_records(&b);
}

#[test]
fn a_real_checkpoint_block_unpacks_and_dequantizes_to_the_oracle() {
    let Some(b) = load() else {
        eprintln!("{ENV} not set: the independent EXL3 fixture check is skipped");
        return;
    };
    let mut c = Cursor {
        b: &b,
        at: check_records(&b),
    };
    let halves = c.u16();
    let (kt, nt) = (c.u32() as usize, c.u32() as usize);
    assert_eq!((kt, nt), (8, 8));
    let trellis_len = c.u32() as usize;
    let trellis = c.u16s(trellis_len);
    let suh_len = c.u32() as usize;
    let suh = c.u16s(suh_len);
    let svh_len = c.u32() as usize;
    let svh = c.u16s(svh_len);
    let wq_len = c.u32() as usize;
    let wq_want = c.u16s(wq_len);
    let dq_len = c.u32() as usize;
    let dq_want: Vec<f64> = b[c.at..c.at + dq_len]
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let (k, n) = (kt * 16, nt * 16);
    let wq = unpack(halves, Exl3Codebook::Mcg, &trellis, k, n);
    assert_eq!(wq, wq_want, "the decoded Wq bits");
    let dq = dequant_f64(&wq, k, n, &suh, &svh);
    let mag = dq_want.iter().map(|v| v.abs()).fold(0.0, f64::max);
    for (g, w) in dq.iter().zip(&dq_want) {
        assert!((g - w).abs() <= 1e-9 * (1.0 + mag), "{g} vs {w}");
    }
    // The bf16 kernel mirror (what the GPU dequant must equal byte for byte) is the oracle rounded.
    let bf = dequant_bf16_mirror(&wq, k, n, &suh, &svh);
    for kk in 0..k {
        for nn in 0..n {
            let w = dq_want[kk * n + nn];
            let g = f64::from(bf16_to_f32(bf[nn * k + kk]));
            assert!(
                (g - w).abs() <= w.abs() * 2f64.powi(-8) + 1e-4 * mag,
                "[{kk},{nn}] {g} vs {w}"
            );
        }
    }
}
