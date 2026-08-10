// SPDX-License-Identifier: AGPL-3.0-only
//! Hand-built single-block oracles. These exact vectors double as the
//! GPU-kernel numeric oracle: any GPU dequant kernel must reproduce them
//! bit-for-bit for the integer-code types.

use super::{GgmlType, dequant_to_f32, f16_to_f32};

// f16 bit patterns for the round constants used below.
const F16_2_0: u16 = 0x4000; // 2.0
const F16_1_0: u16 = 0x3C00; // 1.0
const F16_0_5: u16 = 0x3800; // 0.5

fn push_f16(v: &mut Vec<u8>, bits: u16) {
    v.extend_from_slice(&bits.to_le_bytes());
}

#[test]
fn f16_helper_matches_known_values() {
    assert_eq!(f16_to_f32(F16_2_0), 2.0);
    assert_eq!(f16_to_f32(F16_1_0), 1.0);
    assert_eq!(f16_to_f32(F16_0_5), 0.5);
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0xC000), -2.0);
}

#[test]
fn q8_0_single_block() {
    // d = 2.0, qs[0..3] = 1, 2, -1
    let mut blk = Vec::new();
    push_f16(&mut blk, F16_2_0);
    let mut qs = [0u8; 32];
    qs[0] = 1i8 as u8;
    qs[1] = 2i8 as u8;
    qs[2] = (-1i8) as u8;
    blk.extend_from_slice(&qs);

    let mut out = [0f32; 32];
    dequant_to_f32(GgmlType::Q8_0, &blk, 32, &mut out).unwrap();
    assert_eq!(out[0], 2.0);
    assert_eq!(out[1], 4.0);
    assert_eq!(out[2], -2.0);
    assert_eq!(out[3], 0.0);
}

#[test]
fn q4_1_single_block() {
    // d = 2.0, m = 1.0; qs[0] = 0x13 -> low nibble 3, high nibble 1.
    // y = nibble*d + m
    let mut blk = Vec::new();
    push_f16(&mut blk, F16_2_0);
    push_f16(&mut blk, F16_1_0);
    let mut qs = [0u8; 16];
    qs[0] = 0x13;
    blk.extend_from_slice(&qs);

    let mut out = [0f32; 32];
    dequant_to_f32(GgmlType::Q4_1, &blk, 32, &mut out).unwrap();
    assert_eq!(out[0], 7.0); // 3*2 + 1
    assert_eq!(out[16], 3.0); // 1*2 + 1
    assert_eq!(out[1], 1.0); // 0*2 + 1
}

#[test]
fn q6_k_single_block() {
    // d = 0.5; scales[0] = 2, scales[4] = 1 (i8); ql[0] = 0x0A; qh[0] = 0x01.
    // out[0]  = d * scales[0] * (q1)  with q1 = (10 | (1<<4)) - 32 = -6  -> -6.0
    // out[64] = d * scales[4] * (q3)  with q3 = (0  | 0)      - 32 = -32 -> -16.0
    let mut blk = vec![0u8; 210];
    blk[0] = 0x0A; // ql[0]
    blk[128] = 0x01; // qh[0]
    blk[192] = 2; // scales[0]
    blk[192 + 4] = 1; // scales[4]
    blk[208] = (F16_0_5 & 0xFF) as u8; // d f16 low
    blk[209] = (F16_0_5 >> 8) as u8; // d f16 high

    let mut out = [0f32; 256];
    dequant_to_f32(GgmlType::Q6K, &blk, 256, &mut out).unwrap();
    assert_eq!(out[0], -6.0);
    assert_eq!(out[64], -16.0);
}

#[test]
fn q4_k_single_block() {
    // d = 2.0, dmin = 1.0; scales[0]=1, scales[1]=2, scales[4]=0, scales[5]=0.
    // qs[0] = 0x35 (low nibble 5, high nibble 3).
    // chunk0: is=0 -> (sc,m)=(1,0) -> d1=2, m1=0; is=1 -> (2,0) -> d2=4, m2=0.
    // out[0]  = d1 * 5 - 0 = 10
    // out[32] = d2 * 3 - 0 = 12
    let mut blk = vec![0u8; 144];
    blk[0] = (F16_2_0 & 0xFF) as u8;
    blk[1] = (F16_2_0 >> 8) as u8;
    blk[2] = (F16_1_0 & 0xFF) as u8;
    blk[3] = (F16_1_0 >> 8) as u8;
    blk[4] = 1; // scales[0]
    blk[5] = 2; // scales[1]
    // scales[4], scales[5] already 0
    blk[16] = 0x35; // qs[0]

    let mut out = [0f32; 256];
    dequant_to_f32(GgmlType::Q4K, &blk, 256, &mut out).unwrap();
    assert_eq!(out[0], 10.0);
    assert_eq!(out[32], 12.0);
}

#[test]
fn q2_0_g128_single_block() {
    // PrismML id42, group 128, scale at FRONT. d = 2.0.
    // qs[0] = 0xE4 = 0b11_10_01_00 -> codes 0,1,2,3 (low-bits-first).
    // value = (code - 1)*d: -d, 0, +d, +2d.
    let mut blk = Vec::new();
    push_f16(&mut blk, F16_2_0);
    let mut qs = [0u8; 32]; // 128 / 4
    qs[0] = 0xE4;
    blk.extend_from_slice(&qs);

    let mut out = [0f32; 128];
    dequant_to_f32(GgmlType::Q2_0 { group: 128 }, &blk, 128, &mut out).unwrap();
    assert_eq!(out[0], -2.0); // code 0 -> -d
    assert_eq!(out[1], 0.0); // code 1 ->  0
    assert_eq!(out[2], 2.0); // code 2 -> +d
    assert_eq!(out[3], 4.0); // code 3 -> +2d
    assert_eq!(out[4], -2.0); // qs[1]=0 -> code 0 -> -d
}

#[test]
fn q2_0_g64_block_bytes() {
    // Fork-master variant: group 64 -> block = 2 + 64/4 = 18 bytes.
    assert_eq!(GgmlType::Q2_0 { group: 64 }.block_bytes().unwrap(), 18);
    assert_eq!(GgmlType::Q2_0 { group: 128 }.block_bytes().unwrap(), 34);
    assert_eq!(GgmlType::Q2_0 { group: 64 }.block_size(), 64);
}

#[test]
fn rejects_short_input_and_bad_length() {
    let mut out = [0f32; 32];
    // Too few bytes for one Q8_0 block.
    assert!(dequant_to_f32(GgmlType::Q8_0, &[0u8; 10], 32, &mut out).is_err());
    // n_elements not a multiple of block size.
    assert!(dequant_to_f32(GgmlType::Q8_0, &[0u8; 34], 16, &mut out).is_err());
}

#[test]
fn bf16_bytes_facade_roundtrips_q8_0() {
    // Facade: id 8, d=2.0, qs[0]=3 -> value 6.0. BF16 bits of 6.0 = 0x40C0.
    let mut blk = Vec::new();
    push_f16(&mut blk, F16_2_0);
    let mut qs = [0u8; 32];
    qs[0] = 3;
    blk.extend_from_slice(&qs);
    let bytes = super::to_bf16_bytes(8, 128, &blk, 32).unwrap();
    assert_eq!(bytes.len(), 64);
    let v0 = u16::from_le_bytes([bytes[0], bytes[1]]);
    assert_eq!(v0, 0x40C0); // bf16(6.0)
    assert!(super::supports(8));
    assert!(super::supports(42));
    assert!(!super::supports(999));
}
