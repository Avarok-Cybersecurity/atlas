// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the RDMA LoRA landing byte-transforms (convert + B repack),
//! byte-identical to the disk pack.

use super::*;
use half::{bf16, f16};

#[test]
fn convert_f32_halves_len_and_rounds_like_disk() {
    // Two f32 values → 4 BF16 bytes, matching half::bf16::from_f32.
    let vals = [1.5f32, -2.25f32];
    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let out = convert_to_bf16(&raw, "F32").unwrap();
    assert_eq!(out.len(), 4, "F32 [n] → BF16 [n] halves the byte count");
    let expect: Vec<u8> = vals
        .iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect();
    assert_eq!(out, expect);
}

#[test]
fn convert_f16_preserves_len() {
    let raw: Vec<u8> = [f16::from_f32(0.5), f16::from_f32(3.0)]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = convert_to_bf16(&raw, "F16").unwrap();
    assert_eq!(out.len(), 4, "F16 [n] → BF16 [n] same byte count");
}

#[test]
fn convert_bf16_is_identity() {
    let raw: Vec<u8> = (0..8).collect();
    assert_eq!(convert_to_bf16(&raw, "BF16").unwrap(), raw);
}

#[test]
fn convert_rejects_unknown_dtype() {
    assert!(convert_to_bf16(&[0u8; 4], "F8_E4M3").is_err());
}

#[test]
fn repack_b_pads_columns_zero_and_preserves_rows() {
    // out_dim=2, r=1, max_rank=3. src = 2 rows × 1 bf16 = [A0,A1].
    // dst = 2 rows × 3 bf16, real col at head, pads zero.
    let src: Vec<u8> = vec![0xAA, 0xBB, 0xCC, 0xDD]; // row0=[AA BB], row1=[CC DD]
    let dst = repack_b_to_padded(&src, 2, 1, 3);
    assert_eq!(dst.len(), 2 * 3 * 2);
    assert_eq!(&dst[0..2], &[0xAA, 0xBB]); // row0 real
    assert_eq!(&dst[2..6], &[0, 0, 0, 0]); // row0 pad cols
    assert_eq!(&dst[6..8], &[0xCC, 0xDD]); // row1 real
    assert_eq!(&dst[8..12], &[0, 0, 0, 0]); // row1 pad cols
}

#[test]
fn land_a_is_convert_only() {
    let t = LoraLandTarget {
        tensor_name: "x".into(),
        kind: LoraAbKind::A,
        dst: 0,
        out_dim: 0,
        in_dim: 2,
        rank: 1,
        max_rank: 4,
    };
    let raw: Vec<u8> = [1.0f32, 2.0f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = land_bytes_for_target(&t, &raw, "F32").unwrap();
    assert_eq!(out.len(), 4); // r*in*2 = 1*2*2, no padding for A columns
}

#[test]
fn land_b_converts_then_repacks() {
    let t = LoraLandTarget {
        tensor_name: "y".into(),
        kind: LoraAbKind::B,
        dst: 0,
        out_dim: 2,
        in_dim: 0,
        rank: 1,
        max_rank: 3,
    };
    // B = [out=2, r=1] f32 → convert → repack to [2,3] bf16.
    let raw: Vec<u8> = [5.0f32, 6.0f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let out = land_bytes_for_target(&t, &raw, "F32").unwrap();
    assert_eq!(out.len(), 2 * 3 * 2);
    // Row real cols == bf16 of the source values.
    assert_eq!(&out[0..2], &bf16::from_f32(5.0).to_le_bytes());
    assert_eq!(&out[6..8], &bf16::from_f32(6.0).to_le_bytes());
}
