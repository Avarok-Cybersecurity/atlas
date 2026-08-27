// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side unit tests for the NLLB position tables + language formatting.

use super::super::NllbLang;
use super::*;

#[test]
fn decoder_pos_table_offsets_by_two() {
    // Position row `i` must equal sinusoid `i + 2` (fairseq offset).
    let d = 8;
    let t = decoder_pos_table_bf16(3, d);
    for (row, logical_pos) in [(0, 2.0), (2, 4.0)] {
        let mut expect = vec![half::bf16::from_f32(0.0); d];
        sinusoid_row(logical_pos, d, &mut expect);
        assert_eq!(&t[row * d..(row + 1) * d], expect, "row {row}");
    }
}

#[test]
fn encoder_positions_skip_pad_and_count_from_two() {
    // ids: [lang, pad, tokA] with pad=1 → positions [2, pad(zeroed), 3].
    let d = 8;
    let ids = [256047u32, 1, 100];
    let pos = encoder_pos_bf16(&ids, d, 1);
    assert!(pos[d..2 * d].iter().all(|v| v.to_f32() == 0.0));
    for (row, logical_pos) in [(0, 2.0), (2, 3.0)] {
        let mut expect = vec![half::bf16::from_f32(0.0); d];
        sinusoid_row(logical_pos, d, &mut expect);
        assert_eq!(&pos[row * d..(row + 1) * d], expect, "row {row}");
    }
}

#[test]
fn h2d_byte_views_preserve_element_bytes() {
    let ints = [0x0102_0304u32, 0xa0b0_c0d0];
    assert_eq!(u32_bytes(&ints), &[4, 3, 2, 1, 0xd0, 0xc0, 0xb0, 0xa0]);

    let halves = [half::bf16::from_bits(0x1234), half::bf16::from_bits(0xabcd)];
    assert_eq!(bf16_bytes(&halves), &[0x34, 0x12, 0xcd, 0xab]);
}

#[test]
fn encoder_input_wraps_src_lang_and_eos() {
    let lang = NllbLang {
        src_lang_id: 256047,
        tgt_lang_id: 256057,
        decoder_start_id: 2,
        eos_id: 2,
        pad_id: 1,
    };
    assert_eq!(
        lang.encoder_input(&[10, 20, 30]),
        vec![256047, 10, 20, 30, 2]
    );
}
