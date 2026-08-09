// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side unit tests for the NLLB position tables + language formatting.

use super::super::NllbLang;
use super::*;

#[test]
fn decoder_pos_table_offsets_by_two() {
    // Position row `i` must equal sinusoid `i + 2` (fairseq offset).
    let d = 8;
    let t = decoder_pos_table_bf16(3, d);
    let mut expect = vec![half::bf16::from_f32(0.0); d];
    sinusoid_row(2.0, d, &mut expect); // row 0 → pos 2
    for j in 0..d {
        assert!(
            (t[j].to_f32() - expect[j].to_f32()).abs() < 1e-2,
            "row0 col{j}"
        );
    }
}

#[test]
fn encoder_positions_skip_pad_and_count_from_two() {
    // ids: [lang, tokA, pad] with pad=1 → positions [2, 3, pad(zeroed)].
    let d = 8;
    let ids = [256047u32, 100, 1];
    let pos = encoder_pos_bf16(&ids, d, 1);
    // pad row (index 2) is all zero
    assert!(pos[2 * d..3 * d].iter().all(|v| v.to_f32() == 0.0));
    // first row equals sinusoid(2)
    let mut e = vec![half::bf16::from_f32(0.0); d];
    sinusoid_row(2.0, d, &mut e);
    for j in 0..d {
        assert!((pos[j].to_f32() - e[j].to_f32()).abs() < 1e-2);
    }
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
