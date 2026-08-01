// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_sequence_is_a_well_formed_osc52() {
    let seq = osc52("hi").expect("non-empty text encodes");
    let s = String::from_utf8(seq).expect("ascii escape sequence");
    // ESC ] 52 ; c ; <base64> BEL
    assert!(s.starts_with("\x1b]52;c;"), "{s:?}");
    assert!(s.ends_with('\x07'), "{s:?}");
    assert!(s.contains("aGk="), "base64 of 'hi': {s:?}");
}

#[test]
fn utf8_survives_the_encoding() {
    // Model ids and log lines carry box-drawing and em dashes; mangling them
    // would put broken text on the clipboard rather than failing loudly.
    let text = "Qwen3.6-35B — ✓ 0.8% ▓░";
    let s = String::from_utf8(osc52(text).unwrap()).unwrap();
    let b64 = s.trim_start_matches("\x1b]52;c;").trim_end_matches('\x07');
    let back = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    assert_eq!(String::from_utf8(back).unwrap(), text);
}

#[test]
fn empty_text_is_not_a_copy() {
    assert!(osc52("").is_none());
    assert!(
        copy("").is_err(),
        "an empty selection must not claim success"
    );
}

#[test]
fn an_oversized_selection_is_refused_rather_than_truncated() {
    // Terminals silently drop or truncate an over-long sequence. A truncated
    // clipboard is worse than a refusal, because the user does not find out
    // until they paste.
    let huge = "x".repeat(MAX_BYTES);
    assert!(
        too_large(&huge),
        "should exceed the limit once base64-expanded"
    );
    assert!(osc52(&huge).is_none());
    let err = copy(&huge).unwrap_err();
    assert!(err.contains("too large"), "{err}");
}

#[test]
fn a_selection_just_under_the_limit_is_accepted() {
    // base64 is 4/3, so this is comfortably inside.
    let ok = "y".repeat(MAX_BYTES / 2);
    assert!(!too_large(&ok));
    assert!(osc52(&ok).is_some());
}
