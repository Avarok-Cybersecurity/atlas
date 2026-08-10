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

#[test]
fn the_limit_bites_on_encoded_bytes_not_on_characters() {
    // The cap is on what goes down the wire. A 3-byte character encodes to
    // four base64 bytes just like three ASCII ones, so a character count would
    // let a CJK selection through at three times the real size.
    let ascii = MAX_BYTES / 4 * 3;
    assert!(
        !too_large(&"a".repeat(ascii)),
        "the largest ASCII that fits"
    );
    assert!(too_large(&"a".repeat(ascii + 3)), "one base64 group over");
    // A third as many characters, the same number of bytes, the same verdict.
    assert!(too_large(&"日".repeat(ascii / 3 + 1)));
}

#[test]
fn an_empty_selection_is_not_reported_as_too_large() {
    // `too_large` drives a distinct message; nothing-selected must reach its
    // own branch rather than being explained as a size problem.
    assert!(!too_large(""));
}

#[test]
fn a_successful_copy_reports_characters_not_bytes() {
    // The toast says how much was sent, and a byte count would claim 24 for an
    // eight-character CJK selection.
    //
    // This is the one test that actually writes the escape to stdout — the
    // write bypasses cargo's print capture, so running the suite in a terminal
    // really does set that terminal's clipboard. There is no other way to
    // cover the success path, and OSC 52 has no reply to fake.
    assert_eq!(
        copy("日本語のテキスト").expect("a normal selection copies"),
        8
    );
}

#[test]
fn an_oversized_copy_explains_itself_in_characters() {
    // A byte count here would read as three times the selection the user made.
    let err = copy(&"日".repeat(MAX_BYTES)).unwrap_err();
    assert!(err.contains(&format!("{MAX_BYTES} chars")), "{err}");
}

#[test]
fn control_characters_in_the_selection_are_encoded_not_forwarded() {
    // Log lines carry raw ESC. Passing one through unencoded would end the OSC
    // 52 sequence early and let the rest of the selection execute as escapes.
    let seq = osc52("\x1b]0;pwned\x07 rest").expect("encodes");
    let payload = &seq[b"\x1b]52;c;".len()..seq.len() - 1];
    assert!(
        !payload.contains(&0x1b) && !payload.contains(&0x07),
        "no bare ESC or BEL survives into the payload"
    );
    let back = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .expect("valid base64");
    assert_eq!(
        String::from_utf8(back).expect("utf8"),
        "\x1b]0;pwned\x07 rest"
    );
}

#[test]
fn a_multi_line_selection_is_one_sequence_with_no_embedded_newlines() {
    // A newline inside the escape sequence would be delivered to the shell.
    // Base64 has none, but only if the encoder is not the line-wrapping kind.
    let seq = osc52("line one\nline two\r\nline three").expect("encodes");
    assert_eq!(
        seq.iter().filter(|b| **b == b'\n' || **b == b'\r').count(),
        0
    );
}
