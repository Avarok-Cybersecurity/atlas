// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn ascii_codepoint_packs_to_itself() {
    assert_eq!(codepoint_to_packed_utf8(b'a' as u32), b'a' as u32);
    assert_eq!(codepoint_to_packed_utf8(0x7F), 0x7F);
}

#[test]
fn two_byte_codepoint() {
    // U+00E9 'é' -> 0xC3 0xA9
    assert_eq!(codepoint_to_packed_utf8(0xE9), 0xC3A9);
}

#[test]
fn three_byte_codepoint() {
    // U+20AC '€' -> 0xE2 0x82 0xAC
    assert_eq!(codepoint_to_packed_utf8(0x20AC), 0xE282AC);
}

#[test]
fn ascii_range_single_edge() {
    let mut fsm = FsmWithStartEnd::default();
    let s = fsm.add_state();
    let e = fsm.add_state();
    fsm.set_start_state(s);
    fsm.add_end_state(e);
    add_character_range(&mut fsm, s, e, b'a' as u32, b'z' as u32);
    assert!(fsm.accept_string(b"m"));
    assert!(!fsm.accept_string(b"A"));
}
