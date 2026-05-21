// SPDX-License-Identifier: AGPL-3.0-only
//
// `add_same_length_range` — the same-byte-length half of the UTF-8
// character-range FSM helper. Split out of `char_range.rs` to keep each
// file under the 250-line cap. Port of `AddSameLengthCharacterRange`
// from `cpp/grammar_functor.cc`.

use crate::fsm::FsmWithStartEnd;

/// Decompose `[min, max]` of packed UTF-8 values of the *same byte
/// length* into byte-range edges from state `from` to state `to`.
pub fn add_same_length_range(
    fsm: &mut FsmWithStartEnd,
    from: usize,
    to: usize,
    mut min: u32,
    mut max: u32,
) {
    let bmin = |v: u32| {
        [
            (v & 0xFF) as u8,
            (v >> 8) as u8,
            (v >> 16) as u8,
            (v >> 24) as u8,
        ]
    };
    let mut byte_min = bmin(min);
    let mut byte_max = bmin(max);

    // ASCII (single byte).
    if byte_max[1] == 0 {
        fsm.fsm_mut()
            .add_edge(from, to, byte_min[0] as i16, byte_max[0] as i16);
        return;
    }

    if byte_max[3] != 0 {
        // 4-byte unicode.
        if byte_max[3] == byte_min[3] {
            let tmp = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, tmp, byte_min[3] as i16, byte_max[3] as i16);
            min &= 0x00FF_FFFF;
            max &= 0x00FF_FFFF;
            add_same_length_range(fsm, tmp, to, min, max);
            return;
        }
        if (min & 0x00FF_FFFF) != 0x808080 {
            let tmin = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, tmin, byte_min[3] as i16, byte_min[3] as i16);
            add_same_length_range(fsm, tmin, to, min & 0x00FF_FFFF, 0x00BF_BFBF);
        } else {
            byte_min[3] -= 1;
        }
        if (max & 0x00FF_FFFF) != 0xBFBFBF {
            let tmax = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, tmax, byte_max[3] as i16, byte_max[3] as i16);
            add_same_length_range(fsm, tmax, to, 0x0080_8080, max & 0x00FF_FFFF);
        } else {
            byte_max[3] += 1;
        }
        if byte_max[3] as i16 - byte_min[3] as i16 > 1 {
            let m1 = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, m1, byte_min[3] as i16 + 1, byte_max[3] as i16 - 1);
            let m2 = fsm.add_state();
            fsm.fsm_mut().add_edge(m1, m2, 0x80, 0xBF);
            let m3 = fsm.add_state();
            fsm.fsm_mut().add_edge(m2, m3, 0x80, 0xBF);
            fsm.fsm_mut().add_edge(m3, to, 0x80, 0xBF);
        }
        return;
    }
    if byte_max[2] != 0 {
        // 3-byte unicode.
        if byte_max[2] == byte_min[2] {
            let tmp = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, tmp, byte_min[2] as i16, byte_max[2] as i16);
            min &= 0x00FFFF;
            max &= 0x00FFFF;
            add_same_length_range(fsm, tmp, to, min, max);
            return;
        }
        if (min & 0x00FFFF) != 0x8080 {
            let tmin = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, tmin, byte_min[2] as i16, byte_min[2] as i16);
            add_same_length_range(fsm, tmin, to, min & 0x00FFFF, 0x00BFBF);
        } else {
            byte_min[2] -= 1;
        }
        if (max & 0x00FFFF) != 0xBFBF {
            let tmax = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, tmax, byte_max[2] as i16, byte_max[2] as i16);
            add_same_length_range(fsm, tmax, to, 0x0080, max & 0x00FFFF);
        } else {
            byte_max[2] += 1;
        }
        if byte_max[2] as i16 - byte_min[2] as i16 > 1 {
            let m1 = fsm.add_state();
            fsm.fsm_mut()
                .add_edge(from, m1, byte_min[2] as i16 + 1, byte_max[2] as i16 - 1);
            let m2 = fsm.add_state();
            fsm.fsm_mut().add_edge(m1, m2, 0x80, 0xBF);
            fsm.fsm_mut().add_edge(m2, to, 0x80, 0xBF);
        }
        return;
    }

    // 2-byte unicode.
    if byte_max[1] == byte_min[1] {
        let tmp = fsm.add_state();
        fsm.fsm_mut()
            .add_edge(from, tmp, byte_min[1] as i16, byte_max[1] as i16);
        min &= 0x00FF;
        max &= 0x00FF;
        add_same_length_range(fsm, tmp, to, min, max);
        return;
    }
    if (min & 0x00FF) != 0x80 {
        let tmin = fsm.add_state();
        fsm.fsm_mut()
            .add_edge(from, tmin, byte_min[1] as i16, byte_min[1] as i16);
        add_same_length_range(fsm, tmin, to, min & 0x00FF, 0x00BF);
    } else {
        byte_min[1] -= 1;
    }
    if (max & 0x00FF) != 0xBF {
        let tmax = fsm.add_state();
        fsm.fsm_mut()
            .add_edge(from, tmax, byte_max[1] as i16, byte_max[1] as i16);
        add_same_length_range(fsm, tmax, to, 0x0080, max & 0x00FF);
    } else {
        byte_max[1] += 1;
    }
    if byte_max[1] as i16 - byte_min[1] as i16 > 1 {
        let m1 = fsm.add_state();
        fsm.fsm_mut()
            .add_edge(from, m1, byte_min[1] as i16 + 1, byte_max[1] as i16 - 1);
        fsm.fsm_mut().add_edge(m1, to, 0x80, 0xBF);
    }
}
