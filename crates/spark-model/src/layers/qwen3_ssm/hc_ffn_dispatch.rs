// SPDX-License-Identifier: AGPL-3.0-only

//! Keep native EXL3 replay on the decode router and stable expert grid.

#[derive(Debug, PartialEq, Eq)]
pub(super) enum HcFfnDispatch {
    Single,
    K2,
    K3,
    NativeBatched,
    Prefill,
}

pub(super) fn hc_ffn_dispatch(
    rows: usize,
    small_m: bool,
    exact_replay: bool,
    native_exl3: bool,
) -> HcFfnDispatch {
    match rows {
        1 if small_m => HcFfnDispatch::Single,
        2 if small_m => HcFfnDispatch::K2,
        3 if small_m => HcFfnDispatch::K3,
        // Four rows previously fell through to sorted-expert prefill,
        // bypassing the replay router and stable single-token split-K plan.
        4 if small_m && exact_replay && native_exl3 => HcFfnDispatch::NativeBatched,
        _ => HcFfnDispatch::Prefill,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_row_native_replay_uses_decode_experts() {
        assert_eq!(
            hc_ffn_dispatch(4, true, true, true),
            HcFfnDispatch::NativeBatched
        );
    }

    #[test]
    fn preserve_cold_prefill_other_formats_and_explicit_small_m_disable() {
        for rows in 1..=8 {
            for replay in [false, true] {
                for native in [false, true] {
                    assert_eq!(
                        hc_ffn_dispatch(rows, false, replay, native),
                        HcFfnDispatch::Prefill
                    );
                    let old = match rows {
                        1 => HcFfnDispatch::Single,
                        2 => HcFfnDispatch::K2,
                        3 => HcFfnDispatch::K3,
                        _ => HcFfnDispatch::Prefill,
                    };
                    if !(rows == 4 && replay && native) {
                        assert_eq!(hc_ffn_dispatch(rows, true, replay, native), old);
                    }
                }
            }
        }
    }
}
