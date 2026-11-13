// SPDX-License-Identifier: AGPL-3.0-only
//! The `ffn_fp8_verify_gemv` row: the dense FFN's FP8 speculative-verify rows
//! (M = 2..8) on the sm_89 batch-M GEMV (`kernels/ada/common/fp8_gemv_ada_batchm.cu`,
//! `layers/dense_ffn_ada_kernels/verify.rs`), one weight pass per projection and
//! byte-identical per row to the M=1 decode forward, instead of the prefill
//! ladder. Resolution: `AVAROK_FFN_FP8_VERIFY_GEMV` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's
//! declaration. Off on every target until a GPU receipt (byte parity, greedy
//! SHA, verify time) flips it on Ada.
use crate::layers::ops::target_defaults::Resolved;

pub const ENV: &str = "AVAROK_FFN_FP8_VERIFY_GEMV";

pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .ffn_fp8_verify_gemv
        .value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::target_defaults::Source;

    #[test]
    fn the_toggle_grammar_and_the_declaration() {
        assert!(!resolve(false, None).value);
        assert_eq!(resolve(false, None).source, Source::Target);
        assert!(resolve(false, Some("1")).value);
        assert_eq!(resolve(false, Some("1")).source, Source::Env);
        for off in ["0", "false", "OFF", " no "] {
            assert!(!resolve(true, Some(off)).value, "{off:?}");
        }
    }
}
