// SPDX-License-Identifier: AGPL-3.0-only
//! The `affine4_prefill_int8` row: the affine routes' prompt GEMM (M > 16) quantizes the activation
//! rows to int8 per group and runs the W4A8 int8 tensor-core GEMM over the same 4-bit codes (lane
//! L17b, `port::affine::prefill`) instead of the BF16-activation prompt GEMM; a chunk with a
//! non-finite or overflowing group falls back to BF16. 4-bit weights only; it changes nothing unless an
//! affine route row is on. Resolution: `AVAROK_AFFINE4_PREFILL_INT8` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration. Off on every
//! target until the prompt-row KL against BF16 and a TTFT gain pass on Ada.
use crate::layers::ops::target_defaults::Resolved;

pub const ENV: &str = "AVAROK_AFFINE4_PREFILL_INT8";

pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .affine4_prefill_int8
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
        assert!(resolve(false, Some("true")).value);
        assert_eq!(resolve(false, Some("true")).source, Source::Env);
        for off in ["0", "false", "off", "NO"] {
            assert!(!resolve(true, Some(off)).value, "{off:?}");
        }
    }
}
