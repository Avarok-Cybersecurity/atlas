// SPDX-License-Identifier: AGPL-3.0-only
//! The `attn_affine4_from_bf16` row: on a plain-BF16 dense Qwen3.5 checkpoint, quantize each
//! full-attention layer's q, k, v and o to MLX affine codes once at load (`--w4-bits`, `--w4-group`)
//! and serve them through `W4Linear` (lane L23a's route,
//! `weight_loader/qwen35_dense/attn_affine4_from_bf16.rs`). Resolution:
//! `AVAROK_ATTN_AFFINE4_FROM_BF16` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything
//! else = on), else the target's declaration. Off on every target until the D7 gate passes on Ada.
use crate::layers::ops::target_defaults::Resolved;

pub const ENV: &str = "AVAROK_ATTN_AFFINE4_FROM_BF16";

pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .attn_affine4_from_bf16
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
        for off in ["0", "FALSE", "off", "no"] {
            assert!(!resolve(true, Some(off)).value, "{off:?}");
        }
    }
}
