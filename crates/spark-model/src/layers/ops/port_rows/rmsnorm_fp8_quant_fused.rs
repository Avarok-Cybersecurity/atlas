// SPDX-License-Identifier: AGPL-3.0-only
//! The `rmsnorm_fp8_quant_fused` row: the prefill FFN-norm of a Qwen3 / Qwen3.5
//! layer writes the dense FFN's W8A8 gate/up activation (E4M3 codes plus
//! 1x128 FP32 scales) in the same pass as the norm
//! (`kernels/ada/common/rmsnorm_quant_fp8_ada.cu`,
//! `layers/dense_ffn_w8a8_prefill/norm_fused.rs`), so the separate
//! `per_token_group_quant_fp8` launch and its re-read of the normed rows go.
//! Byte-identical to the unfused pair. Resolution: `AVAROK_RMSNORM_FP8_QUANT`
//! under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else =
//! on), else the target's declaration. Off everywhere until the byte-equal
//! microtest and a TTFT ABBA gain on Ada.
use crate::layers::ops::target_defaults::Resolved;

pub const ENV: &str = "AVAROK_RMSNORM_FP8_QUANT";

pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .rmsnorm_fp8_quant_fused
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
        assert!(resolve(false, Some("on")).value);
        assert!(!resolve(true, Some("0")).value);
    }
}
