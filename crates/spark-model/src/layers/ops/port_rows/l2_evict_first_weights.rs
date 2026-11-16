// SPDX-License-Identifier: AGPL-3.0-only
//! The `l2_evict_first_weights` row: the sm_89 FP8 decode and verify GEMVs
//! load their weights with an explicit evict-first L2 policy (the `_l2ef`
//! twins in `kernels/ada/common/fp8_gemv_ada.cu` and `fp8_gemv_ada_batchm.cu`,
//! picked in `layers/dense_ffn_ada_kernels.rs`). Output bits are unchanged;
//! only the cache policy of the weight loads moves. Resolution:
//! `AVAROK_L2_EVICT_FIRST` under the toggle grammar (`0`/`false`/`off`/`no` =
//! off, anything else = on), else the target's declaration. Off everywhere
//! until the SASS check, byte parity and a bench5 ABBA gain on Ada.
use crate::layers::ops::target_defaults::Resolved;

pub const ENV: &str = "AVAROK_L2_EVICT_FIRST";

pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .l2_evict_first_weights
        .value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::target_defaults::Source;

    #[test]
    fn the_toggle_grammar_and_the_declaration() {
        assert!(!resolve(false, None).value);
        assert_eq!(resolve(true, None).source, Source::Target);
        assert!(resolve(false, Some("yes")).value);
        assert!(!resolve(true, Some("off")).value);
        assert_eq!(resolve(true, Some("off")).source, Source::Env);
    }
}
