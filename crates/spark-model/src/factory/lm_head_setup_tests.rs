// SPDX-License-Identifier: AGPL-3.0-only

//! The guard table for releasing the checkpoint's own `lm_head` tensor.
//!
//! Every `false` here is a pointer someone still holds. The one that is easiest
//! to lose is `source_is_fp8`: it is not a preference or a target check, it is
//! the PROOF that `load_lm_head` made a copy. Drop it and the release is a
//! use-after-free on every BF16 and every NVFP4-packed checkpoint.
//!
//! No GPU, no checkpoint, no environment.

use super::lm_head_source_is_dead;

/// ★ The one configuration that releases: an FP8 checkpoint served with the
/// default head and no drafter. `unsloth/Qwen3.8-27B-NVFP4` under the r9700
/// recipe is exactly this, and it is 1.18 GiB.
#[test]
fn the_default_serve_of_an_fp8_checkpoint_releases() {
    assert!(lm_head_source_is_dead(true, false, false, false));
}

/// `load_lm_head` returns the STORE's pointer for anything that is not FP8:
/// `dense()` for BF16, and `weight_map::quantized` binds the packed bytes for
/// an NVFP4-prepacked head. Releasing either frees memory the model is about
/// to read.
#[test]
fn a_checkpoint_whose_lm_head_was_never_copied_never_releases() {
    for lm_head_fp8 in [false, true] {
        for dflash in [false, true] {
            for speculative in [false, true] {
                assert!(
                    !lm_head_source_is_dead(false, lm_head_fp8, dflash, speculative),
                    "released a store pointer the model still holds \
                     (lm_head_fp8={lm_head_fp8}, dflash={dflash}, spec={speculative})"
                );
            }
        }
    }
}

/// `--lm-head-dtype fp8` is the path whose entire purpose is to bind the
/// checkpoint's own E4M3 bytes zero-copy rather than mirror them.
#[test]
fn the_native_fp8_head_keeps_its_own_bytes() {
    assert!(!lm_head_source_is_dead(true, true, false, false));
}

/// `--dflash` calls `native_fp8_lm_head_share` again for the drafter tail, and
/// that call happens AFTER the release site. `--speculative` is exclusive with
/// it in clap and is guarded anyway, because the two flags reach different code.
#[test]
fn a_drafter_keeps_the_head_it_has_not_bound_yet() {
    assert!(!lm_head_source_is_dead(true, false, true, false));
    assert!(!lm_head_source_is_dead(true, false, false, true));
    assert!(!lm_head_source_is_dead(true, true, true, true));
}
