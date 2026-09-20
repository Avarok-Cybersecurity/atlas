// SPDX-License-Identifier: AGPL-3.0-only

//! The decision table for the FP8 prefill predequant guard.
//!
//! No GPU and no environment: `Fp8PrefillKernels` is built by hand so what is
//! pinned is the rule, not the machine CI happens to run on.

use super::*;

/// Every kernel present: an NVIDIA build, and the state the whole guard has to
/// leave alone.
fn all_present() -> Fp8PrefillKernels {
    Fp8PrefillKernels {
        ldmab_armed: true,
        ldmab: true,
        bf16_to_fp8: true,
        scalar: true,
        m128: true,
    }
}

#[test]
fn a_complete_build_with_the_variable_unset_is_untouched() {
    assert_eq!(decide(None, all_present()), None);
}

/// ★ The R9700 case, stated as the table row it is: `w4a16_fp8_ldmab.cu` is a
/// gfx1201 census failure, so the module is not in the tree and the kernel the
/// default dispatch would launch does not resolve.
#[test]
fn a_missing_ldmab_kernel_skips_the_predequant() {
    let k = Fp8PrefillKernels {
        ldmab: false,
        ..all_present()
    };
    assert_eq!(
        decide(None, k),
        Some(PredequantSkip::KernelMissing(
            "w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab"
        )),
    );
}

/// ...but only while that route is armed. `AVAROK_FP8_LDMAB=0` takes
/// `fp8_gemm_n128` off it, and then a missing ldmab kernel is irrelevant.
#[test]
fn the_ldmab_kernel_only_matters_while_that_route_is_armed() {
    let k = Fp8PrefillKernels {
        ldmab_armed: false,
        ldmab: false,
        bf16_to_fp8: false,
        ..all_present()
    };
    assert_eq!(decide(None, k), None);
}

/// Both non-ldmab arms are required regardless, because which one runs is a
/// property of the request (`K % 32` and the token count), not of the build.
#[test]
fn every_arm_the_dispatch_can_take_must_resolve() {
    for (k, expected) in [
        (
            Fp8PrefillKernels {
                scalar: false,
                ..all_present()
            },
            "w4a16::fp8_gemm_t",
        ),
        (
            Fp8PrefillKernels {
                m128: false,
                ..all_present()
            },
            "w4a16::fp8_gemm_t_m128",
        ),
        (
            Fp8PrefillKernels {
                bf16_to_fp8: false,
                ..all_present()
            },
            "w4a16::bf16_to_fp8",
        ),
    ] {
        assert_eq!(
            decide(None, k),
            Some(PredequantSkip::KernelMissing(expected)),
            "a missing {expected} must skip the predequant"
        );
    }
    // And with the ldmab route disarmed, `bf16_to_fp8` is no longer on any
    // path `fp8_gemm_n128` takes.
    let k = Fp8PrefillKernels {
        ldmab_armed: false,
        bf16_to_fp8: false,
        ..all_present()
    };
    assert_eq!(decide(None, k), None);
}

#[test]
fn the_env_var_wins_over_a_complete_build_and_zero_forces_it_back() {
    assert_eq!(decide(Some("1"), all_present()), Some(PredequantSkip::Env));
    assert_eq!(decide(Some(""), all_present()), Some(PredequantSkip::Env));
    assert_eq!(
        decide(Some("yes"), all_present()),
        Some(PredequantSkip::Env)
    );
    // `0` is the escape hatch: build them even though the operator's shell
    // profile exports the variable.
    assert_eq!(decide(Some("0"), all_present()), None);
    // ...but it cannot override a kernel that does not exist. The env var
    // answers "should we", the probe answers "can we", and the probe is not
    // negotiable.
    let k = Fp8PrefillKernels {
        ldmab: false,
        ..all_present()
    };
    assert!(decide(Some("0"), k).is_some());
}

/// The reason string is what an operator reads out of a serve log, so it has to
/// name the thing they can act on.
#[test]
fn every_reason_names_what_is_missing() {
    assert!(
        PredequantSkip::Env
            .reason()
            .contains("AVAROK_NO_FP8_PREDEQUANT")
    );
    assert!(
        PredequantSkip::KernelMissing("w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab")
            .reason()
            .contains("w4a16_fp8_ldmab::fp8_fp8_gemm_ldmab")
    );
}
