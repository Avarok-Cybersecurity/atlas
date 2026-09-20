// SPDX-License-Identifier: AGPL-3.0-only

//! The wording of the registry's diagnostics, and the one decision behind the
//! symbol guard. Everything under test here is pure, so these run on a host
//! with no CUDA context and no GPU.

use std::collections::HashSet;

use super::{
    func_attribute_failure_message, is_undefined, kernel_label, launch_failure_message,
    undefined_symbol_message,
};

#[test]
fn launch_failure_names_the_kernel() {
    let msg = launch_failure_message(
        &kernel_label(
            Some("qwen3.6-27b_nvfp4::dense_gemv_bf16_batch2"),
            0x7f0c_1234_5678,
        ),
        "CUDA_ERROR_INVALID_IMAGE (200): invalid image",
        [5440, 1, 1],
        [256, 1, 1],
        0,
    );
    assert_eq!(
        msg,
        "cuLaunchKernel failed for qwen3.6-27b_nvfp4::dense_gemv_bf16_batch2 \
         (fn@0x7f0c12345678): CUDA_ERROR_INVALID_IMAGE (200): invalid image \
         (grid=[5440,1,1], block=[256,1,1], shared_mem=0)"
    );
}

#[test]
fn unregistered_handle_still_reports_the_pointer() {
    let msg = launch_failure_message(
        &kernel_label(None, 0x42),
        "CUDA_ERROR_INVALID_VALUE (1): invalid argument",
        [1, 2, 3],
        [64, 1, 1],
        8192,
    );
    assert_eq!(
        msg,
        "cuLaunchKernel failed for <unregistered kernel> (fn@0x42): \
         CUDA_ERROR_INVALID_VALUE (1): invalid argument \
         (grid=[1,2,3], block=[64,1,1], shared_mem=8192)"
    );
}

/// The refusal names both halves and says why, because on SCALE the
/// alternative was a launch-time `CUDA_ERROR_INVALID_IMAGE` with nothing
/// in it about an optional module.
#[test]
fn an_undefined_symbol_is_refused_by_name() {
    assert_eq!(
        undefined_symbol_message("nvfp4_mmq", "avarok_nvfp4_repack"),
        "nvfp4_mmq::avarok_nvfp4_repack: not defined in this target's code object \
         (optional module compiled out?)"
    );
}

/// The three answers a code object can give, and what each one licenses.
#[test]
fn only_a_parsed_set_that_lacks_the_name_refuses_a_lookup() {
    let defines: HashSet<String> = ["avarok_nvfp4_repack".to_string()].into_iter().collect();
    // Present: the driver is asked, as always.
    assert!(!is_undefined(Some(&defines), "avarok_nvfp4_repack"));
    // Parsed and absent: refused here, before any driver call. This is
    // nvfp4_mmq on gfx1201: SCALE would answer SUCCESS for this name.
    assert!(is_undefined(Some(&defines), "avarok_nvfp4_quantize"));
    // An optional module compiled out defines nothing at all.
    assert!(is_undefined(Some(&HashSet::new()), "avarok_nvfp4_repack"));
    // No parsed set (PTX, or an object we could not read): never refused.
    assert!(!is_undefined(None, "avarok_nvfp4_quantize"));
}

#[test]
fn attribute_failure_names_the_kernel() {
    assert_eq!(
        func_attribute_failure_message(
            &kernel_label(Some("common::prefill_paged"), 0xabc),
            65536,
            "CUDA_ERROR_INVALID_VALUE (1): invalid argument"
        ),
        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED=65536) failed for common::prefill_paged \
         (fn@0xabc): CUDA_ERROR_INVALID_VALUE (1): invalid argument"
    );
}
