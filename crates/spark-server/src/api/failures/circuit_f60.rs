// SPDX-License-Identifier: AGPL-3.0-only

//! F60 MTP-disable env switch, hoisted from `circuit.rs` to keep that file
//! under the 500 LoC cap. Re-exported through `failures/mod.rs`.

/// F60 (2026-04-27, fix38): formerly disabled MTP for tool-use turns
/// as a workaround for SSM state corruption observed under high MTP
/// reject rates on agentic workloads.
///
/// F65 (2026-04-27, fix39): SUPERSEDED. F62 implements proper SpecMamba
/// dual-buffer SSM rollback (per arXiv:2509.19873 / arXiv:2505.14969),
/// which makes MTP correctness guaranteed by construction — the live
/// SSM state is never mutated during verify, so the high reject rate
/// no longer stresses the rollback path. F60's default flips to OFF.
///
/// The env-var escape hatch is preserved: setting
/// `ATLAS_DISABLE_MTP_FOR_TOOLS=1` still disables MTP for tool-use
/// (for A/B testing or rollback if F62 has unexpected regressions).
pub fn f60_disable_mtp_for_request(tools_active: bool) -> bool {
    if !tools_active {
        return false;
    }
    matches!(
        std::env::var("ATLAS_DISABLE_MTP_FOR_TOOLS").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}
