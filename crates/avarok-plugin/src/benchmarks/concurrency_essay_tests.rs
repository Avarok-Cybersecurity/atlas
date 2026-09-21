// SPDX-License-Identifier: AGPL-3.0-only

//! THE ESSAY FIXTURE — the published ladder38 request, pinned byte for byte
//! against `bench/ladder38/harness_w55_conc_ladder.py`'s own output.
//!
//! Split from `concurrency_verdict_tests.rs` for the 500-LoC cap. Exact
//! piecewise copy — no test changed in the move. It is a CHILD of that
//! module, not a sibling: a sibling would need its own `gate::coverage`
//! registry row (which re-opens every gate), a child needs none.

use super::*;

fn sweep_with_fixture(name: &str) -> ConcurrencySweep {
    let mut b = ConcurrencySweep::default();
    let mut v = ParamValues::defaults(&b.parameters());
    v.set("prompt_mode", ParamValue::Text(name.into()));
    b.configure(&v).map(|()| b).unwrap()
}

/// Byte-identical to the Python harness: the digests are of
/// `bench/ladder38/harness_w55_conc_ladder.py`'s own output (`set_nonce_base(0)`,
/// `make_prompt(isl)` for isl 64 / 128 / 512, `essay` mode → `[req 000001]`).
#[test]
fn essay_fixture_is_byte_identical_to_the_ladder38_harness() {
    use sha2::Digest;
    let sha256 = |t: &str| format!("{:x}", sha2::Sha256::digest(t.as_bytes()));
    let b = sweep_with_fixture("essay");
    assert_eq!(b.fixture, Fixture::Essay);
    let digests = [
        "f948427566b56c54a2f52610de41de641a0595b91037101d509e81758edcb448",
        "571b4943bc5302ad252def6d06cb2e444f5f8116a8c62a9d92d742d05c25b1d8",
        "0f0a1c4ee1e7b09b68103c02264839b46c903d9ee2240eac2ac87bbcca4282c6",
    ];
    for (isl, digest) in [64usize, 128, 512].into_iter().zip(digests) {
        let p = b.cell_prompt(isl, &essay_nonce_tag(1));
        assert!(p.starts_with("[req 000001] The quick brown fox"), "{p}");
        assert!(p.ends_with(ESSAY_TASK) && !p.contains("MinHeap"), "{p}");
        assert_eq!(sha256(&p), digest, "isl {isl} differs from the harness");
    }
}

#[test]
fn essay_nonce_is_fixed_width_and_the_other_fixtures_keep_their_tags() {
    let plan = prompt_plan(128, 1, Fixture::Essay);
    assert_eq!(plan.measured[0], "req 000000");
    assert_eq!(plan.measured[127], "req 000127");
    let width = "req ".len() + ESSAY_NONCE_WIDTH;
    assert!(plan.measured.iter().all(|t| t.len() == width));
    assert_eq!(plan.warmup_rounds, std::slice::from_ref(&plan.measured));
    assert_eq!(essay_nonce_tag(1_000_007), "req 000007"); // never widens
    // Every committed natural/count record was measured on `c{i}`: unchanged.
    assert_eq!(prompt_plan(2, 0, Fixture::Natural).measured, ["c0", "c1"]);
    assert_eq!(prompt_plan(2, 0, Fixture::Count).measured, ["c0", "c1"]);
}

/// The essay request carries the harness's penalty pins, the fixtures whose
/// floors were calibrated on the server-filled preset do not, and the
/// published instrument is opted into by name, never defaulted into (PCND).
#[test]
fn essay_requests_pin_penalty_free_sampling_and_only_by_name() {
    let essay = sweep_with_fixture("essay").request_body("m", 128, "req 000001");
    assert_eq!(essay["presence_penalty"], 0.0);
    assert_eq!(essay["frequency_penalty"], 0.0);
    assert_eq!(essay["temperature"], 0.0);
    assert_eq!(essay["reasoning_effort"], "none");
    let content = essay["messages"][0]["content"].as_str().unwrap();
    assert!(content.ends_with(ESSAY_TASK), "{content}");
    for name in ["natural", "count"] {
        let body = sweep_with_fixture(name).request_body("m", 128, "c0");
        assert!(body.get("presence_penalty").is_none(), "{name}");
        assert!(body.get("frequency_penalty").is_none(), "{name}");
    }
    assert_eq!(Fixture::parse("ESSAY"), Some(Fixture::Essay));
    assert_eq!(Fixture::parse("prose"), None);
    let mut v = ParamValues::defaults(&ConcurrencySweep::default().parameters());
    assert_eq!(v.text("prompt_mode").unwrap(), "natural");
    v.set("prompt_mode", ParamValue::Text("prose".into()));
    assert!(ConcurrencySweep::default().configure(&v).is_err());
}
