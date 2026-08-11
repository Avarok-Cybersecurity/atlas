// SPDX-License-Identifier: AGPL-3.0-only

//! `/v1/models` must advertise the context window it will actually honour.
//!
//! `max_model_len` is not an OpenAI field — it is a vLLM extension that
//! clients (LiteLLM, aider, Continue, OpenWebUI) read to size a request
//! before sending it. Without it a client either guesses or discovers the
//! ceiling by having a request rejected at admission.
//!
//! The property under test is not "the field exists" but **"the advertised
//! ceiling equals the enforced one"**. Those are two different numbers the
//! moment someone hardcodes a literal here, which is exactly the bug this
//! guards: `AppState::max_seq_len` is what the admission path checks, so it
//! must be what the wire reports.

use crate::openai::ModelInfo;

/// Wire JSON for an entry built the way the handlers build it.
///
/// ★ This goes through `ModelInfo::advertise`, the SAME constructor
/// `list_models` and `get_model` call. An earlier version of this file built
/// the struct literally, and a mutation that hardcoded the ceiling in the
/// handler SURVIVED — the test proved serialization and nothing about
/// derivation. Drive the production constructor or the test is decoration.
///
/// Honest limit, established by mutation: hardcoding the ceiling *inside*
/// `advertise` is caught by **rustc**, not by these tests — `max_seq_len`
/// becomes an unused parameter and `#[deny(warnings)]` rejects it. The tests
/// below own the wire contract (field name, omission-vs-zero); the compiler
/// owns "the parameter is actually used". Both are needed and neither
/// substitutes for the other.
fn advertised(max_seq_len: usize) -> serde_json::Value {
    serde_json::to_value(ModelInfo::advertise("test-model".to_string(), max_seq_len))
        .expect("ModelInfo serializes")
}

/// Wire JSON for the no-model-loaded case, which has no ceiling to report.
fn unknown_ceiling() -> serde_json::Value {
    serde_json::to_value(ModelInfo {
        id: "test-model".to_string(),
        object: "model".to_string(),
        created: 0,
        owned_by: "atlas-spark".to_string(),
        max_model_len: None,
    })
    .expect("ModelInfo serializes")
}

#[test]
fn advertised_ceiling_is_the_served_max_seq_len() {
    // POSITIVE: a loaded model reports its real ceiling, on the wire, under
    // the name clients actually look for.
    let served_max_seq_len = 32768usize;
    let v = advertised(served_max_seq_len);
    assert_eq!(
        v.get("max_model_len").and_then(|x| x.as_u64()),
        Some(served_max_seq_len as u64),
        "the advertised ceiling must equal the value admission enforces \
         (AppState::max_seq_len); a hardcoded literal here silently diverges \
         from the scheduler the first time someone changes --max-seq-len"
    );
}

#[test]
fn no_model_loaded_omits_the_field_rather_than_reporting_zero() {
    // NEGATIVE: before a model is chosen the honest answer is "unknown", and
    // absence says that. A serialized 0 would read as "zero context" — a
    // client sizing against it would refuse to send anything at all, which is
    // a worse failure than not knowing.
    let v = unknown_ceiling();
    assert!(
        v.get("max_model_len").is_none(),
        "unknown must be ABSENT, not 0: {v}"
    );
    // The rest of the object must still be well-formed — omitting one optional
    // field must not make the entry unusable.
    assert_eq!(v.get("id").and_then(|x| x.as_str()), Some("test-model"));
    assert_eq!(v.get("object").and_then(|x| x.as_str()), Some("model"));
}

#[test]
fn the_field_name_is_the_one_clients_probe() {
    // Guards a rename. `max_model_len` is the vLLM spelling; `max_tokens`,
    // `context_length` and `max_seq_len` are all plausible-looking names that
    // no client reads. A rename would be silently inert in production and is
    // invisible to a test that only checks the value.
    let v = advertised(4096);
    let obj = v.as_object().expect("object");
    assert!(obj.contains_key("max_model_len"), "keys: {:?}", obj.keys());
    for wrong in [
        "max_seq_len",
        "context_length",
        "max_tokens",
        "context_window",
    ] {
        assert!(
            !obj.contains_key(wrong),
            "{wrong} is not the field clients read; keys: {:?}",
            obj.keys()
        );
    }
}
