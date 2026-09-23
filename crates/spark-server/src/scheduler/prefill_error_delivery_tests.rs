// SPDX-License-Identifier: AGPL-3.0-only

//! A prefill that fails AFTER the request's response sink was taken must
//! deliver the failure to that request: an `Error` frame on a stream, an
//! `Err` (surfaced as a 5xx) on a blocking call. `handle_prefill_start_error`
//! only logs — the request is consumed by the time it runs — so this is the
//! contract `start_chunked_prefill` itself has to keep, driven here through
//! the real entry point with a model whose `prefill_chunk` refuses (the G25
//! serve's every-request `cuMemsetD8Async` 901 had exactly this shape).

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use futures::StreamExt;

use super::test_support::PreemptStubModel;
use crate::api::InferenceRequest;
use crate::api::inference_types::StreamEvent;

/// One request body; `$variant` and its sink field(s) are the only difference.
macro_rules! request {
    ($variant:ident { $($sink:tt)* }) => {
        InferenceRequest::$variant {
            prompt_tokens: Arc::new(vec![1, 2, 3]),
            session_hash: 0,
            adapter_slot: -1,
            src_lang_id: 0,
            tgt_lang_id: 0,
            num_beams: 1,
            length_penalty: 1.0,
            early_stopping: false,
            image_pixels: vec![],
            max_tokens: 4,
            min_tokens: 0,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            top_n_sigma: 0.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            dry_multiplier: 0.0,
            dry_base: 0.0,
            dry_allowed_length: 0,
            lz_penalty: 0.0,
            logit_bias: vec![],
            stop_tokens: vec![],
            enable_thinking: false,
            thinking_budget: None,
            repetition_detection: None,
            require_tool_call: false,
            tools_present: false,
            suppress_tool_call: false,
            disable_mtp: true,
            grammar_spec: None,
            seed: Some(42),
            top_logprobs: None,
            prompt_logprobs: None,
            echo: false,
            timeout_at: None,
            $($sink)*
        }
    };
}

/// Drive the real chunked-prefill start, inline (not deferred), against a
/// model whose `prefill_chunk` fails.
fn start(req: InferenceRequest) -> anyhow::Result<()> {
    let mut engine = None;
    super::prefill_a_step::start_chunked_prefill(
        &super::sched_ctx::SchedCtx::for_test(),
        None,
        None,
        None,
        None,
        &PreemptStubModel::default(),
        req,
        &[],
        64,
        0,
        0,
        &mut engine,
        0,
        false,
        None,
        None,
    )
    .map(|_| ())
}

#[tokio::test]
async fn a_streaming_prefill_start_error_reaches_the_client_as_an_error_frame() {
    let (token_tx, token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
    let err = start(request!(Streaming {
        token_tx,
        cancel_flag: Arc::new(AtomicBool::new(false)),
    }))
    .expect_err("the stub's prefill_chunk refuses");
    assert!(
        format!("{err:#}").contains("unused in preempt tests"),
        "{err:#}"
    );

    // What the SSE surface reads: the channel through `terminated`, so this
    // asserts the frame the client sees, not merely what was sent.
    let events: Vec<StreamEvent> = crate::api::stream_terminal::terminated(
        tokio_stream::wrappers::ReceiverStream::new(token_rx),
    )
    .collect()
    .await;
    let [StreamEvent::Error(msg)] = events.as_slice() else {
        panic!(
            "expected exactly one Error frame, got {} events",
            events.len()
        );
    };
    assert!(
        msg.starts_with("prefill_chunk failed:") && msg.contains("unused in preempt tests"),
        "the frame must carry the real reason, not the no-result backstop: {msg}"
    );
}

#[tokio::test]
async fn a_blocking_prefill_start_error_reaches_the_caller_as_an_err() {
    let (response_tx, response_rx) = tokio::sync::oneshot::channel();
    start(request!(Blocking { response_tx })).expect_err("the stub's prefill_chunk refuses");
    let got = response_rx
        .await
        .expect("the sink must be answered, not dropped (a drop reads as 'Inference cancelled')");
    let Err(e) = got else {
        panic!("a failed prefill must not produce a response");
    };
    assert!(
        format!("{e:#}").starts_with("prefill_chunk failed:"),
        "{e:#}"
    );
}
