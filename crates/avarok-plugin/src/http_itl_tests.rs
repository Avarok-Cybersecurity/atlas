// SPDX-License-Identifier: AGPL-3.0-only

//! ITL per AIPerf, the server's raw timing window, and the arrival-gap
//! jitter instrument.
//!
//! Split from `http_tests.rs` for the 500-LoC cap when these landed. Exact
//! piecewise copy — no test changed in the move. It is a CHILD of that
//! module rather than a sibling so `sse`/`endpoint_answering` and the rest
//! of the mock harness stay single-sourced.

use super::*;

/// A mock that writes its reply in timed phases, so the client's clocks can
/// be checked against a schedule it did not choose.
async fn endpoint_answering_in_phases(phases: Vec<(String, Duration)>) -> TargetEndpoint {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback");
    let port = listener.local_addr().expect("address").port();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");
        let mut request = [0u8; 4096];
        let _ = socket.read(&mut request).await.expect("read request");
        for (bytes, pause) in phases {
            socket.write_all(bytes.as_bytes()).await.expect("reply");
            tokio::time::sleep(pause).await;
        }
    });
    TargetEndpoint::local(port, "mock")
}

const SSE_HEAD: &str =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";

fn content_delta(text: &str) -> String {
    format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{text}\"}}}}]}}\n\n")
}

/// The server's raw timing components ride in `usage` beside the shipped
/// rate; a server without them leaves them `None`, and the server-clock
/// ITL derived from them is the AIPerf rule on the server's numbers.
#[test]
fn server_raw_timing_components_are_captured_and_the_server_itl_is_derived() {
    let mut out = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":10,"prompt_tokens":12,
                "time_to_first_token_ms":100.0,"response_token/s":10.0,
                "decode_time_ms":900.0,"total_time_ms":1000.0},"choices":[]}"#),
        &mut out,
    );
    assert_eq!(out.server_decode_time_ms, Some(900.0));
    assert_eq!(out.server_total_time_ms, Some(1000.0));
    assert_eq!(out.server_tpot_ms(), Some(100.0));
    // The identity the server publishes: (total − ttft)/(n−1) == decode/(n−1).
    let from_total = (out.server_total_time_ms.unwrap() - out.server_ttft_ms.unwrap()) / 9.0;
    assert!((from_total - out.server_tpot_ms().unwrap()).abs() < 1e-9);

    let mut bare = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":10,"prompt_tokens":2,
                "time_to_first_token_ms":100.0,"response_token/s":10.0},"choices":[]}"#),
        &mut bare,
    );
    assert_eq!(bare.server_decode_time_ms, None);
    assert_eq!(bare.server_total_time_ms, None);
    assert_eq!(
        bare.server_tpot_ms(),
        None,
        "no raw window → no server ITL, not 0"
    );

    let mut one_token = ChatOutcome::default();
    apply_chunk(
        &sse(r#"{"usage":{"completion_tokens":1,"prompt_tokens":2,
                "decode_time_ms":0.4,"total_time_ms":100.4},"choices":[]}"#),
        &mut one_token,
    );
    assert_eq!(one_token.server_decode_time_ms, Some(0.4));
    assert_eq!(
        one_token.server_tpot_ms(),
        None,
        "undefined below two tokens"
    );
}

/// AIPerf: the ITL numerator ends at the FINAL response chunk, not the last
/// content token. Three tokens arrive at once, then the usage chunk and
/// `[DONE]` follow 300 ms later: the old numerator (last delta − first
/// delta) is ~0 and the aligned one is ≥ 300 ms over n−1 = 2.
#[tokio::test]
async fn client_itl_numerator_ends_at_the_final_chunk_not_the_last_token() {
    let tail = Duration::from_millis(300);
    let target = endpoint_answering_in_phases(vec![
        (
            format!(
                "{SSE_HEAD}{}{}{}",
                content_delta("a"),
                content_delta("b"),
                content_delta("c")
            ),
            tail,
        ),
        (
            "data: {\"usage\":{\"completion_tokens\":3,\"prompt_tokens\":1},\"choices\":[]}\n\n\
             data: [DONE]\n\n"
                .to_string(),
            Duration::ZERO,
        ),
    ])
    .await;
    let out = chat_stream(&target, &serde_json::json!({}), Duration::from_secs(5))
        .await
        .expect("stream");
    assert_eq!(out.completion_tokens, 3);
    let tpot = out.tpot_ms.expect("two or more tokens → ITL defined");
    assert!(
        tpot >= tail.as_secs_f64() * 1000.0 / 2.0,
        "ITL {tpot:.1} ms must include the 300 ms to the final chunk over n−1 = 2"
    );
    // The client-side AIPerf identity: (e2e − ttft) / (n − 1) == tpot.
    let identity = (out.e2e_ms - out.ttft_ms.unwrap()) / 2.0;
    assert!((identity - tpot).abs() < 1e-6, "{identity} vs {tpot}");
}

/// A gap is between ARRIVALS (socket reads that carried a token), so a
/// burst of two deltas in one write is one arrival; and a single stalled
/// arrival dominates the max and the p99 while the mean barely moves.
#[tokio::test]
async fn arrival_gaps_are_per_read_and_a_stall_shows_in_the_tail() {
    let step = Duration::from_millis(40);
    let stall = Duration::from_millis(400);
    let mut phases = vec![(format!("{SSE_HEAD}{}", content_delta("t0")), step)];
    for i in 1..10 {
        // Every other arrival is a two-delta burst — one read, one gap.
        let bytes = if i % 2 == 0 {
            format!("{}{}", content_delta("x"), content_delta("y"))
        } else {
            content_delta("z")
        };
        phases.push((bytes, if i == 5 { stall } else { step }));
    }
    phases.push(("data: [DONE]\n\n".to_string(), Duration::ZERO));
    let target = endpoint_answering_in_phases(phases).await;
    let out = chat_stream(&target, &serde_json::json!({}), Duration::from_secs(5))
        .await
        .expect("stream");
    assert_eq!(
        out.completion_tokens, 14,
        "10 arrivals, 4 of them bursts of 2"
    );
    let g = out.arrival_gaps.stats().expect("gaps recorded");
    assert_eq!(g.count, 9, "10 arrivals → 9 gaps: a burst is ONE arrival");
    assert!(g.max_ms >= 400.0, "the stall is the max: {g:?}");
    assert!(g.p99_ms >= 400.0, "…and the p99: {g:?}");
    assert!(g.p50_ms < 200.0, "the median is a normal step: {g:?}");
    let s = g.stability().expect("p50 > 0");
    assert!(
        s > 1.0,
        "one stall in nine steps: stability RISES (lower is better): {s}"
    );
    assert!(g.cv().unwrap() > 0.5, "{g:?}");
}

/// A reply delivered in one read has no gap to time: the jitter instrument
/// reports nothing rather than a perfectly smooth zero, and ITL is `None`
/// below two tokens.
#[tokio::test]
async fn a_single_arrival_yields_no_gaps_and_one_token_yields_no_itl() {
    let target = endpoint_answering_in_phases(vec![(
        format!("{SSE_HEAD}{}data: [DONE]\n\n", content_delta("only")),
        Duration::ZERO,
    )])
    .await;
    let out = chat_stream(&target, &serde_json::json!({}), Duration::from_secs(5))
        .await
        .expect("stream");
    assert_eq!(out.completion_tokens, 1);
    assert_eq!(out.arrival_gaps.stats(), None);
    assert_eq!(out.tpot_ms, None);
    assert!(out.ttft_ms.is_some());
}
