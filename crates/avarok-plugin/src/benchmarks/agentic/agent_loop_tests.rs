// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;
use crate::artifacts::ArtifactStore;
use crate::plugin::{PluginHandle, TargetEndpoint};

async fn two_reply_server(first: &str, first_reason: &str) -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&requests);
    let first = first.to_string();
    let first_reason = first_reason.to_string();
    tokio::spawn(async move {
        for (text, reason) in [
            (first.as_str(), first_reason.as_str()),
            ("finished", "stop"),
        ] {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = socket.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                let Some(header_end) = request.windows(4).position(|w| w == b"\r\n\r\n") else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let length = headers
                    .lines()
                    .find_map(|line| {
                        line.to_ascii_lowercase()
                            .strip_prefix("content-length: ")
                            .map(str::to_string)
                    })
                    .and_then(|n| n.parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= header_end + 4 + length {
                    break;
                }
            }
            seen.lock()
                .push(String::from_utf8_lossy(&request).into_owned());
            let text = serde_json::to_string(text).unwrap();
            let body = format!(
                "data: {{\"choices\":[{{\"delta\":{{\"content\":{text}}},\"finish_reason\":\"{reason}\"}}]}}\n\ndata: [DONE]\n\n"
            );
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        }
    });
    (port, requests)
}

async fn run_two_turns(first: &str, first_reason: &str) -> (Transcript, Vec<String>) {
    let (port, requests) = two_reply_server(first, first_reason).await;
    let (events, receiver) = std::sync::mpsc::channel();
    let sandbox = super::tests::sandbox("loop-recovery");
    let handle = PluginHandle::new(
        1,
        TargetEndpoint::local(port, "test-model"),
        ArtifactStore::with_root(&sandbox),
        events,
        Arc::new(AtomicBool::new(false)),
    );
    let mut config = super::tests::cfg(sandbox);
    config.max_turns = 2;
    config.request_timeout = Duration::from_secs(2);
    let transcript = run_task(&handle, &config, "do the task").await.unwrap();
    drop(receiver);
    let requests = requests.lock().clone();
    (transcript, requests)
}

#[tokio::test]
async fn a_truncated_turn_is_returned_to_the_model_with_a_recovery_instruction() {
    let (transcript, requests) = run_two_turns("partial answer", "length").await;
    assert_eq!(transcript.turns, 2);
    assert_eq!(transcript.truncated_turns, 1);
    assert_eq!(transcript.final_text, "finished");
    assert_eq!(requests.len(), 2);
    assert!(requests[1].contains("partial answer"));
    assert!(requests[1].contains("Continue from where it stopped"));
}

#[tokio::test]
async fn unparsed_tool_syntax_is_returned_with_a_reissue_instruction() {
    let (transcript, requests) = run_two_turns("<tool_call><function=bash>", "stop").await;
    assert_eq!(transcript.turns, 2);
    assert_eq!(transcript.unparsed_call_turns, 1);
    assert_eq!(transcript.final_text, "finished");
    assert_eq!(requests.len(), 2);
    assert!(requests[1].contains("<tool_call><function=bash>"));
    assert!(requests[1].contains("Re-issue exactly that one call"));
}
