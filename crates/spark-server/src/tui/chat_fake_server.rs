// SPDX-License-Identifier: AGPL-3.0-only

//! A loopback HTTP server for the Chat pane's tests.
//!
//! Shared by `chat_stream_tests` (the client) and `chat_more_tests` (the
//! reducer) so one definition of "what the server did" backs both halves, and
//! so an SSE frame that the client mis-frames cannot be papered over by a
//! second, kinder fake in the other file.

use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{Receiver, channel};

/// A running fake, and the channel its captured request arrives on.
pub(crate) struct Fake {
    pub(crate) port: u16,
    pub(crate) request: Receiver<String>,
}

/// Bind a loopback listener, accept ONE connection, capture the raw request,
/// then let `reply` write whatever the test needs. The socket closes when
/// `reply` returns, which is the EOF the client waits for.
pub(crate) fn serve<F>(reply: F) -> Fake
where
    F: FnOnce(&mut TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
    let port = listener.local_addr().expect("local addr").port();
    let (tx, request) = channel();
    std::thread::spawn(move || {
        let Ok((mut sock, _)) = listener.accept() else {
            return;
        };
        let _ = tx.send(read_request(&mut sock));
        reply(&mut sock);
    });
    Fake { port, request }
}

/// Read exactly the headers plus the declared `Content-Length`.
///
/// Reading to EOF instead would deadlock: the client keeps the connection open
/// waiting for a reply this function has not let the test send yet.
fn read_request(sock: &mut TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        if let Some(end) = super::find_subslice(&buf, b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buf[..end]).into_owned();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("Content-Length: "))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buf.len() >= end + 4 + len {
                break;
            }
        }
        match sock.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// A 200 SSE reply, one `write` per chunk.
///
/// The chunk boundaries are the point: a test that hands the client the whole
/// body in one read never exercises the frame that arrives in halves.
pub(crate) fn sse(sock: &mut TcpStream, chunks: &[&[u8]]) {
    let _ = sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n");
    let _ = sock.flush();
    for c in chunks {
        if sock.write_all(c).is_err() {
            return;
        }
        let _ = sock.flush();
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}

/// A complete non-200 response with a `Content-Length` body.
pub(crate) fn refuse(sock: &mut TcpStream, status: &str, ctype: &str, body: &str) {
    let _ = sock.write_all(
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    );
    let _ = sock.flush();
}

/// The port of a listener that has already been closed — nothing answers there.
///
/// Picked by the OS rather than hardcoded so the test never knocks on a port
/// something else on the box is using.
pub(crate) fn dead_port() -> u16 {
    let l = TcpListener::bind(("127.0.0.1", 0)).expect("bind loopback");
    l.local_addr().expect("local addr").port()
}
