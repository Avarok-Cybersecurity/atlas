// SPDX-License-Identifier: AGPL-3.0-only
//
// A peer that REJECTS a client hangs up mid-handshake: the v2 wire carries no
// error frame, so the reason lives only in the peer's log. Bare `read_exact`
// context then surfaces "failed to fill whole buffer", which reads like a
// transport fault and sends operators hunting a network problem that does not
// exist.
//
// This was found by running a real serve against a capped `atlas-cache-peer`
// (client asked for a 31.9 GiB arena against a 4 GiB `--max-blade-gb`; the peer
// logged "paging blade cap" and closed). No unit test caught it, because every
// existing handshake test scripts a COOPERATIVE peer.

use std::io::{Read, Result as IoResult};

use atlas_rdma::handshake::read_rw_server_params;

/// A stream that yields `n` bytes then EOFs — the peer hanging up.
struct TruncatedPeer {
    remaining: Vec<u8>,
}

impl Read for TruncatedPeer {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<usize> {
        if self.remaining.is_empty() {
            return Ok(0); // clean EOF -> read_exact yields UnexpectedEof
        }
        let n = buf.len().min(self.remaining.len());
        buf[..n].copy_from_slice(&self.remaining[..n]);
        self.remaining.drain(..n);
        Ok(n)
    }
}

#[test]
fn peer_hangup_before_rail_params_names_the_rejection_not_a_buffer_error() {
    let mut s = TruncatedPeer { remaining: vec![] };
    let err = read_rw_server_params(&mut s, 1, "test-peer").unwrap_err();
    let msg = format!("{err:#}");

    assert!(
        msg.contains("closed the connection during the rail handshake"),
        "must say the peer hung up: {msg}"
    );
    assert!(
        msg.contains("REJECTED"),
        "must name this as a rejection, not a transport fault: {msg}"
    );
    assert!(
        msg.contains("--max-blade-gb"),
        "must point at the most common cause (arena exceeds the peer cap): {msg}"
    );
    assert!(
        msg.contains("test-peer"),
        "must name WHICH peer rejected us: {msg}"
    );
    // The regression we are pinning: the old message was exactly this.
    assert!(
        !msg.contains("failed to fill whole buffer"),
        "must not surface the raw io error as the headline: {msg}"
    );
}

#[test]
fn a_truncated_params_body_is_still_a_read_error_not_a_rejection() {
    // The peer DID send its n_rails echo, then died partway through the params
    // body. That is a genuine transport/protocol fault, not a clean rejection,
    // and must not be mislabelled as "the peer rejected you".
    let mut s = TruncatedPeer {
        remaining: vec![1u8, 0xde, 0xad],
    };
    let err = read_rw_server_params(&mut s, 1, "test-peer").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        !msg.contains("REJECTED"),
        "a mid-body truncation is not a rejection: {msg}"
    );
    assert!(
        msg.contains("params"),
        "should blame the params read: {msg}"
    );
}

#[test]
fn a_rail_count_mismatch_is_reported_as_a_grant_mismatch() {
    // Peer answers with a DIFFERENT rail count. Distinct failure, distinct
    // message — this path must not be swallowed by the new EOF branch.
    let mut s = TruncatedPeer {
        remaining: vec![3u8],
    };
    let err = read_rw_server_params(&mut s, 1, "test-peer").unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("granted 3 rails, wanted 1"), "{msg}");
}
