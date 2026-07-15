// SPDX-License-Identifier: AGPL-3.0-only

// TRANSCRIPT goldens — the only test class that pins the client's write ORDER
// (struct goldens pin field layout; round-trips are blind to symmetric
// reorders). Drives the exact handshake step sequence `railset::RailSet`
// performs, via the un-gated `handshake` byte functions with FIXED QP
// identities (no ibverbs — runs on the ATLAS_SKIP_BUILD/CI path), against a
// scripted fake peer, and asserts:
//   * the client's COMPLETE emitted byte stream, hand-written, per dialect
//     (RO = expert/weight/LoRA, RW = KV/snapshot) at 1 and 2 rails;
//   * every read happens at the right point of the written stream (the
//     [n_rails] → server params → [n_rails]+client params → ack interleaving);
//   * the parsed server params, against hand-written reply bytes (pins the
//     reader field order independent of the writer).
// These transcripts are frozen: a stable external wire contract.

use std::io::{Read, Write};

use atlas_rdma::handshake::{read_ack, read_rw_server_params, write_client_params, write_n_rails};
use atlas_rdma::wire::{CacheServerParams, VerbsServerParams, read_server_rails};

/// Fake bidirectional stream: scripted input, captured output, plus a log of
/// `out.len()` at every read call — pinning WHEN the client reads relative to
/// what it has written.
struct Duplex {
    inp: std::io::Cursor<Vec<u8>>,
    out: Vec<u8>,
    reads_at: Vec<usize>,
}

impl Duplex {
    fn scripted(reply: Vec<u8>) -> Self {
        Self {
            inp: std::io::Cursor::new(reply),
            out: Vec::new(),
            reads_at: Vec::new(),
        }
    }
}

impl Read for Duplex {
    fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
        self.reads_at.push(self.out.len());
        self.inp.read(b)
    }
}

impl Write for Duplex {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.out.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn gid(start: u8) -> [u8; 16] {
    core::array::from_fn(|i| start + i as u8)
}

/// Fixed client QP identities with per-field-distinct bytes.
const CLIENT_IDS: [(u32, u32, [u8; 16]); 2] = [
    (
        0x1122_3344,
        0x00AA_BBCC,
        [
            0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D,
            0x6E, 0x6F,
        ],
    ),
    (
        0x5566_7788,
        0x00DD_EEFF,
        [
            0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D,
            0x7E, 0x7F,
        ],
    ),
];

/// The client params block: `[u8 n]` + n × `[qpn][psn][gid]` (24 B each).
fn expected_client_params(n: usize) -> Vec<u8> {
    let mut v = vec![n as u8];
    #[rustfmt::skip]
    v.extend_from_slice(&[
        0x44, 0x33, 0x22, 0x11, // qpn 0x1122_3344 LE
        0xCC, 0xBB, 0xAA, 0x00, // psn 0x00AA_BBCC LE
        0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, // gid verbatim
        0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,
    ]);
    if n == 2 {
        #[rustfmt::skip]
        v.extend_from_slice(&[
            0x88, 0x77, 0x66, 0x55, // qpn 0x5566_7788 LE
            0xFF, 0xEE, 0xDD, 0x00, // psn 0x00DD_EEFF LE
            0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, // gid verbatim
            0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E, 0x7F,
        ]);
    }
    v
}

/// Hand-written server-side QP identity block `[qpn][psn][gid]` for rail `i`.
fn server_qp_bytes(i: usize) -> Vec<u8> {
    match i {
        0 => {
            let mut v = vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
            v.extend_from_slice(&gid(0x20));
            v
        }
        _ => {
            let mut v = vec![0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10];
            v.extend_from_slice(&gid(0x30));
            v
        }
    }
}

const BASE0: [u8; 8] = [0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]; // 0x1122_3344_5566_7788
const RKEY0: [u8; 4] = [0xCC, 0xBB, 0xAA, 0x99]; // 0x99AA_BBCC
const BASE1: [u8; 8] = [0x0D, 0xF0, 0xAD, 0x0B, 0xEF, 0xBE, 0xAD, 0xDE]; // 0xDEAD_BEEF_0BAD_F00D
const RKEY1: [u8; 4] = [0x04, 0x03, 0x02, 0x01]; // 0x0102_0304

fn expected_server_param(i: usize) -> VerbsServerParams {
    let (qpn, psn) = if i == 0 {
        (0x0403_0201, 0x0807_0605)
    } else {
        (0x0C0B_0A09, 0x100F_0E0D)
    };
    VerbsServerParams {
        qpn,
        psn,
        gid: gid(if i == 0 { 0x20 } else { 0x30 }),
        layers: vec![if i == 0 {
            (0x1122_3344_5566_7788, 0x99AA_BBCC)
        } else {
            (0xDEAD_BEEF_0BAD_F00D, 0x0102_0304)
        }],
    }
}

/// RO dialect (expert / weight / LoRA), post-preamble: the client writes
/// `[u8 n]`, reads `[u8 n]{VerbsServerParams}×n`, writes `[u8 n]` + client
/// params, reads `[ack]`.
fn drive_ro(n: usize) {
    // Scripted peer reply: [u8 n] + n × (qp identity + [u32 1][base][rkey]),
    // then the STATUS_OK ack byte. Hand-written.
    let mut reply = vec![n as u8];
    for i in 0..n {
        reply.extend_from_slice(&server_qp_bytes(i));
        reply.extend_from_slice(&1u32.to_le_bytes()); // n_layers = 1
        reply.extend_from_slice(if i == 0 { &BASE0 } else { &BASE1 });
        reply.extend_from_slice(if i == 0 { &RKEY0 } else { &RKEY1 });
    }
    reply.push(0x00); // ack = STATUS_OK

    let mut fake = Duplex::scripted(reply);
    // The exact RailSet step sequence (begin → read_server_ro → complete).
    write_n_rails(&mut fake, n).unwrap();
    let server = read_server_rails(&mut fake, n).unwrap();
    write_client_params(&mut fake, &CLIENT_IDS[..n]).unwrap();
    read_ack(&mut fake, "test peer").unwrap();

    // 1. The parsed server params, against the hand bytes (reader order pin).
    let want: Vec<VerbsServerParams> = (0..n).map(expected_server_param).collect();
    assert_eq!(server, want);

    // 2. The complete emitted client stream, hand-written (writer order pin).
    let mut expect = vec![n as u8];
    expect.extend_from_slice(&expected_client_params(n));
    assert_eq!(fake.out, expect, "RO client transcript changed ({n} rails)");

    // 3. Interleaving: every server-params read happened after exactly the
    // 1-byte n_rails write; the ack read after the full client stream.
    let total = expect.len();
    assert_eq!(fake.reads_at.first(), Some(&1));
    assert_eq!(fake.reads_at.last(), Some(&total));
    assert!(fake.reads_at.iter().all(|&a| a == 1 || a == total));
}

/// RW dialect (KV / snapshot), post-preamble: `[u8 n]` out, `[u8 n echo]` +
/// n × CacheServerParams in, `[u8 n]` + client params out, `[ack]` in.
fn drive_rw(n: usize) {
    let mut reply = vec![n as u8];
    for i in 0..n {
        reply.extend_from_slice(&server_qp_bytes(i));
        reply.extend_from_slice(if i == 0 { &BASE0 } else { &BASE1 });
        reply.extend_from_slice(if i == 0 { &RKEY0 } else { &RKEY1 });
    }
    reply.push(0x00); // ack

    let mut fake = Duplex::scripted(reply);
    write_n_rails(&mut fake, n).unwrap();
    let server = read_rw_server_params(&mut fake, n, "test peer").unwrap();
    write_client_params(&mut fake, &CLIENT_IDS[..n]).unwrap();
    read_ack(&mut fake, "test peer").unwrap();

    let want: Vec<CacheServerParams> = (0..n)
        .map(|i| {
            let ro = expected_server_param(i);
            CacheServerParams {
                qpn: ro.qpn,
                psn: ro.psn,
                gid: ro.gid,
                base_addr: ro.layers[0].0,
                rkey: ro.layers[0].1,
            }
        })
        .collect();
    assert_eq!(server, want);

    let mut expect = vec![n as u8];
    expect.extend_from_slice(&expected_client_params(n));
    assert_eq!(fake.out, expect, "RW client transcript changed ({n} rails)");

    let total = expect.len();
    assert_eq!(fake.reads_at.first(), Some(&1));
    assert_eq!(fake.reads_at.last(), Some(&total));
    assert!(fake.reads_at.iter().all(|&a| a == 1 || a == total));
}

#[test]
fn ro_transcript_single_rail() {
    drive_ro(1);
}

#[test]
fn ro_transcript_dual_rail() {
    drive_ro(2);
}

#[test]
fn rw_transcript_single_rail() {
    drive_rw(1);
}

#[test]
fn rw_transcript_dual_rail() {
    drive_rw(2);
}

#[test]
fn rw_echo_mismatch_bails() {
    // Peer echoes 3 rails when the client negotiated 2 → protocol error
    // BEFORE any params are consumed. (Deliberately unbounded otherwise —
    // the RO dialect's 1..=8 bound is its own.)
    let mut fake = Duplex::scripted(vec![3u8]);
    assert!(read_rw_server_params(&mut fake, 2, "test peer").is_err());
}

#[test]
fn non_ok_ack_bails() {
    let mut fake = Duplex::scripted(vec![1u8]); // STATUS_ERR
    let err = read_ack(&mut fake, "test peer").unwrap_err();
    assert!(err.to_string().contains("refused connection (ack 1)"));
}
