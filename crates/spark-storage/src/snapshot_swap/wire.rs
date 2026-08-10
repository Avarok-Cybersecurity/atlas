// SPDX-License-Identifier: AGPL-3.0-only

// ───────────────────────────── control protocol ─────────────────────────────
//
// The TCP control channel between a paging client and the peer. It rides the
// SAME stream the peer used for the RDMA handshake (which today just idles).
// v2-only: EVERY client's first bytes are
// `[u64 PAGING_MAGIC_V2][u8 kind][u64 arena_bytes][u64 blob_bytes]`.
// `blob_bytes == 0` selects the RAW one-sided mode (per-connection arena,
// client-owned allocator, no residency — the legacy data plane,
// now selected explicitly); anything else is a paging arena. The retired v1
// magic and the bare-`total_bytes` legacy escape are affirmatively rejected.
//
// After the shared rail handshake, the loop is: client sends [op][key], peer
// replies [status] (+ [offset] for ALLOC/GET-hit). Data still moves one-sided
// over RDMA into/out of `slot_offset(slot)`; only tiny control messages cross
// TCP. Peer and client halves deliberately share this ONE module so the wire
// format (byte-frozen, golden-pinned — it is what the fleet peer binary
// speaks) can never drift.

use std::io::{Read, Write};

use anyhow::{Context, Result, bail};

use super::{Residency, SlotArena, SwapStore};

/// The RETIRED v1 first-u64 ("PAGE" + 1). Recognized ONLY to reject it with a
/// dedicated diagnostic — the v1 dialect was deleted (no kind byte) and the
/// bare-`total_bytes` legacy escape, so a stale binary fails legibly at
/// handshake instead of being reinterpreted.
const PAGING_MAGIC_V1_RETIRED: u64 = 0x5041_4745_0000_0001;

/// The ONLY accepted first u64 on the RW paging port ("PAGE" + 2, v2-only
///): after the magic comes a `[u8 kind]` byte so ONE peer serves
/// a registry of per-(kind, shape) arenas, then `[u64 arena_bytes]`
/// `[u64 blob_bytes]` — `blob_bytes == 0` selects the RAW one-sided mode.
pub const PAGING_MAGIC_V2: u64 = 0x5041_4745_0000_0002;

/// The tier a paging arena serves. Only the RW paging kinds (SSM, KV-as-paging)
/// ride the `CacheServerParams` single-base+rkey reply; the read-only tiers
/// (experts/weights/lora) speak a different manifest+VerbsServerParams dialect
/// and are NOT accepted on this handshake (rejected in `parse_paging_header`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PagingKind(pub u8);
impl PagingKind {
    pub const SSM: PagingKind = PagingKind(0);
    pub const KV: PagingKind = PagingKind(1);
    /// Whether this kind is servable on the RW paging (CacheServerParams) path.
    pub fn is_paging_rw(self) -> bool {
        self.0 <= 1
    }
}

/// Parse the paging handshake header after the caller has read the first u64.
/// v2-only: the first u64 MUST be [`PAGING_MAGIC_V2`], then
/// `[u8 kind][u64 arena_bytes][u64 blob_bytes]`. `blob_bytes == 0` = the RAW
/// one-sided mode (per-connection arena, client-owned allocator — the caller
/// routes it OFF the paging registry). Rejects unsupported kinds (≥2), the
/// retired v1 magic (dedicated diagnostic so a stale binary fails legibly),
/// and any other first u64 (e.g. a bare legacy `total_bytes`).
pub fn parse_paging_header<R: Read>(first: u64, r: &mut R) -> Result<(PagingKind, u64, u64)> {
    if first == PAGING_MAGIC_V1_RETIRED {
        bail!(
            "paging: v1 client no longer supported (magic {first:#x}); rebuild the client — \
             every connect now sends [u64 PAGING_MAGIC_V2][u8 kind][u64 arena_bytes][u64 blob_bytes]"
        );
    }
    if first != PAGING_MAGIC_V2 {
        bail!(
            "paging: first u64 {first:#x} is not PAGING_MAGIC_V2 (the bare legacy total_bytes \
             handshake was retired; RAW one-sided clients send the v2 header with \
             blob_bytes == 0)"
        );
    }
    let mut kb = [0u8; 1];
    r.read_exact(&mut kb).context("read paging kind")?;
    let kind = PagingKind(kb[0]);
    if !kind.is_paging_rw() {
        bail!(
            "paging: unsupported kind {} (only SSM/KV ride this handshake)",
            kb[0]
        );
    }
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8).context("read paging arena_bytes")?;
    let arena_bytes = u64::from_le_bytes(b8);
    r.read_exact(&mut b8).context("read paging blob_bytes")?;
    let blob_bytes = u64::from_le_bytes(b8);
    Ok((kind, arena_bytes, blob_bytes))
}

/// Encode the CLIENT half of the v2 paging handshake header — what EVERY
/// paging client sends first: KV paging, SSM paging (`connect_paging`), and
/// both RAW one-sided modes (via `blob_bytes == 0`):
/// `[u64 PAGING_MAGIC_V2 LE][u8 kind][u64 arena_bytes LE][u64 blob_bytes LE]`
/// — 25 bytes, followed by the unchanged `[u8 n_rails]` RailSet exchange.
/// Lives in this ONE shared module (beside `parse_paging_header`, its peer
/// half) so writer and reader can never drift; byte-frozen and golden-pinned
/// in `wire_tests.rs`.
pub fn encode_paging_v2_header(kind: PagingKind, arena_bytes: u64, blob_bytes: u64) -> [u8; 25] {
    let mut w = [0u8; 25];
    w[0..8].copy_from_slice(&PAGING_MAGIC_V2.to_le_bytes());
    w[8] = kind.0;
    w[9..17].copy_from_slice(&arena_bytes.to_le_bytes());
    w[17..25].copy_from_slice(&blob_bytes.to_le_bytes());
    w
}

/// Round-robin stripe a `blob_bytes` transfer into `chunk_bytes` chunks across
/// `n_rails`, returning per-rail lists of `(offset, len)`. The offset is the
/// chunk's position in BOTH the (single, contiguous) staging buffer and the peer
/// arena slot — so whichever rail fetches chunk j, it lands at its true offset
/// and one memcpy reassembles the blob (the verified inc-6 reassembly fix). The
/// tail chunk carries the short remainder, never `chunk_bytes`.
pub fn stripe_plan(
    blob_bytes: usize,
    chunk_bytes: usize,
    n_rails: usize,
) -> Vec<Vec<(usize, usize)>> {
    let n = n_rails.max(1);
    let cb = chunk_bytes.max(1);
    let mut rails: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    let mut off = 0usize;
    let mut j = 0usize;
    while off < blob_bytes {
        let len = cb.min(blob_bytes - off);
        rails[j % n].push((off, len));
        off += len;
        j += 1;
    }
    rails
}

/// Chunk size for the striped snapshot pipeline (ATLAS_SSM_CHUNK_BYTES, default
/// 1 MiB) and pipeline depth (ATLAS_SSM_PIPELINE_DEPTH, default 16, clamped
/// 1..=128, mirroring the KV backend).
pub fn staging_chunk_bytes() -> usize {
    std::env::var("ATLAS_SSM_CHUNK_BYTES")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v >= 4096)
        .unwrap_or(1024 * 1024)
}
pub fn staging_depth() -> usize {
    std::env::var("ATLAS_SSM_PIPELINE_DEPTH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16)
        .clamp(1, 128)
}

pub const OP_BYE: u8 = 0;
pub const OP_ALLOC: u8 = 1;
pub const OP_COMMIT: u8 = 2;
pub const OP_GET: u8 = 3;
pub const OP_REMOVE: u8 = 4;

pub const ST_OK: u8 = 0;
pub const ST_MISS: u8 = 1;
pub const ST_ERR: u8 = 2;

/// Result of one control request, ready to serialize.
#[derive(Debug, PartialEq, Eq)]
pub enum PagingReply {
    /// ST_OK with a following u64 arena offset (ALLOC and GET-hit).
    Located(u64),
    /// ST_OK with no payload (COMMIT, REMOVE).
    Ok,
    /// ST_MISS — unknown key (GET).
    Miss,
    /// ST_ERR — operation failed (e.g. arena exhausted by reservations).
    Err,
    /// Client asked to close.
    Bye,
}

/// Execute one control op against the residency and return the reply. Pure over
/// the (already unit-tested) `Residency`, so the protocol is testable
/// without a socket or RDMA.
pub fn dispatch<A: SlotArena, S: SwapStore>(
    res: &mut Residency<A, S>,
    op: u8,
    key: u64,
) -> PagingReply {
    match op {
        OP_BYE => PagingReply::Bye,
        OP_ALLOC => match res.alloc(key) {
            Ok(slot) => PagingReply::Located(res.slot_offset(slot)),
            Err(e) => {
                tracing::warn!("paging ALLOC {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_COMMIT => match res.commit(key) {
            Ok(()) => PagingReply::Ok,
            Err(e) => {
                tracing::warn!("paging COMMIT {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_GET => match res.locate(key) {
            Ok(Some(slot)) => PagingReply::Located(res.slot_offset(slot)),
            Ok(None) => PagingReply::Miss,
            Err(e) => {
                tracing::warn!("paging GET {key:#x} failed: {e:#}");
                PagingReply::Err
            }
        },
        OP_REMOVE => {
            res.remove(key);
            PagingReply::Ok
        }
        other => {
            tracing::warn!("paging: unknown op {other}");
            PagingReply::Err
        }
    }
}

fn write_reply<W: Write>(w: &mut W, reply: &PagingReply) -> Result<()> {
    match reply {
        PagingReply::Located(off) => {
            w.write_all(&[ST_OK])?;
            w.write_all(&off.to_le_bytes())?;
        }
        PagingReply::Ok => w.write_all(&[ST_OK])?,
        PagingReply::Miss => w.write_all(&[ST_MISS])?,
        PagingReply::Err => w.write_all(&[ST_ERR])?,
        PagingReply::Bye => {}
    }
    w.flush()?;
    Ok(())
}

/// The peer-side control loop: read `[op][u64 key]` requests, dispatch against
/// `res`, write replies, until BYE or hangup. Generic over the stream so it runs
/// against a real `TcpStream` in the peer and a fake duplex in tests.
/// One control op with connection-scoped read-pin lifecycle. Releases this
/// connection's previous GET read-pin — its RDMA READ has necessarily drained,
/// because the client is synchronous and only sends its NEXT op after the read
/// completes — then dispatches, and pins a fresh GET hit so a concurrent ALLOC
/// on another connection cannot evict the slot mid-read. `pinned` threads the
/// connection's currently-pinned key across calls. Needs NO new opcode and no
/// client change (auto-release on next op / disconnect) → wire-compatible with
/// the frozen control protocol.
fn handle_paging_op<A: SlotArena, S: SwapStore>(
    res: &mut Residency<A, S>,
    op: u8,
    key: u64,
    pinned: &mut Option<u64>,
) -> PagingReply {
    if let Some(prev) = pinned.take() {
        res.unpin_read(prev);
    }
    let reply = dispatch(res, op, key);
    if op == OP_GET && matches!(reply, PagingReply::Located(_)) {
        res.pin_read(key);
        *pinned = Some(key);
    }
    reply
}

pub fn run_paging_loop<T: Read + Write, A: SlotArena, S: SwapStore>(
    stream: &mut T,
    res: &mut Residency<A, S>,
) -> Result<()> {
    let mut pinned: Option<u64> = None;
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            break; // client hung up
        }
        if op[0] == OP_BYE {
            break;
        }
        let mut kb = [0u8; 8];
        stream.read_exact(&mut kb).context("read paging key")?;
        let key = u64::from_le_bytes(kb);
        let reply = handle_paging_op(res, op[0], key, &mut pinned);
        write_reply(stream, &reply)?;
    }
    // Release this connection's outstanding read-pin on hangup / BYE.
    if let Some(pk) = pinned {
        res.unpin_read(pk);
    }
    Ok(())
}

// ─────────────────────── client-side protocol helpers ───────────────────────
//
// The CLIENT half of the control channel, sharing the wire format above so peer
// and client can never drift. Each sends `[op][u64 key]` and reads the reply.
// The RDMA data-plane WRITE/READ (client-side) happens between `client_alloc`
// and `client_commit` (PUT) or after `client_get` (GET) — see RdmaSnapshotArena.

fn send_req<T: Write>(s: &mut T, op: u8, key: u64) -> Result<()> {
    let mut buf = [0u8; 9];
    buf[0] = op;
    buf[1..].copy_from_slice(&key.to_le_bytes());
    s.write_all(&buf)?;
    s.flush()?;
    Ok(())
}

fn read_status<T: Read>(s: &mut T) -> Result<u8> {
    let mut st = [0u8; 1];
    s.read_exact(&mut st).context("read paging status")?;
    Ok(st[0])
}

fn read_offset<T: Read>(s: &mut T) -> Result<u64> {
    let mut b = [0u8; 8];
    s.read_exact(&mut b).context("read paging offset")?;
    Ok(u64::from_le_bytes(b))
}

/// PUT step 1: reserve a slot for `key`; returns the arena offset to RDMA-WRITE.
pub fn client_alloc<T: Read + Write>(s: &mut T, key: u64) -> Result<u64> {
    send_req(s, OP_ALLOC, key)?;
    match read_status(s)? {
        ST_OK => read_offset(s),
        st => bail!("paging ALLOC {key:#x} refused (status {st})"),
    }
}

/// PUT step 2: the RDMA-WRITE has drained; mark `key` resident.
pub fn client_commit<T: Read + Write>(s: &mut T, key: u64) -> Result<()> {
    send_req(s, OP_COMMIT, key)?;
    match read_status(s)? {
        ST_OK => Ok(()),
        st => bail!("paging COMMIT {key:#x} failed (status {st})"),
    }
}

/// GET: `Some(offset)` to RDMA-READ, or `None` if the peer has no such key.
pub fn client_get<T: Read + Write>(s: &mut T, key: u64) -> Result<Option<u64>> {
    send_req(s, OP_GET, key)?;
    match read_status(s)? {
        ST_OK => Ok(Some(read_offset(s)?)),
        ST_MISS => Ok(None),
        st => bail!("paging GET {key:#x} error (status {st})"),
    }
}

/// Drop `key` from the peer cache.
pub fn client_remove<T: Read + Write>(s: &mut T, key: u64) -> Result<()> {
    send_req(s, OP_REMOVE, key)?;
    match read_status(s)? {
        ST_OK => Ok(()),
        st => bail!("paging REMOVE {key:#x} failed (status {st})"),
    }
}

/// Politely tell the peer to close the paging loop.
pub fn client_bye<T: Write>(s: &mut T) -> Result<()> {
    send_req(s, OP_BYE, 0)
}

/// Shared variant of [`run_paging_loop`]: many connection threads drive ONE
/// process-global residency, locking it per request. This is what makes the
/// peer a SHARED warm cache — a snapshot PUT by one client is GET-able by
/// another (same namespace). The lock is held only for the (fast) map op + any
/// spill/fault byte move, never across a TCP read.
pub fn run_paging_loop_shared<T: Read + Write, A: SlotArena, S: SwapStore>(
    stream: &mut T,
    res: &std::sync::Mutex<Residency<A, S>>,
) -> Result<()> {
    // Per-connection read-pin (see `handle_paging_op`): a GET hit is pinned OUT
    // of the LRU under the same lock as the dispatch, so a concurrent ALLOC on
    // another connection can't evict the slot while THIS client RDMA-reads it.
    // Released on the connection's next op (its read has drained) or disconnect.
    let mut pinned: Option<u64> = None;
    loop {
        let mut op = [0u8; 1];
        if stream.read_exact(&mut op).is_err() {
            break;
        }
        if op[0] == OP_BYE {
            break;
        }
        let mut kb = [0u8; 8];
        stream.read_exact(&mut kb).context("read paging key")?;
        let key = u64::from_le_bytes(kb);
        let reply = {
            let mut g = res.lock().expect("shared residency mutex poisoned");
            handle_paging_op(&mut g, op[0], key, &mut pinned)
        };
        write_reply(stream, &reply)?;
    }
    if let Some(pk) = pinned {
        res.lock()
            .expect("shared residency mutex poisoned")
            .unpin_read(pk);
    }
    Ok(())
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
