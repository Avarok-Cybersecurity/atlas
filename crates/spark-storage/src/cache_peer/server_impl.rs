// SPDX-License-Identifier: AGPL-3.0-only
//
// Accept loop + per-connection lifecycle for the RW blade: the v2-only
// handshake parse (paging vs RAW one-sided mode, selected by `blob_bytes`),
// the server-side rail handshake holding the crate's SINGLE `reg_mr_rw` call
// site (the access flag is a security invariant and stays AT the call site —
// census-pinned by `tests/reg_mr_flag_audit.rs`), and the two data planes:
// the shared paging control loop vs the RAW idle-until-hangup blade.

use std::net::{TcpListener, TcpStream, ToSocketAddrs};

use anyhow::{Context, Result, bail};

/// RDMA rail selection for the blade. One `(dev, gid_idx)` per CX7 adapter;
/// a client requests N rails and the peer registers its arena on each so the
/// client can stripe traffic across both adapters (~1.75x aggregate on GB10).
#[derive(Clone, Debug)]
pub struct RdmaConfig {
    /// `(device, gid_idx)` per rail, in link order (rail 0 = .178, 1 = .177).
    pub rails: Vec<(String, u32)>,
    /// Ceiling on total committed (registered) blade RAM across all
    /// concurrent connections, in bytes. `0` = unlimited (the default).
    pub max_blade_bytes: u64,
    /// Directory for NVMe swap files backing paging-mode connections
    ///. `None` = paging clients are refused (RAM-only). When set, a
    /// paging connection's RDMA arena becomes a page-cache over an O_DIRECT
    /// swap file here, bounded by `swap_cap_bytes`.
    pub swap_dir: Option<std::path::PathBuf>,
    /// Disk cap for the paging swap file, in bytes: bounds the on-disk
    /// snapshot count (coldest dropped when full → later GET misses →
    /// recompute). 0 = unbounded. Default 50 GiB (operator sanity limit).
    /// In the multi-arena registry this is the SHARED ceiling carved across
    /// kinds unless a kind has a `per_kind_swap_cap_bytes` override.
    pub swap_cap_bytes: u64,
    /// Per-`PagingKind` disk-cap overrides (`kind.0 → bytes`). When a kind
    /// is present here, its arena gets this FIXED disk budget instead of
    /// carving from the shared `swap_cap_bytes` remainder — so one kind
    /// (e.g. KV) can't starve another (e.g. SSM snapshots). 0 = unbounded
    /// for that kind. Set via `--swap-cap-gb-<kind>`.
    pub per_kind_swap_cap_bytes: std::collections::HashMap<u8, u64>,
}

impl Default for RdmaConfig {
    fn default() -> Self {
        Self {
            rails: vec![("roceP2p1s0f1".into(), 3), ("rocep1s0f1".into(), 3)],
            max_blade_bytes: 0,
            swap_dir: None,
            swap_cap_bytes: 50 * 1024 * 1024 * 1024,
            per_kind_swap_cap_bytes: std::collections::HashMap::new(),
        }
    }
}

/// Serve a KV overflow blade on `addr` until interrupted. One thread per
/// connection; each connection gets its own RW arena sized by the client.
pub fn serve<A: ToSocketAddrs>(addr: A, rdma: RdmaConfig) -> Result<()> {
    let listener = TcpListener::bind(addr).context("bind cache-peer listener")?;
    let local = listener.local_addr().ok();
    // One process-global ledger, shared by every connection thread; a
    // connection reserves its arena size before it maps/registers any RAM.
    let ledger = std::sync::Arc::new(crate::blade_cap::CommitLedger::new(rdma.max_blade_bytes));
    tracing::info!(
        "cache-peer (RW RDMA overflow blade) listening on {:?} (rails {:?}, cap {})",
        local,
        rdma.rails,
        if rdma.max_blade_bytes == 0 {
            "unlimited".to_string()
        } else {
            format!(
                "{:.1} GiB",
                rdma.max_blade_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        },
    );
    // Explicit memlock ceiling: paging arenas are anon-mmap'd AND
    // RDMA-registered (pinned = memlocked). With no `--max-blade-gb` the
    // registry can pin unbounded RAM across (kind, shape) arenas as clients
    // of new shapes connect — on a shared box that can exhaust host RAM /
    // hit the memlock rlimit. Warn so the operator sets an explicit cap.
    if rdma.max_blade_bytes == 0 && rdma.swap_dir.is_some() {
        tracing::warn!(
            "cache-peer paging registry active with NO blade ceiling (--max-blade-gb 0 = \
             unlimited): each new (kind, shape) arena pins RDMA-registered RAM without bound. \
             Set --max-blade-gb <G> to cap total memlocked blade RAM."
        );
    }
    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("cache-peer accept error: {e}");
                continue;
            }
        };
        let rdma = rdma.clone();
        let ledger = ledger.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, &rdma, &ledger) {
                tracing::warn!("cache-peer connection ended: {e}");
            }
        });
    }
    Ok(())
}

#[cfg(not(atlas_rdma_verbs))]
fn handle_conn(
    _stream: TcpStream,
    _rdma: &RdmaConfig,
    _ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
) -> Result<()> {
    bail!("cache-peer needs a build with rdma-core (atlas_rdma_verbs)");
}

#[cfg(atlas_rdma_verbs)]
fn handle_conn(
    mut stream: TcpStream,
    rdma: &RdmaConfig,
    ledger: &std::sync::Arc<crate::blade_cap::CommitLedger>,
) -> Result<()> {
    use super::registry::{self, Mmap, SharedPaging};
    use atlas_rdma::verbs::Verbs;
    use atlas_rdma::wire::{CacheServerParams, STATUS_OK, VerbsClientParams};
    use std::io::{Read, Write};
    stream.set_nodelay(true).ok();

    // 1. Client handshake (v2-only): EVERY client sends
    //    `[u64 PAGING_MAGIC_V2][u8 kind][u64 arena_bytes][u64 blob_bytes]`.
    //    blob_bytes > 0 → paging mode (peer-owned residency over the shared
    //    per-(kind, blob) registry arena). blob_bytes == 0 → RAW one-sided
    //    mode: a per-connection anonymous arena with a client-owned allocator
    //    (the legacy data plane, selected explicitly).
    //    `parse_paging_header` rejects the retired v1 magic and any bare
    //    legacy total_bytes with a legible diagnostic.
    let mut b8 = [0u8; 8];
    stream.read_exact(&mut b8).context("read paging magic")?;
    let first = u64::from_le_bytes(b8);
    let (kind, arena_bytes, blob) = crate::snapshot_swap::parse_paging_header(first, &mut stream)?;
    let total = arena_bytes as usize;
    let blob = blob as usize;
    // Explicit arena sanity bound. Pre-Step-C the `1<<42` check did double
    // duty (legacy-vs-magic dispatch AND size sanity); the dispatch role is
    // gone but the bound stays — arena size must never be limited only by
    // the (default-unlimited, warn-only) blade ledger.
    if total == 0 || total > (1usize << 42) {
        bail!("implausible blade arena size: {total}");
    }
    let paging: Option<(u8, usize)> = if blob == 0 {
        None // RAW one-sided mode
    } else {
        if !total.is_multiple_of(blob) {
            bail!("paging: arena_bytes {total} not a multiple of blob_bytes {blob}");
        }
        // Reject BEFORE the rail handshake when this peer has no swap
        // dir — the client's connect_paging then errors cleanly and
        // falls back to the bounded/host-RAM tier.
        if rdma.swap_dir.is_none() {
            bail!("paging client but peer started without --swap-dir; refusing");
        }
        Some((kind.0, blob))
    };
    let mut b1 = [0u8; 1];
    stream.read_exact(&mut b1).context("read n_rails")?;
    let n_rails = b1[0] as usize;
    if n_rails == 0 || n_rails > rdma.rails.len() {
        bail!(
            "client asked for {n_rails} rails; peer has {}",
            rdma.rails.len()
        );
    }

    // Acquire the arena to register. RAW mode: a per-connection anonymous
    // mapping, charged per-conn. Paging: the process-global SHARED
    // arena (charged ONCE at init) so every client's QPs point at the SAME
    // physical slots → a snapshot PUT by one client is GET-able by another.
    let pid = std::process::id();
    let shared: Option<std::sync::Arc<SharedPaging>> = match paging {
        Some((kind, blob)) => Some(registry::get_or_init_shared_paging(
            rdma, kind, total, blob, ledger,
        )?),
        None => None,
    };
    // Per-connection arena + blade reservation (RAW mode only), kept alive
    // until teardown; the shared arena's reservation lives in the static.
    let local: Option<(crate::blade_cap::Reservation, Mmap)> = if shared.is_none() {
        let reservation = ledger.try_reserve(total as u64).context("kv blade cap")?;
        let arena = Mmap::anon(total).context("mmap kv blade arena")?;
        Some((reservation, arena))
    } else {
        None
    };
    let (arena_base, arena_len): (*mut libc::c_void, usize) = match (&shared, &local) {
        (Some(sh), _) => (sh.arena.addr, sh.arena.len),
        (None, Some((_, arena))) => (arena.addr, arena.len),
        _ => unreachable!("exactly one of shared/local is set"),
    };
    // Register the arena ONCE per rail (each device its own PD/rkey; shared
    // refcounted pages, so N rails cost N MR handles + rkeys, not N× RAM).
    let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
    let mut rkeys: Vec<u32> = Vec::with_capacity(n_rails);
    for (i, (dev, gid)) in rdma.rails.iter().take(n_rails).enumerate() {
        let psn = (0x5a5a5a ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
        let mut v = Verbs::create(dev, *gid, psn)?;
        // SAFETY: the arena (shared or local) outlives every rail below.
        let keys = unsafe { v.reg_mr_rw(arena_base as *mut _, arena_len)? };
        rkeys.push(keys.rkey);
        rails.push(v);
    }

    // 2. Publish rail count + each rail's QP + rkey (shared base).
    stream.write_all(&[n_rails as u8]).context("send n_rails")?;
    for (v, rkey) in rails.iter().zip(&rkeys) {
        CacheServerParams {
            qpn: v.qpn(),
            psn: v.psn(),
            gid: v.gid(),
            base_addr: arena_base as u64,
            rkey: *rkey,
        }
        .write_to(&mut stream)
        .context("send kv server params")?;
    }

    // 3-4. Learn each client rail's QP, connect, ack.
    stream.read_exact(&mut b1).context("read client n_rails")?;
    if b1[0] as usize != n_rails {
        bail!("client rail count mismatch");
    }
    for v in rails.iter_mut() {
        let cp = VerbsClientParams::read_from(&mut stream).context("read kv client params")?;
        v.connect(cp.qpn, cp.psn, &cp.gid)?;
    }
    stream
        .write_all(&[STATUS_OK])
        .context("send kv ready ack")?;
    let mode = if paging.is_some() {
        "paging"
    } else {
        "raw one-sided"
    };
    tracing::info!(
        "cache-peer client connected: kind {}, {n_rails} rail(s), {:.1} GiB RW blade ({mode})",
        kind.0,
        total as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    // 5. Data plane.
    if let Some(sh) = shared {
        // Paging mode: drive the SHARED residency — a snapshot PUT by
        // one client is GET-able by another (cross-connection warm cache).
        // Bytes move one-sided over RDMA into/out of the shared arena slots;
        // only tiny [op][key] control messages cross this TCP stream. The MR
        // is never re-registered — swap happens under the stable rkey.
        tracing::info!("cache-peer PAGING client joined shared arena ({n_rails} rail(s))");
        let r = crate::snapshot_swap::run_paging_loop_shared(&mut stream, &sh.residency);
        drop(rails); // dereg this conn's MRs; the shared arena stays mapped
        return r;
    }

    // RAW one-sided blade (v2, blob_bytes == 0): the client owns allocation
    // against the fixed arena; the peer just idles until hangup.
    let mut sink = [0u8; 8];
    loop {
        match stream.read(&mut sink) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
    // Dereg (rails) before unmap (arena): drop rails first.
    drop(rails);
    drop(local);
    Ok(())
}
