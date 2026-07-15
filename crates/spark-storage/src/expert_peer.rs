// SPDX-License-Identifier: AGPL-3.0-only
//
// Expert weight-serving peer + wire protocol (Stage 4, Phase A: TCP).
//
// The RDMA weight-tier's first incarnation: a peer that holds the resident
// expert store and serves records to a streaming client over a socket, into the
// client's pinned arena. This proves the residency-tier abstraction (a peer as
// a fetch tier, distinct from EP sharding) with zero verbs risk, over the RoCE
// Ethernet netdev. Phase B swaps the transport for one-sided RDMA_READ into the
// SAME arena (see RESEARCH-RDMA-TIER.md) — the protocol geometry is identical.
//
// Wire protocol (little-endian), connection-oriented:
//   1. On accept the server sends the manifest: [u32 len][len bytes of JSON].
//      The client parses it to size its arena (same ExpertIndex geometry).
//   2. Request loop: client sends [u32 layer][u32 expert]; server replies
//      [u8 status][record_stride bytes] (status 0 = OK, nonzero = error, no
//      payload). A layer/expert of u32::MAX/u32::MAX is a graceful shutdown.
//
// The peer is pure I/O (no CUDA); the client half lives in the cuda-gated
// `expert_tier_rdma` module because it lands bytes in the pinned arena.

use anyhow::{Context, Result, bail};

// The handshake wire codecs moved verbatim to the CUDA-free `atlas-rdma`
// crate (extracted to atlas-rdma); re-exported here at their old paths so
// the server below and every external user are zero-diff. The byte layouts
// are golden-pinned in `tests/rdma_wire_golden.rs` and frozen vs the live
// gx10 peer.
pub use atlas_rdma::wire::{
    MODE_TCP, MODE_VERBS, STATUS_ERR, STATUS_OK, VerbsClientParams, VerbsServerParams,
    read_server_rails, write_server_rails,
};

/// Sentinel request that asks the server to close the connection.
pub const SHUTDOWN_MARKER: u32 = u32::MAX;

/// Serialize a request: `(layer, expert)`.
pub fn encode_request(layer: u32, expert: u32) -> [u8; 8] {
    let mut b = [0u8; 8];
    b[0..4].copy_from_slice(&layer.to_le_bytes());
    b[4..8].copy_from_slice(&expert.to_le_bytes());
    b
}

/// Parse a request buffer.
pub fn decode_request(b: &[u8; 8]) -> (u32, u32) {
    let layer = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let expert = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    (layer, expert)
}

#[cfg(unix)]
pub use server_impl::{RdmaConfig, serve};

#[cfg(unix)]
mod server_impl {
    use super::*;
    use crate::expert::ExpertKey;
    use crate::expert_pack::ExpertFileReader;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    /// RDMA device selection for the verbs (`MODE_VERBS`) transport. Ignored for
    /// TCP clients. One `(device, gid_idx)` per CX7 adapter: a dual-rail client
    /// requests N rails and the peer registers each layer mmap on every rail so
    /// the client can stripe expert fetches across both adapters. The default is
    /// a SINGLE rail (`roceP2p1s0f1`, RoCEv2 GID index 3) — the pre-dual-rail
    /// behavior, unchanged for `verify_verbs.sh`; add a second `--rail` to serve
    /// dual-rail clients.
    #[derive(Clone, Debug)]
    pub struct RdmaConfig {
        /// `(device, gid_idx)` per rail, in link order (rail 0 = the cabled link).
        pub rails: Vec<(String, u32)>,
        /// Ceiling on total registered store RAM across concurrent verbs
        /// connections, in bytes. `0` = unlimited (the default). Each verbs
        /// connection registers the whole store (`index.total_bytes()`) once (the
        /// N per-rail MRs share the same mmap pages), so this bounds the number of
        /// concurrent store registrations.
        pub max_blade_bytes: u64,
    }

    impl Default for RdmaConfig {
        fn default() -> Self {
            Self {
                rails: vec![("roceP2p1s0f1".into(), 3)],
                max_blade_bytes: 0,
            }
        }
    }

    /// Serve records from `dir` on `addr` until interrupted. One thread per
    /// connection. Blocking; intended to run as its own process (`atlas-expert-peer`).
    pub fn serve<A: ToSocketAddrs>(dir: &Path, addr: A, rdma: RdmaConfig) -> Result<()> {
        let reader = Arc::new(ExpertFileReader::open(dir)?);
        let manifest = serde_json::to_vec(reader.index())?;
        let dir: Arc<PathBuf> = Arc::new(dir.to_path_buf());
        let rdma = Arc::new(rdma);
        // One process-global ledger; each verbs connection reserves the store
        // size before it mmaps/registers the layer files.
        let ledger = Arc::new(crate::blade_cap::CommitLedger::new(rdma.max_blade_bytes));
        let listener = TcpListener::bind(addr).context("bind expert-peer listener")?;
        let local = listener.local_addr().ok();
        tracing::info!(
            "expert-peer serving {} ({} layers, {} experts, stride {}) on {:?} \
             (verbs rails {:?}, cap {})",
            dir.display(),
            reader.index().num_moe_layers,
            reader.index().num_experts,
            reader.index().record_stride,
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
        for conn in listener.incoming() {
            let stream = match conn {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("expert-peer accept error: {e}");
                    continue;
                }
            };
            let reader = reader.clone();
            let manifest = manifest.clone();
            let dir = dir.clone();
            let rdma = rdma.clone();
            let ledger = ledger.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_conn(stream, &reader, &manifest, &dir, &rdma, &ledger) {
                    tracing::warn!("expert-peer connection ended: {e}");
                }
            });
        }
        Ok(())
    }

    fn handle_conn(
        mut stream: TcpStream,
        reader: &ExpertFileReader,
        manifest: &[u8],
        dir: &Path,
        rdma: &RdmaConfig,
        ledger: &Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        stream.set_nodelay(true).ok();
        // 1. Send the manifest.
        stream.write_all(&(manifest.len() as u32).to_le_bytes())?;
        stream.write_all(manifest)?;

        // 2. The client picks a transport. TCP `pread`s on demand and pins no
        // RAM, so only the verbs path (which mmaps + registers the store) is
        // charged against the ledger.
        let mut mode = [0u8; 1];
        stream
            .read_exact(&mut mode)
            .context("read transport mode")?;
        match mode[0] {
            MODE_TCP => serve_tcp(stream, reader),
            MODE_VERBS => serve_verbs(stream, reader, dir, rdma, ledger),
            other => bail!("client requested unknown transport mode {other}"),
        }
    }

    /// Two-sided record streaming (Phase A). The request loop is unchanged.
    fn serve_tcp(mut stream: TcpStream, reader: &ExpertFileReader) -> Result<()> {
        let stride = reader.index().record_stride as usize;
        let mut req = [0u8; 8];
        loop {
            if stream.read_exact(&mut req).is_err() {
                break; // client hung up
            }
            let (layer, expert) = decode_request(&req);
            if layer == SHUTDOWN_MARKER && expert == SHUTDOWN_MARKER {
                break;
            }
            match reader.read_record_raw(ExpertKey::new(layer, expert)) {
                Ok(rec) => {
                    debug_assert_eq!(rec.len(), stride);
                    stream.write_all(&[STATUS_OK])?;
                    stream.write_all(&rec)?;
                }
                Err(e) => {
                    tracing::warn!("expert-peer read {layer}/{expert}: {e}");
                    stream.write_all(&[STATUS_ERR])?;
                }
            }
        }
        Ok(())
    }

    #[cfg(not(atlas_rdma_verbs))]
    fn serve_verbs(
        _stream: TcpStream,
        _reader: &ExpertFileReader,
        _dir: &Path,
        _rdma: &RdmaConfig,
        _ledger: &Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        bail!("client requested verbs transport but this peer was built without rdma-core");
    }

    /// One-sided RDMA READ (Phase B). The server `mmap`s + registers each layer
    /// file with REMOTE_READ, publishes the MRs' `{base, rkey}` + its QP params,
    /// connects to the client's QP, then goes idle: the client pulls records
    /// directly out of these MRs. The server CPU never touches a record byte.
    #[cfg(atlas_rdma_verbs)]
    fn serve_verbs(
        mut stream: TcpStream,
        reader: &ExpertFileReader,
        dir: &Path,
        rdma: &RdmaConfig,
        ledger: &Arc<crate::blade_cap::CommitLedger>,
    ) -> Result<()> {
        use atlas_rdma::verbs::Verbs;

        let index = reader.index();
        let num_layers = index.num_moe_layers;

        // The client requests how many rails it wants to stripe across (default
        // 1). Validate against what this peer is configured to serve.
        let mut b1 = [0u8; 1];
        stream.read_exact(&mut b1).context("read n_rails")?;
        let n_rails = b1[0] as usize;
        if n_rails == 0 || n_rails > rdma.rails.len() {
            bail!(
                "client asked for {n_rails} rails; peer has {}",
                rdma.rails.len()
            );
        }

        // Admission gate: charge the deterministic store size (identical for
        // every client) ONCE — the N per-rail MRs pin the SAME refcounted mmap
        // pages, so the committed footprint is `total_bytes`, not total*n_rails.
        // Reserve BEFORE any mmap/reg_mr; the RAII guard releases on every exit
        // below (early bail, reg_mr error, hangup).
        let _reservation = ledger
            .try_reserve(index.total_bytes())
            .context("expert blade cap")?;

        // One QP per rail (distinct per-rail PSN so successive clients/rails
        // don't collide).
        let pid = std::process::id();
        let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
        for (i, (dev, gid)) in rdma.rails.iter().take(n_rails).enumerate() {
            let psn = (0x424242 ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
            rails.push(Verbs::create(dev, *gid, psn)?);
        }

        // mmap each layer file ONCE (REMOTE_READ) and register that SAME virtual
        // range on EVERY rail's PD — one rkey per (rail, layer), identical base
        // VA, shared physical pages (not N× RAM). Keep the mappings alive for the
        // whole connection — the NIC DMAs out of them.
        let mut mmaps: Vec<Mmap> = Vec::with_capacity(num_layers as usize);
        let mut per_rail_layers: Vec<Vec<(u64, u32)>> = (0..n_rails)
            .map(|_| Vec::with_capacity(num_layers as usize))
            .collect();
        for l in 0..num_layers {
            let path = dir.join(index.file_name(l));
            let m = Mmap::open_ro(&path).with_context(|| format!("mmap {}", path.display()))?;
            for (ri, v) in rails.iter_mut().enumerate() {
                // SAFETY: the mapping covers `m.len` bytes at `m.addr` and
                // outlives every rail's Verbs (mmaps dropped after rails below).
                let keys = unsafe { v.reg_mr(m.addr as *mut _, m.len, true)? };
                per_rail_layers[ri].push((m.addr as u64, keys.rkey));
            }
            mmaps.push(m);
        }

        // Publish one VerbsServerParams per rail (shared base, per-rail rkey).
        let sp: Vec<VerbsServerParams> = rails
            .iter()
            .enumerate()
            .map(|(ri, v)| VerbsServerParams {
                qpn: v.qpn(),
                psn: v.psn(),
                gid: v.gid(),
                layers: std::mem::take(&mut per_rail_layers[ri]),
            })
            .collect();
        write_server_rails(&mut stream, &sp).context("send verbs server params")?;

        // Learn each client rail's QP, connect, ack.
        stream.read_exact(&mut b1).context("read client n_rails")?;
        if b1[0] as usize != n_rails {
            bail!("client rail count mismatch");
        }
        for v in rails.iter_mut() {
            let cp =
                VerbsClientParams::read_from(&mut stream).context("read verbs client params")?;
            v.connect(cp.qpn, cp.psn, &cp.gid)?;
        }
        stream
            .write_all(&[STATUS_OK])
            .context("send verbs ready ack")?;
        tracing::info!(
            "expert-peer verbs client connected ({n_rails} rail(s), {} layer MRs/rail)",
            num_layers,
        );

        // Idle until the client hangs up. All record movement is one-sided RDMA
        // READ initiated by the client; the server just holds the MRs open.
        let mut sink = [0u8; 8];
        loop {
            match stream.read(&mut sink) {
                Ok(0) => break, // client closed
                Ok(_) => {}     // ignore (shutdown marker or stray bytes)
                Err(_) => break,
            }
        }
        // Drop order: `rails` (which dereg's every MR) must fall before `mmaps`
        // are unmapped, so dereg happens over live mappings. Drop rails first.
        drop(rails);
        drop(mmaps);
        Ok(())
    }

    /// A read-only `mmap` of a whole file, unmapped on drop.
    #[cfg(atlas_rdma_verbs)]
    struct Mmap {
        addr: *mut libc::c_void,
        len: usize,
    }

    #[cfg(atlas_rdma_verbs)]
    impl Mmap {
        fn open_ro(path: &Path) -> Result<Self> {
            use std::os::fd::AsRawFd;
            let f = std::fs::File::open(path)?;
            let len = f.metadata()?.len() as usize;
            if len == 0 {
                bail!("empty layer file {}", path.display());
            }
            // SAFETY: fd is a valid open RO file; MAP_SHARED read mapping of `len`
            // bytes. The kernel keeps the mapping valid after the fd closes.
            let addr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    f.as_raw_fd(),
                    0,
                )
            };
            if addr == libc::MAP_FAILED {
                bail!(
                    "mmap {} failed: {}",
                    path.display(),
                    std::io::Error::last_os_error()
                );
            }
            Ok(Self { addr, len })
        }
    }

    #[cfg(atlas_rdma_verbs)]
    impl Drop for Mmap {
        fn drop(&mut self) {
            // SAFETY: addr/len came from a successful mmap and are unmapped once.
            unsafe { libc::munmap(self.addr, self.len) };
        }
    }
}

/// Read the length-prefixed manifest from a freshly-connected stream and parse
/// it. Shared by the client (`expert_tier_rdma`).
#[cfg(unix)]
pub fn read_manifest<R: std::io::Read>(stream: &mut R) -> Result<crate::expert_pack::ExpertIndex> {
    let mut lenb = [0u8; 4];
    stream
        .read_exact(&mut lenb)
        .context("read manifest length")?;
    let len = u32::from_le_bytes(lenb) as usize;
    if len == 0 || len > 16 * 1024 * 1024 {
        bail!("implausible peer manifest length: {len}");
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).context("read manifest json")?;
    let index: crate::expert_pack::ExpertIndex =
        serde_json::from_slice(&buf).context("parse peer manifest")?;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_round_trips() {
        let b = encode_request(7, 42);
        assert_eq!(decode_request(&b), (7, 42));
        let s = encode_request(SHUTDOWN_MARKER, SHUTDOWN_MARKER);
        assert_eq!(decode_request(&s), (SHUTDOWN_MARKER, SHUTDOWN_MARKER));
    }

    // The codec round-trip / validation tests moved WITH the codecs to
    // `crates/atlas-rdma/tests/wire_roundtrip.rs` (extracted to atlas-rdma);
    // the exact byte layouts stay pinned by `tests/rdma_wire_golden.rs` here.
}
