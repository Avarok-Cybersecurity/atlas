// SPDX-License-Identifier: AGPL-3.0-only
//
// The `atlas-weight-peer` daemon (unix, server-side). Accepts a model request,
// stages the model (warm mmaps + parsed manifest via `shard`), publishes the
// manifest (`wire`), then serves the shards one-sided REMOTE_READ over verbs.
// This module holds the sole `reg_mr(.., true)` — the REMOTE_READ registration
// pinned by `tests/reg_mr_flag_audit.rs`.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::io::Read;
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::manifest::WeightManifest;
use super::shard::{Mmap, build_manifest};
use super::wire::{read_model_request, write_weight_manifest};
use crate::expert_peer::MODE_VERBS;

/// Peer configuration: the RDMA rails, the memory ceiling, and how the peer
/// resolves a model request to a directory.
#[derive(Clone, Debug)]
pub struct WeightPeerConfig {
    /// `(device, gid_idx)` per rail, in link order (rail 0 = the cabled
    /// link). Mirrors `expert_peer::RdmaConfig::rails`.
    pub rails: Vec<(String, u32)>,
    /// Ceiling on total registered (staged) RAM in bytes across ALL staged
    /// models. `0` = unlimited. Each model is charged its shard bytes once,
    /// at first stage (the per-connection MRs share the same warm pages).
    pub max_blade_bytes: u64,
    /// Model directories to pre-stage at startup. Also the allow-list when
    /// `allow_any_path` is false: a client may only request a model whose
    /// resolved path matches one of these (or its basename).
    pub staged_dirs: Vec<PathBuf>,
    /// When true, a client may request ANY filesystem path and the peer
    /// stages it on demand (convenient for a trusted LAN; off by default).
    pub allow_any_path: bool,
}

impl Default for WeightPeerConfig {
    fn default() -> Self {
        Self {
            rails: vec![("roceP2p1s0f1".into(), 3)],
            max_blade_bytes: 0,
            staged_dirs: Vec::new(),
            allow_any_path: false,
        }
    }
}

/// A staged model held resident: the persistent shard mmaps (kept mapped so
/// their pages stay warm in RAM across connections), the manifest, and the
/// ledger reservation released when the model is dropped. Per-connection
/// `reg_mr` re-registers these same base VAs on each client QP's PD.
struct StagedModel {
    // Read only by the verbs serve path (reg_mr each shard); on a build
    // without rdma-core the mmaps still hold pages warm but aren't iterated.
    #[cfg_attr(not(atlas_rdma_verbs), allow(dead_code))]
    shard_mmaps: Vec<Mmap>,
    manifest: WeightManifest,
    _reservation: crate::blade_cap::Reservation,
}

type StagedMap = Arc<Mutex<HashMap<String, Arc<StagedModel>>>>;

/// Serve staged models on `addr` until interrupted. One thread per
/// connection; blocking. Intended to run as its own process
/// (`atlas-weight-peer`).
pub fn serve<A: ToSocketAddrs>(addr: A, cfg: WeightPeerConfig) -> Result<()> {
    let cfg = Arc::new(cfg);
    let ledger = Arc::new(crate::blade_cap::CommitLedger::new(cfg.max_blade_bytes));
    let staged: StagedMap = Arc::new(Mutex::new(HashMap::new()));

    // Pre-stage the configured directories (first stage is the slow one —
    // do it up front so the first client swap is already warm).
    for dir in &cfg.staged_dirs {
        match stage_model(&staged, &ledger, dir) {
            Ok(m) => tracing::info!(
                "weight-peer pre-staged {} ({} shards, {} tensors, {:.1} GiB)",
                m.manifest.model_id,
                m.manifest.num_shards(),
                m.manifest.tensors.len(),
                m.manifest.total_shard_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
            ),
            Err(e) => tracing::warn!("weight-peer pre-stage {} failed: {e}", dir.display()),
        }
    }

    let listener = TcpListener::bind(addr).context("bind weight-peer listener")?;
    let local = listener.local_addr().ok();
    tracing::info!(
        "weight-peer serving on {:?} (verbs rails {:?}, cap {}, allow_any_path {})",
        local,
        cfg.rails,
        if cfg.max_blade_bytes == 0 {
            "unlimited".to_string()
        } else {
            format!(
                "{:.1} GiB",
                cfg.max_blade_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        },
        cfg.allow_any_path,
    );

    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("weight-peer accept error: {e}");
                continue;
            }
        };
        let cfg = cfg.clone();
        let ledger = ledger.clone();
        let staged = staged.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, &cfg, &ledger, &staged) {
                tracing::warn!("weight-peer connection ended: {e}");
            }
        });
    }
    Ok(())
}

fn handle_conn(
    mut stream: TcpStream,
    cfg: &WeightPeerConfig,
    ledger: &Arc<crate::blade_cap::CommitLedger>,
    staged: &StagedMap,
) -> Result<()> {
    stream.set_nodelay(true).ok();

    // 1. Client tells us which model it wants.
    let request = read_model_request(&mut stream)?;
    let dir = resolve_request(cfg, &request)?;

    // 2. Stage it (or reuse the warm one) and publish the manifest.
    let model = stage_model(staged, ledger, &dir)?;
    write_weight_manifest(&mut stream, &model.manifest).context("send manifest")?;

    // 3. Transport selection. Only verbs is served for weights.
    let mut mode = [0u8; 1];
    stream
        .read_exact(&mut mode)
        .context("read transport mode")?;
    match mode[0] {
        MODE_VERBS => serve_verbs(stream, &model, cfg),
        other => bail!("weight-peer only serves verbs; client asked for mode {other}"),
    }
}

/// Resolve a client's model request string to a directory, honoring the
/// allow-list / `allow_any_path` policy.
fn resolve_request(cfg: &WeightPeerConfig, request: &str) -> Result<PathBuf> {
    let req = Path::new(request);
    // Exact path match against a staged dir, or basename match.
    for d in &cfg.staged_dirs {
        if d == req
            || d.file_name().and_then(|n| n.to_str()) == Some(request)
            || d.to_string_lossy() == request
        {
            return Ok(d.clone());
        }
    }
    if cfg.allow_any_path && req.is_dir() {
        return Ok(req.to_path_buf());
    }
    bail!(
        "model '{request}' is not staged (and allow_any_path is off); \
         pass it to the peer with --stage <dir>"
    );
}

/// Look up a warm staged model or stage it now (mmap shards + parse headers
/// + charge the ledger). Idempotent per resolved-path key.
fn stage_model(
    staged: &StagedMap,
    ledger: &Arc<crate::blade_cap::CommitLedger>,
    dir: &Path,
) -> Result<Arc<StagedModel>> {
    let key = dir.to_string_lossy().into_owned();
    {
        let map = staged.lock().unwrap();
        if let Some(m) = map.get(&key) {
            return Ok(m.clone());
        }
    }

    // Build the manifest (resolve shards + parse each shard's header) and
    // mmap every shard REMOTE-read + warm.
    let (shard_paths, manifest) = build_manifest(dir, &key)?;
    // Charge the ledger BEFORE pinning any pages; the RAII guard lives in
    // the StagedModel and releases if we bail below or when it's dropped.
    let reservation = ledger
        .try_reserve(manifest.total_shard_bytes())
        .context("weight blade cap")?;

    let mut shard_mmaps = Vec::with_capacity(shard_paths.len());
    for p in &shard_paths {
        shard_mmaps.push(Mmap::open_ro(p).with_context(|| format!("mmap {}", p.display()))?);
    }

    let model = Arc::new(StagedModel {
        shard_mmaps,
        manifest,
        _reservation: reservation,
    });
    let mut map = staged.lock().unwrap();
    // Another thread may have staged it while we worked; prefer the existing
    // one (drops ours, releasing its reservation).
    Ok(map.entry(key).or_insert(model).clone())
}

/// One-sided RDMA READ weight serving. Registers each shard mmap REMOTE_READ
/// on every rail, publishes the per-shard `(base, rkey)`, connects to the
/// client's QPs, then idles — the client pulls all tensor bytes one-sided.
#[cfg(not(atlas_rdma_verbs))]
fn serve_verbs(
    _stream: TcpStream,
    _model: &Arc<StagedModel>,
    _cfg: &WeightPeerConfig,
) -> Result<()> {
    bail!("client requested verbs transport but this peer was built without rdma-core");
}

#[cfg(atlas_rdma_verbs)]
fn serve_verbs(
    mut stream: TcpStream,
    model: &Arc<StagedModel>,
    cfg: &WeightPeerConfig,
) -> Result<()> {
    use crate::expert_peer::{STATUS_OK, VerbsClientParams, VerbsServerParams, write_server_rails};
    use atlas_rdma::verbs::Verbs;
    use std::io::Write;

    let num_shards = model.shard_mmaps.len();

    // Negotiate the rail count.
    let mut b1 = [0u8; 1];
    stream.read_exact(&mut b1).context("read n_rails")?;
    let n_rails = b1[0] as usize;
    if n_rails == 0 || n_rails > cfg.rails.len() {
        bail!(
            "client asked for {n_rails} rails; peer has {}",
            cfg.rails.len()
        );
    }

    // One QP per rail (distinct per-rail PSN so successive clients don't
    // collide). No ledger charge here — staging already charged the pages;
    // the N per-rail MRs share those same refcounted mmap pages.
    let pid = std::process::id();
    let mut rails: Vec<Verbs> = Vec::with_capacity(n_rails);
    for (i, (dev, gid)) in cfg.rails.iter().take(n_rails).enumerate() {
        let psn = (0x77_7777 ^ pid ^ ((i as u32) << 20)) & 0xff_ffff;
        rails.push(Verbs::create(dev, *gid, psn)?);
    }

    // Register each shard mmap (REMOTE_READ) on EVERY rail's PD — one rkey
    // per (rail, shard), identical base VA, shared physical pages. The mmaps
    // live in the persistent StagedModel, so they outlive these MRs.
    let mut per_rail_shards: Vec<Vec<(u64, u32)>> = (0..n_rails)
        .map(|_| Vec::with_capacity(num_shards))
        .collect();
    for m in &model.shard_mmaps {
        for (ri, v) in rails.iter_mut().enumerate() {
            // SAFETY: the mapping covers `m.len` bytes at `m.addr` and lives
            // in the StagedModel Arc, which outlives every rail here.
            let keys = unsafe { v.reg_mr(m.addr as *mut _, m.len, true)? };
            per_rail_shards[ri].push((m.addr as u64, keys.rkey));
        }
    }

    // Publish one VerbsServerParams per rail; `layers` carries per-SHARD
    // (base, rkey) in shard order (shards play experts' per-layer role).
    let sp: Vec<VerbsServerParams> = rails
        .iter()
        .enumerate()
        .map(|(ri, v)| VerbsServerParams {
            qpn: v.qpn(),
            psn: v.psn(),
            gid: v.gid(),
            layers: std::mem::take(&mut per_rail_shards[ri]),
        })
        .collect();
    write_server_rails(&mut stream, &sp).context("send verbs server params")?;

    // Learn each client rail's QP, connect, ack.
    stream.read_exact(&mut b1).context("read client n_rails")?;
    if b1[0] as usize != n_rails {
        bail!("client rail count mismatch");
    }
    for v in rails.iter_mut() {
        let cp = VerbsClientParams::read_from(&mut stream).context("read verbs client params")?;
        v.connect(cp.qpn, cp.psn, &cp.gid)?;
    }
    stream
        .write_all(&[STATUS_OK])
        .context("send verbs ready ack")?;
    tracing::info!(
        "weight-peer verbs client connected to {} ({n_rails} rail(s), {num_shards} shard MRs/rail)",
        model.manifest.model_id,
    );

    // Idle until the client hangs up. All movement is one-sided RDMA READ.
    let mut sink = [0u8; 8];
    loop {
        match stream.read(&mut sink) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
    // Drop rails (dereg MRs) BEFORE the StagedModel Arc frees anything — the
    // mmaps persist in the map regardless, but dropping rails first keeps
    // dereg strictly over live mappings.
    drop(rails);
    Ok(())
}
