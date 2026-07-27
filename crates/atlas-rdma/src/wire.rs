// SPDX-License-Identifier: AGPL-3.0-only
//
// The RDMA handshake wire codecs, shared by every Atlas RDMA client and the
// peer daemons (the daemons re-export these, so client and server speak one
// codec).
//
// ** GOLDEN WIRE FORMAT ** — every byte layout here is a frozen external
// contract (already-deployed peers depend on it). All integers little-endian. The byte
// vectors are pinned by `tests/wire_roundtrip.rs` and `tests/transcript_golden.rs`,
// and the frozen constant values by `frozen_wire_constants` (wire_roundtrip.rs).
//
// Un-gated on purpose: pure `std::io` + anyhow, so it compiles and unit-tests
// on the metal/ATLAS_SKIP_BUILD build with no rdma-core.

use anyhow::{Context, Result, bail};

// ── Shared status / transport-mode bytes ──
pub const STATUS_OK: u8 = 0;
pub const STATUS_ERR: u8 = 1;
/// Two-sided TCP record streaming (the expert record path).
pub const MODE_TCP: u8 = 0;
/// One-sided RDMA READ over verbs: the server publishes its store's MRs and
/// the client READs records directly into its arena.
pub const MODE_VERBS: u8 = 1;

/// The server's half of the verbs handshake (RO dialect: expert / weight /
/// LoRA tiers): its QP identity plus, per MoE layer (or per shard), the base
/// virtual address + rkey of that entry's registered MR.
/// `remote_addr(layer, expert) = layers[layer].0 + expert * record_stride`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerbsServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    /// `(mr_base_addr, rkey)` for each MoE layer (expert tier) or shard
    /// (weight/LoRA tiers), index-addressed.
    pub layers: Vec<(u64, u32)>,
}

/// The client's half: just its QP identity (its arena MR is local-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerbsClientParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
}

/// The peer's half of the RW-blade handshake (KV overflow / SSM snapshots):
/// its QP identity + the single RW MR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheServerParams {
    pub qpn: u32,
    pub psn: u32,
    pub gid: [u8; 16],
    pub base_addr: u64,
    pub rkey: u32,
}

/// Anything that carries a remote QP identity a client rail can `connect` to.
/// Implemented by both server-param dialects so the RailSet handshake tail is
/// shared without homogenizing the two wire layouts.
pub trait RemoteQp {
    fn qp_identity(&self) -> (u32, u32, [u8; 16]);
}

impl RemoteQp for VerbsServerParams {
    fn qp_identity(&self) -> (u32, u32, [u8; 16]) {
        (self.qpn, self.psn, self.gid)
    }
}

impl RemoteQp for CacheServerParams {
    fn qp_identity(&self) -> (u32, u32, [u8; 16]) {
        (self.qpn, self.psn, self.gid)
    }
}

impl VerbsServerParams {
    /// Wire form: `[u32 qpn][u32 psn][16 gid][u32 n_layers]{[u64 base][u32 rkey]}*`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        w.write_all(&(self.layers.len() as u32).to_le_bytes())?;
        for (base, rkey) in &self.layers {
            w.write_all(&base.to_le_bytes())?;
            w.write_all(&rkey.to_le_bytes())?;
        }
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let qpn = read_u32(r)?;
        let psn = read_u32(r)?;
        let mut gid = [0u8; 16];
        r.read_exact(&mut gid).context("read server gid")?;
        let n = read_u32(r)? as usize;
        if n == 0 || n > 4096 {
            bail!("implausible verbs layer count: {n}");
        }
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let mut b8 = [0u8; 8];
            r.read_exact(&mut b8).context("read mr base")?;
            let base = u64::from_le_bytes(b8);
            let rkey = read_u32(r)?;
            layers.push((base, rkey));
        }
        Ok(Self {
            qpn,
            psn,
            gid,
            layers,
        })
    }
}

impl VerbsClientParams {
    /// Wire form: `[u32 qpn][u32 psn][16 gid]`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let qpn = read_u32(r)?;
        let psn = read_u32(r)?;
        let mut gid = [0u8; 16];
        r.read_exact(&mut gid).context("read client gid")?;
        Ok(Self { qpn, psn, gid })
    }
}

impl CacheServerParams {
    /// Wire form: `[u32 qpn][u32 psn][16 gid][u64 base_addr][u32 rkey]`.
    pub fn write_to<W: std::io::Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.qpn.to_le_bytes())?;
        w.write_all(&self.psn.to_le_bytes())?;
        w.write_all(&self.gid)?;
        w.write_all(&self.base_addr.to_le_bytes())?;
        w.write_all(&self.rkey.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from<R: std::io::Read>(r: &mut R) -> Result<Self> {
        let mut b4 = [0u8; 4];
        let mut b8 = [0u8; 8];
        let mut gid = [0u8; 16];
        r.read_exact(&mut b4).context("kv qpn")?;
        let qpn = u32::from_le_bytes(b4);
        r.read_exact(&mut b4).context("kv psn")?;
        let psn = u32::from_le_bytes(b4);
        r.read_exact(&mut gid).context("kv gid")?;
        r.read_exact(&mut b8).context("kv base")?;
        let base_addr = u64::from_le_bytes(b8);
        r.read_exact(&mut b4).context("kv rkey")?;
        let rkey = u32::from_le_bytes(b4);
        Ok(Self {
            qpn,
            psn,
            gid,
            base_addr,
            rkey,
        })
    }
}

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read u32")?;
    Ok(u32::from_le_bytes(b))
}

/// Frame N per-rail `VerbsServerParams` for the dual-rail RO tiers: a leading
/// `[u8 n_rails]` count followed by each rail's params — exactly how the RW
/// blade frames its per-rail `CacheServerParams`. Single-rail (`n == 1`) is the
/// default, byte-for-byte the pre-dual-rail path plus the one-byte count prefix.
pub fn write_server_rails<W: std::io::Write>(w: &mut W, rails: &[VerbsServerParams]) -> Result<()> {
    if rails.is_empty() || rails.len() > 8 {
        bail!("implausible server rail count: {}", rails.len());
    }
    w.write_all(&[rails.len() as u8])?;
    for sp in rails {
        sp.write_to(w)?;
    }
    Ok(())
}

/// Read `want` per-rail `VerbsServerParams` framed by a leading `[u8 n_rails]`.
/// Bails if the framed count is zero, absurd (> 8), or != `want` — the client
/// already negotiated `want` rails, so any other count is a protocol error.
pub fn read_server_rails<R: std::io::Read>(
    r: &mut R,
    want: usize,
) -> Result<Vec<VerbsServerParams>> {
    let mut b1 = [0u8; 1];
    r.read_exact(&mut b1).context("read server rail count")?;
    let n = b1[0] as usize;
    if n == 0 || n > 8 {
        bail!("implausible server rail count: {n}");
    }
    if n != want {
        bail!("server framed {n} rails but client negotiated {want}");
    }
    let mut rails = Vec::with_capacity(n);
    for _ in 0..n {
        rails.push(VerbsServerParams::read_from(r)?);
    }
    Ok(rails)
}
