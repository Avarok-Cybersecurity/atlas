// SPDX-License-Identifier: AGPL-3.0-only

//! Expert storage backend for K3 routed MXFP4 packs.
//!
//! Official `moonshotai/Kimi-K3` is ~1.56 TB / 96 shards. Two Sparks cannot
//! hold that resident. Trunk + embed + lm_head + KDA/MLA + shared experts
//! stay in RAM; routed experts come from per-id NVMe packs.
//!
//! `K3_EXPERT_BACKEND=resident | mmap | prefetch` (default resident).
//! Do not download the 96-shard repo in this module.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

pub const ENV: &str = "K3_EXPERT_BACKEND";
const MAGIC: &[u8; 4] = b"K3E1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertBackendKind {
    /// Twin / dummy experts already in RAM.
    Resident,
    /// Per-expert MXFP4 packs on NVMe (`e{id}.mxfp4`).
    Mmap,
    /// mmap + next-token expert ids from the last router call.
    Prefetch,
}

pub fn kind_from_env() -> ExpertBackendKind {
    match std::env::var(ENV).as_deref() {
        Ok("mmap") => ExpertBackendKind::Mmap,
        Ok("prefetch") => ExpertBackendKind::Prefetch,
        _ => ExpertBackendKind::Resident,
    }
}

/// Dummy pack path: `{dir}/e{id}.mxfp4`.
pub fn pack_path(dir: &Path, expert_id: usize) -> PathBuf {
    dir.join(format!("e{expert_id}.mxfp4"))
}

pub fn write_dummy_pack(dir: &Path, expert_id: usize, payload: &[u8]) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let path = pack_path(dir, expert_id);
    let mut f = File::create(&path)?;
    f.write_all(MAGIC)?;
    f.write_all(&(expert_id as u32).to_le_bytes())?;
    f.write_all(&(payload.len() as u32).to_le_bytes())?;
    f.write_all(payload)?;
    Ok(path)
}

fn parse_pack(bytes: &[u8]) -> Result<(u32, &[u8])> {
    if bytes.len() < 12 || &bytes[..4] != MAGIC {
        bail!("K3 expert pack: bad magic");
    }
    let id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if bytes.len() != 12 + n {
        bail!("K3 expert pack: length mismatch");
    }
    Ok((id, &bytes[12..]))
}

/// mmap'd per-expert packs. Missing id fails closed (no silent zero tensor).
pub struct MmapExpertStore {
    maps: HashMap<usize, Mmap>,
}

impl MmapExpertStore {
    pub fn open(dir: &Path) -> Result<Self> {
        let mut maps = HashMap::new();
        if !dir.is_dir() {
            bail!("K3 expert mmap dir {} missing", dir.display());
        }
        for ent in std::fs::read_dir(dir)? {
            let ent = ent?;
            let name = ent.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with('e') || !name.ends_with(".mxfp4") {
                continue;
            }
            let f = File::open(ent.path())?;
            // SAFETY: packs are immutable after write_dummy_pack / rental stage.
            let mmap =
                unsafe { Mmap::map(&f) }.with_context(|| ent.path().display().to_string())?;
            let (id, _) = parse_pack(&mmap)?;
            maps.insert(id as usize, mmap);
        }
        Ok(Self { maps })
    }

    pub fn get(&self, expert_id: usize) -> Result<&[u8]> {
        let mmap = self
            .maps
            .get(&expert_id)
            .with_context(|| format!("K3 expert {expert_id} pack missing (no silent host F32)"))?;
        let (id, payload) = parse_pack(mmap)?;
        if id as usize != expert_id {
            bail!("K3 expert pack id {id} != requested {expert_id}");
        }
        Ok(payload)
    }
}

/// Prefetch: remember last router top-k so the next token can hint NVMe.
#[derive(Clone, Debug, Default)]
pub struct PrefetchPlanner {
    last_ids: Vec<usize>,
}

impl PrefetchPlanner {
    pub fn note_router(&mut self, ids: &[usize]) {
        self.last_ids = ids.to_vec();
    }

    pub fn next_hint(&self) -> &[usize] {
        &self.last_ids
    }
}

/// Read a pack without mmap (tests / tiny payloads).
pub fn read_pack_file(path: &Path) -> Result<(u32, Vec<u8>)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let (id, payload) = parse_pack(&bytes)?;
    Ok((id, payload.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch() -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("k3-exp-{}-{nanos}", std::process::id()))
    }

    #[test]
    fn default_kind_is_resident() {
        if std::env::var_os(ENV).is_some() {
            return;
        }
        assert_eq!(kind_from_env(), ExpertBackendKind::Resident);
    }

    #[test]
    fn mmap_dummy_pack_roundtrips() {
        let dir = scratch();
        write_dummy_pack(&dir, 3, &[1, 2, 3, 4]).unwrap();
        let store = MmapExpertStore::open(&dir).unwrap();
        assert_eq!(store.get(3).unwrap(), &[1, 2, 3, 4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_pack_does_not_silent_zero() {
        let dir = scratch();
        write_dummy_pack(&dir, 0, &[9]).unwrap();
        let store = MmapExpertStore::open(&dir).unwrap();
        let err = store.get(1).unwrap_err().to_string();
        assert!(
            err.contains("missing") && err.contains("no silent host F32"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prefetch_notes_router_ids() {
        let mut p = PrefetchPlanner::default();
        assert!(p.next_hint().is_empty());
        p.note_router(&[7, 1]);
        assert_eq!(p.next_hint(), &[7, 1]);
    }

    #[test]
    fn mmap_env_child() {
        const THIS: &str = "kimi_k3::expert_backend::tests::mmap_env_child";
        const MARKER: &str = "K3_EXPERT_BACKEND_CHILD";
        if std::env::var_os(MARKER).is_some() {
            assert_eq!(kind_from_env(), ExpertBackendKind::Mmap);
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", THIS])
            .env(MARKER, "1")
            .env(ENV, "mmap")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
