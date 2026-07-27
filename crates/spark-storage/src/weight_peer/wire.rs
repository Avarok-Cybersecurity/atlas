// SPDX-License-Identifier: AGPL-3.0-only
//
// Weight-staging wire codec (un-gated, CUDA-free, verbs-free).
//
// The length-prefixed framing the daemon and its clients speak: the model
// request `[u32 len][bytes]` and the manifest `[u32 len][JSON]`. Kept beside
// `manifest` (which it references) and free of any server dependency so the
// LoRA client (`weight_lora_rdma`) imports only this + `manifest`.

use anyhow::{Context, Result, bail};

use super::manifest::WeightManifest;

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read u32")?;
    Ok(u32::from_le_bytes(b))
}

/// Longest model id/path the wire accepts (a corrupt/hostile length must not
/// trigger a huge allocation).
pub const MODEL_REQUEST_MAX: usize = 8192;

/// Wire form of the model request: `[u32 len][len bytes UTF-8 id/path]`.
pub fn write_model_request<W: std::io::Write>(w: &mut W, id: &str) -> Result<()> {
    let bytes = id.as_bytes();
    if bytes.is_empty() || bytes.len() > MODEL_REQUEST_MAX {
        bail!("implausible model request length: {}", bytes.len());
    }
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)?;
    Ok(())
}

/// Read a model request written by [`write_model_request`].
pub fn read_model_request<R: std::io::Read>(r: &mut R) -> Result<String> {
    let len = read_u32(r)? as usize;
    if len == 0 || len > MODEL_REQUEST_MAX {
        bail!("implausible model request length: {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).context("read model request body")?;
    String::from_utf8(buf).context("model request is not valid UTF-8")
}

/// Serialize + frame a manifest as `[u32 len][len bytes JSON]`.
pub fn write_weight_manifest<W: std::io::Write>(w: &mut W, m: &WeightManifest) -> Result<()> {
    let json = serde_json::to_vec(m).context("serialize weight manifest")?;
    w.write_all(&(json.len() as u32).to_le_bytes())?;
    w.write_all(&json)?;
    Ok(())
}

/// Read + parse a length-prefixed manifest. Shared by the client tier.
pub fn read_weight_manifest<R: std::io::Read>(r: &mut R) -> Result<WeightManifest> {
    let len = read_u32(r)? as usize;
    if len == 0 || len > 256 * 1024 * 1024 {
        bail!("implausible weight manifest length: {len}");
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .context("read weight manifest json")?;
    let m: WeightManifest = serde_json::from_slice(&buf).context("parse weight manifest json")?;
    if m.version != WeightManifest::VERSION {
        bail!(
            "weight manifest version {} != supported {}",
            m.version,
            WeightManifest::VERSION
        );
    }
    if m.shard_files.len() != m.shard_lens.len() {
        bail!(
            "manifest shard_files ({}) / shard_lens ({}) length mismatch",
            m.shard_files.len(),
            m.shard_lens.len()
        );
    }
    Ok(m)
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
