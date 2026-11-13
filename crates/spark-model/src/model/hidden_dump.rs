// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: opt-in dump of the prefill hidden state (after chosen layers, after the final
//! norm, and the last-position logits) so a GPU parity check can compare Atlas's Qwen3.5 path with
//! a CPU reference (`route_gate/encoder/GPU-PARITY.md`, runner `~/atlas-wt/bench/gpu_parity.py`).
//!
//! Environment (all read once per prefill call; with `AVAROK_DUMP_HIDDEN` unset a prefill pays one
//! `std::env::var` miss and nothing else, and no device work is added):
//!
//! * `AVAROK_DUMP_HIDDEN=<dir>`: switch on, write into `<dir>` (created if missing).
//! * `AVAROK_DUMP_LAYERS=11,23` (0-based layer indices, or `all`): layers whose output is dumped.
//!   Unset: no layer dump, only the final norm and the logits.
//! * `AVAROK_DUMP_ALL_POSITIONS=1`: dump every processed position, not only the last token.
//!
//! Files, per dump a raw little-endian f32 payload plus a JSON header written last (the header is
//! the completion marker; both files are renamed into place from `*.tmp`):
//!
//! * `hidden_L<i>.bin/.json`: the residual stream after layer `i`. The layer loop leaves the full
//!   stream in `hidden` (`residual` is scratch), so this IS what layer `i + 1` sees.
//! * `final_norm.bin/.json`: the last token after the final RMSNorm (the input of the LM head).
//! * `logits.bin/.json`: the last-position logits, `vocab_size` values, in the dtype the LM head
//!   wrote (`device_dtype` in the header; bf16 unless the model runs fp32 logits).
//!
//! The header carries `shape`, `layer`, `token_count`, `proc_count`, `seq_len_start`,
//! `position_start`, `positions` ("last" or "all"), `dtype` (always "f32", the file) and
//! `device_dtype` (what the GPU computed in, widened exactly to f32).
//!
//! Limits, by design: ONE request at a time (every dump overwrites the file of the same name, so
//! the last prefill wins); the single-sequence prefill paths are hooked (`prefill` and the last
//! `prefill_chunk`), the batched co-dispatch path is not; a prefix-cache or Marconi hit makes
//! `seq_len_start > 0` and the earlier positions are not recomputed, so run with prefix caching
//! off (the default) and check `seq_len_start == 0` in the header. A failed dump is logged and
//! never fails the request.

use std::io;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::json;
use spark_runtime::gpu::DevicePtr;

use super::types::TransformerModel;

pub(crate) const ENV_DIR: &str = "AVAROK_DUMP_HIDDEN";
pub(crate) const ENV_LAYERS: &str = "AVAROK_DUMP_LAYERS";
pub(crate) const ENV_ALL_POSITIONS: &str = "AVAROK_DUMP_ALL_POSITIONS";
/// Header `version`; bump when a field changes meaning.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// What to dump, parsed from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HiddenDumpPlan {
    pub(crate) dir: PathBuf,
    all_layers: bool,
    layers: Vec<usize>,
    pub(crate) all_positions: bool,
    /// Layer-list entries that did not parse (reported once per prefill).
    pub(crate) ignored: Vec<String>,
}

impl HiddenDumpPlan {
    /// `None` unless `AVAROK_DUMP_HIDDEN` names a directory.
    pub(crate) fn from_env() -> Option<Self> {
        let dir = std::env::var(ENV_DIR).ok()?;
        let layers = std::env::var(ENV_LAYERS).ok();
        let all_positions = std::env::var(ENV_ALL_POSITIONS).ok();
        let plan = Self::parse(&dir, layers.as_deref(), all_positions.as_deref())?;
        if !plan.ignored.is_empty() {
            tracing::warn!(
                "{ENV_LAYERS}: ignoring unparsable entries {:?} (want 0-based indices or `all`)",
                plan.ignored
            );
        }
        Some(plan)
    }

    /// Pure parser behind [`Self::from_env`] (unit-tested without touching the process environment).
    pub(crate) fn parse(
        dir: &str,
        layers: Option<&str>,
        all_positions: Option<&str>,
    ) -> Option<Self> {
        let dir = dir.trim();
        if dir.is_empty() {
            return None;
        }
        let mut all_layers = false;
        let mut list: Vec<usize> = Vec::new();
        let mut ignored = Vec::new();
        for tok in layers
            .unwrap_or("")
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            if tok == "all" || tok == "*" {
                all_layers = true;
            } else if let Ok(i) = tok.parse::<usize>() {
                if !list.contains(&i) {
                    list.push(i);
                }
            } else {
                ignored.push(tok.to_string());
            }
        }
        list.sort_unstable();
        let all_positions = matches!(
            all_positions.map(str::trim),
            Some("1" | "true" | "yes" | "on")
        );
        Some(Self {
            dir: PathBuf::from(dir),
            all_layers,
            layers: list,
            all_positions,
            ignored,
        })
    }

    pub(crate) fn wants_layer(&self, layer: usize) -> bool {
        self.all_layers || self.layers.contains(&layer)
    }
}

/// Where in a prefill a hook fires; copied into the header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DumpSite {
    /// "prefill" (single shot) or "prefill_chunk" (the last chunk).
    pub(crate) prefill_path: &'static str,
    /// Prompt length in tokens.
    pub(crate) token_count: usize,
    /// Rows the layers processed this pass (`token_count - seq_len_start`).
    pub(crate) proc_count: usize,
    /// Absolute position of row 0 (0 on a cold prompt).
    pub(crate) seq_len_start: usize,
}

/// Header fields of one dump file.
#[derive(Debug, Clone)]
pub(crate) struct DumpMeta {
    /// "layer", "final_norm" or "logits".
    pub(crate) kind: &'static str,
    pub(crate) layer: Option<usize>,
    pub(crate) device_dtype: &'static str,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// Absolute position of row 0.
    pub(crate) position_start: usize,
    /// "last" (one row) or "all" (every processed position).
    pub(crate) positions: &'static str,
    pub(crate) site: DumpSite,
}

pub(crate) fn layer_stem(layer: usize) -> String {
    format!("hidden_L{layer}")
}

/// Write `<dir>/<stem>.bin` (f32 little endian) and then `<dir>/<stem>.json`; returns the header path.
pub(crate) fn write_dump(
    dir: &Path,
    stem: &str,
    meta: &DumpMeta,
    values: &[f32],
) -> io::Result<PathBuf> {
    if values.len() != meta.rows * meta.cols {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "dump {stem}: {} values for shape [{}, {}]",
                values.len(),
                meta.rows,
                meta.cols
            ),
        ));
    }
    std::fs::create_dir_all(dir)?;
    let mut payload = Vec::with_capacity(values.len() * 4);
    for v in values {
        payload.extend_from_slice(&v.to_le_bytes());
    }
    let bin_name = format!("{stem}.bin");
    let bin_tmp = dir.join(format!("{bin_name}.tmp"));
    std::fs::write(&bin_tmp, &payload)?;
    std::fs::rename(&bin_tmp, dir.join(&bin_name))?;

    let header = json!({
        "version": FORMAT_VERSION,
        "name": stem,
        "kind": meta.kind,
        "layer": meta.layer,
        "bin": bin_name,
        "dtype": "f32",
        "endian": "little",
        "device_dtype": meta.device_dtype,
        "shape": [meta.rows, meta.cols],
        "bytes": payload.len(),
        "positions": meta.positions,
        "position_start": meta.position_start,
        "token_count": meta.site.token_count,
        "proc_count": meta.site.proc_count,
        "seq_len_start": meta.site.seq_len_start,
        "prefill_path": meta.site.prefill_path,
    });
    let json_tmp = dir.join(format!("{stem}.json.tmp"));
    std::fs::write(
        &json_tmp,
        serde_json::to_vec_pretty(&header).map_err(io::Error::other)?,
    )?;
    let json_path = dir.join(format!("{stem}.json"));
    std::fs::rename(&json_tmp, &json_path)?;
    Ok(json_path)
}

/// Read a dump back: the header and the payload as f32 (used by the round-trip test; the runner
/// has its own stdlib reader).
#[cfg(test)]
pub(crate) fn read_dump(json_path: &Path) -> io::Result<(serde_json::Value, Vec<f32>)> {
    let header: serde_json::Value =
        serde_json::from_slice(&std::fs::read(json_path)?).map_err(io::Error::other)?;
    let bin = json_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(header["bin"].as_str().unwrap_or(""));
    let bytes = std::fs::read(bin)?;
    let rows = header["shape"][0].as_u64().unwrap_or(0) as usize;
    let cols = header["shape"][1].as_u64().unwrap_or(0) as usize;
    if bytes.len() != rows * cols * 4 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "payload length does not match shape",
        ));
    }
    Ok((header, f32_le_to_f32(&bytes)))
}

/// BF16 (little endian) to f32, exact.
pub(crate) fn bf16_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

pub(crate) fn f32_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn report(what: &str, result: Result<PathBuf>) {
    match result {
        Ok(path) => tracing::info!("{ENV_DIR}: wrote {what} to {}", path.display()),
        Err(e) => tracing::warn!("{ENV_DIR}: {what} not written: {e:#}"),
    }
}

impl TransformerModel {
    /// Copy `elems` values at `ptr` to the host as f32 (waits for the stream first).
    fn hidden_dump_read(
        &self,
        ptr: DevicePtr,
        elems: usize,
        fp32: bool,
        stream: u64,
    ) -> Result<Vec<f32>> {
        self.gpu.synchronize(stream)?;
        let mut buf = vec![0u8; elems * if fp32 { 4 } else { 2 }];
        self.gpu.copy_d2h(ptr, &mut buf)?;
        Ok(if fp32 {
            f32_le_to_f32(&buf)
        } else {
            bf16_le_to_f32(&buf)
        })
    }

    /// After layer `layer` of a prefill: dump the residual stream in `hidden` (BF16 `[proc_count, H]`)
    /// if the plan names this layer. No-op for `None`.
    pub(crate) fn dump_layer_hidden(
        &self,
        plan: Option<&HiddenDumpPlan>,
        layer: usize,
        hidden: DevicePtr,
        site: DumpSite,
        stream: u64,
    ) {
        let Some(plan) = plan else { return };
        if !plan.wants_layer(layer) || site.proc_count == 0 {
            return;
        }
        let h = self.config.hidden_size;
        let (rows, first, positions) = if plan.all_positions {
            (site.proc_count, 0, "all")
        } else {
            (1, site.proc_count - 1, "last")
        };
        let result = self
            .hidden_dump_read(hidden.offset(first * h * 2), rows * h, false, stream)
            .and_then(|vals| {
                let meta = DumpMeta {
                    kind: "layer",
                    layer: Some(layer),
                    device_dtype: "bf16",
                    rows,
                    cols: h,
                    position_start: site.seq_len_start + first,
                    positions,
                    site,
                };
                write_dump(&plan.dir, &layer_stem(layer), &meta, &vals).map_err(anyhow::Error::from)
            });
        report(&format!("layer {layer} hidden"), result);
    }

    /// The last token after the final RMSNorm: `normed` holds one BF16 row of `H` values (the LM head input).
    pub(crate) fn dump_final_norm(
        &self,
        plan: Option<&HiddenDumpPlan>,
        normed: DevicePtr,
        site: DumpSite,
        stream: u64,
    ) {
        let Some(plan) = plan else { return };
        if site.proc_count == 0 {
            return;
        }
        let h = self.config.hidden_size;
        let result = self
            .hidden_dump_read(normed, h, false, stream)
            .and_then(|vals| {
                let meta = DumpMeta {
                    kind: "final_norm",
                    layer: None,
                    device_dtype: "bf16",
                    rows: 1,
                    cols: h,
                    position_start: site.seq_len_start + site.proc_count - 1,
                    positions: "last",
                    site,
                };
                write_dump(&plan.dir, "final_norm", &meta, &vals).map_err(anyhow::Error::from)
            });
        report("final norm", result);
    }

    /// The last-position logits the LM head just wrote (`vocab_size` values, bf16 or fp32 by model).
    pub(crate) fn dump_logits(&self, plan: Option<&HiddenDumpPlan>, site: DumpSite, stream: u64) {
        let Some(plan) = plan else { return };
        if site.proc_count == 0 {
            return;
        }
        let fp32 = self.use_fp32_logits;
        let v = self.config.vocab_size;
        let result = self
            .hidden_dump_read(self.decode_logits_ptr(), v, fp32, stream)
            .and_then(|vals| {
                let meta = DumpMeta {
                    kind: "logits",
                    layer: None,
                    device_dtype: if fp32 { "f32" } else { "bf16" },
                    rows: 1,
                    cols: v,
                    position_start: site.seq_len_start + site.proc_count - 1,
                    positions: "last",
                    site,
                };
                write_dump(&plan.dir, "logits", &meta, &vals).map_err(anyhow::Error::from)
            });
        report("logits", result);
    }
}

#[cfg(test)]
#[path = "hidden_dump_tests.rs"]
mod tests;
