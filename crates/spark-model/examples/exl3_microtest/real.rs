// SPDX-License-Identifier: AGPL-3.0-only
// Group naming after TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/exl3_load.zig:7-13 and 120-165 (`{prefix}.trellis`, `.suh`/`.svh` or `.su`/`.sv`, the
// `.mcg` / `.mul1` marker tensors and their magic values). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes: no code copied; a minimal safetensors header reader written for the microtest.

//! Responsibility: the microtest's REAL-tensor cells: one EXL3 group read straight from a checkpoint's
//! safetensors shards (`--real-dir <dir> --prefix <tensor prefix>`, for example a projection of
//! `turboderp/Qwen3.8-27B-exl3`), and TensorFold's public-checkpoint fixture block (`ATLAS_EXL3_ORACLE`).
//! Both are decoded by the GPU dequant and checked byte for byte against the host mirror, and the real
//! group also runs the GEMV gates. The reader is independent of lane L20a's, so these cells also check
//! that reader's assumptions about the stored layout.

use anyhow::{Context, Result, bail, ensure};
use spark_model::port::exl3::format::{
    Exl3Codebook, Exl3Parts, Exl3Scales, MARKER_MCG, MARKER_MUL1,
};
use spark_model::port::exl3::reference::unpack;
use spark_model::port::exl3::reference_math::{bf16_to_f32, dequant_bf16_mirror, dequant_f64};
use spark_model::port::exl3::{Exl3Linear, release};
use spark_runtime::gpu::GpuBackend;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::util::{Tally, download_u16, unequal};

struct Entry {
    file: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

/// Every tensor header of the `*.safetensors` files in `dir`.
fn scan(dir: &Path) -> Result<HashMap<String, Entry>> {
    let mut out = HashMap::new();
    for e in std::fs::read_dir(dir).with_context(|| format!("{}", dir.display()))? {
        let path = e?.path();
        if path.extension().and_then(|x| x.to_str()) != Some("safetensors") {
            continue;
        }
        let mut f = std::fs::File::open(&path)?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len)?;
        let n = u64::from_le_bytes(len);
        ensure!(n < 1 << 30, "{}: header of {n} bytes", path.display());
        let mut head = vec![0u8; n as usize];
        f.read_exact(&mut head)?;
        let json: serde_json::Value = serde_json::from_slice(&head)?;
        let obj = json
            .as_object()
            .context("safetensors header is not an object")?;
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let offs = v["data_offsets"].as_array().context("data_offsets")?;
            let shape = v["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize).context("shape dim"))
                .collect::<Result<Vec<_>>>()?;
            out.insert(
                name.clone(),
                Entry {
                    file: path.clone(),
                    dtype: v["dtype"].as_str().unwrap_or("").to_owned(),
                    shape,
                    begin: 8 + n + offs[0].as_u64().context("begin")?,
                    end: 8 + n + offs[1].as_u64().context("end")?,
                },
            );
        }
    }
    Ok(out)
}

fn read(e: &Entry) -> Result<Vec<u8>> {
    let mut f = std::fs::File::open(&e.file)?;
    f.seek(SeekFrom::Start(e.begin))?;
    let mut b = vec![0u8; (e.end - e.begin) as usize];
    f.read_exact(&mut b)?;
    Ok(b)
}

/// One stored group, owned.
pub struct Group {
    pub k: usize,
    pub n: usize,
    pub halves: u16,
    pub cb: Exl3Codebook,
    pub trellis: Vec<u8>,
    pub su: Vec<u8>,
    pub sv: Vec<u8>,
    pub packed: bool,
}

impl Group {
    pub fn parts(&self) -> Exl3Parts<'_> {
        let (su, sv) = if self.packed {
            (
                Exl3Scales::Signs(&self.su[..]),
                Exl3Scales::Signs(&self.sv[..]),
            )
        } else {
            (Exl3Scales::F16(&self.su[..]), Exl3Scales::F16(&self.sv[..]))
        };
        Exl3Parts {
            k: self.k,
            n: self.n,
            halves: self.halves,
            codebook: self.cb,
            trellis: &self.trellis,
            su,
            sv,
            bias: None,
        }
    }
}

/// Read `{prefix}.*` from the shards in `dir`.
pub fn load_group(dir: &Path, prefix: &str) -> Result<Group> {
    let all = scan(dir)?;
    let get = |s: &str| all.get(&format!("{prefix}.{s}"));
    let t = get("trellis").with_context(|| format!("no {prefix}.trellis in {}", dir.display()))?;
    ensure!(
        t.dtype == "I16" && t.shape.len() == 3,
        "{prefix}.trellis is {} {:?}",
        t.dtype,
        t.shape
    );
    let (k, n, halves) = (t.shape[0] * 16, t.shape[1] * 16, (t.shape[2] / 8) as u16);
    let mut cb = Exl3Codebook::Inst3;
    for (name, magic) in [("mcg", MARKER_MCG), ("mul1", MARKER_MUL1)] {
        if let Some(m) = get(name) {
            let b = read(m)?;
            ensure!(b.len() == 4, "{prefix}.{name} is not a scalar");
            cb = Exl3Codebook::from_marker(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))?;
            ensure!(
                u32::from_le_bytes([b[0], b[1], b[2], b[3]]) == magic,
                "{prefix}.{name} magic"
            );
        }
    }
    if get("bias").is_some() {
        bail!("{prefix} has a bias, which the W4Linear route refuses");
    }
    let (su, sv, packed) = match (get("suh"), get("svh"), get("su"), get("sv")) {
        (Some(a), Some(b), _, _) => (read(a)?, read(b)?, false),
        (_, _, Some(a), Some(b)) => (read(a)?, read(b)?, true),
        _ => bail!("{prefix}: neither suh/svh nor su/sv"),
    };
    Ok(Group {
        k,
        n,
        halves,
        cb,
        trellis: read(t)?,
        su,
        sv,
        packed,
    })
}

/// The GPU dequant of a real group against the host mirror, byte for byte.
pub fn dequant_gate(gpu: &dyn GpuBackend, g: &Group, label: &str, t: &mut Tally) -> Result<()> {
    let parts = g.parts();
    parts.check()?;
    let w = Exl3Linear::upload(gpu, &parts, 0)?;
    let out = gpu.alloc(g.n * g.k * 2)?;
    w.dequant_bf16(gpu, out, 0)?;
    let got = download_u16(gpu, out, g.n * g.k)?;
    let wq = unpack(g.halves, g.cb, &parts.trellis_u16(), g.k, g.n);
    let want = dequant_bf16_mirror(&wq, g.k, g.n, &parts.suh_bits(), &parts.svh_bits());
    t.gate(
        &format!("{label}: GPU dequant == host mirror"),
        unequal(&got, &want),
        &format!("[{} -> {}] {} half-bits {:?}", g.k, g.n, g.halves, g.cb),
    );
    gpu.free(out)?;
    release(gpu, &*w)?;
    Ok(())
}

/// TensorFold's fixture block (`ATLAS_EXL3_ORACLE`): the real MCG 8.00 bpw 128 x 128 block, its GPU
/// dequant against the host mirror (bytes) and against the fixture's FP64 dequant (bf16 rounding).
pub fn fixture_gate(gpu: &dyn GpuBackend, t: &mut Tally) -> Result<()> {
    let Some(path) = std::env::var_os("ATLAS_EXL3_ORACLE") else {
        t.info("fixture block", "ATLAS_EXL3_ORACLE not set, skipped");
        return Ok(());
    };
    let b = std::fs::read(&path)?;
    ensure!(&b[..8] == b"EXL3ORAC", "not an EXL3 oracle file");
    let u32_at = |at: usize| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;
    let u16_at = |at: usize| u16::from_le_bytes([b[at], b[at + 1]]);
    let count = u64::from_le_bytes(b[8..16].try_into()?) as usize;
    let mut at = 16;
    for _ in 0..count {
        let words = 8 * u16_at(at) as usize;
        let (kt, nt) = (u32_at(at + 4), u32_at(at + 8));
        at += 12 + kt * nt * (words * 2 + 256 * 4);
    }
    at += 4 + u32_at(at) * 6;
    let halves = u16_at(at);
    let (kt, nt) = (u32_at(at + 2), u32_at(at + 6));
    let tl = u32_at(at + 10);
    at += 14;
    let trellis = b[at..at + tl].to_vec();
    at += tl;
    let sl = u32_at(at);
    let su = b[at + 4..at + 4 + sl].to_vec();
    at += 4 + sl;
    let vl = u32_at(at);
    let sv = b[at + 4..at + 4 + vl].to_vec();
    at += 4 + vl;
    at += 4 + u32_at(at); // the Wq bits (the CPU tests check them)
    let dl = u32_at(at);
    let dq: Vec<f64> = b[at + 4..at + 4 + dl]
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let g = Group {
        k: kt * 16,
        n: nt * 16,
        halves,
        cb: Exl3Codebook::Mcg,
        trellis,
        su,
        sv,
        packed: false,
    };
    dequant_gate(gpu, &g, "fixture block", t)?;
    let parts = g.parts();
    let w = Exl3Linear::upload(gpu, &parts, 0)?;
    let out = gpu.alloc(g.n * g.k * 2)?;
    w.dequant_bf16(gpu, out, 0)?;
    let got = download_u16(gpu, out, g.n * g.k)?;
    let mag = dq.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let mut bad = 0;
    for kk in 0..g.k {
        for nn in 0..g.n {
            let want = dq[kk * g.n + nn];
            let gv = f64::from(bf16_to_f32(got[nn * g.k + kk]));
            bad += usize::from((gv - want).abs() > want.abs() * 2f64.powi(-8) + 1e-4 * mag);
        }
    }
    t.gate(
        "fixture block: GPU dequant within bf16 rounding of the FP64 oracle",
        bad,
        "",
    );
    // Cross-check the host's own FP64 path on the same block.
    let wq = unpack(g.halves, g.cb, &parts.trellis_u16(), g.k, g.n);
    let mine = dequant_f64(&wq, g.k, g.n, &parts.suh_bits(), &parts.svh_bits());
    let off = mine
        .iter()
        .zip(&dq)
        .filter(|(a, b)| (*a - *b).abs() > 1e-9 * (1.0 + mag))
        .count();
    t.gate("fixture block: host FP64 dequant == oracle FP64", off, "");
    gpu.free(out)?;
    release(gpu, &*w)?;
    Ok(())
}
