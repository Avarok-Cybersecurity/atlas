// SPDX-License-Identifier: AGPL-3.0-only
// The workspace rules follow vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/marlin.cu:528-541 (the FP32 reduce buffer) and marlin_utils.py:407-415
// (`marlin_make_workspace_new`: zeroed locks, `sms * max_blocks_per_sm`). Copyright contributors to the
// vLLM project, Apache-2.0. Changes: the locks are owned per weight (no shared workspace manager), the
// reduce buffer is the caller's `W4Scratch::ws`, and the decode configuration is fixed at upload.

//! Responsibility: [`MarlinWeight`], one Marlin weight resident on the device (codes, scales, zero
//! points, its own lock array, the resolved kernels and the decode configuration fixed at upload), with
//! [`MarlinWeight::upload`] and [`MarlinWeight::release`].
//!
//! The lock array belongs to the weight: the kernel returns every lock to zero, so stream-ordered
//! launches of one weight reuse it, and two weights never share one. A weight is decoded by one stream
//! at a time.

use std::sync::Arc;

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::MarlinType;
use super::config::{self, Problem, ThreadConfig};
use super::kernels::{self, MarlinKernels};
use super::repack::MarlinHost;

/// One Marlin weight on the device.
#[derive(Clone, Debug)]
pub struct MarlinWeight {
    /// `[k / 16][2 n]` u32 repacked codes.
    pub words: DevicePtr,
    /// `[groups][n]` permuted BF16 scales.
    pub scales: DevicePtr,
    /// `[groups][n / 8]` packed zero points, NULL for `kU4B8`.
    pub zeros: DevicePtr,
    /// `config::locks_bytes(sms)` zeroed i32.
    pub locks: DevicePtr,
    pub btype: MarlinType,
    /// Outputs.
    pub n: usize,
    /// Inputs.
    pub k: usize,
    pub group: usize,
    /// The grid size of every launch.
    pub sms: usize,
    /// Dynamic shared bytes of every launch.
    pub max_shared_mem: usize,
    /// The one decode configuration (row invariance).
    pub decode_cfg: ThreadConfig,
    pub kernels: Arc<MarlinKernels>,
}

/// Bytes of a host image's device buffers: codes, scales, zero points (locks excluded).
pub fn image_bytes(h: &MarlinHost) -> usize {
    h.words.len() * 4 + h.scales.len() * 2 + h.zeros.as_ref().map_or(0, |z| z.len() * 4)
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Allocations made so far by an upload, freed if a later step fails.
struct Staged<'a> {
    gpu: &'a dyn GpuBackend,
    ptrs: Vec<DevicePtr>,
}

impl Staged<'_> {
    fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        let p = self.gpu.alloc(bytes)?;
        self.ptrs.push(p);
        Ok(p)
    }

    fn abandon(self) {
        for p in self.ptrs {
            let _ = self.gpu.free(p);
        }
    }
}

/// Allocate and fill the four buffers of `h` (codes, scales, zero points or NULL, zeroed locks), then
/// wait for the copies.
fn upload_buffers(
    st: &mut Staged<'_>,
    h: &MarlinHost,
    sms: usize,
    stream: u64,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, DevicePtr)> {
    let gpu = st.gpu;
    let words = st.alloc(h.words.len() * 4)?;
    gpu.copy_h2d_async(&u32_bytes(&h.words), words, stream)?;
    let scales = st.alloc(h.scales.len() * 2)?;
    gpu.copy_h2d_async(&u16_bytes(&h.scales), scales, stream)?;
    let zeros = match &h.zeros {
        Some(z) => {
            let p = st.alloc(z.len() * 4)?;
            gpu.copy_h2d_async(&u32_bytes(z), p, stream)?;
            p
        }
        None => DevicePtr::NULL,
    };
    let lb = config::locks_bytes(sms);
    let locks = st.alloc(lb)?;
    gpu.memset_async(locks, 0, lb, stream)?;
    gpu.synchronize(stream)?;
    Ok((words, scales, zeros, locks))
}

impl MarlinWeight {
    /// The problem this weight's launches compute.
    pub fn problem(&self) -> Problem {
        Problem::of(self.n, self.k, self.group, self.btype.has_zp())
    }

    /// Groups along K.
    pub fn groups(&self) -> usize {
        self.k / self.group
    }

    /// Resident device bytes (codes, scales, zero points, locks).
    pub fn device_bytes(&self) -> usize {
        let zeros = if self.btype.has_zp() {
            self.groups() * self.n / 8 * 4
        } else {
            0
        };
        self.k * self.n / 2 + self.groups() * self.n * 2 + zeros + config::locks_bytes(self.sms)
    }

    /// Bytes of `W4Scratch::ws` a call of at most `rows` rows needs (the FP32 reduce buffer).
    pub fn scratch_bytes(&self, rows: usize) -> usize {
        config::c_tmp_bytes(rows, self.sms)
    }

    /// Resolve the kernels, fix the decode configuration, allocate and upload `h` on `stream`.
    pub fn upload(gpu: &dyn GpuBackend, h: &MarlinHost, stream: u64) -> Result<Self> {
        config::supports_shape(h.n, h.k, h.group).map_err(|e| anyhow::anyhow!("marlin: {e}"))?;
        ensure!(
            kernels::group_supported(h.group, h.k),
            "marlin: group {} at K={} has no kernel instance",
            h.group,
            h.k
        );
        ensure!(
            h.btype.has_zp() == h.zeros.is_some(),
            "marlin: a {:?} image with{} zero points",
            h.btype,
            if h.zeros.is_some() { "" } else { "out" }
        );
        let p = Problem::of(h.n, h.k, h.group, h.btype.has_zp());
        let kern = kernels::resolved(gpu, h.btype, p.group_blocks())?;
        let sms = gpu.sm_count()? as usize;
        ensure!(sms > 0, "marlin: the backend reports no SMs");
        let max_shared_mem = config::ADA_MAX_SHARED_MEM_OPTIN;
        let avail = |t: usize, m8: bool, c: &ThreadConfig| kern.has(t, m8, c);
        let Some(decode_cfg) =
            config::invariant_decode_config(&p, &kernels::DECODE_TMBS, max_shared_mem, &avail)
        else {
            anyhow::bail!(
                "marlin: [{}x{}] g{} has no 128-thread configuration valid at every decode tile",
                h.n,
                h.k,
                h.group
            );
        };

        let mut st = Staged {
            gpu,
            ptrs: Vec::new(),
        };
        let res = upload_buffers(&mut st, h, sms, stream);
        let (words, scales, zeros, locks) = match res {
            Ok(v) => v,
            Err(e) => {
                st.abandon();
                return Err(e);
            }
        };
        Ok(Self {
            words,
            scales,
            zeros,
            locks,
            btype: h.btype,
            n: h.n,
            k: h.k,
            group: h.group,
            sms,
            max_shared_mem,
            decode_cfg,
            kernels: kern,
        })
    }

    /// Free the device buffers. The caller guarantees no launch still reads them.
    pub fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.words, self.scales, self.zeros, self.locks] {
            if !p.is_null() {
                gpu.free(p)?;
            }
        }
        Ok(())
    }

    /// Read the device image back to the host (synchronizes `stream`; load and oracle use only).
    pub fn download(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<MarlinHost> {
        gpu.synchronize(stream)?;
        let mut wb = vec![0u8; self.k * self.n / 2];
        gpu.copy_d2h_on_stream(self.words, &mut wb, stream)?;
        let mut sb = vec![0u8; self.groups() * self.n * 2];
        gpu.copy_d2h_on_stream(self.scales, &mut sb, stream)?;
        let zeros = if self.btype.has_zp() {
            let mut zb = vec![0u8; self.groups() * self.n / 2];
            gpu.copy_d2h_on_stream(self.zeros, &mut zb, stream)?;
            Some(
                zb.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            )
        } else {
            None
        };
        Ok(MarlinHost {
            btype: self.btype,
            n: self.n,
            k: self.k,
            group: self.group,
            words: wb
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            scales: sb
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect(),
            zeros,
        })
    }
}
