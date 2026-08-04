// SPDX-License-Identifier: AGPL-3.0-only
//
// ExpertTier — the residency abstraction the expert streamer fetches through.
//
// Sits ABOVE the record layer (`ExpertFileReader` / `ExpertIndex`), not as a
// `StorageBackend` impl: that trait is GroupKey/KV-tile shaped and synchronizes
// the stream on return, the wrong shape for pull-on-demand MoE experts. A tier
// lands one expert record into a slot and returns the six device addresses the
// fused kernels read (already resident / prefill-transposed layout — nothing is
// transformed at fetch time, invariant D).
//
// Three tiers, one interface (residency order device < UMA-over-NVMe < RDMA):
//   * `PosixTier`    — deterministic bounce oracle (pread -> copy_h2d into a
//                      device buffer). The bit-identical acceptance reference.
//   * `UmaArenaTier` — the zero-copy path: O_DIRECT NVMe fill straight into the
//                      pinned arena; the ptr table points at the pinned VA, no
//                      HtoD copy.
//   * RdmaTier       — Stage 4: one-sided RDMA_READ into the SAME pinned arena.
//
// All three feed the identical ptr-table patch, so swapping tiers cannot change
// a single byte the GEMM reads — which the Tier-1 parity test proves.

use anyhow::{Context, Result, bail};
use atlas_tier::pio;
use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::cuda_min::{DeviceBuffer, copy_h_to_d_async, stream_sync};
use crate::expert::{ExpertKey, ExpertLayout, ExpertRecordHeader, ExpertRecordSpec, Proj};
use crate::expert_arena::ExpertArena;
use crate::expert_pack::{ExpertFileReader, ExpertIndex};

/// The six sub-buffer device addresses (+ scalars) of one resident expert —
/// exactly what the ptr-table patcher writes into the shadow tables.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertResidency {
    /// gate/up/down B_packed device VA.
    pub packed_addr: [u64; 3],
    /// gate/up/down B_scale device VA.
    pub scale_addr: [u64; 3],
    /// gate/up/down per-tensor weight_scale_2.
    pub scale2: [f32; 3],
    /// gate/up/down input_scale (None = weight-only W4A16).
    pub input_scale: [Option<f32>; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TierKind {
    Posix,
    Uma,
    Rdma,
}

/// A destination slot in the residency ring: which slab, which slot within it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaSlot {
    pub slab: u32,
    pub slot: u32,
}

impl ArenaSlot {
    pub fn new(slab: u32, slot: u32) -> Self {
        Self { slab, slot }
    }
}

/// Fetch an expert record into a slot; return the addresses to patch.
pub trait ExpertTier: Send {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, stream: u64) -> Result<ExpertResidency>;
    fn kind(&self) -> TierKind;
    /// Graceful-degradation probe (RDMA link health at Stage 4). Default: up.
    fn healthy(&self) -> bool {
        true
    }
}

/// Turn a record buffer's header + a base device VA into an `ExpertResidency`
/// using the record spec's sub-offsets. Shared by every tier so they cannot
/// disagree on layout.
pub(crate) fn residency_from(
    spec: &ExpertRecordSpec,
    record: &[u8],
    base_dev_va: u64,
    key: ExpertKey,
) -> Result<ExpertResidency> {
    let hdr = ExpertRecordHeader::from_bytes(record)
        .context("expert record header magic/version mismatch")?;
    // Invariant D at fetch time: the header carries identity precisely so a
    // misplaced/corrupt record is caught here, not served silently.
    if hdr.layer != key.layer || hdr.expert != key.expert {
        bail!(
            "expert record identity mismatch: header ({},{}) != requested {:?}",
            hdr.layer,
            hdr.expert,
            key
        );
    }
    let mut packed_addr = [0u64; 3];
    let mut scale_addr = [0u64; 3];
    for p in Proj::ALL {
        packed_addr[p as usize] = base_dev_va + spec.packed_off(p);
        scale_addr[p as usize] = base_dev_va + spec.scale_off(p);
    }
    Ok(ExpertResidency {
        packed_addr,
        scale_addr,
        scale2: hdr.scale2,
        input_scale: hdr.input_scale,
    })
}

/// Deterministic bounce oracle: pread the record (no O_DIRECT) into a host
/// buffer, `copy_h2d` into a per-slot device buffer, stream-synced. This is the
/// reference every other tier must match byte-for-byte.
pub struct PosixTier {
    reader: ExpertFileReader,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    /// One contiguous device buffer holding num_slabs*slots_per_slab records.
    dev: DeviceBuffer,
    slots_per_slab: u32,
    num_slabs: u32,
}

impl PosixTier {
    pub fn open(dir: &Path, num_slabs: u32, slots_per_slab: u32) -> Result<Self> {
        let reader = ExpertFileReader::open(dir)?;
        let index: &ExpertIndex = reader.index();
        let spec = index.spec();
        let layout = index.layout();
        if num_slabs == 0 || slots_per_slab == 0 {
            bail!("PosixTier: zero geometry ({num_slabs},{slots_per_slab})");
        }
        let stride = layout.record_stride as usize;
        // Checked like the sibling ExpertArena ctor — a wrapped product must
        // never yield a small alloc that slot_dev_va then addresses past.
        let total = (num_slabs as usize)
            .checked_mul(slots_per_slab as usize)
            .and_then(|v| v.checked_mul(stride))
            .context("PosixTier: arena size overflow")?;
        let dev = DeviceBuffer::new(total)?;
        Ok(Self {
            reader,
            spec,
            layout,
            dev,
            slots_per_slab,
            num_slabs,
        })
    }

    fn slot_dev_va(&self, slot: ArenaSlot) -> Result<u64> {
        if slot.slab >= self.num_slabs || slot.slot >= self.slots_per_slab {
            bail!("PosixTier: slot {:?} out of range", slot);
        }
        let i = (slot.slab as u64) * (self.slots_per_slab as u64) + (slot.slot as u64);
        Ok(self.dev.ptr + i * self.layout.record_stride)
    }
}

impl ExpertTier for PosixTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, stream: u64) -> Result<ExpertResidency> {
        let record = self.reader.read_record_raw(key)?; // host bytes
        let dev_va = self.slot_dev_va(slot)?;
        copy_h_to_d_async(dev_va, record.as_ptr() as *const _, record.len(), stream)?;
        stream_sync(stream)?; // single bounce would be overwritten otherwise
        residency_from(&self.spec, &record, dev_va, key)
    }
    fn kind(&self) -> TierKind {
        TierKind::Posix
    }
}

/// The zero-copy path: O_DIRECT the record straight into the pinned arena slot;
/// the returned addresses point INTO the arena (GPU-addressable at the same
/// VA), no `copy_h2d`.
pub struct UmaArenaTier {
    files: Vec<File>, // one O_DIRECT fd per MoE layer
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    arena: ExpertArena,
}

impl UmaArenaTier {
    pub fn open(dir: &Path, num_slabs: u32, slots_per_slab: u32) -> Result<Self> {
        let reader = ExpertFileReader::open(dir)?;
        let index: &ExpertIndex = reader.index();
        let spec = index.spec();
        let layout = index.layout();
        // Re-open each layer file with O_DIRECT for aligned zero-copy reads.
        let mut files = Vec::with_capacity(index.num_moe_layers as usize);
        for l in 0..index.num_moe_layers {
            let p = dir.join(index.file_name(l));
            let mut opts = OpenOptions::new();
            opts.read(true);
            set_direct_flag(&mut opts);
            let f = opts
                .open(&p)
                .with_context(|| format!("open {}", p.display()))?;
            files.push(f);
        }
        let arena = ExpertArena::new(num_slabs, slots_per_slab, layout.record_stride as usize)?;
        Ok(Self {
            files,
            spec,
            layout,
            arena,
        })
    }

    pub fn arena(&self) -> &ExpertArena {
        &self.arena
    }
}

impl ExpertTier for UmaArenaTier {
    fn fetch(&mut self, key: ExpertKey, slot: ArenaSlot, _stream: u64) -> Result<ExpertResidency> {
        let stride = self.layout.record_stride as usize;
        let host = self.arena.slot_host_ptr(slot.slab, slot.slot)?;
        let file = self
            .files
            .get(key.layer as usize)
            .with_context(|| format!("UmaArenaTier: no file for layer {}", key.layer))?;
        let off = self.layout.file_offset(key);
        // Positional read of the whole (4 KiB-aligned) record straight into the
        // pinned, GPU-addressable slot — this is the "delete the bounce copy"
        // step. O_DIRECT on Linux makes it zero-copy; elsewhere it is buffered.
        // SAFETY: `host` points at a slot of `stride` bytes inside the pinned
        // arena, and the slice covers exactly that slot.
        let dst = unsafe { std::slice::from_raw_parts_mut(host, stride) };
        pio::read_exact_at(file, dst, off)
            .with_context(|| format!("UmaArenaTier read {key:?} at {off}"))?;
        // SAFETY: the slot holds `stride` valid bytes just read from disk.
        let record = unsafe { std::slice::from_raw_parts(host, stride) };
        let dev_va = self.arena.slot_dev_va(slot.slab, slot.slot)?;
        residency_from(&self.spec, record, dev_va, key)
    }
    fn kind(&self) -> TierKind {
        TierKind::Uma
    }
}

/// Open the tier named by `backend` over a built store:
///   * `posix` / `uma` — read `dir` locally (bounce oracle / zero-copy).
///   * `rdma`          — connect to `$ATLAS_EXPERT_PEER` over TWO-SIDED TCP.
///   * `rdma-verbs`    — connect to `$ATLAS_EXPERT_PEER` over ONE-SIDED RDMA READ
///     (verbs); device/GID from `$ATLAS_EXPERT_RDMA_DEV`/`$ATLAS_EXPERT_RDMA_GID`.
///
/// Both peer backends serve the store's records over the RoCE fabric.
pub fn open_tier(
    backend: &str,
    dir: &Path,
    num_slabs: u32,
    slots_per_slab: u32,
) -> Result<Box<dyn ExpertTier>> {
    // The RDMA expert tiers need rdma-core, so they exist on unix only. The
    // local `posix`/`uma` tiers below are portable; asking for an RDMA backend
    // elsewhere fails with a clear message rather than being silently absent
    // from the match.
    let rdma = |use_verbs: bool| -> Result<Box<dyn ExpertTier>> {
        let flag = if use_verbs { "rdma-verbs" } else { "rdma" };
        #[cfg(unix)]
        {
            let addr = std::env::var("ATLAS_EXPERT_PEER").map_err(|_| {
                anyhow::anyhow!("--expert-backend {flag} needs $ATLAS_EXPERT_PEER=host:port")
            })?;
            Ok(Box::new(crate::expert_tier_rdma::RdmaTier::connect(
                &addr,
                num_slabs,
                slots_per_slab,
                use_verbs,
            )?))
        }
        #[cfg(not(unix))]
        {
            bail!("--expert-backend {flag} requires rdma-core, which is unix-only")
        }
    };
    match backend {
        "posix" => Ok(Box::new(PosixTier::open(dir, num_slabs, slots_per_slab)?)),
        "uma" => Ok(Box::new(UmaArenaTier::open(
            dir,
            num_slabs,
            slots_per_slab,
        )?)),
        "rdma" => rdma(false),
        "rdma-verbs" => rdma(true),
        other => bail!("unknown expert backend '{other}' (want posix|uma|rdma|rdma-verbs)"),
    }
}

/// Copy `len` bytes from a device VA to a fresh host `Vec` (test/verify helper).
pub fn read_device(dev_va: u64, len: usize, stream: u64) -> Result<Vec<u8>> {
    use crate::cuda_min::copy_d_to_h_async;
    let mut out = vec![0u8; len];
    copy_d_to_h_async(out.as_mut_ptr() as *mut _, dev_va, len, stream)?;
    stream_sync(stream)?;
    Ok(out)
}

/// O_DIRECT on Linux (the zero-copy arena read depends on it); no equivalent
/// flag elsewhere — see the `layout` module header for why Windows stays
/// buffered rather than using FILE_FLAG_NO_BUFFERING.
#[cfg(target_os = "linux")]
fn set_direct_flag(opts: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    opts.custom_flags(libc::O_DIRECT);
}

#[cfg(not(target_os = "linux"))]
fn set_direct_flag(_opts: &mut OpenOptions) {}
