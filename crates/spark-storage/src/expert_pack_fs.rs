// SPDX-License-Identifier: AGPL-3.0-only

//! Unix file reader/writer for expert packs — the `fs_impl` submodule of
//! `expert_pack`, split out to keep the parent under the 500-LoC cap.

use super::*;
use std::fs::{File, OpenOptions};
// Positional I/O via the shared helper rather than `std::os::unix::fs::FileExt`:
// the only thing that made this 328-line module unix-only was the trait import.
use atlas_tier::pio;
use std::path::{Path, PathBuf};

/// Offline writer: creates the manifest + one file per MoE layer and places
/// records at their strided offsets. Plain buffered writes (no O_DIRECT) —
/// alignment only matters on the streamer's read path, and the record stride
/// is already a 4 KiB multiple, so the files are O_DIRECT-readable.
pub struct ExpertFileWriter {
    dir: PathBuf,
    index: ExpertIndex,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    files: Vec<File>,
}

impl ExpertFileWriter {
    pub fn create(dir: &Path, index: ExpertIndex) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
        let spec = index.spec();
        let layout = index.layout();
        let mut files = Vec::with_capacity(index.num_moe_layers as usize);
        for l in 0..index.num_moe_layers {
            let p = dir.join(index.file_name(l));
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&p)
                .with_context(|| format!("create {}", p.display()))?;
            f.set_len(layout.bytes_per_layer())
                .with_context(|| format!("set_len {}", p.display()))?;
            files.push(f);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            index,
            spec,
            layout,
            files,
        })
    }

    pub fn spec(&self) -> &ExpertRecordSpec {
        &self.spec
    }

    /// Assemble and write one expert record at its strided offset.
    pub fn write_record(
        &self,
        key: ExpertKey,
        header: &ExpertRecordHeader,
        projs: &[ProjData; 3],
    ) -> Result<()> {
        if key.layer >= self.index.num_moe_layers {
            bail!("layer {} out of range", key.layer);
        }
        if key.expert >= self.index.num_experts {
            bail!("expert {} out of range", key.expert);
        }
        let rec = pack_record(&self.spec, self.layout.record_stride, header, projs)?;
        let off = self.layout.file_offset(key);
        pio::write_all_at(&self.files[key.layer as usize], &rec, off)
            .with_context(|| format!("write record {:?} at {off}", key))?;
        Ok(())
    }

    /// Flush the manifest to `manifest.json`. Call once, last.
    pub fn finish(self) -> Result<()> {
        for f in &self.files {
            f.sync_all().context("fsync layer file")?;
        }
        let p = self.dir.join(ExpertIndex::MANIFEST_NAME);
        let json = serde_json::to_string_pretty(&self.index)?;
        std::fs::write(&p, json).with_context(|| format!("write {}", p.display()))?;
        Ok(())
    }
}

/// Reader used by tests / tooling to verify a built store without a GPU.
/// The production streamer reads via the O_DIRECT `backend::*` engine; this
/// is a plain-pread reference for the acceptance round-trip.
pub struct ExpertFileReader {
    index: ExpertIndex,
    spec: ExpertRecordSpec,
    layout: ExpertLayout,
    files: Vec<File>,
}

impl ExpertFileReader {
    pub fn open(dir: &Path) -> Result<Self> {
        let mp = dir.join(ExpertIndex::MANIFEST_NAME);
        let json =
            std::fs::read_to_string(&mp).with_context(|| format!("read {}", mp.display()))?;
        let index: ExpertIndex =
            serde_json::from_str(&json).with_context(|| format!("parse {}", mp.display()))?;
        if index.version != ExpertRecordHeader::VERSION {
            bail!(
                "manifest version {} != supported {}",
                index.version,
                ExpertRecordHeader::VERSION
            );
        }
        let spec = index.spec();
        let layout = index.layout();
        let mut files = Vec::with_capacity(index.num_moe_layers as usize);
        for l in 0..index.num_moe_layers {
            let p = dir.join(index.file_name(l));
            files.push(File::open(&p).with_context(|| format!("open {}", p.display()))?);
        }
        Ok(Self {
            index,
            spec,
            layout,
            files,
        })
    }

    pub fn index(&self) -> &ExpertIndex {
        &self.index
    }
    pub fn spec(&self) -> &ExpertRecordSpec {
        &self.spec
    }

    /// Read one record's raw `record_stride` bytes into a fresh buffer.
    pub fn read_record_raw(&self, key: ExpertKey) -> Result<Vec<u8>> {
        // Graceful Err on a bad layer (a direct Vec index would panic —
        // the sibling UmaArenaTier bounds-checks, so match it).
        if key.layer as usize >= self.files.len() {
            bail!(
                "ExpertFileReader: layer {} out of range ({} layer files)",
                key.layer,
                self.files.len()
            );
        }
        let mut buf = vec![0u8; self.layout.record_stride as usize];
        let off = self.layout.file_offset(key);
        pio::read_exact_at(&self.files[key.layer as usize], &mut buf, off)
            .with_context(|| format!("read record {:?} at {off}", key))?;
        Ok(buf)
    }
}

#[cfg(test)]
mod fs_tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "atlas-xpr-{}-{}-{}",
            tag,
            std::process::id(),
            // cheap unique-ish suffix without pulling in rand here
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    // Tiny synthetic model: 2 MoE layers, 3 experts, small dims.
    fn synth_index() -> ExpertIndex {
        // inter/hidden multiples of 32 so packed/scale byte counts are exact.
        ExpertIndex::new(64, 128, 16, 256, 4096, vec![0, 1], 3)
    }

    fn synth_projs(spec: &ExpertRecordSpec, seed: u8) -> [Vec<(Vec<u8>, Vec<u8>)>; 1] {
        let mut out = Vec::new();
        for p in Proj::ALL {
            let pb = spec.proj_bytes(p);
            let packed: Vec<u8> = (0..pb.packed_bytes)
                .map(|i| (i as u8).wrapping_add(seed).wrapping_add(p as u8))
                .collect();
            let scale: Vec<u8> = (0..pb.scale_bytes)
                .map(|i| (i as u8).wrapping_mul(3).wrapping_add(seed))
                .collect();
            out.push((packed, scale));
        }
        [out]
    }

    #[test]
    fn write_then_read_round_trips_bit_identical() {
        let dir = tmpdir("rt");
        let index = synth_index();
        let spec = index.spec();

        // Build expected payloads per (layer, expert).
        let mut expected = std::collections::HashMap::new();
        {
            let w = ExpertFileWriter::create(&dir, index.clone()).unwrap();
            for layer in 0..index.num_moe_layers {
                for expert in 0..index.num_experts {
                    let seed = (layer as u8) << 4 | expert as u8;
                    let raw = synth_projs(&spec, seed);
                    let projs = [
                        ProjData {
                            packed: &raw[0][0].0,
                            scale: &raw[0][0].1,
                        },
                        ProjData {
                            packed: &raw[0][1].0,
                            scale: &raw[0][1].1,
                        },
                        ProjData {
                            packed: &raw[0][2].0,
                            scale: &raw[0][2].1,
                        },
                    ];
                    let header = ExpertRecordHeader {
                        layer,
                        expert,
                        inter: index.inter as u32,
                        hidden: index.hidden as u32,
                        group_size: index.group_size as u32,
                        scale2: [seed as f32, seed as f32 + 0.5, seed as f32 + 1.0],
                        input_scale: [Some(1.0), None, Some(2.0)],
                    };
                    w.write_record(ExpertKey::new(layer, expert), &header, &projs)
                        .unwrap();
                    expected.insert((layer, expert), (raw, header));
                }
            }
            w.finish().unwrap();
        }

        // Read back and compare bit-for-bit.
        let r = ExpertFileReader::open(&dir).unwrap();
        assert_eq!(r.index(), &index);
        for layer in 0..index.num_moe_layers {
            for expert in 0..index.num_experts {
                let key = ExpertKey::new(layer, expert);
                let buf = r.read_record_raw(key).unwrap();
                let (hdr, views) = unpack_record(r.spec(), &buf).unwrap();
                let (raw, exp_hdr) = &expected[&(layer, expert)];
                assert_eq!(&hdr, exp_hdr, "header {:?}", key);
                for p in Proj::ALL {
                    assert_eq!(
                        views[p as usize].packed,
                        &raw[0][p as usize].0[..],
                        "packed {:?} {:?}",
                        key,
                        p
                    );
                    assert_eq!(
                        views[p as usize].scale,
                        &raw[0][p as usize].1[..],
                        "scale {:?} {:?}",
                        key,
                        p
                    );
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wrong_projection_length_errors() {
        let index = synth_index();
        let spec = index.spec();
        let header = ExpertRecordHeader {
            layer: 0,
            expert: 0,
            inter: index.inter as u32,
            hidden: index.hidden as u32,
            group_size: index.group_size as u32,
            scale2: [1.0; 3],
            input_scale: [Some(1.0); 3],
        };
        let bad = vec![0u8; 8]; // deliberately wrong length
        let ok_scale = vec![0u8; spec.proj_bytes(Proj::Gate).scale_bytes as usize];
        let projs = [
            ProjData {
                packed: &bad,
                scale: &ok_scale,
            },
            ProjData {
                packed: &bad,
                scale: &ok_scale,
            },
            ProjData {
                packed: &bad,
                scale: &ok_scale,
            },
        ];
        let err = pack_record(&spec, index.record_stride, &header, &projs);
        assert!(err.is_err(), "short packed buffer must error");
    }

    #[test]
    fn read_record_raw_rejects_out_of_range_layer() {
        let dir = tmpdir("oob");
        let index = synth_index(); // 2 MoE layers
        ExpertFileWriter::create(&dir, index)
            .unwrap()
            .finish()
            .unwrap();
        let r = ExpertFileReader::open(&dir).unwrap();
        // Valid layer is fine; an out-of-range layer is a graceful Err, not a panic.
        assert!(r.read_record_raw(ExpertKey::new(0, 0)).is_ok());
        assert!(r.read_record_raw(ExpertKey::new(99, 0)).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn manifest_geometry_round_trips_through_json() {
        let index = synth_index();
        let json = serde_json::to_string(&index).unwrap();
        let back: ExpertIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(index, back);
        // Derived geometry is stable across the JSON hop.
        assert_eq!(index.layout().record_stride, back.layout().record_stride);
        assert_eq!(index.spec().raw_bytes(), back.spec().raw_bytes());
    }
}
