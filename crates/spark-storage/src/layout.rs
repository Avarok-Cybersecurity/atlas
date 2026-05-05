// SPDX-License-Identifier: AGPL-3.0-only
//
// On-disk layout for `--high-speed-swap`. One file per layer under
// `--high-speed-swap-dir`, pre-allocated via `posix_fallocate` so the
// filesystem reserves the bytes up-front (no surprise ENOSPC mid-decode).
//
// File names: `layer_{:05}.kv`. File contents are an opaque
// `GroupLayout`-defined stripe; the `Layout` type owns the open `File`s plus
// `O_DIRECT` fds for the I/O backends.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use crate::group::{GroupKey, GroupLayout};

pub struct Layout {
    pub dir: PathBuf,
    pub spec: GroupLayout,
    /// One `File` per layer, opened with O_DIRECT for the io_uring / cuFile path.
    files: Vec<OwnedFd>,
}

impl Layout {
    pub fn create(dir: &Path, spec: GroupLayout) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
        let mut files = Vec::with_capacity(spec.num_layers as usize);
        for layer in 0..spec.num_layers {
            let p = dir.join(format!("layer_{layer:05}.kv"));
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .custom_flags(libc::O_DIRECT)
                .open(&p)
                .with_context(|| format!("open O_DIRECT {}", p.display()))?;
            preallocate(&f, spec.bytes_per_layer())
                .with_context(|| format!("fallocate {}", p.display()))?;
            files.push(f.into());
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            spec,
            files,
        })
    }

    /// Open an existing layout (panics if a file is missing or undersized).
    pub fn open(dir: &Path, spec: GroupLayout) -> Result<Self> {
        let mut files = Vec::with_capacity(spec.num_layers as usize);
        for layer in 0..spec.num_layers {
            let p = dir.join(format!("layer_{layer:05}.kv"));
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .custom_flags(libc::O_DIRECT)
                .open(&p)
                .with_context(|| format!("open O_DIRECT {}", p.display()))?;
            let len = f.metadata()?.len();
            if len < spec.bytes_per_layer() {
                anyhow::bail!(
                    "layer file {} is undersized: {} < {}",
                    p.display(),
                    len,
                    spec.bytes_per_layer()
                );
            }
            files.push(f.into());
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            spec,
            files,
        })
    }

    pub fn fd(&self, layer: u32) -> RawFd {
        self.files[layer as usize].as_raw_fd()
    }

    pub fn offset(&self, key: GroupKey) -> u64 {
        self.spec.file_offset(key)
    }

    pub fn group_bytes(&self) -> u64 {
        self.spec.group_bytes()
    }
}

#[cfg(unix)]
fn preallocate(file: &File, size: u64) -> Result<()> {
    // posix_fallocate is portable across ext4/xfs and reserves space without
    // writing zeros; FALLOC_FL_KEEP_SIZE would be wrong here because we *do*
    // want the file size to grow.
    let fd = file.as_raw_fd();
    let res = unsafe { libc::posix_fallocate(fd, 0, size as libc::off_t) };
    if res != 0 {
        anyhow::bail!("posix_fallocate({size}) failed: {res}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{GroupKey, KvKind};

    #[test]
    fn create_open_round_trip() {
        let tmp = tempdir();
        let spec = GroupLayout::new(2, 4, 2, 16, 128, 2, 4096);
        {
            let l = Layout::create(&tmp, spec).unwrap();
            assert_eq!(l.spec.num_layers, 2);
            // File should be size bytes_per_layer.
            let p = tmp.join("layer_00000.kv");
            let len = std::fs::metadata(&p).unwrap().len();
            assert_eq!(len, spec.bytes_per_layer());
        }
        {
            let l = Layout::open(&tmp, spec).unwrap();
            let off = l.offset(GroupKey::new(0, 1, 1, KvKind::V));
            assert_eq!(off, spec.file_offset(GroupKey::new(0, 1, 1, KvKind::V)));
        }
        std::fs::remove_dir_all(&tmp).ok();
    }

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("atlas-storage-test-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
