// SPDX-License-Identifier: AGPL-3.0-only
//
// On-disk layout for `--high-speed-swap`. One file per layer under
// `--high-speed-swap-dir`, pre-allocated so the filesystem reserves the bytes
// up-front (no surprise ENOSPC mid-decode).
//
// File names: `layer_{:05}.kv`. File contents are an opaque
// `GroupLayout`-defined stripe; the `Layout` type owns the open `File`s.
//
// PLATFORMS. On Linux the files are opened `O_DIRECT` (the io_uring / cuFile
// path needs it) and reserved with `posix_fallocate`. On Windows they are
// opened buffered and reserved with `set_len`:
//   * `FILE_FLAG_NO_BUFFERING` is the nearest O_DIRECT analogue but imposes
//     sector alignment on every buffer, offset and length; the tier's records
//     are 4 KiB-aligned but its bounce buffers are pinned host allocations
//     whose alignment is CUDA's to choose, so the flag would be unsafe to
//     assume. Buffered is correct, just not zero-copy.
//   * `SetFileValidData` is the true fallocate analogue but needs
//     SE_MANAGE_VOLUME_NAME; `set_len` reserves the range without it.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};

use crate::group::{GroupKey, GroupLayout};

pub struct Layout {
    pub dir: PathBuf,
    pub spec: GroupLayout,
    /// One `File` per layer. O_DIRECT on Linux for the io_uring / cuFile path;
    /// buffered elsewhere. Held as `File` rather than `OwnedFd` so the portable
    /// positional-I/O backend can use it on every platform.
    files: Vec<File>,
}

impl Layout {
    pub fn create(dir: &Path, spec: GroupLayout) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
        let mut files = Vec::with_capacity(spec.num_layers as usize);
        for layer in 0..spec.num_layers {
            let p = dir.join(format!("layer_{layer:05}.kv"));
            let mut opts = OpenOptions::new();
            opts.read(true).write(true).create(true).truncate(false);
            set_direct_flag(&mut opts);
            let f = opts
                .open(&p)
                .with_context(|| format!("open {}", p.display()))?;
            preallocate(&f, spec.bytes_per_layer())
                .with_context(|| format!("preallocate {}", p.display()))?;
            files.push(f);
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
            let mut opts = OpenOptions::new();
            opts.read(true).write(true);
            set_direct_flag(&mut opts);
            let f = opts
                .open(&p)
                .with_context(|| format!("open {}", p.display()))?;
            let len = f.metadata()?.len();
            if len < spec.bytes_per_layer() {
                anyhow::bail!(
                    "layer file {} is undersized: {} < {}",
                    p.display(),
                    len,
                    spec.bytes_per_layer()
                );
            }
            files.push(f);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            spec,
            files,
        })
    }

    /// Raw fd for the io_uring / cuFile paths, which are Linux-only.
    #[cfg(unix)]
    pub fn fd(&self, layer: u32) -> RawFd {
        self.files[layer as usize].as_raw_fd()
    }

    /// The layer file itself — what the portable positional-I/O backend uses,
    /// and the only accessor available on Windows.
    pub fn file(&self, layer: u32) -> &File {
        &self.files[layer as usize]
    }

    pub fn offset(&self, key: GroupKey) -> u64 {
        self.spec.file_offset(key)
    }

    pub fn group_bytes(&self) -> u64 {
        self.spec.group_bytes()
    }
}

/// O_DIRECT on Linux; nothing to set elsewhere (see the header note).
#[cfg(target_os = "linux")]
fn set_direct_flag(opts: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    opts.custom_flags(libc::O_DIRECT);
}

#[cfg(not(target_os = "linux"))]
fn set_direct_flag(_opts: &mut OpenOptions) {}

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

// Windows: `set_len` extends the file to the requested size. NTFS keeps the
// tail sparse until written, so this reserves the RANGE rather than the blocks
// -- weaker than posix_fallocate, and the honest trade for not requiring the
// SE_MANAGE_VOLUME privilege that SetFileValidData needs.
#[cfg(windows)]
fn preallocate(file: &File, size: u64) -> Result<()> {
    file.set_len(size)
        .with_context(|| format!("set_len({size})"))?;
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
