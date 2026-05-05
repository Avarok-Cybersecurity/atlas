// SPDX-License-Identifier: AGPL-3.0-only

//! File-backed KV cache spill manager.
//!
//! Manages swap files for sequence-level KV cache + SSM state overflow.
//! The serialization format is owned by the Model; this module handles
//! file lifecycle and disk space enforcement.

use anyhow::Result;
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

/// Manages swap file creation, opening, deletion, and space limits.
pub struct KvSpillManager {
    spill_dir: PathBuf,
    next_id: u64,
    max_bytes: u64,
    used_bytes: u64,
}

impl KvSpillManager {
    /// Create a new spill manager rooted at `spill_dir` with a byte budget.
    ///
    /// Creates the directory if it doesn't exist. Removes any stale files
    /// from prior runs.
    pub fn new(spill_dir: PathBuf, max_bytes: u64) -> Result<Self> {
        if spill_dir.exists() {
            // Clean stale swap files from prior runs.
            for entry in fs::read_dir(&spill_dir)? {
                let entry = entry?;
                if entry.file_name().to_string_lossy().starts_with("swap_") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        } else {
            fs::create_dir_all(&spill_dir)?;
        }
        Ok(Self {
            spill_dir,
            next_id: 0,
            max_bytes,
            used_bytes: 0,
        })
    }

    /// Create a new swap file. Returns `(id, buffered_writer)`.
    pub fn create_file(&mut self) -> Result<(u64, BufWriter<fs::File>)> {
        let id = self.next_id;
        self.next_id += 1;
        let path = self.file_path(id);
        let file = fs::File::create(&path)?;
        Ok((id, BufWriter::new(file)))
    }

    /// Open an existing swap file for reading.
    pub fn open_file(&self, id: u64) -> Result<BufReader<fs::File>> {
        let path = self.file_path(id);
        let file = fs::File::open(&path)?;
        Ok(BufReader::new(file))
    }

    /// Remove a swap file and reclaim its disk usage.
    pub fn remove_file(&mut self, id: u64) -> Result<()> {
        let path = self.file_path(id);
        let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        fs::remove_file(&path)?;
        self.used_bytes = self.used_bytes.saturating_sub(size);
        Ok(())
    }

    /// Record bytes written to a swap file (call after flush).
    pub fn record_usage(&mut self, id: u64) {
        let path = self.file_path(id);
        if let Ok(meta) = fs::metadata(&path) {
            self.used_bytes += meta.len();
        }
    }

    /// Check if `estimated_bytes` can fit within the space budget.
    pub fn has_space(&self, estimated_bytes: u64) -> bool {
        self.used_bytes + estimated_bytes <= self.max_bytes
    }

    /// Current disk usage in bytes.
    pub fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    fn file_path(&self, id: u64) -> PathBuf {
        self.spill_dir.join(format!("swap_{id}.bin"))
    }
}

impl Drop for KvSpillManager {
    fn drop(&mut self) {
        // Best-effort cleanup of remaining swap files.
        if let Ok(entries) = fs::read_dir(&self.spill_dir) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("swap_") {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "atlas_spill_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn test_create_open_remove_lifecycle() {
        let dir = temp_dir();
        let mut mgr = KvSpillManager::new(dir.clone(), 1024 * 1024).unwrap();

        // Create and write data.
        let (id, mut writer) = mgr.create_file().unwrap();
        writer.write_all(&[1u8; 256]).unwrap();
        writer.flush().unwrap();
        drop(writer);
        mgr.record_usage(id);
        assert_eq!(mgr.used_bytes(), 256);

        // Open and read back.
        let mut reader = mgr.open_file(id).unwrap();
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut buf).unwrap();
        assert_eq!(buf.len(), 256);
        assert!(buf.iter().all(|&b| b == 1));

        // Remove.
        mgr.remove_file(id).unwrap();
        assert_eq!(mgr.used_bytes(), 0);
        assert!(!dir.join("swap_0.bin").exists());

        // Cleanup test dir.
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_has_space() {
        let dir = temp_dir();
        let mut mgr = KvSpillManager::new(dir.clone(), 512).unwrap();

        assert!(mgr.has_space(512));
        assert!(!mgr.has_space(513));

        let (id, mut writer) = mgr.create_file().unwrap();
        writer.write_all(&[0u8; 256]).unwrap();
        writer.flush().unwrap();
        drop(writer);
        mgr.record_usage(id);

        assert!(mgr.has_space(256));
        assert!(!mgr.has_space(257));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_stale_cleanup_on_new() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).unwrap();

        // Create a stale swap file.
        fs::write(dir.join("swap_99.bin"), [0u8; 64]).unwrap();
        assert!(dir.join("swap_99.bin").exists());

        // Creating a new manager should clean it up.
        let _mgr = KvSpillManager::new(dir.clone(), 1024).unwrap();
        assert!(!dir.join("swap_99.bin").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_drop_cleanup() {
        let dir = temp_dir();
        {
            let mut mgr = KvSpillManager::new(dir.clone(), 1024).unwrap();
            let (_, mut writer) = mgr.create_file().unwrap();
            writer.write_all(&[0u8; 32]).unwrap();
            writer.flush().unwrap();
            drop(writer);
            // mgr drops here.
        }
        // File should be cleaned up by Drop.
        assert!(!dir.join("swap_0.bin").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sequential_ids() {
        let dir = temp_dir();
        let mut mgr = KvSpillManager::new(dir.clone(), 1024 * 1024).unwrap();

        let (id0, w0) = mgr.create_file().unwrap();
        drop(w0);
        let (id1, w1) = mgr.create_file().unwrap();
        drop(w1);

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);

        let _ = fs::remove_dir_all(&dir);
    }
}
