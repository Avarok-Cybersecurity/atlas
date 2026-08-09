// SPDX-License-Identifier: AGPL-3.0-only

//! Positional file I/O, one implementation per platform.
//!
//! `pread`/`pwrite` (unix) and `seek_read`/`seek_write` (Windows) are both
//! positional — they take an explicit offset and do NOT move the file cursor —
//! which is what every tier here relies on to share one `File` across
//! concurrent record accesses. Only the syscall differs, so only the syscall
//! lives here; bounds checks and record semantics stay with the callers.
//!
//! This exists because the same six lines were about to be written a third
//! time (`atlas-tier::direct_swap`, `spark-model`'s snapshot arena, and
//! `spark-storage`'s file backend). `atlas-tier` is the crate all three
//! already depend on.
//!
//! Both platforms may transfer fewer bytes than requested, so both loop.

use std::fs::File;
use std::io;

/// Write all of `buf` at `offset`, looping over short writes.
pub fn write_all_at(f: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    let (mut off, mut done) = (offset, 0usize);
    while done < buf.len() {
        let n = write_at(f, &buf[done..], off)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                format!("positional write returned 0 bytes at offset {off}"),
            ));
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

/// Fill `buf` from `offset`, looping over short reads. A zero-length read
/// before `buf` is full is EOF and is reported as an error rather than
/// leaving the tail of the buffer holding stale bytes.
pub fn read_exact_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    let (mut off, mut done) = (offset, 0usize);
    let total = buf.len();
    while done < total {
        let n = read_at(f, &mut buf[done..], off)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("positional read hit EOF after {done} of {total} bytes at offset {off}"),
            ));
        }
        done += n;
        off += n as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn write_at(f: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.write_at(buf, offset)
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, offset)
}

#[cfg(windows)]
fn write_at(f: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_write(buf, offset)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round-trips at a non-zero offset on whichever platform runs the tests:
    // the point is that both arms agree on positional semantics, including
    // leaving the cursor untouched.
    #[test]
    fn round_trip_at_offset() {
        let dir = std::env::temp_dir().join(format!("atlas_pio_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pio.bin");
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();

        write_all_at(&f, &[0u8; 4096], 0).unwrap();
        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        write_all_at(&f, &payload, 2048).unwrap();

        let mut out = vec![0u8; payload.len()];
        read_exact_at(&f, &mut out, 2048).unwrap();
        assert_eq!(out, payload);

        // Reading past the end must fail, not silently return short.
        let mut past = vec![0u8; 8192];
        assert!(read_exact_at(&f, &mut past, 4096).is_err());

        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
