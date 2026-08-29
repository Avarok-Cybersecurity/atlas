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

/// Fill as much of `buf` as the file holds, and succeed as long as the first
/// `needed` bytes arrived. Returns how many bytes were read.
///
/// This exists for O_DIRECT block reads at the tail of a file. A block read is
/// issued in whole blocks, but a file's length is whatever its content is: a
/// safetensors file is `8 + header + tensors` and nothing pads it to a block. So
/// the block covering a tensor's LAST rows runs past EOF, the kernel returns a
/// short read, and [`read_exact_at`] calls that an error -- failing a request
/// over bytes that were never part of any row. The rows themselves are always
/// inside the file; only the padding is not.
///
/// A short read before `needed` is still an error: that is a real truncation.
pub fn read_at_least_at(f: &File, buf: &mut [u8], offset: u64, needed: usize) -> io::Result<usize> {
    debug_assert!(needed <= buf.len());
    let (mut off, mut done) = (offset, 0usize);
    while done < buf.len() {
        let n = read_at(f, &mut buf[done..], off)?;
        // EOF. Under O_DIRECT this is the only way a block-aligned read comes up
        // short, and it is expected on the final block of a file.
        if n == 0 {
            break;
        }
        done += n;
        off += n as u64;
    }
    if done < needed {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "positional read got {done} bytes at offset {offset}, needed {needed} \
                 — the file is shorter than the data it is supposed to hold"
            ),
        ));
    }
    Ok(done)
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

    /// The tail case this exists for: a file whose length is NOT a multiple of
    /// the block size, read in whole blocks. `read_exact_at` calls that EOF and
    /// fails; `read_at_least_at` must succeed as long as the real bytes arrived,
    /// and must still fail when they did not.
    #[test]
    fn a_block_read_past_eof_succeeds_for_the_bytes_that_exist() {
        let dir = std::env::temp_dir().join(format!("atlas_pio_tail_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tail.bin");
        // 4096 + 288: exactly the shape of the shipped checkpoint's shards, which
        // end 288 bytes into their final block.
        let len = 4096usize + 288;
        std::fs::write(&path, vec![7u8; len]).unwrap();
        let f = std::fs::File::open(&path).unwrap();

        let mut buf = vec![0u8; 8192];
        // The row lives at 4096..4096+288 — inside the file, inside a block that
        // is not.
        let n = read_at_least_at(&f, &mut buf, 4096, 288).unwrap();
        assert_eq!(n, 288, "should read to EOF and no further");
        assert!(
            buf[..288].iter().all(|&b| b == 7),
            "the row's bytes must arrive"
        );

        // Demanding more than the file holds is a real truncation and must fail.
        let e = read_at_least_at(&f, &mut buf, 4096, 289).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);

        // And the strict reader still refuses the same read, which is the bug.
        assert!(read_exact_at(&f, &mut buf[..8192], 4096).is_err());

        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Round-trips at a non-zero offset on whichever platform runs the tests:
    // the point is that both arms agree on positional semantics, including
    // leaving the cursor untouched.
    #[test]
    fn round_trip_at_offset() {
        use std::io::{Seek, SeekFrom};

        let dir = std::env::temp_dir().join(format!("atlas_pio_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pio.bin");
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        f.seek(SeekFrom::Start(137)).unwrap();

        write_all_at(&f, &[0u8; 4096], 0).unwrap();
        assert_eq!(f.stream_position().unwrap(), 137);
        let payload: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        write_all_at(&f, &payload, 2048).unwrap();
        assert_eq!(f.stream_position().unwrap(), 137);

        let mut out = vec![0u8; payload.len()];
        read_exact_at(&f, &mut out, 2048).unwrap();
        assert_eq!(out, payload);
        assert_eq!(f.stream_position().unwrap(), 137);

        // Reading past the end must fail, not silently return short.
        let mut past = vec![0u8; 8192];
        assert!(read_exact_at(&f, &mut past, 4096).is_err());
        assert_eq!(f.stream_position().unwrap(), 137);

        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
