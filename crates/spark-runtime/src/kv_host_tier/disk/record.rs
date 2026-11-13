// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from LMCache (https://github.com/LMCache/LMCache), commit 3aefae1, Apache-2.0, file csrc/storage_backends/fs/connector.cpp (the O_DIRECT gate `try_enable_odirect` :50-70, the block size query :109-115, write to a temp name then atomic rename :220-240) and lmcache/v1/distributed/key_codec.py:100-142 (key to file name and back). Changes: rewritten in Rust; the record carries an Atlas header (model fingerprint, block key, tokens, checksums) so a restart scan can validate it without a manifest; a failed O_DIRECT open or transfer falls back to buffered I/O with one log line instead of an error, by Atlas contributors.

//! Responsibility: the on-disk record of one KV block of the disk tier (lane L27): its byte layout,
//! the checksums, the aligned bounce buffer, and the file I/O (temp file plus atomic rename,
//! O_DIRECT when buffer, size and offset are aligned, buffered I/O otherwise).
//!
//! # Layout (little endian)
//!
//! ```text
//! 0   magic        b"ATLASKVB"
//! 8   version      u32 (1)
//! 12  header_len   u32   bytes before the payload, a multiple of `align`
//! 16  namespace    u64   model fingerprint (`ssm_tier/fingerprint.rs`, opaque here)
//! 24  key          [u8; 16]
//! 40  payload_len  u64   the KV block bytes (all layers, K then V)
//! 48  payload_sum  u64   checksum of the payload
//! 56  n_tokens     u32   0 when the writer did not know the block's tokens
//! 60  align        u32   the alignment the file was written with
//! 64  created_ms   u64   unix milliseconds (diagnostic only)
//! 72  reserved     u64   0
//! 80  header_sum   u64   checksum of bytes 0..80 and the token words
//! 88  reserved     u64   0
//! 96  tokens       n_tokens x u32
//! ..  zero padding to header_len, then the payload, then zero padding to a multiple of `align`
//! ```
//!
//! A file whose size, header checksum or key does not match is corrupt and is removed by the
//! restart scan; the payload checksum is checked on every read (a torn write after a crash fails
//! it), so the scan never has to read payloads.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};

/// File magic.
pub const MAGIC: &[u8; 8] = b"ATLASKVB";
/// Record format version.
pub const VERSION: u32 = 1;
/// Bytes of the fixed header part (before the token words).
pub const FIXED: usize = 96;
/// Smallest alignment used (also the O_DIRECT floor on every Linux filesystem we target).
pub const MIN_ALIGN: usize = 4096;
/// Largest alignment accepted from a filesystem's preferred block size.
pub const MAX_ALIGN: usize = 1 << 20;
/// File name suffix of a published record.
pub const SUFFIX: &str = ".kvb";
/// File name suffix of a record being written.
pub const TMP_SUFFIX: &str = ".tmp";

/// Round `n` up to a multiple of `align` (a power of two).
pub fn round_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

/// A filesystem block size clamped to a power of two in `[MIN_ALIGN, MAX_ALIGN]`.
pub fn clamp_align(blksize: u64) -> usize {
    let b = (blksize as usize).clamp(MIN_ALIGN, MAX_ALIGN);
    if b.is_power_of_two() {
        b
    } else {
        b.next_power_of_two().min(MAX_ALIGN)
    }
}

/// Lower-case hex of a block key: the record's file stem.
pub fn key_hex(key: &[u8; 16]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(32);
    for b in key {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 15) as usize] as char);
    }
    s
}

/// Inverse of [`key_hex`]; `None` for anything that is not exactly 32 lower- or upper-case hex digits.
pub fn key_from_hex(s: &str) -> Option<[u8; 16]> {
    if s.len() != 32 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

const P1: u64 = 0x9E37_79B1_85EB_CA87;
const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;

fn mix(acc: u64, w: u64) -> u64 {
    (acc ^ w.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
}

/// A fast 64-bit integrity checksum (four independent lanes over 32-byte strides, a
/// splitmix64 finalizer). It detects torn and corrupted records; it is not a cryptographic hash
/// (the block key already is one). The two odd constants are the well-known xxHash64 primes.
pub fn checksum(bytes: &[u8]) -> u64 {
    let mut lanes = [P1, P2, P1 ^ P2, P1.wrapping_add(P2)];
    let mut chunks = bytes.chunks_exact(32);
    for c in &mut chunks {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let mut w = [0u8; 8];
            w.copy_from_slice(&c[i * 8..i * 8 + 8]);
            *lane = mix(*lane, u64::from_le_bytes(w));
        }
    }
    let mut h = lanes[0].rotate_left(1)
        ^ lanes[1].rotate_left(7)
        ^ lanes[2].rotate_left(12)
        ^ lanes[3].rotate_left(18);
    for c in chunks.remainder().chunks(8) {
        let mut w = [0u8; 8];
        w[..c.len()].copy_from_slice(c);
        h = mix(h, u64::from_le_bytes(w));
    }
    h ^= bytes.len() as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ (h >> 33)
}

/// A heap buffer whose usable part starts at an `align`-aligned address (safe code: an
/// over-allocated `Vec` and an offset), reused across records as the O_DIRECT bounce buffer.
pub struct AlignedBuf {
    raw: Vec<u8>,
    off: usize,
    len: usize,
    align: usize,
}

impl AlignedBuf {
    /// A zeroed buffer of `len` usable bytes aligned to `align` (a power of two).
    pub fn new(len: usize, align: usize) -> Self {
        let raw = vec![0u8; len + align];
        let addr = raw.as_ptr() as usize;
        let off = (align - addr % align) % align;
        Self {
            raw,
            off,
            len,
            align,
        }
    }

    /// Make room for `len` bytes at `align`, reallocating only when the buffer is too small or
    /// aligned differently. The contents are unspecified afterwards.
    pub fn ensure(&mut self, len: usize, align: usize) {
        if self.align != align || self.raw.len() < self.off + len {
            *self = Self::new(len, align);
        } else {
            self.len = len;
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.raw[self.off..self.off + self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }

    /// Whether the usable part may be handed to an O_DIRECT transfer of its full length.
    pub fn direct_ok(&self) -> bool {
        let addr = self.as_slice().as_ptr() as usize;
        addr.is_multiple_of(self.align) && self.len.is_multiple_of(self.align) && self.len > 0
    }
}

/// The parsed header of a record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub header_len: usize,
    pub namespace: u64,
    pub key: [u8; 16],
    pub payload_len: u64,
    pub payload_sum: u64,
    pub tokens: Vec<u32>,
    pub align: usize,
}

impl Header {
    /// Total file length of the record this header describes.
    pub fn file_len(&self) -> u64 {
        self.header_len as u64 + round_up(self.payload_len as usize, self.align) as u64
    }
}

/// File length of a record with `n_tokens` tokens and `payload_len` payload bytes.
pub fn record_len(n_tokens: usize, payload_len: usize, align: usize) -> usize {
    round_up(FIXED + 4 * n_tokens, align) + round_up(payload_len, align)
}

fn put_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

fn get_u32(b: &[u8], at: usize) -> u32 {
    let mut w = [0u8; 4];
    w.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(w)
}

fn get_u64(b: &[u8], at: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(w)
}

fn header_sum(fixed: &[u8], token_words: &[u8]) -> u64 {
    checksum(&fixed[..80]) ^ checksum(token_words).rotate_left(17)
}

/// Encode a full record (header, payload, padding) into `buf`; returns the record length.
pub fn encode(
    buf: &mut AlignedBuf,
    namespace: u64,
    key: &[u8; 16],
    tokens: &[u32],
    payload: &[u8],
    align: usize,
    created_ms: u64,
) -> usize {
    let header_len = round_up(FIXED + 4 * tokens.len(), align);
    let total = header_len + round_up(payload.len(), align);
    buf.ensure(total, align);
    let b = buf.as_mut_slice();
    b[..header_len].fill(0);
    b[..8].copy_from_slice(MAGIC);
    put_u32(b, 8, VERSION);
    put_u32(b, 12, header_len as u32);
    put_u64(b, 16, namespace);
    b[24..40].copy_from_slice(key);
    put_u64(b, 40, payload.len() as u64);
    put_u64(b, 48, checksum(payload));
    put_u32(b, 56, tokens.len() as u32);
    put_u32(b, 60, align as u32);
    put_u64(b, 64, created_ms);
    for (i, t) in tokens.iter().enumerate() {
        put_u32(b, FIXED + 4 * i, *t);
    }
    let sum = header_sum(&b[..FIXED], &b[FIXED..FIXED + 4 * tokens.len()]);
    put_u64(b, 80, sum);
    b[header_len..header_len + payload.len()].copy_from_slice(payload);
    b[header_len + payload.len()..total].fill(0);
    total
}

/// Bytes of the record start that [`parse_header`] needs: the fixed part, then (from it) the
/// whole header. Returns `None` while `bytes` is too short to say.
pub fn header_len_hint(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < FIXED {
        return None;
    }
    Some(get_u32(bytes, 12) as usize)
}

/// Parse and validate a header from the start of a record. Checks magic, version, the header
/// checksum and the internal lengths; never reads the payload.
pub fn parse_header(bytes: &[u8]) -> Result<Header> {
    if bytes.len() < FIXED {
        bail!(
            "record shorter than its fixed header ({} bytes)",
            bytes.len()
        );
    }
    if &bytes[..8] != MAGIC {
        bail!("bad magic");
    }
    let version = get_u32(bytes, 8);
    if version != VERSION {
        bail!("unsupported record version {version}");
    }
    let header_len = get_u32(bytes, 12) as usize;
    let n_tokens = get_u32(bytes, 56) as usize;
    let align = get_u32(bytes, 60) as usize;
    if !align.is_power_of_two() || !(MIN_ALIGN..=MAX_ALIGN).contains(&align) {
        bail!("bad alignment {align}");
    }
    if header_len != round_up(FIXED + 4 * n_tokens, align) {
        bail!("header length {header_len} does not match {n_tokens} tokens at alignment {align}");
    }
    if bytes.len() < FIXED + 4 * n_tokens {
        bail!("record shorter than its token list");
    }
    let token_words = &bytes[FIXED..FIXED + 4 * n_tokens];
    if header_sum(&bytes[..FIXED], token_words) != get_u64(bytes, 80) {
        bail!("header checksum mismatch");
    }
    let mut key = [0u8; 16];
    key.copy_from_slice(&bytes[24..40]);
    let tokens = (0..n_tokens).map(|i| get_u32(token_words, 4 * i)).collect();
    Ok(Header {
        header_len,
        namespace: get_u64(bytes, 16),
        key,
        payload_len: get_u64(bytes, 40),
        payload_sum: get_u64(bytes, 48),
        tokens,
        align,
    })
}

/// The payload of a full record whose header is `h`, checksum verified.
pub fn payload<'a>(h: &Header, record: &'a [u8]) -> Result<&'a [u8]> {
    let start = h.header_len;
    let end = start + h.payload_len as usize;
    if record.len() < end {
        bail!("record truncated: {} of {end} bytes", record.len());
    }
    let p = &record[start..end];
    if checksum(p) != h.payload_sum {
        bail!("payload checksum mismatch");
    }
    Ok(p)
}

/// Read only the header of the record at `path` (buffered; the restart scan's read).
pub fn read_header(path: &Path) -> Result<Header> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut fixed = vec![0u8; FIXED];
    f.read_exact(&mut fixed).context("read fixed header")?;
    let len = header_len_hint(&fixed).unwrap_or(FIXED);
    if !(FIXED..=MAX_ALIGN * 4).contains(&len) {
        bail!("implausible header length {len}");
    }
    let mut all = fixed;
    all.resize(len, 0);
    f.read_exact(&mut all[FIXED..])
        .context("read header tail")?;
    parse_header(&all)
}

/// Whether O_DIRECT is still tried. Starts at the configured value; the first O_DIRECT failure
/// that buffered I/O then survives turns it off for the life of the tier, with one log line.
pub struct IoMode {
    direct: AtomicBool,
    warned: AtomicBool,
}

impl IoMode {
    pub fn new(direct: bool) -> Self {
        Self {
            direct: AtomicBool::new(direct && cfg!(target_os = "linux")),
            warned: AtomicBool::new(false),
        }
    }

    pub fn direct(&self) -> bool {
        self.direct.load(Ordering::Relaxed)
    }

    fn fall_back(&self, what: &str, e: &std::io::Error) {
        self.direct.store(false, Ordering::Relaxed);
        if !self.warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "kv tier disk: O_DIRECT {what} failed ({e}); using buffered I/O from now on"
            );
        }
    }
}

fn open_opts(write: bool, direct: bool) -> OpenOptions {
    let mut o = OpenOptions::new();
    if write {
        o.write(true).create(true).truncate(true);
    } else {
        o.read(true);
    }
    #[cfg(target_os = "linux")]
    if direct {
        use std::os::unix::fs::OpenOptionsExt;
        o.custom_flags(libc::O_DIRECT);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = direct;
    o
}

fn write_file(path: &Path, data: &[u8], direct: bool) -> std::io::Result<()> {
    let mut f = open_opts(true, direct).open(path)?;
    f.write_all(data)
}

/// Write `buf` (a whole encoded record) to `tmp`, then rename it to `final_path`. O_DIRECT is used
/// when `io` allows it and the buffer qualifies; a failed O_DIRECT attempt is retried buffered. On
/// any error the temp file is removed and `final_path` is untouched.
pub fn write_atomic(tmp: &Path, final_path: &Path, buf: &AlignedBuf, io: &IoMode) -> Result<()> {
    let data = buf.as_slice();
    let res = if io.direct() && buf.direct_ok() {
        match write_file(tmp, data, true) {
            Ok(()) => Ok(()),
            Err(e) => {
                let retry = write_file(tmp, data, false);
                if retry.is_ok() {
                    io.fall_back("write", &e);
                }
                retry
            }
        }
    } else {
        write_file(tmp, data, false)
    };
    if let Err(e) = res {
        let _ = std::fs::remove_file(tmp);
        return Err(e).with_context(|| format!("write {}", tmp.display()));
    }
    if let Err(e) = std::fs::rename(tmp, final_path) {
        let _ = std::fs::remove_file(tmp);
        return Err(e)
            .with_context(|| format!("rename {} -> {}", tmp.display(), final_path.display()));
    }
    Ok(())
}

fn read_file(path: &Path, out: &mut [u8], direct: bool) -> std::io::Result<()> {
    let mut f = open_opts(false, direct).open(path)?;
    f.read_exact(out)
}

/// Read the whole record at `path` (`len` bytes, from its metadata) into `buf`, O_DIRECT when
/// allowed and aligned, buffered otherwise.
pub fn read_all(
    path: &Path,
    len: usize,
    align: usize,
    buf: &mut AlignedBuf,
    io: &IoMode,
) -> Result<()> {
    buf.ensure(len, align);
    let res = if io.direct() && buf.direct_ok() {
        match read_file(path, buf.as_mut_slice(), true) {
            Ok(()) => Ok(()),
            Err(e) => {
                let retry = read_file(path, buf.as_mut_slice(), false);
                if retry.is_ok() {
                    io.fall_back("read", &e);
                }
                retry
            }
        }
    } else {
        read_file(path, buf.as_mut_slice(), false)
    };
    res.with_context(|| format!("read {}", path.display()))
}
