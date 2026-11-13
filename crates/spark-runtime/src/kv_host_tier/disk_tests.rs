// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the disk KV tier (lane L27) on temporary directories: roundtrip,
//! warm restart and adoption, crash leftovers, corruption, fingerprint isolation, the byte budget
//! and its LRU order across a restart, token checks, the directory lock, validation and metrics.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use super::*;

/// Payload bytes of one test block; deliberately not a multiple of 4096.
const BB: usize = 10_000;
const NS: u64 = 0x5eed_0000_0000_0001;

fn cfg(dir: &Path, budget: u64) -> DiskConfig {
    DiskConfig {
        dir: dir.to_path_buf(),
        budget_bytes: budget,
        direct_io: true,
    }
}

fn block(seed: u8) -> Vec<u8> {
    (0..BB)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

fn key(n: u8) -> BlockKey {
    let mut k = [n; 16];
    k[15] = n.wrapping_mul(7);
    BlockKey(k)
}

/// File length of one token-less test record in `dir`.
fn rec_len(dir: &Path) -> u64 {
    record::record_len(0, BB, dir_align(dir)) as u64
}

fn open(dir: &Path, budget: u64) -> DiskTier {
    DiskTier::open(&cfg(dir, budget), BB).expect("open disk tier")
}

fn file_of(dir: &Path, k: &BlockKey) -> std::path::PathBuf {
    dir.join(format!("{}{}", record::key_hex(&k.0), record::SUFFIX))
}

fn promote(t: &DiskTier, k: &BlockKey) -> Option<Vec<u8>> {
    let mut out = vec![0u8; BB];
    t.promote(k, &mut out).expect("promote").then_some(out)
}

#[test]
fn roundtrip_then_warm_restart_adopts_and_promotes() {
    let d = tempfile::tempdir().unwrap();
    {
        let t = open(d.path(), 64 * GIB);
        assert!(t.adopt_on_restart(NS).unwrap().is_empty());
        let cs: Arc<dyn ColdStore> = Arc::new(t);
        cs.demote(&key(1), &block(1)).unwrap();
        cs.demote(&key(2), &block(2)).unwrap();
        let mut out = vec![0u8; BB];
        assert!(cs.promote(&key(1), &mut out).unwrap());
        assert_eq!(out, block(1));
        assert!(!cs.promote(&key(9), &mut out).unwrap(), "never demoted");
    }
    // The "restart": a new tier over the same directory.
    let t = open(d.path(), 64 * GIB);
    let mut adopted = t.adopt_on_restart(NS).unwrap();
    adopted.sort_by_key(|k| k.0);
    let mut want = [key(1), key(2)];
    want.sort_by_key(|k| k.0);
    assert_eq!(
        adopted.iter().map(|k| k.0).collect::<Vec<_>>(),
        want.iter().map(|k| k.0).collect::<Vec<_>>()
    );
    assert_eq!(promote(&t, &key(2)).unwrap(), block(2));
    let s = t.stats();
    assert_eq!((s.adopted, s.hits, s.files), (2, 1, 2));
    assert_eq!(s.bytes, 2 * rec_len(d.path()));
}

#[test]
fn another_fingerprint_is_neither_adopted_nor_promoted() {
    let d = tempfile::tempdir().unwrap();
    {
        let t = open(d.path(), 64 * GIB);
        t.adopt_on_restart(NS).unwrap();
        t.demote(&key(1), &block(1)).unwrap();
    }
    {
        let t = open(d.path(), 64 * GIB);
        assert!(t.adopt_on_restart(NS + 1).unwrap().is_empty());
        assert!(promote(&t, &key(1)).is_none());
        assert!(
            file_of(d.path(), &key(1)).exists(),
            "a foreign record is kept, not deleted"
        );
    }
    let t = open(d.path(), 64 * GIB);
    assert_eq!(t.adopt_on_restart(NS).unwrap().len(), 1);
    assert_eq!(promote(&t, &key(1)).unwrap(), block(1));
}

#[test]
fn crash_leftovers_are_removed_on_open_and_foreign_files_kept() {
    let d = tempfile::tempdir().unwrap();
    {
        let t = open(d.path(), 64 * GIB);
        t.adopt_on_restart(NS).unwrap();
        t.demote(&key(1), &block(1)).unwrap();
        t.demote(&key(2), &block(2)).unwrap();
    }
    // A writer that died before its rename, a record cut short, and a file that is not ours.
    let tmp = d.path().join(format!(
        "{}.4242.0{}",
        record::key_hex(&key(3).0),
        record::TMP_SUFFIX
    ));
    File::create(&tmp)
        .unwrap()
        .write_all(b"half a record")
        .unwrap();
    let cut = file_of(d.path(), &key(2));
    let len = std::fs::metadata(&cut).unwrap().len();
    File::options()
        .write(true)
        .open(&cut)
        .unwrap()
        .set_len(len - 100)
        .unwrap();
    let foreign = d.path().join("notes.txt");
    std::fs::write(&foreign, b"keep me").unwrap();

    let t = open(d.path(), 64 * GIB);
    assert!(!tmp.exists(), "stray temp file removed");
    assert!(!cut.exists(), "truncated record removed");
    assert!(
        foreign.exists(),
        "files that are not records are left alone"
    );
    let s = t.stats();
    assert_eq!((s.stray_tmp, s.err_corrupt, s.files), (1, 1, 1));
    assert_eq!(t.adopt_on_restart(NS).unwrap().len(), 1);
    assert_eq!(promote(&t, &key(1)).unwrap(), block(1));
}

#[test]
fn a_corrupt_header_is_removed_by_the_scan() {
    let d = tempfile::tempdir().unwrap();
    {
        let t = open(d.path(), 64 * GIB);
        t.adopt_on_restart(NS).unwrap();
        t.demote(&key(1), &block(1)).unwrap();
    }
    let p = file_of(d.path(), &key(1));
    flip_byte(&p, 17); // inside the namespace field
    let t = open(d.path(), 64 * GIB);
    assert!(!p.exists());
    assert_eq!(t.stats().err_corrupt, 1);
    assert!(t.adopt_on_restart(NS).unwrap().is_empty());
}

fn flip_byte(p: &Path, at: u64) {
    let mut f = File::options().read(true).write(true).open(p).unwrap();
    f.seek(SeekFrom::Start(at)).unwrap();
    let mut b = [0u8; 1];
    f.read_exact(&mut b).unwrap();
    f.seek(SeekFrom::Start(at)).unwrap();
    f.write_all(&[b[0] ^ 0x40]).unwrap();
}

#[test]
fn a_corrupt_payload_is_a_miss_and_the_record_is_dropped() {
    let d = tempfile::tempdir().unwrap();
    let t = open(d.path(), 64 * GIB);
    t.adopt_on_restart(NS).unwrap();
    t.demote(&key(1), &block(1)).unwrap();
    let p = file_of(d.path(), &key(1));
    let header_len = record::read_header(&p).unwrap().header_len as u64;
    flip_byte(&p, header_len + 5);
    assert!(promote(&t, &key(1)).is_none());
    assert!(!p.exists());
    let s = t.stats();
    assert_eq!((s.err_corrupt, s.misses, s.files, s.bytes), (1, 1, 0, 0));
    // Demoting again works.
    t.demote(&key(1), &block(1)).unwrap();
    assert_eq!(promote(&t, &key(1)).unwrap(), block(1));
}

#[test]
fn the_budget_evicts_least_recently_used_first() {
    let d = tempfile::tempdir().unwrap();
    let t = open(d.path(), 2 * rec_len(d.path()));
    t.adopt_on_restart(NS).unwrap();
    t.demote(&key(1), &block(1)).unwrap();
    t.demote(&key(2), &block(2)).unwrap();
    assert!(promote(&t, &key(1)).is_some()); // key 1 is now the most recent
    t.demote(&key(3), &block(3)).unwrap();
    assert!(
        promote(&t, &key(2)).is_none(),
        "least recently used evicted"
    );
    assert!(!file_of(d.path(), &key(2)).exists());
    assert_eq!(promote(&t, &key(1)).unwrap(), block(1));
    assert_eq!(promote(&t, &key(3)).unwrap(), block(3));
    let s = t.stats();
    assert_eq!((s.evictions, s.files), (1, 2));
    assert!(s.bytes <= t.budget_bytes());
    // A block larger than the whole budget is skipped, not written.
    let small = tempfile::tempdir().unwrap();
    let tiny = open(small.path(), 1024);
    tiny.adopt_on_restart(NS).unwrap();
    tiny.demote(&key(1), &block(1)).unwrap();
    assert_eq!((tiny.stats().skipped, tiny.stats().files), (1, 0));
}

#[test]
fn the_lru_order_survives_a_restart_through_mtimes() {
    let d = tempfile::tempdir().unwrap();
    {
        let t = open(d.path(), 64 * GIB);
        t.adopt_on_restart(NS).unwrap();
        t.demote(&key(1), &block(1)).unwrap();
        t.demote(&key(2), &block(2)).unwrap();
    }
    let now = SystemTime::now();
    File::open(file_of(d.path(), &key(1)))
        .unwrap()
        .set_modified(now)
        .unwrap();
    File::open(file_of(d.path(), &key(2)))
        .unwrap()
        .set_modified(now - Duration::from_secs(100))
        .unwrap();
    let t = open(d.path(), 2 * rec_len(d.path()));
    t.adopt_on_restart(NS).unwrap();
    t.demote(&key(3), &block(3)).unwrap();
    assert!(promote(&t, &key(2)).is_none(), "the older mtime goes first");
    assert!(promote(&t, &key(1)).is_some());
}

#[test]
fn a_smaller_budget_on_restart_trims_at_open() {
    let d = tempfile::tempdir().unwrap();
    {
        let t = open(d.path(), 64 * GIB);
        t.adopt_on_restart(NS).unwrap();
        for i in 1..=4 {
            t.demote(&key(i), &block(i)).unwrap();
        }
    }
    let t = open(d.path(), 2 * rec_len(d.path()));
    assert_eq!(t.usage().0, 2);
    assert_eq!(t.stats().evictions, 2);
}

#[test]
fn stored_tokens_must_match_the_probe() {
    let d = tempfile::tempdir().unwrap();
    let t = open(d.path(), 64 * GIB);
    t.adopt_on_restart(NS).unwrap();
    let toks: Vec<u32> = (0..64).collect();
    t.demote_with_tokens(&key(1), &toks, &block(1)).unwrap();
    let mut out = vec![0u8; BB];
    let mut other = toks.clone();
    other[10] = 99_999;
    assert!(
        !t.promote_with_tokens(&key(1), &other, &mut out).unwrap(),
        "same key, other tokens: miss"
    );
    assert_eq!(t.stats().err_corrupt, 1);
    t.demote_with_tokens(&key(1), &toks, &block(1)).unwrap();
    assert!(t.promote_with_tokens(&key(1), &toks, &mut out).unwrap());
    assert_eq!(out, block(1));
    // A record written without tokens is validated by key and checksums alone.
    t.demote(&key(2), &block(2)).unwrap();
    assert!(t.promote_with_tokens(&key(2), &toks, &mut out).unwrap());
    // Through the trait object, as the host tier calls it.
    let cs: &dyn ColdStore = &t;
    assert!(cs.promote_with_tokens(&key(1), &toks, &mut out).unwrap());
}

#[test]
fn wrong_sizes_and_unnamed_fingerprints_are_refused() {
    let d = tempfile::tempdir().unwrap();
    let t = open(d.path(), 64 * GIB);
    assert!(
        t.demote(&key(1), &block(1)).is_err(),
        "no fingerprint named yet"
    );
    t.adopt_on_restart(NS).unwrap();
    assert!(t.demote(&key(1), &[0u8; 17]).is_err(), "wrong block size");
    t.demote(&key(1), &block(1)).unwrap();
    let mut small = vec![0u8; BB - 1];
    assert!(
        !t.promote(&key(1), &mut small).unwrap(),
        "wrong destination size is a miss"
    );
    assert!(DiskTier::open(&cfg(d.path(), GIB), 0).is_err());
}

#[test]
fn demoting_the_same_key_twice_touches_instead_of_rewriting() {
    let d = tempfile::tempdir().unwrap();
    let t = open(d.path(), 64 * GIB);
    t.adopt_on_restart(NS).unwrap();
    t.demote(&key(1), &block(1)).unwrap();
    t.demote(&key(1), &block(1)).unwrap();
    let s = t.stats();
    assert_eq!((s.writes, s.skipped, s.files), (1, 1, 1));
}

#[test]
fn buffered_mode_roundtrips() {
    let d = tempfile::tempdir().unwrap();
    let mut c = cfg(d.path(), 64 * GIB);
    c.direct_io = false;
    let t = DiskTier::open(&c, BB).unwrap();
    assert_eq!(t.stats().direct_io, 0);
    t.adopt_on_restart(NS).unwrap();
    t.demote(&key(5), &block(5)).unwrap();
    assert_eq!(promote(&t, &key(5)).unwrap(), block(5));
}

#[test]
fn concurrent_demote_and_promote() {
    let d = tempfile::tempdir().unwrap();
    let t = Arc::new(open(d.path(), 64 * GIB));
    t.adopt_on_restart(NS).unwrap();
    let handles: Vec<_> = (0..4u8)
        .map(|w| {
            let t = t.clone();
            std::thread::spawn(move || {
                for i in 0..8u8 {
                    let n = w * 16 + i;
                    t.demote(&key(n), &block(n)).unwrap();
                    assert_eq!(promote(&t, &key(n)).unwrap(), block(n));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(t.usage().0, 32);
}

#[cfg(unix)]
#[test]
fn a_second_tier_on_the_same_directory_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let first = open(d.path(), GIB);
    let second = DiskTier::open(&cfg(d.path(), GIB), BB);
    assert!(second.is_err());
    drop(first);
    assert!(
        DiskTier::open(&cfg(d.path(), GIB), BB).is_ok(),
        "lock released on drop"
    );
}

#[test]
fn validation_rules() {
    let d = tempfile::tempdir().unwrap();
    let dir = Some(d.path());
    assert_eq!(
        validate(false, dir, Some(4), true).unwrap(),
        None,
        "row off: nothing, even under --hermetic"
    );
    assert!(
        validate(true, None, Some(4), false).is_err(),
        "needs a directory"
    );
    assert!(validate(true, dir, None, false).is_err(), "needs a budget");
    assert!(
        validate(true, dir, Some(0), false).is_err(),
        "needs a non-zero budget"
    );
    let e = validate(true, dir, Some(4), true).unwrap_err().to_string();
    assert!(e.contains("--hermetic"), "{e}");
    let file = d.path().join("f");
    std::fs::write(&file, b"x").unwrap();
    assert!(
        validate(true, Some(&file), Some(4), false).is_err(),
        "not a directory"
    );
    let ok = validate(true, dir, Some(4), false).unwrap().unwrap();
    assert_eq!(ok.budget_bytes, 4 * GIB);
    assert_eq!(ok.dir, d.path());
}

#[test]
fn records_encode_parse_and_detect_damage() {
    let align = record::MIN_ALIGN;
    let payload = block(7);
    let toks = [3u32, 1, 4, 1, 5];
    let mut buf = record::AlignedBuf::new(0, align);
    let len = record::encode(&mut buf, NS, &key(7).0, &toks, &payload, align, 123);
    assert_eq!(len, record::record_len(toks.len(), payload.len(), align));
    assert!(buf.direct_ok());
    let rec = buf.as_slice().to_vec();
    let h = record::parse_header(&rec).unwrap();
    assert_eq!(
        (h.namespace, h.key, h.payload_len),
        (NS, key(7).0, BB as u64)
    );
    assert_eq!(h.tokens, toks);
    assert_eq!(h.file_len(), len as u64);
    assert_eq!(record::payload(&h, &rec).unwrap(), &payload[..]);
    let mut bad = rec.clone();
    bad[h.header_len + 3] ^= 1;
    assert!(record::payload(&h, &bad).is_err(), "payload flip");
    for at in [0usize, 9, 20, 30, 45, 57, 100] {
        let mut bad = rec.clone();
        bad[at] ^= 1;
        assert!(record::parse_header(&bad).is_err(), "header flip at {at}");
    }
}

#[test]
fn checksums_see_every_single_bit_flip_sampled() {
    let base = block(3);
    let c0 = record::checksum(&base);
    for at in (0..BB).step_by(97) {
        for bit in [0u8, 3, 7] {
            let mut b = base.clone();
            b[at] ^= 1 << bit;
            assert_ne!(record::checksum(&b), c0, "byte {at} bit {bit}");
        }
    }
    assert_ne!(
        record::checksum(&base[..BB - 1]),
        c0,
        "length is part of the sum"
    );
}

#[test]
fn key_names_roundtrip() {
    let k = key(0xab).0;
    let hex = record::key_hex(&k);
    assert_eq!(hex.len(), 32);
    assert_eq!(record::key_from_hex(&hex), Some(k));
    assert_eq!(record::key_from_hex("LOCK"), None);
    assert_eq!(record::key_from_hex(&"g".repeat(32)), None);
    assert_eq!(record::clamp_align(512), 4096);
    assert_eq!(record::clamp_align(65536), 65536);
    assert_eq!(record::clamp_align(5000), 8192);
}

#[test]
fn metrics_text_names_every_series() {
    let d = tempfile::tempdir().unwrap();
    let _t = open(d.path(), GIB);
    let mut s = String::new();
    render_prometheus(&mut s);
    for name in [
        "atlas_kv_tier_disk_bytes",
        "atlas_kv_tier_disk_capacity_bytes",
        "atlas_kv_tier_disk_hits_total",
        "atlas_kv_tier_disk_adopted_total",
        "atlas_kv_tier_disk_evictions_total",
        "atlas_kv_tier_disk_errors_total{kind=\"corrupt\"}",
    ] {
        assert!(s.contains(name), "{name} missing from:\n{s}");
    }
    assert!(s.contains("# TYPE atlas_kv_tier_disk_bytes gauge"));
    assert!(s.contains("# TYPE atlas_kv_tier_disk_hits_total counter"));
}
