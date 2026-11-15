# Disk KV tier, warm restart and asynchronous restore: LMCache attribution

Lane L27 of the consumer-card port (`port/consumer-opt`). Two rows, both off on every
target: `kv_tier_disk` (`AVAROK_KV_TIER_DISK`) and `kv_restore_async` (`AVAROK_KV_RESTORE_ASYNC`).

## Upstream

- Project: LMCache, https://github.com/LMCache/LMCache, commit `3aefae1`.
- License: Apache License, Version 2.0 (text in `LICENSES/Apache-2.0.txt`). LMCache ships no NOTICE
  file; the `NOTICE` entry names the project, the commit and the files below.
- Copyright: the LMCache contributors.

## What was taken, and how

| Upstream file and lines | Kind | Atlas file |
|---|---|---|
| `csrc/storage_backends/fs/connector.cpp:50-70` (`try_enable_odirect`: O_DIRECT only when the length is a multiple of the disk block size and the buffer address is aligned) | translation | `crates/spark-runtime/src/kv_host_tier/disk/record.rs` (`AlignedBuf::direct_ok`, `write_atomic`, `read_all`) |
| `csrc/storage_backends/fs/connector.cpp:109-115` (disk block size from `statvfs`) | translation | `record.rs` `clamp_align` plus `disk.rs` `dir_align` (`st_blksize` of the directory, clamped to 4096..1 MiB) |
| `csrc/storage_backends/fs/connector.cpp:220-240` (write to a temp path, rename to the final name, remove the temp file on failure) | translation | `record.rs` `write_atomic` |
| `lmcache/v1/distributed/key_codec.py:100-142` (storage key to file name and back, rejecting bad suffixes and fields) | translation | `record.rs` `key_hex`, `key_from_hex`; `disk.rs` scan |
| `lmcache/integration/vllm/lmcache_mp_connector.py:1360-1420` (load between scheduler steps; a failed asynchronous load is bypassed until the request is admitted for local computation) | design only, no code | `crates/spark-server/src/scheduler/restore_queue.rs` |

## Changes for Atlas

- Rewritten in Rust; no C++, Python or build glue is carried over.
- Atlas record format: a fixed little-endian header (magic, version, model fingerprint, 128-bit
  block key, payload length and checksum, optional token list, alignment, header checksum) padded
  to the alignment, then the payload. The restart scan validates headers only, without a
  manifest; payload checksums are checked on every promote.
- Upstream raises an error when O_DIRECT is requested with an unaligned buffer; Atlas always
  writes from its own aligned bounce buffer, and when the filesystem refuses O_DIRECT it retries
  buffered and keeps buffered I/O, with one log line.
- Least recently used eviction under a byte budget, seeded from file mtimes so the order survives
  a restart; an advisory `flock` on `LOCK` keeps one server per directory.
- The asynchronous restore queue is an Atlas design for its single-threaded scheduler: requests
  park outside the batch, are released to the front of the pending queue, and are parked at most
  once (the LMCache bypass rule).

## Not taken

LMCache's worker pools, read budgets, remote and object-store connectors, and its Python
storage manager are not used.
