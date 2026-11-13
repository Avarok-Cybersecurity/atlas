// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the disk tier's data path (lane L27): writing a demoted block (budget,
//! eviction, temp file plus rename), reading a promoted one (every check, and removal of a
//! record that fails one), and naming the served fingerprint at adoption.

use std::fs::File;
use std::sync::atomic::Ordering;
use std::time::SystemTime;

use anyhow::{Result, anyhow, bail};

use super::index::{State, unix_nanos};
use super::record::{self, AlignedBuf, TMP_SUFFIX};
use super::{BlockKey, DiskTier};

/// Why a promote did not return a block.
enum ReadFail {
    Io(anyhow::Error),
    Corrupt(anyhow::Error),
}

impl DiskTier {
    fn take_buf(&self) -> AlignedBuf {
        self.scratch
            .lock()
            .pop()
            .unwrap_or_else(|| AlignedBuf::new(0, self.align))
    }

    fn give_buf(&self, b: AlignedBuf) {
        let mut s = self.scratch.lock();
        if s.len() < 4 {
            s.push(b);
        }
    }

    pub(super) fn publish_usage(&self, st: &State) {
        self.stats.set_usage(st.used, st.entries.len() as u64);
        self.stats.set_direct_io(self.io.direct());
    }

    /// Write one block. Already on disk for the same namespace: touched, not rewritten (a key
    /// names immutable content). Larger than the whole budget: skipped.
    pub(super) fn put(
        &self,
        key: &[u8; 16],
        namespace: u64,
        tokens: &[u32],
        payload: &[u8],
    ) -> Result<()> {
        if payload.len() != self.payload_bytes {
            bail!(
                "kv tier disk: a {}-byte block offered to a tier of {}-byte blocks",
                payload.len(),
                self.payload_bytes
            );
        }
        let need = record::record_len(tokens.len(), payload.len(), self.align) as u64;
        {
            let mut st = self.state.lock();
            let same = st
                .entries
                .get(key)
                .is_some_and(|e| e.namespace == namespace);
            if need > self.budget || same || st.inflight.contains(key) {
                if same {
                    st.touch(key);
                }
                self.stats.skipped();
                return Ok(());
            }
            // A record of another namespace under this key: replace it.
            if st.remove(key).is_some() {
                self.unlink(&[*key]);
            }
            let victims = st.victims(need, self.budget);
            self.unlink(&victims);
            self.stats.evicted(victims.len() as u64);
            st.used += need;
            st.inflight.insert(*key);
        }
        let res = self.write_record(key, namespace, tokens, payload);
        let mut st = self.state.lock();
        st.inflight.remove(key);
        st.used -= need;
        match res {
            Ok(()) => {
                let now = unix_nanos(SystemTime::now());
                st.insert(*key, namespace, payload.len() as u64, need, now);
                self.stats.wrote(need);
                self.publish_usage(&st);
                Ok(())
            }
            Err(e) => {
                self.stats.io_error();
                self.publish_usage(&st);
                Err(e)
            }
        }
    }

    fn write_record(
        &self,
        key: &[u8; 16],
        namespace: u64,
        tokens: &[u32],
        payload: &[u8],
    ) -> Result<()> {
        let mut buf = self.take_buf();
        let now_ms = unix_nanos(SystemTime::now()) / 1_000_000;
        record::encode(
            &mut buf, namespace, key, tokens, payload, self.align, now_ms,
        );
        let hex = record::key_hex(key);
        let seq = self.tmp_seq.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let tmp = self.dir.join(format!("{hex}.{pid}.{seq}{TMP_SUFFIX}"));
        let res = record::write_atomic(&tmp, &self.path_of(key), &buf, &self.io);
        self.give_buf(buf);
        res
    }

    /// Read one block into `out`. `tokens`: when given and the record stores tokens, they must be
    /// equal. A record that fails any check is removed and counted; the result is then a miss.
    pub(super) fn get(
        &self,
        key: &[u8; 16],
        tokens: Option<&[u32]>,
        out: &mut [u8],
    ) -> Result<bool> {
        let found = {
            let st = self.state.lock();
            st.namespace.and_then(|ns| {
                st.entries
                    .get(key)
                    .filter(|e| e.namespace == ns && e.payload_len == out.len() as u64)
                    .map(|e| (ns, e.generation, e.file_bytes))
            })
        };
        let Some((ns, generation, file_bytes)) = found else {
            self.stats.miss();
            return Ok(false);
        };
        let path = self.path_of(key);
        let mut buf = self.take_buf();
        let verdict = record::read_all(&path, file_bytes as usize, self.align, &mut buf, &self.io)
            .map_err(ReadFail::Io)
            .and_then(|()| {
                Self::verify(buf.as_slice(), key, ns, tokens, out).map_err(ReadFail::Corrupt)
            });
        self.give_buf(buf);
        let mut st = self.state.lock();
        let current = st
            .entries
            .get(key)
            .is_some_and(|e| e.generation == generation);
        let e = match verdict {
            Ok(()) => {
                if current {
                    st.touch(key);
                    if let Ok(f) = File::open(&path) {
                        let _ = f.set_modified(SystemTime::now());
                    }
                }
                self.stats.hit();
                return Ok(true);
            }
            Err(ReadFail::Io(e)) => {
                self.stats.io_error();
                e
            }
            Err(ReadFail::Corrupt(e)) => {
                self.stats.corrupt();
                e
            }
        };
        tracing::warn!("kv tier disk: dropping record {}: {e:#}", path.display());
        if current {
            st.remove(key);
            self.unlink(&[*key]);
        }
        self.publish_usage(&st);
        self.stats.miss();
        Ok(false)
    }

    fn verify(
        rec: &[u8],
        key: &[u8; 16],
        ns: u64,
        tokens: Option<&[u32]>,
        out: &mut [u8],
    ) -> Result<()> {
        let h = record::parse_header(rec)?;
        if &h.key != key || h.namespace != ns || h.payload_len != out.len() as u64 {
            bail!("record identity does not match its index entry");
        }
        if let Some(t) = tokens
            && !h.tokens.is_empty()
            && h.tokens != t
        {
            bail!("stored tokens differ from the probe's tokens");
        }
        out.copy_from_slice(record::payload(&h, rec)?);
        Ok(())
    }

    /// Name the served model's fingerprint and return the keys recorded for it.
    pub(super) fn adopt(&self, namespace: u64) -> Vec<BlockKey> {
        let mut st = self.state.lock();
        st.namespace = Some(namespace);
        let block = self.payload_bytes as u64;
        let keys: Vec<BlockKey> = st
            .entries
            .iter()
            .filter(|(_, e)| e.namespace == namespace && e.payload_len == block)
            .map(|(k, _)| BlockKey(*k))
            .collect();
        self.stats.adopted(keys.len() as u64);
        tracing::info!(
            "kv tier disk: adopted {} record(s) of fingerprint {namespace:016x} ({} on disk)",
            keys.len(),
            st.entries.len()
        );
        keys
    }

    pub(super) fn namespace(&self) -> Result<u64> {
        self.state.lock().namespace.ok_or_else(|| {
            anyhow!("kv tier disk: demote before adopt_on_restart named the model fingerprint")
        })
    }
}
