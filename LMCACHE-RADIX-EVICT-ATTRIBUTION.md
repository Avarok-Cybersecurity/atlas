# LMCache attribution: radix eviction without a full scan (port sprint lane L28)

The radix prefix cache's ordered victim index (row `radix_lru_index`, `AVAROK_RADIX_LRU_INDEX`, off)
follows a design idea from LMCache: keep eviction candidates in an ordered structure so selecting a
victim is a bounded walk from the least recently used end and never holds or scans a snapshot of
every entry. No LMCache code was copied; the Atlas file is an independent Rust implementation for
Atlas's own radix tree, and the victim rule is Atlas's, unchanged.

- Upstream: https://github.com/LMCache/LMCache, commit `3aefae1`
- License: Apache License, Version 2.0 (`LICENSES/Apache-2.0.txt`). LMCache ships no NOTICE file;
  the repository `NOTICE` carries an LMCache entry for this lane.

## What was taken from where

| Atlas file | Upstream design source | What it follows | Changes for Atlas |
|---|---|---|---|
| `crates/spark-runtime/src/radix_tree/inner/lru_index.rs` | `lmcache/v1/distributed/eviction_policy/isolated_lru.py:118-190`, `lmcache/v1/distributed/eviction_policy/lru.py:72-95` | An ordered map of eviction candidates; victim selection walks from the LRU end and holds no whole-bucket snapshot | A `BTreeSet<(last_access, node id)>` of evictable radix leaves, maintained at every transition of "not a root, `ref_count <= 1`, no children, block assigned"; ties break toward the lowest node id so the pick equals Atlas's existing full scan exactly; a stale entry is detected in O(1) per pick and never trusted (panic in test and debug builds, a rebuild otherwise) |
| `crates/spark-runtime/src/radix_tree/inner/evict.rs` | as above | One removal path for both selection modes | The scan stays as the row-off path and as the oracle |

Not taken: LMCache's per-request reverse insertion (later chunks first) is already implied by
Atlas's leaf-only rule, and ARC (`eviction_policy/arc.py`) is not ported.

Verification: `crates/spark-runtime/src/radix_tree/tests/lru_index.rs` drives a scan tree and an
index tree through thousands of randomized operations with forced same-tick ties and asserts
identical victim sequences and identical final trees; every radix test also cross-checks the index
against the scan on every pick.

## The Apache-2.0 notice

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except
in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0 (also `LICENSES/Apache-2.0.txt` in this repository).
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for the specific language governing permissions and limitations under
the License.
