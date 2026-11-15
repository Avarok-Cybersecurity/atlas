# SPDX-License-Identifier: AGPL-3.0-only
# Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: tools/convert/proposal.py:21-42 (select_shortlist). Changes: translated from NumPy to the Python standard library, a tokenizer identity check and a held-out coverage table added, the output written as Atlas's u32 rank file plus a provenance JSON, by Atlas contributors.
"""Generate `mtp_vocab_rank_qwen35.u32` and its provenance JSON.

The MTP draft head under `mtp_vocab_rank = "freq"` holds the LM-head rows of
the most frequent tokens instead of the first N token ids. This script builds
the ranking Atlas ships as v1: the top ids of Cinference's
`ranking.train.counts.i64` (Apache-2.0 data; token counts by NInfer authors,
source corpus not published), ordered exactly as Cinference's
`select_shortlist` orders them, with the tokenizer's added tokens forced in.

It also runs the CPU tokenizer identity check that must pass before the
table may be used with a Qwen3.5 checkpoint: the counts were made with the
Qwen3.6-27B tokenizer, so every (id, token) pair the three Cinference
manifests print must decode to the same text through the local Qwen3.5
`tokenizer.json`. The JSON records the FNV-1a-64 of every tokenizer file the
check passed for; Atlas uses the table only when the served model's
`tokenizer.json` hashes to one of them (`layers/mtp_head/vocab_rank.rs`).

Standard library only. Usage (paths are explicit; nothing is downloaded):

    python3 -I gen_mtp_vocab_rank.py \
        --ranking-dir <cinference checkout>/tools/freq_corpus/fixtures/ranking \
        --tokenizer <model dir>/Qwen3.5-4B/tokenizer.json \
        --tokenizer <model dir>/Qwen3.5-0.8B/tokenizer.json \
        --out-dir crates/spark-model/data

The output is deterministic: the same inputs give byte-identical files.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path

VOCAB_ROWS = 248320  # rows of every Cinference counts file (manifest "vocab")
COUNT_ROWS = 6  # row 0 = total, rows 1..5 = strata (manifest "counts_layout")
SHIP_ROWS = 131072  # Cinference's official proposal width (proposal.py:50)
GRID = (32768, 65536, 98304, 100000, 131072)
SOURCE_COMMIT = "383e5db"


def fnv1a64(data: bytes) -> int:
    """FNV-1a 64-bit, the same function as `vocab_rank::fnv1a64` in Rust."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def read_counts(path: Path) -> list[list[int]]:
    """int64 little-endian, COUNT_ROWS rows of VOCAB_ROWS each."""
    raw = path.read_bytes()
    if len(raw) != COUNT_ROWS * VOCAB_ROWS * 8:
        sys.exit(f"{path}: {len(raw)} bytes, want {COUNT_ROWS * VOCAB_ROWS * 8}")
    a = array.array("q")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return [list(a[r * VOCAB_ROWS : (r + 1) * VOCAB_ROWS]) for r in range(COUNT_ROWS)]


def select_shortlist(total: list[int], n: int, forced: list[int]) -> list[int]:
    """Cinference `select_shortlist`: the n most frequent ids, forced ids in.

    `np.argsort(-counts, kind="stable")` is a stable sort on descending count,
    so ties keep ascending id order; the final order is again a stable sort
    by descending count over [picked..., forced...].
    """
    order = sorted(range(len(total)), key=lambda i: -total[i])
    forced_sorted = sorted({i for i in forced if 0 <= i < len(total)})
    if len(forced_sorted) >= n:
        return sorted(forced_sorted, key=lambda i: -total[i])[:n]
    forced_set = set(forced_sorted)
    wanted = n - len(forced_sorted)
    picked = []
    for i in order:
        if i not in forced_set:
            picked.append(i)
            if len(picked) == wanted:
                break
    selected = picked + forced_sorted
    return sorted(selected, key=lambda i: -total[i])


def select_rows(table: list[int], vocab: int, n: int, forced: list[int]) -> list[int]:
    """Atlas's runtime row selection (`vocab_rank::select_rows`), mirrored
    here so the coverage table describes the head Atlas actually builds."""
    n = min(n, vocab)
    forced_set = sorted({i for i in forced if i < vocab})
    if len(forced_set) >= n:
        return forced_set[:n]
    seen = set(forced_set)
    picked = []
    want = n - len(forced_set)
    for i in table:
        if len(picked) == want:
            break
        if i < vocab and i not in seen:
            seen.add(i)
            picked.append(i)
    i = 0
    while len(picked) < want and i < vocab:
        if i not in seen:
            seen.add(i)
            picked.append(i)
        i += 1
    return picked + forced_set


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE alphabet (the Qwen tokenizers use it)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def decode_vocab(tok: dict) -> tuple[dict[int, str], list[int]]:
    """id -> decoded text, and the ids of the added tokens."""
    inv = {v: k for k, v in bytes_to_unicode().items()}
    out: dict[int, str] = {}
    for piece, idx in tok["model"]["vocab"].items():
        try:
            raw = bytes(inv[ch] for ch in piece)
        except KeyError:
            raw = piece.encode("utf-8")
        out[idx] = raw.decode("utf-8", errors="replace")
    added = []
    for a in tok.get("added_tokens", []):
        out[a["id"]] = a["content"]
        added.append(a["id"])
    return out, sorted(added)


def coverage(rows: list[int], counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return sum(counts[i] for i in rows) / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranking-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=SHIP_ROWS)
    args = ap.parse_args()

    rd = args.ranking_dir
    train_path = rd / "ranking.train.counts.i64"
    train = read_counts(train_path)
    manifests = {}
    for name in ("ranking.train", "ranking.heldout", "accept.heldout"):
        manifests[name] = json.loads((rd / f"{name}.manifest.json").read_text("utf-8"))
    strata = manifests["ranking.train"]["strata_order"]

    # Tokenizer identity check: every listed tokenizer must decode every
    # manifest (id, token) pair identically, and all listed tokenizers must
    # agree with each other on the decoded vocabulary.
    tok_records = []
    decoded_ref = None
    added_ref = None
    checked = 0
    for tpath in args.tokenizer:
        raw = tpath.read_bytes()
        tok = json.loads(raw)
        decoded, added = decode_vocab(tok)
        mismatches = []
        for mname, m in manifests.items():
            for entry in m.get("top_tokens", []):
                checked += 1
                got = decoded.get(entry["id"])
                if got != entry["token"]:
                    mismatches.append((mname, entry["id"], entry["token"], got))
        if mismatches:
            for mm in mismatches[:20]:
                print("MISMATCH", mm, file=sys.stderr)
            sys.exit(f"{tpath}: {len(mismatches)} manifest token(s) differ; table not applicable")
        if decoded_ref is None:
            decoded_ref, added_ref = decoded, added
        elif decoded != decoded_ref or added != added_ref:
            sys.exit(f"{tpath}: vocabulary differs from {args.tokenizer[0]}")
        tok_records.append(
            {
                "file": str(tpath.name),
                "model_dir": str(tpath.parent.name),
                "bytes": len(raw),
                "fnv1a64": f"{fnv1a64(raw):016x}",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    forced = added_ref or []
    tokenizer_vocab = len(decoded_ref or {})

    table = select_shortlist(train[0], args.rows, forced)
    assert len(table) == args.rows and len(set(table)) == args.rows
    assert all(f in set(table) for f in forced)

    # Held-out coverage at the runtime vocabulary (tokenizer-capped) for the
    # rows Atlas builds: frequency ranking vs Atlas's first-N ids.
    cov = {}
    for split in ("ranking.heldout", "accept.heldout"):
        held = read_counts(rd / f"{split}.counts.i64")
        per = {}
        for n in GRID:
            freq_rows = select_rows(table, tokenizer_vocab, n, forced)
            id_rows = list(range(min(n, tokenizer_vocab)))
            row = {"freq": {}, "id": {}}
            for r, label in enumerate(["total"] + list(strata)):
                row["freq"][label] = round(coverage(freq_rows, held[r]), 6)
                row["id"][label] = round(coverage(id_rows, held[r]), 6)
            per[str(n)] = row
        cov[split] = per

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = array.array("I", table)
    if sys.byteorder != "little":
        blob.byteswap()
    data = blob.tobytes()
    (out_dir / "mtp_vocab_rank_qwen35.u32").write_bytes(data)

    prov = {
        "schema": 1,
        "artifact": "mtp_vocab_rank",
        "file": "mtp_vocab_rank_qwen35.u32",
        "layout": "u32 little-endian token ids, most frequent first; head row r holds token table[r]",
        "rows": len(table),
        "sha256": hashlib.sha256(data).hexdigest(),
        "source": {
            "project": "Cinference (derived from NInfer by Neroued and contributors)",
            "url": "https://github.com/satellitedown/cinference",
            "commit": SOURCE_COMMIT,
            "path": "tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64",
            "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "row": "0 (total)",
            "license": "Apache-2.0",
            "note": "token counts by NInfer authors; source corpus not published",
            "mode": manifests["ranking.train"].get("mode"),
            "counting_tokenizer": manifests["ranking.train"].get("tokenizer_dir"),
        },
        "selection": "Cinference select_shortlist (tools/convert/proposal.py:21-42): "
        "descending total count, ties by ascending id, forced ids included",
        "forced_ids": forced,
        "tokenizer_check": {
            "manifest_pairs_checked": checked,
            "mismatches": 0,
            "tokenizer_vocab": tokenizer_vocab,
            "accepted_vocab_sizes": [tokenizer_vocab, VOCAB_ROWS],
            "tokenizers": tok_records,
        },
        "coverage_grid": list(GRID),
        "coverage": cov,
        "generator": "crates/spark-model/data/gen_mtp_vocab_rank.py",
    }
    (out_dir / "mtp_vocab_rank_qwen35.json").write_text(
        json.dumps(prov, indent=2, sort_keys=False) + "\n", "utf-8"
    )
    c = cov["ranking.heldout"]["100000"]
    print(
        f"wrote {len(table)} rows; held-out coverage at 100000: "
        f"freq {c['freq']['total']:.4f} vs id {c['id']['total']:.4f}; "
        f"{checked} manifest pairs matched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
