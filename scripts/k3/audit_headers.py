#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fetch pinned K3 headers + tiny A_log tensors, never full checkpoint shards.

Python standard library only. Six bounded Range readers; fail if a server
ignores a Range. Output is suitable for core example k3_rank_memory.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
BASE = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{REVISION}/"


def read_range(shard, start, end):
    request = urllib.request.Request(
        BASE + shard + f"?k3-header-audit={start}-{end}",
        headers={"Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"range not honored for {shard}: {content_range}")
        data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise RuntimeError(f"wrong range length for {shard}")
        return data


def read_header(shard):
    length = struct.unpack("<Q", read_range(shard, 0, 7))[0]
    if not 0 < length < 8_000_000:
        raise RuntimeError(f"unreasonable header length: {length}")
    data = read_range(shard, 8, length + 7)
    json.loads(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    headers = args.output / "headers"
    a_log_dir = args.output / "a-log"
    headers.mkdir()
    a_log_dir.mkdir()

    def header(index):
        shard = f"model-{index:05d}-of-000096.safetensors"
        path = headers / (shard + ".json")
        data = read_header(shard)
        path.write_bytes(data)
        return {"shard": shard, "header_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        header_records = list(pool.map(header, range(1, 97)))
    tasks = []
    count = 0
    text_count = 0
    for path in headers.glob("*.json"):
        data = path.read_bytes()
        for name, meta in json.loads(data).items():
            if name == "__metadata__":
                continue
            count += 1
            if name.startswith("language_model.") or name.startswith("lm_head."):
                text_count += 1
            if name.endswith(".self_attn.A_log"):
                if meta["dtype"] != "F32" or meta["shape"] != [128]:
                    raise RuntimeError(f"unexpected A_log storage: {name}")
                start, end = meta["data_offsets"]
                tasks.append((path.name.removesuffix(".json"), name, len(data)+8+start, len(data)+8+end-1))

    def tail(task):
        shard, name, start, end = task
        data = read_range(shard, start, end)
        values = struct.unpack("<128f", data)
        if any(value != 0.0 for value in values[96:]):
            raise RuntimeError(f"nonzero/nonfinite A_log tail: {name}")
        layer = int(name.split("layers.")[1].split(".")[0])
        (a_log_dir / f"{layer}.bin").write_bytes(data)
        return {"layer": layer, "tensor": name, "shard": shard,
                "byte_range": [start, end], "sha256": hashlib.sha256(data).hexdigest(),
                "tail_zero": True, "active_nonzero": sum(v != 0.0 for v in values[:96])}

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = sorted(pool.map(tail, tasks), key=lambda record: record["layer"])
    if len(records) != 69:
        raise RuntimeError(f"expected 69 KDA layers, got {len(records)}")
    receipt = {"revision": REVISION, "header_count": len(header_records),
               "header_bytes": sum(r["header_bytes"] for r in header_records),
               "all_tensors": count, "text_tensors": text_count,
               "a_log_payload_bytes": 512 * len(records),
               "headers": header_records, "a_log_records": records}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("headers", "a_log_records")}))


if __name__ == "__main__":
    main()
