#!/usr/bin/env python3
"""Fire the canonical 10382-token prompt at the dgx2 Atlas op-drift server
to trigger ATLAS_OP_DUMP / ATLAS_GDN_DUMP / ATLAS_NEMO_DUMP.

We send the *token IDs* directly via /v1/completions `prompt` field (Atlas
accepts both strings and integer arrays per OpenAI spec).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import requests

TOKENS_PATH = pathlib.Path(
    "/workspace/atlas-mtp/bench/fp8_dgx2_drift/atlas_tokens_dgx2.json"
)
URL = "http://10.10.10.2:8888/v1/completions"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"


def main():
    tok = json.loads(TOKENS_PATH.read_text())
    prompt_tokens = tok["all_tokens"]
    print(f"sending {len(prompt_tokens)} tokens to {URL}", flush=True)
    t0 = time.time()
    body = {
        "model": MODEL,
        "prompt": prompt_tokens,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    r = requests.post(URL, json=body, timeout=600)
    dt = time.time() - t0
    print(f"status={r.status_code} elapsed={dt:.1f}s", flush=True)
    if r.status_code != 200:
        print("BODY:", r.text[:1000])
        sys.exit(1)
    j = r.json()
    print("response:", json.dumps(j, indent=2)[:1500])


if __name__ == "__main__":
    main()
