#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Download + serve + smoke-test the Qwen3.6-35B-A3B (vision) model on Atlas,
# exercising the issue-#165 fix: an image attached to a *tool result* must
# reach the vision encoder (it used to be silently dropped).
#
# Run this on the GB10 box, from the repo root, with a binary built from the
# `feat/canonical-chat-ir` branch (the tool-result-image fix is NOT in the
# published Docker image yet).
#
#   ./scripts/test-qwen36-tool-image.sh
#
# Override any setting via env, e.g.:
#   HF_REPO=Sehyo/Qwen3.6-35B-A3B-NVFP4 PORT=8889 ./scripts/test-qwen36-tool-image.sh
set -euo pipefail

# ─────────────────────────── config (edit / override via env) ───────────────
# HuggingFace checkpoint to download. The in-tree kernel target
# kernels/gb10/qwen3.6-35b-a3b/MODEL.toml declares hf_id="Qwen/Qwen3.6-35B-A3B-FP8"
# (FP8 weights + NVFP4 lm-head). If you have a pre-quantized NVFP4 weights repo,
# set HF_REPO to it instead.
HF_REPO="${HF_REPO:-Qwen/Qwen3.6-35B-A3B-FP8}"
# Name the server advertises (so `@atlas/<SERVED_NAME>` in your client matches).
SERVED_NAME="${SERVED_NAME:-qwen3.6-35b-a3b-nvfp4}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8888}"
BASE_URL="http://${HOST}:${PORT}"

SPARK_BIN="${SPARK_BIN:-./target/release/spark}"
KV_DTYPE="${KV_DTYPE:-nvfp4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
SCHED_POLICY="${SCHED_POLICY:-slai}"

# auto = start a server only if one isn't already answering on $PORT.
# yes  = always start one. no = never (test an already-running server).
START_SERVER="${START_SERVER:-auto}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"   # seconds to wait for weights to load
# This model defaults thinking ON (768-tok budget) — that would eat a small
# max_tokens before any content. Disable it for deterministic, fast tests.
DISABLE_THINKING="${DISABLE_THINKING:-1}"   # 1 = pass --disable-thinking
# Dir holding libnccl.so / libnccl.so.2 for a source-built binary (optional).
# Atlas links NCCL; if your shell doesn't already resolve it, point this at a
# dir containing the symlinks and it's prepended to LD_LIBRARY_PATH.
NCCL_DIR="${NCCL_DIR:-}"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/atlas-qwen36-test.XXXXXX")"
SERVER_LOG="${WORKDIR}/server.log"
SERVER_PID=""
STARTED_SERVER=0
# ─────────────────────────────────────────────────────────────────────────────

# Progress goes to stderr so it never pollutes a captured stdout (e.g. the
# image data URL or a chat reply).
log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*" >&2; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[FATAL] %s\033[0m\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

cleanup() {
  if [[ "${STARTED_SERVER}" == "1" && -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    log "stopping server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

need curl
need python3
export SERVED_NAME            # consumed by the inline python request builders
# huggingface-cli / hf is checked (with fallback) inside download_model.

# ── a small, recognizable test image (white bg, red circle, "ATLAS") ─────────
# Returns a `data:image/png;base64,...` URL on stdout. Uses Pillow if present,
# else falls back to a pure-stdlib solid-red PNG (still describable as "red").
make_image_data_url() {
  local png="${WORKDIR}/test.png"
  if python3 -c 'import PIL' >/dev/null 2>&1; then
    python3 - "$png" <<'PY'
import sys
from PIL import Image, ImageDraw
img = Image.new("RGB", (256, 256), "white")
d = ImageDraw.Draw(img)
d.ellipse([48, 48, 208, 208], fill="red", outline="black", width=5)
d.text((96, 120), "ATLAS", fill="white")
img.save(sys.argv[1], "PNG")
PY
  else
    warn "Pillow not installed — using a plain red square (model should still say 'red')"
    python3 - "$png" <<'PY'
import sys, zlib, struct
w = h = 128
raw = bytearray()
for _ in range(h):
    raw.append(0)                       # filter byte per scanline
    raw.extend((220, 30, 30) * w)       # solid red RGB
def chunk(tag, data):
    c = tag + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
png += chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
PY
  fi
  printf 'data:image/png;base64,%s' "$(base64 -w0 < "$png" 2>/dev/null || base64 < "$png" | tr -d '\n')"
}

server_up() { curl -fsS -m 3 "${BASE_URL}/health" >/dev/null 2>&1; }

download_model() {
  if command -v huggingface-cli >/dev/null 2>&1; then
    log "downloading ${HF_REPO} into the HF cache (large; resumes if interrupted)…"
    huggingface-cli download "${HF_REPO}" --revision main
  elif command -v hf >/dev/null 2>&1; then
    log "downloading ${HF_REPO} into the HF cache (large; resumes if interrupted)…"
    hf download "${HF_REPO}" --revision main
  else
    warn "no 'huggingface-cli'/'hf' found — skipping download; assuming ${HF_REPO} is"
    warn "already in the HF cache (the server errors clearly if it isn't)."
    return 0
  fi
  ok "checkpoint present in cache"
}

start_server() {
  [[ -x "${SPARK_BIN}" ]] || die "spark binary not found/executable at '${SPARK_BIN}'.
Build it from the feat/canonical-chat-ir branch first. Atlas links NCCL, so the
linker needs a dir containing libnccl.so on LIBRARY_PATH (NCCL_DIR below is only
the RUNTIME path):
    LIBRARY_PATH=\$NCCL_DIR ATLAS_TARGET_MODEL=qwen3.6-35b-a3b \\
        cargo build --release -p spark-server
(then re-run, or set SPARK_BIN=/path/to/spark)"

  if [[ -n "${NCCL_DIR}" ]]; then
    export LIBRARY_PATH="${NCCL_DIR}:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${NCCL_DIR}:${LD_LIBRARY_PATH:-}"
  fi

  local extra=()
  [[ "${DISABLE_THINKING}" == "1" ]] && extra+=(--disable-thinking)

  log "starting server: ${SPARK_BIN} serve ${HF_REPO} (logs → ${SERVER_LOG})"
  "${SPARK_BIN}" serve "${HF_REPO}" \
    --model-name "${SERVED_NAME}" \
    --bind "${HOST}" \
    --port "${PORT}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --kv-cache-dtype "${KV_DTYPE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --scheduling-policy "${SCHED_POLICY}" \
    "${extra[@]}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  STARTED_SERVER=1

  log "waiting up to ${READY_TIMEOUT}s for weights to load…"
  local deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while ! server_up; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 40 "${SERVER_LOG}" >&2 || true
      die "server process exited before becoming ready (see ${SERVER_LOG})"
    fi
    [[ $(date +%s) -lt ${deadline} ]] || { tail -n 40 "${SERVER_LOG}" >&2; die "timed out waiting for readiness"; }
    sleep 3
  done
  ok "server ready"
}

# chat <request-json-file> → prints assistant content (and stashes raw json)
chat() {
  local body_file="$1" raw="${WORKDIR}/resp.json"
  curl -fsS -m 180 "${BASE_URL}/v1/chat/completions" \
    -H 'Content-Type: application/json' --data @"${body_file}" >"${raw}"
  python3 - "${raw}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
msg = (d.get("choices") or [{}])[0].get("message", {})
print((msg.get("content") or "").strip())
PY
}

# ─────────────────────────────── run ────────────────────────────────────────
log "Atlas Qwen3.6-35B-A3B vision / tool-image test"
log "repo=${HF_REPO}  served-as=${SERVED_NAME}  endpoint=${BASE_URL}"

download_model

case "${START_SERVER}" in
  no)   server_up || die "START_SERVER=no but nothing is answering ${BASE_URL}/health" ;;
  yes)  start_server ;;
  auto) if server_up; then ok "reusing server already listening on ${PORT}"; else start_server; fi ;;
  *)    die "START_SERVER must be auto|yes|no" ;;
esac

log "building test image…"
DATA_URL="$(make_image_data_url)"; export DATA_URL
ok "image ready (${#DATA_URL} base64 chars)"

# 1) smoke — plain text round-trip
cat >"${WORKDIR}/req_smoke.json" <<JSON
{"model":"${SERVED_NAME}","temperature":0,"max_tokens":16,
 "messages":[{"role":"user","content":"Reply with exactly one word: pong"}]}
JSON
log "test 1/8 — text smoke"
SMOKE="$(chat "${WORKDIR}/req_smoke.json")"
printf '    model said: %q\n' "${SMOKE}"
grep -qi pong <<<"${SMOKE}" && ok "smoke passed" || warn "unexpected smoke reply (model may be chatty)"

# 2) vision on a USER message (sanity: the vision path works at all)
python3 - <<PY
import json, os
body = {
  "model": os.environ["SERVED_NAME"], "temperature": 0, "max_tokens": 48,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": os.environ["DATA_URL"]}},
    {"type": "text", "text": "What is the dominant color in this image? Answer in one word."},
  ]}],
}
json.dump(body, open("${WORKDIR}/req_user_img.json", "w"))
PY
log "test 2/8 — image on a USER message"
USER_IMG="$(chat "${WORKDIR}/req_user_img.json")"
printf '    model said: %q\n' "${USER_IMG}"
grep -qi red <<<"${USER_IMG}" && ok "user-image vision passed (saw 'red')" || warn "did not say 'red' — inspect ${WORKDIR}/resp.json"

# 3) THE FEATURE — image on a TOOL RESULT (issue #165). Before the fix this
#    image was dropped (image_count hardcoded to 0); now it must render.
python3 - <<PY
import json, os
body = {
  "model": os.environ["SERVED_NAME"], "temperature": 0, "max_tokens": 64,
  "tools": [{"type": "function", "function": {
      "name": "capture", "description": "Capture a screenshot",
      "parameters": {"type": "object", "properties": {}}}}],
  "messages": [
    {"role": "user", "content": "Use the capture tool, then tell me the dominant color of the captured image in one word."},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "capture", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": [
        {"type": "image_url", "image_url": {"url": os.environ["DATA_URL"]}},
        {"type": "text", "text": "screenshot captured"}]},
  ],
}
json.dump(body, open("${WORKDIR}/req_tool_img.json", "w"))
PY
log "test 3/8 — image on a TOOL RESULT (the issue #165 fix)"
TOOL_IMG="$(chat "${WORKDIR}/req_tool_img.json")"
printf '    model said: %q\n' "${TOOL_IMG}"
if grep -qi red <<<"${TOOL_IMG}"; then
  ok "tool-result image reached the model (saw 'red') — fix confirmed end-to-end"
else
  warn "model did not describe the tool-result image as 'red'."
  warn "If it claims it sees no image, the fix may not be in this binary."
  warn "Raw response: ${WORKDIR}/resp.json   Server log: ${SERVER_LOG}"
fi

# ── IR-migration coverage: the Anthropic + Responses surfaces must behave
#    identically to the chat surface (same IR pipeline underneath). ──────────

# 4) /v1/messages non-streaming: usage + content sanity
cat >"${WORKDIR}/req_anthropic.json" <<JSON
{"model":"${SERVED_NAME}","max_tokens":32,"temperature":0,
 "messages":[{"role":"user","content":"Reply with exactly one word: pong"}]}
JSON
log "test 4/8 — /v1/messages non-streaming"
curl -fsS -m 180 "${BASE_URL}/v1/messages" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_anthropic.json" >"${WORKDIR}/anthropic.json"
python3 - "${WORKDIR}/anthropic.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["type"] == "message" and d["role"] == "assistant", d
text = "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")
assert "pong" in text.lower(), f"unexpected content: {text!r}"
u = d["usage"]
assert u["input_tokens"] > 0, f"input_tokens must be > 0: {u}"
assert u["output_tokens"] > 0, f"output_tokens must be > 0: {u}"
assert "cache_read_input_tokens" in u, f"cache accounting missing: {u}"
print(f"    input_tokens={u['input_tokens']} output_tokens={u['output_tokens']} cache_read={u['cache_read_input_tokens']}")
PY
ok "/v1/messages non-streaming passed (usage + content)"

# 5) /v1/messages streaming: event order + non-zero input_tokens in the
#    final message_delta (the B1 fix — used to always report 0).
log "test 5/8 — /v1/messages streaming (event framing + input_tokens)"
cat >"${WORKDIR}/req_anthropic_stream.json" <<JSON
{"model":"${SERVED_NAME}","max_tokens":32,"temperature":0,"stream":true,
 "messages":[{"role":"user","content":"Reply with exactly one word: pong"}]}
JSON
curl -fsSN -m 180 "${BASE_URL}/v1/messages" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_anthropic_stream.json" >"${WORKDIR}/anthropic_stream.txt"
python3 - "${WORKDIR}/anthropic_stream.txt" <<'PY'
import json, sys
events = []
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("event:"):
        events.append({"name": line.split(":", 1)[1].strip()})
    elif line.startswith("data:") and events:
        try:
            events[-1]["data"] = json.loads(line.split(":", 1)[1].strip())
        except json.JSONDecodeError:
            pass
names = [e["name"] for e in events]
assert names[0] == "message_start", names
assert "content_block_start" in names and "content_block_delta" in names, names
assert names[-2:] == ["message_delta", "message_stop"], names[-4:]
md = next(e for e in reversed(events) if e["name"] == "message_delta")
u = md["data"]["usage"]
assert u["input_tokens"] > 0, f"streaming input_tokens must be > 0 (B1): {u}"
assert u["output_tokens"] > 0, f"streaming output_tokens must be > 0: {u}"
print(f"    events={len(events)} final usage={u}")
PY
ok "/v1/messages streaming passed (framing + input_tokens > 0)"

# 6) /v1/messages/count_tokens: must be consistent with what serving reports
#    (shared prompt pipeline — the old divergent count path is gone).
log "test 6/8 — /v1/messages/count_tokens vs served input_tokens"
curl -fsS -m 60 "${BASE_URL}/v1/messages/count_tokens" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_anthropic.json" >"${WORKDIR}/count.json"
python3 - "${WORKDIR}/count.json" "${WORKDIR}/anthropic.json" <<'PY'
import json, sys
count = json.load(open(sys.argv[1]))["input_tokens"]
served = json.load(open(sys.argv[2]))["usage"]["input_tokens"]
assert count > 0, count
drift = abs(count - served) / max(served, 1)
assert drift <= 0.10, f"count_tokens {count} vs served {served} drifts {drift:.0%} (>10%)"
print(f"    count_tokens={count} served input_tokens={served} drift={drift:.1%}")
PY
ok "count_tokens matches served usage (<=10% drift)"

# 7) /v1/responses — image on a function_call_output (the #165 parity fix
#    for the Responses surface: it used to drop tool-result images).
python3 - <<PY
import json, os
body = {
  "model": os.environ["SERVED_NAME"], "temperature": 0, "max_output_tokens": 64,
  "input": [
    {"type": "message", "role": "user",
     "content": [{"type": "input_text", "text": "Use the capture tool, then tell me the dominant color of the captured image in one word."}]},
    {"type": "function_call", "call_id": "call_1", "name": "capture", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": [
        {"type": "output_text", "text": "screenshot captured"},
        {"type": "input_image", "image_url": os.environ["DATA_URL"]}]},
  ],
}
json.dump(body, open("${WORKDIR}/req_responses_img.json", "w"))
PY
log "test 7/8 — /v1/responses with a function_call_output image"
curl -fsS -m 180 "${BASE_URL}/v1/responses" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_responses_img.json" >"${WORKDIR}/responses.json"
python3 - "${WORKDIR}/responses.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("status") == "completed", d.get("status")
text = ""
for item in d.get("output", []):
    if item.get("type") == "message":
        for part in item.get("content", []):
            text += part.get("text", "")
assert "red" in text.lower(), f"model did not describe the tool image as red: {text!r}"
print(f"    model said: {text.strip()!r}")
PY
ok "/v1/responses carried the function_call_output image (saw 'red')"

# 8) /v1/responses streaming smoke: completes with usage
log "test 8/8 — /v1/responses streaming smoke"
python3 - <<PY
import json, os
body = {"model": os.environ["SERVED_NAME"], "temperature": 0,
        "max_output_tokens": 32, "stream": True,
        "input": "Reply with exactly one word: pong"}
json.dump(body, open("${WORKDIR}/req_responses_stream.json", "w"))
PY
curl -fsSN -m 180 "${BASE_URL}/v1/responses" -H 'Content-Type: application/json' \
  --data @"${WORKDIR}/req_responses_stream.json" >"${WORKDIR}/responses_stream.txt"
python3 - "${WORKDIR}/responses_stream.txt" <<'PY'
import json, sys
completed = None
for line in open(sys.argv[1]):
    line = line.strip()
    if line.startswith("data:"):
        try:
            d = json.loads(line.split(":", 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if d.get("type") == "response.completed":
            completed = d
assert completed is not None, "no response.completed event seen"
usage = completed["response"].get("usage") or {}
assert usage.get("input_tokens", 0) > 0, f"usage missing/zero: {usage}"
print(f"    completed with usage={usage}")
PY
ok "/v1/responses streaming completed with usage"

log "done. artifacts in ${WORKDIR}"
