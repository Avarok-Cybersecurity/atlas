#!/usr/bin/env bash
# Stream a chat completion from Atlas via SSE.
# Atlas exposes the OpenAI-compatible /v1/chat/completions endpoint,
# so any OpenAI streaming client works.
#
# Usage:
#   bash examples/curl/streaming.sh
#   ATLAS_URL=http://192.168.1.10:8888 ATLAS_MODEL=Sehyo/Qwen3.5-35B-A3B-NVFP4 \
#     bash examples/curl/streaming.sh

set -euo pipefail

ATLAS_URL="${ATLAS_URL:-http://localhost:8888}"
ATLAS_MODEL="${ATLAS_MODEL:-Sehyo/Qwen3.5-35B-A3B-NVFP4}"
PROMPT="${1:-Write a haiku about the ocean.}"

echo "Streaming from $ATLAS_URL with $ATLAS_MODEL ..."
echo "Prompt: $PROMPT"
echo

# -N flag disables curl's buffering so the SSE deltas arrive live.
# We pipe through a small awk filter that extracts the `delta.content`
# field from each SSE chunk and prints it without quotes/newlines.
curl -sN -X POST "$ATLAS_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "$ATLAS_MODEL",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": true
}
EOF
)" | awk '
  /^data: \[DONE\]/ { print ""; exit }
  /^data: / {
    # Strip the "data: " prefix
    line = substr($0, 7)
    # Cheap JSON content extraction: find "content":"...".
    # For production, pipe through `jq` instead.
    if (match(line, /"content":"[^"]*"/)) {
      content = substr(line, RSTART + 11, RLENGTH - 12)
      # Unescape \n and \"
      gsub(/\\n/, "\n", content)
      gsub(/\\"/, "\"", content)
      printf "%s", content
      fflush()
    }
  }
'
