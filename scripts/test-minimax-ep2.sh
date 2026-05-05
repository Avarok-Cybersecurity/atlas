#!/usr/bin/env bash
# End-to-end MiniMax EP=2 bring-up + test harness.
#
# 1. Assumes image `atlas-gb10:minimax-ep2` exists on head.
# 2. Distributes image to worker via docker save | ssh | docker load.
# 3. Starts EP=2 via start-minimax-ep2.sh.
# 4. Waits up to STARTUP_TIMEOUT (default 900s) for rank 0 "Listening on".
# 5. Runs tests/single_gpu_suite.py against http://localhost:8888/v1.
# 6. On failure, dumps last 100 log lines from both ranks.
# 7. Stops containers on both nodes at the end.
#
# Env:
#   MODEL   — HF model id (default MiniMaxAI/MiniMax-M2)
#   IMAGE   — docker image tag (default atlas-gb10:minimax-ep2)
#   SKIP_TRANSFER=1 — skip save/load image dance (already on worker)
#   KEEP_RUNNING=1  — leave containers up after tests for manual inspection

set -euo pipefail

MODEL="${MODEL:-MiniMaxAI/MiniMax-M2}"
IMAGE="${IMAGE:-atlas-gb10:minimax-ep2}"
WORKER_IP="${WORKER_IP:-127.0.0.1}"
HEAD_PORT="${HEAD_PORT:-8888}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
RESULT_DIR="/workspace/atlas/tests/all_models_results"
LABEL="minimax-m2-ep2"
mkdir -p "$RESULT_DIR"

ATLAS_ROOT="/workspace/atlas"
STARTUP_SCRIPT="$ATLAS_ROOT/scripts/start-minimax-ep2.sh"

echo "=== MiniMax EP=2 end-to-end test ==="
echo "Model: $MODEL"
echo "Image: $IMAGE"
echo ""

# 1. Verify image on head
if ! sudo docker images "$IMAGE" --format '{{.ID}}' | grep -q '.'; then
  echo "[FATAL] Image $IMAGE not found on head"
  exit 1
fi

# 2. Transfer image to worker unless told to skip
if [[ "${SKIP_TRANSFER:-0}" != "1" ]]; then
  if ! ssh "$WORKER_IP" "sudo docker images '$IMAGE' --format '{{.ID}}' | grep -q ."; then
    echo "=== Transferring image to worker ($WORKER_IP) ==="
    sudo docker save "$IMAGE" | ssh "$WORKER_IP" 'sudo docker load'
  else
    echo "Image already present on worker, skipping transfer"
  fi
fi

# 3. Stop any stale containers before starting
sudo docker rm -f atlas-minimax-ep0 2>/dev/null || true
ssh "$WORKER_IP" 'sudo docker rm -f atlas-minimax-ep1 2>/dev/null' || true

# 4. Start EP=2
IMAGE="$IMAGE" bash "$STARTUP_SCRIPT" "$MODEL"

# 5. Wait for both ranks to be ready
echo ""
echo "=== Waiting for ranks to start (up to ${STARTUP_TIMEOUT}s) ==="
DEADLINE=$(( $(date +%s) + STARTUP_TIMEOUT ))
RANK0_READY=0
RANK1_READY=0
while [[ $(date +%s) -lt $DEADLINE ]]; do
  if [[ $RANK0_READY -eq 0 ]]; then
    if sudo docker logs atlas-minimax-ep0 2>&1 | grep -q 'Listening on'; then
      RANK0_READY=1
      echo "  [rank0] listening"
    fi
  fi
  if [[ $RANK1_READY -eq 0 ]]; then
    if ssh "$WORKER_IP" "sudo docker logs atlas-minimax-ep1 2>&1 | grep -qE 'EP worker ready|Listening on|worker ready'"; then
      RANK1_READY=1
      echo "  [rank1] worker ready"
    fi
  fi
  if [[ $RANK0_READY -eq 1 && $RANK1_READY -eq 1 ]]; then
    break
  fi
  # Detect early exit
  if ! sudo docker ps -q -f name=atlas-minimax-ep0 | grep -q .; then
    echo "[FATAL] rank 0 container exited"
    sudo docker logs --tail 100 atlas-minimax-ep0 2>&1
    exit 1
  fi
  if ! ssh "$WORKER_IP" 'sudo docker ps -q -f name=atlas-minimax-ep1' | grep -q .; then
    echo "[FATAL] rank 1 container exited"
    ssh "$WORKER_IP" 'sudo docker logs --tail 100 atlas-minimax-ep1 2>&1'
    exit 1
  fi
  sleep 15
done

if [[ $RANK0_READY -eq 0 || $RANK1_READY -eq 0 ]]; then
  echo "[FATAL] Startup timeout after ${STARTUP_TIMEOUT}s — rank0=$RANK0_READY rank1=$RANK1_READY"
  echo "--- rank 0 log tail ---"
  sudo docker logs --tail 120 atlas-minimax-ep0 2>&1
  echo "--- rank 1 log tail ---"
  ssh "$WORKER_IP" 'sudo docker logs --tail 120 atlas-minimax-ep1 2>&1'
  exit 1
fi

# 6. Warmup request
echo ""
echo "=== Warmup request ==="
curl -sS -o /dev/null -w "HTTP %{http_code} in %{time_total}s\n" --max-time 180 \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"temperature\":0}" \
  "http://localhost:$HEAD_PORT/v1/chat/completions" || true

# 7. Run test suite
echo ""
echo "=== Running tests/single_gpu_suite.py ==="
OUT_JSON="$RESULT_DIR/${LABEL}.json"
LOG="$RESULT_DIR/${LABEL}.log"
set +e
python3 "$ATLAS_ROOT/tests/single_gpu_suite.py" \
  --base-url "http://localhost:$HEAD_PORT/v1" \
  --model "$MODEL" \
  --output "$OUT_JSON" \
  2>&1 | tee "$LOG"
STATUS=$?
set -e

echo ""
echo "=== Test suite exit: $STATUS ==="
echo "Results: $OUT_JSON"
echo "Log:     $LOG"

if [[ $STATUS -ne 0 ]]; then
  echo "--- rank 0 log tail ---"
  sudo docker logs --tail 80 atlas-minimax-ep0 2>&1
  echo "--- rank 1 log tail ---"
  ssh "$WORKER_IP" 'sudo docker logs --tail 80 atlas-minimax-ep1 2>&1'
fi

# 8. Cleanup
if [[ "${KEEP_RUNNING:-0}" != "1" ]]; then
  echo ""
  echo "=== Stopping EP=2 containers ==="
  sudo docker stop atlas-minimax-ep0 2>/dev/null || true
  sudo docker rm   atlas-minimax-ep0 2>/dev/null || true
  ssh "$WORKER_IP" 'sudo docker stop atlas-minimax-ep1 2>/dev/null' || true
  ssh "$WORKER_IP" 'sudo docker rm   atlas-minimax-ep1 2>/dev/null' || true
fi

exit $STATUS
