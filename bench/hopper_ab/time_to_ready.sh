#!/usr/bin/env bash
# Measure how long an engine takes to become servable, and how long its first
# token takes once it is.
#
# The PRD makes a 30-minute boot cap a GATE, so this has to be a measurement
# rather than an impression. Two things it deliberately does not assume:
#
#  * The clock starts at --start-epoch, which the CALLER supplies from just
#    before it launches the serve process. A script that starts its own clock
#    measures its own startup and forgives whatever the launcher did first.
#  * Connection-refused is a LOADING state, not an error. Atlas answers
#    503 {"status":"loading"} on /health while weights load and then 200; vLLM
#    refuses the TCP connection outright until its server binds and then
#    answers 200. Treating refusal as failure would score vLLM as never booting.
#
# Readiness and usability are different claims, so both are reported: an engine
# can answer /health before its graphs are captured, and the one-token request
# afterwards is what says so.
#
# Usage:
#   time_to_ready.sh --url URL --model MODEL [--engine atlas|vllm]
#                    [--start-epoch SECS] [--timeout-s 1800] [--out FILE]
#                    [--process-owner FILE] (Linux process-mode owner record)
#   time_to_ready.sh --selftest
#
# Exits non-zero unless health and a nonempty one-token completion both pass
# within --timeout-s of the caller-supplied process start.
set -uo pipefail

URL=""
MODEL=""
ENGINE="unknown"
START_EPOCH=""
TIMEOUT_S=1800
OUT=""
SELFTEST=0
PROCESS_OWNER=""

usage() { sed -n '2,26p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --start-epoch) START_EPOCH="$2"; shift 2 ;;
    --timeout-s) TIMEOUT_S="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --process-owner) PROCESS_OWNER="$2"; shift 2 ;;
    --selftest) SELFTEST=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

measure() {
  # curl supplies a total request deadline (including a slowly streamed body).
  # Python owns JSON validation and the single process-start deadline so a
  # healthy endpoint with a failed first completion cannot pass the boot gate.
  python3 - "$ENGINE" "$URL" "$MODEL" "$START_EPOCH" "$TIMEOUT_S" \
    "$PROCESS_OWNER" "$(dirname "$0")/../campaign" <<'PY'
import json
import math
import subprocess
import sys
import time

engine, url, model, start, timeout, process_owner, campaign_path = sys.argv[1:]
now = time.time()
try:
    started = float(start) if start else now
    budget = float(timeout)
    if not math.isfinite(started) or not math.isfinite(budget) or budget <= 0 or started > now:
        raise ValueError('start must not be in the future; timeout must be finite and positive')
except ValueError as exc:
    sys.exit(f'invalid timing arguments: {exc}')
if not model:
    sys.exit('--model is required to verify the first-token gate')
clock = time.monotonic()
deadline = clock + budget - (now - started)
out = dict(schema=1, engine=engine, url=url, model=model, start_epoch=started,
           time_to_ready_s=None, first_token_s=None, polls=0,
           http_codes_seen=[], timeout_s=budget, status='timeout', passed=False,
           http_exchanges=[])


def elapsed():
    return now - started + time.monotonic() - clock


def emit(status, detail=None):
    out['status'] = status
    out['passed'] = status == 'ready'
    out['total_s'] = round(elapsed(), 3)
    if detail:
        out['detail'] = detail
    print(json.dumps(out, indent=2))
    sys.exit(0 if out['passed'] else 1)


guard = None
if process_owner:
    sys.path.insert(0, campaign_path)
    from process_readiness import ProcessFailure, ProcessGuard
    try:
        guard = ProcessGuard(process_owner)
    except ProcessFailure as error:
        emit(error.status, str(error))


def process_failure():
    if guard is not None:
        try:
            guard.check()
        except ProcessFailure as error:
            return error
    return None


def request(path, payload=None):
    failure = process_failure()
    if failure:
        emit(failure.status, str(failure))
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        emit('timeout', 'process-start deadline exhausted')
    cmd = ['curl', '--silent', '--show-error', '--max-time', str(remaining),
           '--write-out', '\n%{http_code}']
    if payload is None:
        cmd += ['--max-time', str(min(5, remaining))]
    else:
        cmd += ['--fail-with-body', '--header', 'Content-Type: application/json', '--data-binary', '@-']
    cmd.append(url + path)
    request_json = None if payload is None else json.dumps(payload)
    # Poll the pinned leader even while curl waits for headers or a completion.
    # Only this probe's direct child is killed; run_cell owns engine teardown.
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as probe:
        pending_input = request_json
        while True:
            try:
                stdout, stderr = probe.communicate(input=pending_input,
                                                   timeout=0.1 if guard else None)
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                failure = process_failure()
                if failure:
                    probe.kill()
                    stdout, stderr = probe.communicate()
                    break
        resp = subprocess.CompletedProcess(cmd, probe.returncode, stdout, stderr)
    raw, separator, code = resp.stdout.rpartition('\n')
    resp.http_code = code if separator else '000'
    resp.stdout = raw if separator else resp.stdout
    out['http_exchanges'].append(dict(
        path=path, request_json=request_json,
        response_status=int(code) if separator and code.isdigit() and code != '000' else None,
        response_body=resp.stdout, response_complete=resp.returncode in (0, 22)))
    failure = failure or process_failure()
    if failure:
        emit(failure.status, str(failure))
    return resp


while time.monotonic() < deadline:
    resp = request('/health')
    code = resp.http_code
    if code == '000':
        # Includes refusal, reset and timeout; curl does not distinguish them
        # through its HTTP code. Preserve the exit code rather than guessing.
        code = 'transport-error'
    out['polls'] += 1
    if code not in out['http_codes_seen']:
        out['http_codes_seen'].append(code)
    if resp.returncode:
        out['last_health_curl_exit'] = resp.returncode
    if time.monotonic() >= deadline:
        emit('timeout', 'health did not pass within process-start deadline')
    if resp.returncode == 0 and code == '200':
        out['time_to_ready_s'] = round(elapsed(), 3)
        break
    time.sleep(min(1, max(0, deadline - time.monotonic())))
else:
    emit('timeout', 'health did not pass within process-start deadline')

payload = dict(model=model, messages=[dict(role='user', content='hi')], max_tokens=1,
               temperature=0.0, seed=42, presence_penalty=0.0, frequency_penalty=0.0,
               chat_template_kwargs={'enable_thinking': False})
token_start = time.monotonic()
resp = request('/v1/chat/completions', payload)
if time.monotonic() >= deadline or resp.returncode == 28:
    emit('timeout', 'first completion exceeded process-start deadline')
if resp.returncode:
    emit('first-token-failed', f'curl exit {resp.returncode}: {resp.stderr.strip()}')
try:
    result = json.loads(resp.stdout)
    if not isinstance(result, dict) or result.get('error'):
        raise ValueError('completion returned an error or a non-object')
    content = result['choices'][0]['message']['content']
    if not isinstance(content, str) or not content.strip():
        raise ValueError('empty first-token reply')
except (ValueError, KeyError, IndexError, TypeError) as exc:
    emit('first-token-failed', f'invalid first-token completion: {exc}')
# This is one-token, non-streaming completion latency, including response
# framing. It is not the ladder's SSE TTFT measurement.
out['first_token_s'] = round(time.monotonic() - token_start, 3)
failure = process_failure()
if failure:
    emit(failure.status, str(failure))
emit('ready')
PY
}

selftest() {
  # Validate the INSTRUMENT against a known case: a stub that reports loading
  # twice and then ready. The measured value must be >= 2 s (two 1 s polls at
  # minimum), the JSON must parse, and both non-200 shapes must appear in the
  # codes list -- otherwise the poll loop is silently treating a loading
  # response as readiness, which is the failure that would make every boot
  # number in the campaign too small.
  local dir port stub_pid out rc
  dir="$(mktemp -d)"
  port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
  cat > "$dir/stub.py" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {"health_hits": 0}

class H(BaseHTTPRequestHandler):
    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        STATE["health_hits"] += 1
        # Atlas's shape: 503 loading, 503 loading, then 200 ready.
        if STATE["health_hits"] <= 2:
            return self._send(503, {"status": "loading"})
        self._send(200, {"status": "ready"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send(200, {"choices": [{"message": {"content": "x"},
                                      "finish_reason": "length"}]})

    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
  python3 "$dir/stub.py" "$port" & stub_pid=$!
  # Wait for the stub to bind before starting the clock, so the selftest
  # measures the poll loop rather than python's import time.
  python3 - "$port" <<'PY'
import socket, sys, time
deadline = time.time() + 10
while time.time() < deadline:
    try:
        socket.create_connection(("127.0.0.1", int(sys.argv[1])), 0.2).close()
        sys.exit(0)
    except OSError:
        time.sleep(0.05)
sys.exit("stub never bound")
PY
  URL="http://127.0.0.1:$port" MODEL="stub-model" ENGINE="selftest" TIMEOUT_S=30
  out="$(measure)"; rc=$?
  kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null
  rm -rf "$dir"
  [ $rc -eq 0 ] || { echo "SELFTEST FAIL: measure exited $rc" >&2; return 1; }
  echo "$out"
  python3 - "$out" <<'PY' || return 1
import json, sys
d = json.loads(sys.argv[1])
assert d["status"] == "ready", d
assert d["time_to_ready_s"] >= 2.0, f"two loading polls must cost >= 2 s, got {d['time_to_ready_s']}"
assert d["polls"] >= 3, d
assert "503" in d["http_codes_seen"], f"loading state never observed: {d}"
assert "200" in d["http_codes_seen"], d
assert d["first_token_s"] is not None, d
print("SELFTEST OK: measured", d["time_to_ready_s"], "s over", d["polls"], "polls", d["http_codes_seen"])
PY
}

if [ "$SELFTEST" = "1" ]; then
  selftest || exit $?
  python3 "$(dirname "$0")/readiness_selftest.py"
  exit $?
fi

[ -n "$URL" ] || { echo "--url is required" >&2; usage >&2; exit 2; }
URL="${URL%/}"

if [ -n "$OUT" ]; then
  measure | tee "$OUT"
  exit "$?"  # pipefail preserves either measurement failure or tee write failure
fi
measure
