# Completion lifecycle soak

`soak.py` exercises a running server using Python's standard library. It does
not start or stop the server. Use an isolated development server and record its
binary, model, launch flags, process identifiers, and logs alongside the receipts.
This is development validation, not a benchmark certification campaign.

```bash
python3 scripts/k3/soak.py \
  --endpoint http://127.0.0.1:8000 --model kimi-k3 \
  --cases scripts/k3/smoke-cases.json \
  --baseline-receipt docs/k3/evidence/spark-prep-20260919/generation.json \
  --cycles 10 --concurrency 2 --cancel-max-tokens 128 \
  --request-deadline 60 --deadline 1800 \
  --output /tmp/k3-soak-new-run
```

The optional baseline receipt above is specific to the packed small K3 fixture
used in Spark preparation. Do not use its expected text for the official K3
checkpoint. Without a trusted receipt, the tool checks consistency against a
fresh isolated baseline; consistent but incorrect generation can therefore pass.
Use independent known outputs for the model under test when available.

The output directory must not exist. The tool creates an append-only
`receipts.jsonl` and final `summary.json`. Every completed or failed request is
flushed immediately. A failed assertion or deadline produces a nonzero exit and
preserves partial evidence. A forcibly killed harness can leave only JSONL.

The tool first generates every case in isolation. Each cycle then:

1. Repeats every case, comparing text, finish reason, and token counts with its
   isolated baseline. Timing and cache hit metadata can legitimately differ.
2. Streams a rotating case, comparing assembled text and finish reason. Each
   stream must contain a terminal choice followed by `[DONE]`; malformed,
   oversized, error, or incomplete streams fail.
3. Starts the rotating case with the larger cancellation token budget and closes
   the connection after the first nonempty content event. A first content event
   already marked terminal does not count as cancellation. It then reruns the
   first canary and compares the result.
4. Launches the configured number of client requests together, comparing each
   result to its isolated baseline.

With 16 cases, 10 cycles, and concurrency 2 this issues 226 requests. Keep both
prompt length and the cancellation token budget within the server context limit.
The default cancellation budget is 128 tokens. Every HTTP request runs in a
separate subprocess, with a hard wall-clock deadline that also bounds trickling
responses. The overall deadline bounds the scheduled work; process cleanup and
receipt writing can add a small amount of wall time.

Client cancellation proves that the client closed its connection and a later
canary succeeded. Inspect server logs for request release and slot reuse; this
alone does not prove immediate GPU cancellation. Concurrent client launches do
not prove overlap inside the scheduler or GPU batching. Streaming uses the text
completion API; chat rendering and tool calls need separate tests.

Sample the server's process RSS and available memory separately before warmup,
during the cycles, and after quiescence. Pair any growth with server logs and
allocation ownership before calling it a leak. This tool does not measure memory
or claim leak freedom. On unified-memory Spark devices, GPU memory telemetry may
be unavailable; do not substitute a zero for an unavailable measurement.

Run the tool's CPU protocol tests with:

```bash
python3 -m unittest discover -s scripts/k3 -p test_soak.py
```
