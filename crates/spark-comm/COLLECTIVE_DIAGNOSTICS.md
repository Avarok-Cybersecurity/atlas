# Collective diagnostics for rental bring-up

Set `AVAROK_COMM_DIAGNOSTICS=1` on every rank and enable the
`avarok::comm=info` tracing target (for example through the application's
`RUST_LOG` filter). `0` or unset disables events; other values fail at startup.
No tensor contents or device buffer addresses are logged.

Each host submission emits rank, world size, local sequence number, operation,
dtype, element count, bytes, CUDA stream and optional peer/root. `count` has
NCCL's operation-specific meaning: per-rank send count for all-gather,
per-rank receive count for reduce-scatter, and buffer count for broadcast and
all-reduce. The all-reduce dtype is BF16; gather, scatter and point-to-point
operations retain their existing byte/U8 API. Barrier uses a zero-count F32
all-reduce. These events describe submissions, not GPU completion. Capturing
or replaying CUDA graphs does not emit an event for each replay.

For a controlled mismatch experiment, disable graph capture, retain separate
logs for every rank, and change one rank's collective operation or count in
the test harness. Compare the first divergent sequence across ranks. This
logging does not add a peer handshake or automatically detect cross-rank
mismatches. Local invalid group/root/peer values and non-integral BF16 byte
counts fail before NCCL; valid but different counts on two ranks may hang.

Broadcast's synchronous completion now polls `cuStreamQuery` and NCCL async
error state, stopping after the existing 30-second deadline instead of timing
an unbounded `cuStreamSynchronize` after it returns. A timeout or completion
error marks the communicator unhealthy and returns rank/root/byte context.
Only the first worker command word uses an idle receive without a request
deadline; it continues polling transport errors. Later command words and
payloads retain the 30-second completion deadline. This prevents healthy
workers from being poisoned merely because the server has been idle.

Subsequent collective submissions fail until the existing explicit reconnect
path is used; this change does not initiate reconnect or abort automatically.

The deadline bounds that polling loop only. NCCL submission/bootstrap, a
wedged driver call, asynchronous all-reduce, graph replay and communicator
teardown are not made globally time-bounded. The rental launcher must apply
an external deadline and terminate **all ranks** on one rank's failure,
escalating to a forced process kill after a bounded grace period. Do not
retry one rank against peers still running an old collective sequence.

CPU checks: `cargo test -p spark-comm --no-default-features` exercises the
production descriptor validation, polling deadline and poison guard without
linking CUDA/NCCL. These tests do not establish GPU failure containment;
run the matched 2/4/8-rank collective checks, deliberate mismatch and killed
rank experiments under an external deadline before full-model inference.
