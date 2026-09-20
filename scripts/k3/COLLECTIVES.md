# Compare collective submission logs

Enable `AVAROK_COMM_DIAGNOSTICS=1` and the `avarok::comm=info` tracing target
on every rank. Capture one fresh process log per rank, starting at sequence 0.
Do not combine logs from separate launches or communicators. Both ordinary
ANSI-colored tracing text and tracing JSON lines are accepted automatically.

```bash
python3 scripts/k3/check_collectives.py --world-size 2 \
  --rank-log 0=/path/to/rank0.log --rank-log 1=/path/to/rank1.log
```

Supply all 8 rank logs for a TP8 run. When the harness knows the exact number of
submissions per rank, add `--expected-submissions N`; this also detects the
case where every log was truncated at the same point. Without an independent
expected count, identical truncated prefixes cannot be distinguished from a
complete run.

Exit 0 with `MATCH_SUBMISSIONS` means the recorded operations agree. Exit 1
reports the first malformed, missing, truncated or mismatched observation,
including rank/sequence or source/destination channel. The checker requires
contiguous local sequences starting at 0 and validates dtype/count/bytes.
It compares collective order, operation, dtype, count and broadcast root.
Device stream values are deliberately ignored. Send/receive operations are
paired FIFO by source/destination and checked for matching dtype/count;
those ranks' local sequence numbers need not coincide.

This is an offline submission check, **not evidence of GPU completion or
freedom from deadlocks**. In particular, mixed point-to-point/collective
ordering can deadlock despite matching descriptors, and CUDA graph replays
are not represented by these host events. Keep the external all-rank timeout,
process exit status and numerical oracle checks. Run on quiesced final logs;
a live log may be reported as truncated while another rank is still writing.

The CPU tests include deliberate count/root mismatches, missing tails,
interior sequence gaps, malformed events and mismatched peer transfers:

```bash
python3 -m unittest discover -s scripts/k3 -p test_check_collectives.py -v
```
