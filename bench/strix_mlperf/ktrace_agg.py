import csv, collections, sys

PATH = "/tmp/prof_out/ktrace_kernel_trace.csv"
rows = list(csv.DictReader(open(PATH)))
if not rows:
    print("empty trace"); sys.exit(1)

def col(r, *names):
    for n in names:
        if n in r: return r[n]
    return None

starts = [int(col(r, "Start_Timestamp")) for r in rows]
ends = [int(col(r, "End_Timestamp")) for r in rows]
t_end = max(ends)
# The prefill is the final burst. Take the last 30s of GPU activity so model
# load / warmup kernels don't dilute the breakdown.
WINDOW_NS = 30 * 1_000_000_000
cutoff = t_end - WINDOW_NS

agg = collections.defaultdict(lambda: [0, 0])  # name -> [total_ns, count]
for r in rows:
    s = int(col(r, "Start_Timestamp")); e = int(col(r, "End_Timestamp"))
    if s < cutoff: continue
    name = col(r, "Kernel_Name") or "?"
    name = name.split("(")[0].strip()
    agg[name][0] += (e - s); agg[name][1] += 1

total = sum(v[0] for v in agg.values())
print("kernels in window: %d dispatches, %.3f s total GPU time\n" % (
    sum(v[1] for v in agg.values()), total / 1e9))
print("%9s %7s %8s %11s  %s" % ("time_s", "pct", "count", "us/disp", "kernel"))
for name, (ns, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:22]:
    print("%9.3f %6.1f%% %8d %11.1f  %s" % (ns / 1e9, 100.0 * ns / total, cnt, ns / 1e3 / cnt, name[:70]))
