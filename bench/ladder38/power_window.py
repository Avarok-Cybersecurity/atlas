# SPDX-License-Identifier: AGPL-3.0-only
"""In-window GPU-rail energy for the concurrency ladder.

★ WHY THIS EXISTS. The ladder used to record `clock_sample_at_rep_start`, a
single `nvidia-smi` shot whose comment claimed it sampled "INSIDE the rep
window, not before it". It did the opposite: `os.popen(...).read()` blocks to
completion BEFORE `await run_rep` issues a request, so every value was the
driver's 1-second trailing average over the *previous* batch's tail. Two
engines were then compared on it, and the resulting "Atlas draws 2x vLLM"
finding was an artifact. Three properties made it unusable:

  1. the sample never covered the rep it was attached to;
  2. one sample per rep, +-5 W stated accuracy, with same-rung rep-to-rep
     scatter reaching 20 W -- repeatability ~30% of the reading;
  3. what it captured differed BY ENGINE, because it caught each engine's
     end-of-batch behaviour (vLLM ends all requests on one step; Atlas
     drains), so it was an asymmetry of the instrument, not of the workload.

This module replaces it with a real integral over the measured window.

WHAT IS AND IS NOT MEASURED. `nvidia-smi` exposes the GPU rail only on GB10:
`power.limit`, Module Power and GPU Memory Power all read N/A, and
`Power Samples: Not Found`. Grace CPU and LPDDR5X are NOT in this number, and
under load the GPU rail is roughly half of system power. Every figure here is
therefore a LOWER BOUND on system energy, and the omitted half is where
CPU-side serving work lives -- which differs between a Python stack and a Rust
one. Say "GPU rail" wherever these numbers are shown.
"""

import json
import shutil
import subprocess
import threading
import time

# Fields sampled together. `power.draw.average` is the driver's 1 s trailing
# mean and is what we integrate (already low-passed, so alias-safe at 4 Hz);
# `.instant` is the raw sensor and is kept only to report the lag between
# them -- they disagreed by 22 W in one call on this box, so a run that
# quotes one must say which.
_QUERY = (
    "power.draw.average,power.draw.instant,clocks.sm,utilization.gpu,"
    "temperature.gpu,clocks_event_reasons.sw_power_cap,"
    "clocks_event_reasons.hw_power_brake_slowdown"
)
_PERIOD_MS = 250

# `.average` lags its own window, so a short measurement is mostly edges.
# Below this we record the samples but refuse to derive J/token from them.
MIN_TRUSTWORTHY_WINDOW_S = 10.0


def _as_float(tok):
    tok = tok.strip()
    if not tok or tok in ("[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _as_active(tok):
    return tok.strip().lower() in ("active", "1", "true")


class PowerWindow:
    """One long-lived `nvidia-smi -lms` child, read on a background thread.

    Deliberately NOT a shot per sample: spawning `nvidia-smi` costs ~10-20 ms
    of CPU each time, and at 4 Hz beside a benchmark whose TTFT ceilings are
    being tightened to +5% that is a visible fraction of a core perturbing the
    thing it measures. In loop mode the same tool costs ~0.2% of one core.
    """

    def __init__(self, period_ms=_PERIOD_MS):
        self.period_ms = period_ms
        self._proc = None
        self._thread = None
        self._rows = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.unavailable = None
        if shutil.which("nvidia-smi") is None:
            self.unavailable = "nvidia-smi not on PATH"

    def _reader(self):
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = line.split(",")
            if len(parts) < 7:
                continue
            row = (
                time.perf_counter(),
                _as_float(parts[0]),
                _as_float(parts[1]),
                _as_float(parts[2]),
                _as_float(parts[3]),
                _as_float(parts[4]),
                _as_active(parts[5]),
                _as_active(parts[6]),
            )
            if row[1] is None:
                continue
            with self._lock:
                self._rows.append(row)

    def __enter__(self):
        if self.unavailable:
            return self
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={_QUERY}",
                 "--format=csv,noheader,nounits", f"--loop-ms={self.period_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except OSError as e:
            self.unavailable = f"spawn failed: {e}"
            return self
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False

    def snapshot(self):
        with self._lock:
            return list(self._rows)

    def integrate(self, t_start, t_end):
        """Trapezoidal integral of GPU-rail power over [t_start, t_end].

        Returns a dict that is ALWAYS explicit about why a number is missing.
        An absent key is never a zero: a zero joule count reads as "free",
        which is the one thing this must never claim.
        """
        window_s = t_end - t_start
        if self.unavailable:
            return {"gpu_rail_status": f"unavailable: {self.unavailable}"}
        rows = [r for r in self.snapshot() if t_start <= r[0] <= t_end]
        if len(rows) < 2:
            return {"gpu_rail_status": f"too few samples ({len(rows)}) in a {window_s:.1f}s window",
                    "gpu_rail_power_samples": len(rows),
                    "gpu_rail_window_s": window_s}
        rows.sort(key=lambda r: r[0])
        energy_j = 0.0
        gaps = []
        for a, b in zip(rows, rows[1:]):
            dt = b[0] - a[0]
            gaps.append(dt)
            energy_j += 0.5 * (a[1] + b[1]) * dt   # trapezoid, not hold-forward
        covered = rows[-1][0] - rows[0][0]
        powers = [r[1] for r in rows]
        lag = [abs(r[1] - r[2]) for r in rows if r[2] is not None]
        period_s = self.period_ms / 1000.0
        out = {
            "gpu_rail_status": "ok",
            "gpu_rail_energy_j": energy_j,
            "gpu_rail_mean_power_w": energy_j / covered if covered > 0 else None,
            "gpu_rail_max_power_w": max(powers),
            "gpu_rail_min_power_w": min(powers),
            "gpu_rail_power_samples": len(rows),
            "gpu_rail_window_s": window_s,
            "gpu_rail_covered_s": covered,
            # What fraction of the window the samples actually span. A number
            # derived from partial coverage is not an integral of that window.
            "gpu_rail_coverage_frac": covered / window_s if window_s > 0 else None,
            "gpu_rail_sample_period_ms": self.period_ms,
            "gpu_rail_max_gap_s": max(gaps),
            # `.average` is a 1 s trailing mean, so the first and last second
            # of any window are partly outside it. Report the size of that
            # edge effect rather than pretending it is absent.
            "gpu_rail_edge_error_frac": 1.0 / window_s if window_s > 0 else None,
            "gpu_rail_avg_instant_lag_w": (sum(lag) / len(lag)) if lag else None,
            "gpu_rail_sw_power_cap_frac": sum(1 for r in rows if r[6]) / len(rows),
            "gpu_rail_hw_power_brake_frac": sum(1 for r in rows if r[7]) / len(rows),
            "sm_clock_mhz_mean": sum(r[3] for r in rows if r[3] is not None) / max(1, sum(1 for r in rows if r[3] is not None)),
            "gpu_util_mean_pct": sum(r[4] for r in rows if r[4] is not None) / max(1, sum(1 for r in rows if r[4] is not None)),
            "gpu_temp_c_max": max((r[5] for r in rows if r[5] is not None), default=None),
        }
        # Refusals, stated rather than silently tolerated.
        reasons = []
        if window_s < MIN_TRUSTWORTHY_WINDOW_S:
            reasons.append(f"window {window_s:.1f}s < {MIN_TRUSTWORTHY_WINDOW_S}s")
        if max(gaps) > 2 * period_s:
            reasons.append(f"max sample gap {max(gaps):.2f}s > 2x period")
        if out["gpu_rail_coverage_frac"] is not None and out["gpu_rail_coverage_frac"] < 0.9:
            reasons.append(f"coverage {out['gpu_rail_coverage_frac']:.2f} < 0.90")
        out["gpu_rail_trustworthy"] = not reasons
        if reasons:
            out["gpu_rail_untrustworthy_because"] = reasons
        return out


def measure_idle_baseline(window, seconds=10.0):
    """Verified-idle baseline: model resident, nothing in flight.

    Verified, not assumed. A baseline taken right after model load catches a
    clocked-up GPU and would subtract far too much. Requires util 0 and a low
    clock across the whole sample, and says so when it cannot get them.
    """
    if window.unavailable:
        return {"gpu_rail_idle_status": f"unavailable: {window.unavailable}"}
    t0 = time.perf_counter()
    time.sleep(seconds)
    rows = [r for r in window.snapshot() if r[0] >= t0]
    if len(rows) < 4:
        return {"gpu_rail_idle_status": f"too few idle samples ({len(rows)})"}
    utils = [r[4] for r in rows if r[4] is not None]
    clocks = [r[3] for r in rows if r[3] is not None]
    busy = [u for u in utils if u > 0]
    if busy:
        return {"gpu_rail_idle_status": f"not idle: utilization reached {max(busy)}%"}
    if clocks and max(clocks) > 400:
        return {"gpu_rail_idle_status": f"not idle: sm clock reached {max(clocks)} MHz"}
    p = [r[1] for r in rows]
    return {
        "gpu_rail_idle_status": "ok",
        "gpu_rail_idle_power_w": sum(p) / len(p),
        "gpu_rail_idle_samples": len(rows),
    }
