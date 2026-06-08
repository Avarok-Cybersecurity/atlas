#!/usr/bin/env python3
"""Edit-flow probe v3 — push the two paths most likely to surface a real Atlas
defect (claude-code; opencode down).

  bigedit  : ~1000-line file (~13k tokens) with ONE buried todo!() stub near the
             middle. Forces DEEP prefill where FP8 deep-layer drift (#211) is
             worst AND an Edit whose old_string must EXACTLY match a deep line.
             If drift corrupts the model's reproduction of old_string, the Edit
             mismatches (or clobbers the wrong region) -> failed edits / churn.
  bigwrite : ask claude to author a single ~400-line file in ONE response — re-
             stresses the CC6 envelope fix at far larger scale than L4 (verify no
             truncation: file compiles + 0 'Stuck in tool-call ENVELOPE').

Trivial coding on purpose -> failures point at Atlas mechanics, not competence.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

SHARED = "/tmp/ovn-cargo-target"
AGENT_TIMEOUT = int(os.environ.get("EDIT3_TIMEOUT", "600"))
CLAUDE_BIN = "/workspace/.local/bin/claude"


def run(cmd, cwd=None, timeout=None, stdin=None, env=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, input=stdin,
                           capture_output=True, text=True,
                           env={**os.environ, **(env or {})})
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or ""), (e.stderr or "") + "\n[timeout]"
    except Exception as e:  # noqa: BLE001
        return 125, "", f"[launch error: {e}]"


def claude(target, prompt):
    return run(["sudo", "-n", "-u", "claude", "timeout", "-k", "10", str(AGENT_TIMEOUT),
                "env", "ANTHROPIC_BASE_URL=http://localhost:8888", "ANTHROPIC_AUTH_TOKEN=dummy",
                f"CARGO_TARGET_DIR={SHARED}", CLAUDE_BIN, "-p", "--output-format", "json",
                "--permission-mode", "bypassPermissions"],
               cwd=str(target), stdin=prompt, timeout=AGENT_TIMEOUT + 30)


def cargo_test(target):
    rc, out, err = run(["bash", "-lc", "cargo test 2>&1"], cwd=str(target), timeout=600,
                       env={"CARGO_TARGET_DIR": SHARED})
    return rc == 0, (out[-1500:] + "\n" + err[-500:]).strip()


def fresh(name):
    t = pathlib.Path(f"/tmp/editprobe3-{name}")
    subprocess.run(["rm", "-rf", str(t)])
    (t / "src").mkdir(parents=True, exist_ok=True)
    return t


CARGO = "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n"


def envelope_cuts():
    rc, out, err = run(["bash", "-lc",
                        "sudo docker logs --since 12m atlas-camp 2>&1 | "
                        "grep -c 'Stuck in tool-call ENVELOPE'"])
    return out.strip()


# ── bigedit: deep-context buried-stub edit ───────────────────────────
def seed_bigedit(t):
    N = 200          # 200 fns * ~5 lines ≈ 1000 lines ≈ ~13k tokens
    STUB = 123
    lines = ['//! Large generated crate; every fn is implemented except ONE stub.', '']
    for i in range(N):
        body = f'    todo!("return base * {i}")' if i == STUB else f'    base * {i}'
        lines += [f'/// scale_{i}: returns base * {i}.',
                  f'pub fn scale_{i}(base: i64) -> i64 {{', body, '}', '']
    lines += ['#[cfg(test)]', 'mod tests {', '    use super::*;',
              f'    #[test] fn t_stub() {{ assert_eq!(scale_{STUB}(2), {2*STUB}); }}',
              '    #[test] fn t_a() { assert_eq!(scale_5(3), 15); }',
              f'    #[test] fn t_b() {{ assert_eq!(scale_{N-1}(1), {N-1}); }}', '}', '']
    (t / "Cargo.toml").write_text(CARGO.format("bigkit"))
    (t / "src" / "lib.rs").write_text("\n".join(lines))
    return (STUB, f"src/lib.rs has {N} functions scale_0..scale_{N-1}. Exactly ONE, "
            f"`scale_{STUB}`, is an unimplemented `todo!()` stub; all others are done. "
            f"Read the file, locate scale_{STUB}, and EDIT only its body in place so it "
            f"returns `base * {STUB}` (matching its neighbors). Do not modify any other "
            f"function and do not rewrite the file. Then run `cargo test`.")


def check_bigedit(t, stub):
    src = (t / "src/lib.rs").read_text()
    code = [l for l in src.splitlines() if not l.lstrip().startswith("//")]
    # neighbor integrity: the fns immediately around the stub must be intact
    return {"still_todo": any("todo!(" in l for l in code),
            "lines": len(src.splitlines()),
            "neighbor122_ok": "base * 122" in src,
            "neighbor124_ok": "base * 124" in src,
            "stub_filled": f"base * {stub}" in src}


# ── bigwrite: large single-shot Write (re-stress CC6) ────────────────
def seed_bigwrite(t):
    (t / "Cargo.toml").write_text(CARGO.format("statemachine"))
    (t / "src" / "lib.rs").write_text("// implement here\n")
    return (None,
            "Create a single Rust file src/lib.rs implementing a small stack-based "
            "calculator for postfix (RPN) integer expressions. Requirements, all in ONE "
            "file written in ONE go: (1) an enum Token { Num(i64), Add, Sub, Mul, Div }; "
            "(2) fn tokenize(&str) -> Result<Vec<Token>, String>; (3) fn eval_rpn(&str) -> "
            "Result<i64, String> that handles +,-,*,/ with division-by-zero and "
            "malformed-input errors; (4) AT LEAST 15 #[cfg(test)] unit tests covering "
            "normal cases, operator precedence via RPN ordering, division by zero, empty "
            "input, and malformed tokens. Make it a substantial, complete single file "
            "(aim for 250+ lines). Then run `cargo test` to confirm everything passes.")


def check_bigwrite(t, _):
    src = (t / "src/lib.rs").read_text()
    return {"lines": len(src.splitlines()),
            "has_eval_rpn": "fn eval_rpn" in src,
            "ends_clean": src.rstrip().endswith("}"),
            "test_count": src.count("#[test]")}


SCEN = {"bigedit": (seed_bigedit, check_bigedit),
        "bigwrite": (seed_bigwrite, check_bigwrite)}


def run_one(name):
    seed_fn, check_fn = SCEN[name]
    t = fresh(name)
    key, prompt = seed_fn(t)
    subprocess.run(["chmod", "-R", "777", str(t)])
    seed_lines = len((t / "src/lib.rs").read_text().splitlines())
    print(f"[{name}] seed={seed_lines} lines; running claude @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    claude(t, prompt)
    ok, tail = cargo_test(t)
    extra = check_fn(t, key)
    rec = {"scenario": name, "complete": ok, "wall_s": round(time.time() - t0, 1),
           "envelope_cuts_12m": envelope_cuts(), "target": str(t), **extra,
           "check_tail": "" if ok else tail}
    print(f"[{name}] complete={ok} cuts={rec['envelope_cuts_12m']} {extra} wall={rec['wall_s']}s", flush=True)
    pathlib.Path(f"/workspace/editprobe3_{name}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    for n in (sys.argv[1:] or list(SCEN)):
        run_one(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
