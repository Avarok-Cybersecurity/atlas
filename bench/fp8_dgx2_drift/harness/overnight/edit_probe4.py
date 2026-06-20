#!/usr/bin/env python3
"""Edit-flow probe v4 — sustained MULTI-TURN session (claude-code).

Targets the documented "degrades after 5-8 turns" FP8 weak spot. One claude
session must implement 8 stubs ONE AT A TIME, running `cargo test` after EACH —
forcing ~16+ tool round-trips with monotonically growing context. Watches for
late-session degradation: a stub left unimplemented, a corrupted/clobbered edit,
a loop/garble, or a crash. Each stub is trivial so failure points at multi-turn
mechanics, not competence.

Runs N times (default 2) to separate model temp-variance (stochastic no-op) from
a real Atlas defect. Writes /workspace/editprobe4_run<i>.json.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

SHARED = "/tmp/ovn-cargo-target"
AGENT_TIMEOUT = int(os.environ.get("EDIT4_TIMEOUT", "700"))
CLAUDE_BIN = "/workspace/.local/bin/claude"
NRUNS = int(os.environ.get("EDIT4_NRUNS", "2"))

STUBS = [("add", "a + b", "i64, b: i64", "i64"),
         ("sub", "a - b", "i64, b: i64", "i64"),
         ("mul", "a * b", "i64, b: i64", "i64"),
         ("max2", "if a > b { a } else { b }", "i64, b: i64", "i64"),
         ("min2", "if a < b { a } else { b }", "i64, b: i64", "i64"),
         ("is_pos", "a > 0", "i64", "bool"),
         ("abs_v", "if a < 0 { -a } else { a }", "i64", "i64"),
         ("clamp_nonneg", "if a < 0 { 0 } else { a }", "i64", "i64")]


def lib_src():
    out = ["//! 8 unimplemented functions. Implement them ONE AT A TIME.", ""]
    for name, _, params, ret in STUBS:
        p0 = "a: " + params
        out += [f"pub fn {name}({p0}) -> {ret} {{", f'    todo!("{name}")', "}", ""]
    out += ["#[cfg(test)]", "mod tests {", "    use super::*;",
            "    #[test] fn t_add() { assert_eq!(add(2,3),5); }",
            "    #[test] fn t_sub() { assert_eq!(sub(5,3),2); }",
            "    #[test] fn t_mul() { assert_eq!(mul(4,3),12); }",
            "    #[test] fn t_max() { assert_eq!(max2(4,9),9); }",
            "    #[test] fn t_min() { assert_eq!(min2(4,9),4); }",
            "    #[test] fn t_pos() { assert!(is_pos(3)); assert!(!is_pos(-1)); }",
            "    #[test] fn t_abs() { assert_eq!(abs_v(-7),7); }",
            "    #[test] fn t_clamp() { assert_eq!(clamp_nonneg(-2),0); assert_eq!(clamp_nonneg(5),5); }",
            "}", ""]
    return "\n".join(out)


PROMPT = (
    "src/lib.rs has 8 unimplemented functions (add, sub, mul, max2, min2, is_pos, "
    "abs_v, clamp_nonneg), each a todo!() stub. Implement them ONE AT A TIME, in "
    "order. After implementing EACH single function, run `cargo test` before moving "
    "to the next one (so you will run cargo test about 8 times total). Edit src/lib.rs "
    "in place for each — do not rewrite the whole file and do not implement more than "
    "one function before re-running the tests. When all 8 are done and `cargo test` is "
    "fully green, stop."
)


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


def one(i):
    t = pathlib.Path(f"/tmp/editprobe4-run{i}")
    subprocess.run(["rm", "-rf", str(t)])
    (t / "src").mkdir(parents=True, exist_ok=True)
    (t / "Cargo.toml").write_text(
        '[package]\nname="turns"\nversion="0.1.0"\nedition="2021"\n\n[dependencies]\n')
    (t / "src" / "lib.rs").write_text(lib_src())
    subprocess.run(["chmod", "-R", "777", str(t)])
    print(f"[run{i}] start @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    run(["sudo", "-n", "-u", "claude", "timeout", "-k", "10", str(AGENT_TIMEOUT),
         "env", "ANTHROPIC_BASE_URL=http://localhost:8888", "ANTHROPIC_AUTH_TOKEN=dummy",
         f"CARGO_TARGET_DIR={SHARED}", CLAUDE_BIN, "-p", "--output-format", "json",
         "--permission-mode", "bypassPermissions"],
        cwd=str(t), stdin=PROMPT, timeout=AGENT_TIMEOUT + 30)
    rc, out, err = run(["bash", "-lc", "cargo test 2>&1"], cwd=str(t), timeout=600,
                       env={"CARGO_TARGET_DIR": SHARED})
    src = (t / "src/lib.rs").read_text()
    code = [l for l in src.splitlines() if not l.lstrip().startswith("//")]
    remaining = [n for (n, _, _, _) in STUBS
                 if f'todo!("{n}")' in src]
    rec = {"run": i, "complete": rc == 0, "wall_s": round(time.time() - t0, 1),
           "stubs_remaining": remaining, "n_remaining": len(remaining),
           "lines": len(src.splitlines()), "target": str(t),
           "check_tail": "" if rc == 0 else (out[-1200:] + "\n" + err[-400:]).strip()}
    print(f"[run{i}] complete={rec['complete']} remaining={remaining} wall={rec['wall_s']}s", flush=True)
    pathlib.Path(f"/workspace/editprobe4_run{i}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    for i in range(1, NRUNS + 1):
        one(i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
