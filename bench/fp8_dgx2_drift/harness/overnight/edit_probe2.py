#!/usr/bin/env python3
"""Edit-flow probe v2 — harder, Atlas-bug-seeking scenarios (claude-code only;
opencode is currently down with the known stale-cache hang).

Scenarios:
  S1 longctx_edit  : a ~600-line lib.rs (60 tiny fns) with ONE buried todo!()
                     stub. Forces long prefill (deep-layer FP8 drift, #211) AND
                     an Edit whose old_string must EXACTLY match a line deep in
                     context — if drift corrupts the model's reproduction of
                     old_string, the Edit fails to match -> retries / multi-turn
                     degradation. Highest Atlas-bug yield.
  S2 multifile_rename: rename a fn across 2 files + its call site. Stresses
                     multiple Edit calls across files in one session.

Each: seed -> run claude-code (Read+Edit+Bash) -> cargo test -> inspect.
Writes /workspace/editprobe2_<scenario>.json. Coding is trivial on purpose so a
failure points at Atlas mechanics, not model competence.
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

SHARED = "/tmp/ovn-cargo-target"
AGENT_TIMEOUT = int(os.environ.get("EDIT2_TIMEOUT", "480"))
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
    t = pathlib.Path(f"/tmp/editprobe2-{name}")
    subprocess.run(["rm", "-rf", str(t)])
    (t / "src").mkdir(parents=True, exist_ok=True)
    return t


def write(t, rel, content):
    p = t / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


CARGO = "[package]\nname = \"{}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n"


# ── S1: long-context edit ────────────────────────────────────────────
def seed_longctx(t):
    lines = ['//! Large utility crate. Exactly ONE function below is a stub',
             '//! (`todo!()`); every other function is already implemented.', '']
    STUB = 37
    for i in range(60):
        if i == STUB:
            lines += [f'/// Stepwise value number {i}: return base plus {i}.',
                      f'pub fn step_{i}(base: i64) -> i64 {{',
                      f'    todo!("return base + {i}")',
                      '}', '']
        else:
            lines += [f'/// Stepwise value number {i}: return base plus {i}.',
                      f'pub fn step_{i}(base: i64) -> i64 {{',
                      f'    base + {i}',
                      '}', '']
    lines += ['#[cfg(test)]', 'mod tests {', '    use super::*;',
              '    #[test] fn t_stub() { assert_eq!(step_37(100), 137); }',
              '    #[test] fn t_other() { assert_eq!(step_3(10), 13); assert_eq!(step_59(0), 59); }',
              '}', '']
    write(t, "Cargo.toml", CARGO.format("longkit"))
    write(t, "src/lib.rs", "\n".join(lines))
    return ("src/lib.rs contains 60 functions step_0..step_59. Exactly ONE, `step_37`, is an "
            "unimplemented `todo!()` stub; all others are implemented. Read the file, find "
            "step_37, and EDIT only its body in place so it returns `base + 37` (matching the "
            "pattern of the neighboring functions). Do not touch any other function and do not "
            "rewrite the file. Then run `cargo test` to confirm all tests pass.")


def check_longctx(t):
    src = (t / "src/lib.rs").read_text()
    code = [l for l in src.splitlines() if not l.lstrip().startswith("//")]
    return {"still_todo": any("todo!(" in l for l in code),
            "lines": len(src.splitlines()),
            # did they accidentally clobber a neighbor?
            "step36_ok": "base + 36" in src, "step38_ok": "base + 38" in src}


# ── S2: multi-file rename ────────────────────────────────────────────
def seed_multifile(t):
    write(t, "Cargo.toml", CARGO.format("renamekit"))
    write(t, "src/lib.rs",
          "pub mod math;\n\npub fn compute(x: i64) -> i64 {\n"
          "    // calls the helper in math.rs\n    math::doubler(x) + 1\n}\n\n"
          "#[cfg(test)]\nmod tests {\n    use super::*;\n"
          "    #[test] fn t() { assert_eq!(compute(10), 21); }\n}\n")
    write(t, "src/math.rs",
          "/// Doubles the input. NOTE: this name is being changed to `times_two`.\n"
          "pub fn doubler(x: i64) -> i64 {\n    x * 2\n}\n")
    return ("This crate has src/lib.rs and src/math.rs. Rename the function `doubler` to "
            "`times_two` EVERYWHERE: its definition in src/math.rs AND its call site in "
            "src/lib.rs (compute). Edit both files in place; do not change behavior. Then run "
            "`cargo test` to confirm it still passes.")


def check_multifile(t):
    lib = (t / "src/lib.rs").read_text()
    math = (t / "src/math.rs").read_text()
    return {"no_doubler_left": "doubler" not in lib and "doubler" not in math,
            "times_two_def": "fn times_two" in math,
            "times_two_call": "times_two" in lib}


SCENARIOS = {
    "longctx": (seed_longctx, check_longctx),
    "multifile": (seed_multifile, check_multifile),
}


def run_one(name):
    seed_fn, check_fn = SCENARIOS[name]
    t = fresh(name)
    prompt = seed_fn(t)
    subprocess.run(["chmod", "-R", "777", str(t)])
    seed_lines = len((t / "src/lib.rs").read_text().splitlines())
    print(f"[{name}] seed lib.rs={seed_lines} lines; running claude @ {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    claude(t, prompt)
    ok, tail = cargo_test(t)
    extra = check_fn(t)
    rec = {"scenario": name, "complete": ok, "wall_s": round(time.time() - t0, 1),
           "target": str(t), **extra, "check_tail": "" if ok else tail}
    print(f"[{name}] complete={ok} {extra} wall={rec['wall_s']}s", flush=True)
    pathlib.Path(f"/workspace/editprobe2_{name}.json").write_text(json.dumps(rec, indent=2))
    return rec


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    for n in names:
        run_one(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
