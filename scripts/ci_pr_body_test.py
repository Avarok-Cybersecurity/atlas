#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the actual workflow body sanitizer, including pipe-sized inputs."""

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"


def sanitizer_step():
    text = WORKFLOW.read_text()
    step = text.split("      - name: Sanitize the PR body\n", 1)[1]
    run = step.split("        run: |\n", 1)[1]
    lines = []
    for line in run.splitlines(keepends=True):
        if line.strip() and not line.startswith("          "):
            break
        lines.append(line)
    return textwrap.dedent("".join(lines))


class BodySanitizerTests(unittest.TestCase):
    def run_step(self, body, *, fail_perl=False):
        with tempfile.TemporaryDirectory(prefix="atlas-pr-body-") as directory:
            root = Path(directory)
            output = root / "output"
            env = dict(os.environ, PR_BODY=body, GITHUB_OUTPUT=str(output))
            if fail_perl:
                perl = root / "perl"
                perl.write_text("#!/bin/sh\ncat >/dev/null\nexit 73\n")
                perl.chmod(0o755)
                env["PATH"] = str(root) + os.pathsep + env["PATH"]
            result = subprocess.run(
                ["bash", "-c", sanitizer_step()],
                cwd=root,
                env=env,
                capture_output=True,
                timeout=10,
                check=False,
            )
            data = (root / "body.txt").read_bytes()
            emitted = output.read_bytes() if output.exists() else None
            return result, data, emitted, (root / "injected").exists()

    def assert_sanitized(self, body, expected):
        result, data, emitted, injected = self.run_step(body)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stderr, b"")
        self.assertEqual(data, expected)
        self.assertLessEqual(len(data), 2000)
        self.assertFalse(injected)
        first, payload = emitted.split(b"\n", 1)
        self.assertRegex(first, rb"^text<<__ATLAS_EOF_[0-9]+$")
        delimiter = first.removeprefix(b"text<<")
        framed = data + (b"\n" if data and not data.endswith(b"\n") else b"")
        self.assertEqual(payload, framed + delimiter + b"\n")

    def test_empty_short_and_newline_framing(self):
        for body in ("", "hello", "hello\n", "one\ntwo\n"):
            with self.subTest(body=body):
                self.assert_sanitized(body, body.encode())

    def test_long_bodies_across_pipe_buffer_boundaries(self):
        # The old early-closing head consumer failed intermittently on Linux.
        for size in (2000, 2001, 8000, 10000, 12000, 16384, 20000, 60000):
            for repetition in range(3):
                with self.subTest(size=size, repetition=repetition):
                    self.assert_sanitized("x" * size, b"x" * min(size, 2000))

    def test_controls_and_multiline_comments_removed_before_cap(self):
        body = "\x01\x1bvisible\t\n<!--" + "hidden\n" * 2000 + "-->after\x7f"
        self.assert_sanitized(body, b"visible\t\nafter")

    def test_byte_limit_preserves_existing_utf8_truncation(self):
        body = "z" * 1999 + "\u00e9" * 4000
        self.assert_sanitized(body, body.encode()[:2000])

    def test_shell_metacharacters_remain_data(self):
        body = "$(touch injected) `touch injected` ${PATH}\nEOF\n"
        self.assert_sanitized(body, body.encode())

    def test_real_upstream_failure_is_not_suppressed(self):
        result, _, emitted, injected = self.run_step("text", fail_perl=True)
        self.assertEqual(result.returncode, 73)
        self.assertIsNone(emitted)
        self.assertFalse(injected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
