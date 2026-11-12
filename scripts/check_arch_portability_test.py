#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for check_arch_portability.py: the flag merge, the nvcc command line,
the error-line picker and the target enumeration against the real kernels/ tree.

Run: python3 scripts/check_arch_portability_test.py
No nvcc or GPU needed.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_arch_portability as probe  # noqa: E402


class FlagMerge(unittest.TestCase):
    def test_least_specific_first_and_deduped(self):
        merged = probe.merge_flags(["-DA"], ["--fmad=false", "-DA"], ["-DB", "--fmad=false"])
        self.assertEqual(merged, ["-DA", "--fmad=false", "-DB"])


class NvccArgs(unittest.TestCase):
    def job(self):
        return probe.Job(Path("/k/x.cu"), ("-DA",), ("-DA",), ("m/q",))

    def test_matches_the_build_command_line(self):
        os.environ.pop("AVAROK_KERNEL_NO_STRICT", None)
        os.environ.pop("AVAROK_EXTRA_NVCC_FLAGS", None)
        args = probe.nvcc_args(self.job(), "sm_89", Path("/o/x.ptx"))
        self.assertEqual(
            args,
            ["--ptx", "-arch=sm_89", "-O3", "-DA", "--Werror", "all-warnings",
             "/k/x.cu", "-o", "/o/x.ptx"],
        )

    def test_strict_off_and_extra_flags_are_honored(self):
        os.environ["AVAROK_KERNEL_NO_STRICT"] = "1"
        os.environ["AVAROK_EXTRA_NVCC_FLAGS"] = "-DX=1 -DY"
        try:
            args = probe.nvcc_args(self.job(), "sm_89", Path("/o/x.ptx"))
        finally:
            del os.environ["AVAROK_KERNEL_NO_STRICT"]
            del os.environ["AVAROK_EXTRA_NVCC_FLAGS"]
        self.assertNotIn("--Werror", args)
        self.assertEqual(args[4:6], ["-DX=1", "-DY"])


class FirstError(unittest.TestCase):
    def test_picks_the_first_diagnostic(self):
        text = "note: foo\nk.cu(3): error: identifier \"x\" is undefined\nerror: 2nd\n"
        self.assertEqual(probe.first_error_line(text), 'k.cu(3): error: identifier "x" is undefined')

    def test_falls_back_to_the_first_line(self):
        self.assertEqual(probe.first_error_line("\n  boom \n"), "boom")
        self.assertEqual(probe.first_error_line(""), "(no output)")


class Enumeration(unittest.TestCase):
    def test_gb10_has_jobs_and_shadows_common_by_stem(self):
        jobs, hw = probe.enumerate_jobs("gb10", "*", "nvfp4")
        self.assertTrue(jobs)
        self.assertEqual(hw["hardware"]["vendor"], "nvidia")
        keys = {(j.source, j.flags) for j in jobs}
        self.assertEqual(len(keys), len(jobs), "jobs are de-duplicated")

    def test_unknown_hardware_and_non_nvidia_are_refused(self):
        with self.assertRaises(probe.ProbeError):
            probe.enumerate_jobs("no-such-hw", "*", "nvfp4")
        if (probe.KERNELS / "metal").is_dir():
            with self.assertRaises(probe.ProbeError):
                probe.enumerate_jobs("metal", "*", "nvfp4")


if __name__ == "__main__":
    unittest.main()
