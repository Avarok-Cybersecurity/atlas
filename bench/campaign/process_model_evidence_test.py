#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU CLI oracles for model pins from an owned Linux process snapshot."""

import copy
import datetime
import unittest

from cell_assemble_test import AssemblyTest, HF_ID, REVISION, fixture, launch


def process_launch(engine="vllm"):
    boot = fixture("stub_boot.json")
    utc = datetime.timezone.utc
    stamp = lambda epoch: datetime.datetime.fromtimestamp(epoch, utc).isoformat()
    model = HF_ID if engine == "vllm" else (
        "/cache/models--" + HF_ID.replace("/", "--") + "/snapshots/" + REVISION)
    argv = ["/opt/vllm/bin/python3", "/opt/vllm/bin/vllm", "serve", model,
            "--revision", REVISION] if engine == "vllm" else ["/workspace/spark", "serve", model]
    marker = "fixture-process-" + "c" * 32
    owner = dict(schema=1, kind="linux-proc-owner", pid=321, pgid=321, sid=321,
                 start_ticks=1234567, boot_id="01234567-89ab-4cde-8fab-0123456789ab",
                 run_marker=marker, executable=("/usr/bin/python3.12" if engine == "vllm"
                                               else "/workspace/spark"),
                 executable_sha256="d" * 64, argv=argv,
                 environment={"ATLAS_CAMPAIGN_RUN_TOKEN": marker},
                 created_at=stamp(boot["start_epoch"]))
    observed = copy.deepcopy(owner)
    observed.update(kind="linux-proc", running=True, captured_pid=owner["pid"],
                    captured_start_ticks=owner["start_ticks"], captured_boot_id=owner["boot_id"],
                    captured_at=stamp(boot["start_epoch"] + boot["total_s"] + 1))
    return observed, owner


class ProcessAssemblyTest(unittest.TestCase):
    assemble = AssemblyTest.assemble

    def assert_refused(self, process, owner, **kwargs):
        result = self.assemble(process=process, owner=owner, **kwargs)
        self.assertEqual((result["verdict"], result["failing_stage"]), ("NO-GO", "serve"))
        self.assertIsNone(result["model"]["revision"])
        self.assertIn("invalid model launch evidence", result["notes"])

    def test_a_foreign_or_reused_process_is_refused(self):
        for key, value in (("pid", 999), ("start_ticks", 999), ("pgid", 999), ("sid", 999),
                           ("boot_id", "11111111-1111-4111-8111-111111111111"),
                           ("run_marker", "another-run")):
            observed, owner = process_launch()
            observed[key] = value
            with self.subTest(key=key):
                self.assert_refused(observed, owner)

    def test_b_stopped_or_changed_capture_is_refused(self):
        for key, value in (("running", False), ("captured_pid", 999),
                           ("captured_start_ticks", 999), ("captured_boot_id", "old-boot"),
                           ("captured_at", "2000-01-01T00:00:00Z"),
                           ("executable", "/usr/bin/foreign"),
                           ("executable_sha256", "e" * 64), ("environment", {})):
            observed, owner = process_launch()
            observed[key] = value
            with self.subTest(key=key):
                self.assert_refused(observed, owner)

    def test_c_unowned_or_malformed_records_are_refused(self):
        observed, owner = process_launch()
        for bad_process, bad_owner in ((observed, None), (None, owner), ([], owner),
                                       (observed, {}), ({}, owner)):
            with self.subTest(process=bad_process, owner=bad_owner):
                self.assert_refused(bad_process, bad_owner)
        for key, value in (("pid", True), ("start_ticks", 0), ("boot_id", ""),
                           ("run_marker", ""), ("executable_sha256", "missing")):
            observed, owner = process_launch()
            observed[key] = owner[key] = value
            with self.subTest(key=key):
                self.assert_refused(observed, owner)

    def test_d_wrong_or_mutated_argv_is_refused(self):
        observed, owner = process_launch()
        observed["argv"][3] = "foreign/model"
        self.assert_refused(observed, owner)
        for argv in (["/usr/bin/python3", "-c", "print('not an engine')"],
                     ["/opt/vllm/bin/python3", "/opt/vllm/bin/vllm", "serve", HF_ID,
                      "--revision", "main"],
                     ["/opt/vllm/bin/python3", "/opt/vllm/bin/vllm", "serve", HF_ID,
                      "--revision", REVISION, "--model", "foreign/model"]):
            observed, owner = process_launch()
            observed["argv"] = owner["argv"] = argv
            with self.subTest(argv=argv):
                self.assert_refused(observed, owner)

    def test_e_conflicting_docker_and_process_proof_is_refused(self):
        observed, owner = process_launch()
        self.assert_refused(observed, owner, evidence=launch())

    def test_f_failed_or_mismatched_boot_cannot_prove_revision(self):
        observed, owner = process_launch()
        result = self.assemble(process=observed, owner=owner, boot=fixture("stub_boot_timeout.json"))
        self.assertEqual(result["failing_stage"], "boot")
        self.assertIsNone(result["model"]["revision"])
        self.assert_refused(observed, owner, boot_engine="atlas")
        self.assert_refused(observed, owner, boot_model="foreign/model")
        stale = copy.deepcopy(observed)
        stale["captured_at"] = owner["created_at"]
        self.assert_refused(stale, owner)

    def test_g_owned_pin_survives_supported_actual_process_forms(self):
        for mode in ("shebang", "module", "direct", "atlas"):
            engine = "atlas" if mode == "atlas" else "vllm"
            observed, owner = process_launch(engine)
            if mode == "module":
                observed["argv"] = owner["argv"] = ["/opt/vllm/bin/python3", "-m",
                    "vllm.entrypoints.cli.main", "serve", HF_ID, "--revision", REVISION]
            elif mode == "direct":
                observed["argv"] = owner["argv"] = ["/opt/vllm/bin/vllm", "serve", HF_ID,
                                                    "--revision", REVISION]
                observed["executable"] = owner["executable"] = "/opt/vllm/bin/vllm"
            with self.subTest(mode=mode):
                result = self.assemble(process=observed, owner=owner, engine=engine)
                self.assertEqual(result["verdict"], "CERTIFIED")
                self.assertEqual(result["model"]["revision"], REVISION)
                self.assertIn("Linux /proc", result["notes"])

    def test_h_floating_launch_does_not_inherit_recipe_revision(self):
        observed, owner = process_launch()
        observed["argv"] = owner["argv"] = observed["argv"][:-2]
        result = self.assemble(process=observed, owner=owner)
        self.assertEqual(result["verdict"], "CERTIFIED")
        self.assertIsNone(result["model"]["revision"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
