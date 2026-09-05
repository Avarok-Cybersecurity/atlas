#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU regressions for intended HF identity; no claim about loaded bytes."""
import argparse
import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import vllm_render as renderer

RECIPES = pathlib.Path(__file__).with_name('vllm_recipes.json')


class PinTests(unittest.TestCase):
    def setUp(self):
        self.doc = json.loads(RECIPES.read_text())
        self.entry = self.doc['entries'][0]

    def render(self, entry=None, extra='', spec='off', digest=None, image=None):
        entry = entry or self.entry
        with tempfile.TemporaryDirectory() as stage:
            args = argparse.Namespace(model=entry['model_key'], sku=entry['sku'],
                spec=spec, extra=extra, image=image, image_digest=digest,
                stage=stage, container='pins-test', hf_cache='/unused-cache',
                docker='docker', label=[])
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = renderer.cmd_render(self.doc, args)
            files = {p.name: p.read_bytes() for p in pathlib.Path(stage).iterdir()}
        return rc, out.getvalue(), err.getvalue(), files

    def refused(self, **kwargs):
        rc, out, err, files = self.render(**kwargs)
        self.assertEqual(rc, 8, (out, err))
        self.assertIn('revision identity', err)
        self.assertEqual(files, {}, 'refusal must precede argv staging')

    def test_every_head_and_worker_uses_declared_full_revision(self):
        commands = 0
        for entry in self.doc['entries']:
            with self.subTest(model=entry['model_key'], sku=entry['sku']):
                self.assertRegex(entry.get('revision', ''), r'^[0-9a-f]{40}$')
                rc, out, err, files = self.render(entry=entry)
                self.assertEqual(rc, 0, err)
                self.assertIn('original recipe evidence', out)
                self.assertIn('not loaded-byte proof', out)
                for name, value in files.items():
                    if name.startswith('node') and name.endswith('.argv'):
                        argv = value.decode().strip('\0').split('\0')
                        self.assertEqual(argv.count('--revision'), 1)
                        self.assertEqual(argv[argv.index('--revision') + 1], entry['revision'])
                        commands += 1
        self.assertEqual(commands, 37)

    def test_external_draft_revision_and_exact_spec_arithmetic(self):
        drafts = 0
        for entry in self.doc['entries']:
            if not entry['spec_args']:
                continue
            with self.subTest(model=entry['model_key'], sku=entry['sku']):
                digest = None
                if entry.get('draft_revision'):
                    # Synthetic CPU support receipt for argv arithmetic only.
                    digest = 'sha256:' + 'a' * 64
                    entry['draft_revision_image_refs'] = [renderer.image_ref(entry['image'], digest)]
                rc, _, err, off_files = self.render(entry=entry, digest=digest)
                self.assertEqual(rc, 0, err)
                rc, _, err, on_files = self.render(entry=entry, spec='on', digest=digest)
                self.assertEqual(rc, 0, err)
                for name, off in off_files.items():
                    if not (name.startswith('node') and name.endswith('.argv')):
                        continue
                    off = off.decode().strip('\0').split('\0')
                    on = on_files[name].decode().strip('\0').split('\0')
                    self.assertNotIn('--speculative-config', off)
                    self.assertEqual(on.count('--speculative-config'), 1)
                    self.assertEqual(on, off + entry['spec_args'])
                    config = json.loads(on[on.index('--speculative-config') + 1])
                    if 'model' in config:
                        self.assertRegex(config.get('revision', ''), r'^[0-9a-f]{40}$')
                config = json.loads(entry['spec_args'][1])
                if 'model' in config:
                    drafts += 1
        self.assertEqual(drafts, 5)

    def test_missing_pin_refused(self):
        self.entry.pop('revision', None)
        self.refused()

    def test_branch_name_pin_refused(self):
        self.entry['revision'] = 'main'
        self.refused()

    def test_short_pin_refused(self):
        self.entry['revision'] = '01234567'
        self.refused()

    def test_missing_primary_flag_refused(self):
        args = self.entry['args']
        if '--revision' in args:
            index = args.index('--revision')
            del args[index:index + 2]
        self.refused()

    def test_duplicate_primary_flag_refused(self):
        self.entry['args'] += ['--revision', 'a' * 40]
        self.refused()

    def test_wrong_repo_refused(self):
        self.entry['args'][2] = 'different/repo'
        self.refused()

    def test_worker_pin_mismatch_refused(self):
        entry = next(e for e in self.doc['entries'] if e['worker_args'])
        worker = entry['worker_args'][0]
        worker[worker.index('--revision') + 1] = 'a' * 40
        self.refused(entry=entry)

    def test_same_repo_different_pin_refused(self):
        other = next(e for e in self.doc['entries'] if e is not self.entry and e['hf_id'] == self.entry['hf_id'])
        other['revision'] = 'a' * 40
        self.refused()

    def test_external_draft_missing_revision_refused(self):
        entry = next(e for e in self.doc['entries'] if e['model_key'] == 'kimi-k3')
        config = json.loads(entry['spec_args'][1])
        config.pop('revision', None)
        entry['spec_args'][1] = json.dumps(config)
        self.refused(entry=entry, spec='on')
        self.refused(entry=entry, spec='off')

    def test_internal_spec_cannot_override_primary_revision(self):
        config = json.loads(self.entry['spec_args'][1])
        config['revision'] = 'main'
        self.entry['spec_args'][1] = json.dumps(config)
        self.refused(spec='on')

    def test_spec_args_cannot_override_primary_identity(self):
        original = list(self.entry['spec_args'])
        for flag, value in [('--revision', 'main'), ('--model', 'different/repo'),
                            ('--tokenizer', 'different/tokenizer')]:
            self.entry['spec_args'] = original + [flag, value]
            for mode in ('off', 'on'):
                with self.subTest(flag=flag, spec=mode):
                    self.refused(spec=mode)

    def test_external_draft_repo_mismatch_refused(self):
        entry = next(e for e in self.doc['entries'] if e['model_key'] == 'minimax-m3')
        config = json.loads(entry['spec_args'][1])
        config['model'] = 'different/draft'
        entry['spec_args'][1] = json.dumps(config)
        self.refused(entry=entry, spec='on')

    def test_stale_embedded_spec_config_refused_even_when_off(self):
        self.entry['args'] += ['--speculative-config', '{"model":"unpinned/draft"}']
        self.refused()

    def test_identity_overrides_refused(self):
        for extra in ['--revision main', '--revision=main', '--rev main',
                      '--speculative-config {"model":"other/draft"}',
                      '--speculative-config.revision=main',
                      '--speculative_config.revision=main',
                      '--model other/repo', '--tokenizer other/repo',
                      '--tokenizer-revision main', '--code-revision main',
                      '--config elsewhere.yaml']:
            with self.subTest(extra=extra):
                self.refused(extra=extra)

    def test_non_identity_extra_remains_available(self):
        rc, _, err, files = self.render(extra='--port 9001')
        self.assertEqual(rc, 0, err)
        self.assertTrue(files['head.argv'].endswith(b'--port\x009001\x00'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
