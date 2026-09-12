#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""PRD section 16 and captured Qwen/H200 recipe are the first-cell oracles."""
import json
import pathlib
import subprocess
import unittest

HERE = pathlib.Path(__file__).resolve().parent
MODEL = 'qwen3.8-27b-fp8'
REPO = 'Qwen/Qwen3.8-27B-FP8'
SHA = '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'


class Qwen27BRecipeTest(unittest.TestCase):
    def recipe(self, engine):
        doc = json.loads((HERE / (engine + '_recipes.json')).read_text())
        entry = next((e for e in doc['entries'] if (e['model_key'], e['sku']) == (MODEL, 'h200')), None)
        self.assertIsNotNone(entry, 'PRD first paid H200 pair has no recipe')
        return entry

    def test_both_recipes_pin_same_fp8_checkpoint(self):
        for engine in ('atlas', 'vllm'):
            with self.subTest(engine=engine):
                entry = self.recipe(engine)
                self.assertEqual((entry['hf_id'], entry['revision'], entry['quant']), (REPO, SHA, 'fp8'))
                self.assertEqual(entry.get('ngpus', entry.get('gpus')), 1)

    def test_atlas_uses_target_context_and_calibration(self):
        entry = self.recipe('atlas')
        self.assertFalse(entry['spec_supported'])
        result = subprocess.run(['python3', str(HERE / 'atlas_render.py'), '--recipes',
            str(HERE / 'atlas_recipes.json'), '--model', MODEL, '--sku', 'h200',
            '--spec', 'off', '--think', 'off', '--extra-args'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--max-seq-len 24576', result.stdout)
        self.assertIn('--fp8-kv-calibration-tokens 256', result.stdout)

    def test_vllm_matches_captured_h200_argv(self):
        entry = self.recipe('vllm')
        self.assertEqual(entry['image'], 'vllm/vllm-openai:qwen38')
        self.assertEqual(entry['args'], ['vllm', 'serve', REPO,
            '--tensor-parallel-size', '1', '--enable-auto-tool-choice',
            '--tool-call-parser', 'qwen3_coder', '--reasoning-parser', 'qwen3',
            '--mm-encoder-tp-mode', 'data', '--revision', SHA])
        self.assertFalse(entry['spec_args'], 'No evidenced MTP profile for the dense 27B checkpoint')

    def test_both_engines_refuse_unevidenced_mtp(self):
        for engine in ('atlas', 'vllm'):
            with self.subTest(engine=engine):
                command = ['bash', str(HERE / 'run_cell.sh'), '--engine', engine,
                    '--model', MODEL, '--sku', 'h200', '--workload', 'lat',
                    '--concurrency', '1', '--spec', 'on', '--think', 'off',
                    '--out', '/unused-qwen27b-test', '--dry-run']
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 4, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
