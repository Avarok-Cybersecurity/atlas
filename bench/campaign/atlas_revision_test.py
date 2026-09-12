#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""D3 identity oracle: every Atlas HF artifact has an explicit intended pin."""
import json
import pathlib
import unittest

HERE = pathlib.Path(__file__).resolve().parent


class AtlasRevisionTest(unittest.TestCase):
    def test_every_hf_artifact_has_consistent_full_revision(self):
        doc = json.loads((HERE / 'atlas_recipes.json').read_text())
        revisions = {}
        for entry in doc['entries']:
            with self.subTest(model=entry['model_key'], sku=entry['sku']):
                self.assertRegex(entry.get('revision', ''), r'^[0-9a-f]{40}$')
                self.assertEqual(revisions.setdefault(entry['hf_id'], entry['revision']), entry['revision'])
        fp8 = revisions['nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8']
        nvfp4 = revisions['nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4']
        self.assertNotEqual(fp8, nvfp4, 'Different artifacts cannot borrow each other\'s revision')
        self.assertEqual(fp8, '9bee19446c0dfd01f356e10979d225b2a6621944')

    def test_metadata_does_not_claim_loaded_bytes(self):
        doc = json.loads((HERE / 'atlas_recipes.json').read_text())
        self.assertIs(doc.get('revision_adaptation', {}).get('loaded_bytes_proven'), False)
        self.assertIn('intended', doc['revision_adaptation']['pin_semantics'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
