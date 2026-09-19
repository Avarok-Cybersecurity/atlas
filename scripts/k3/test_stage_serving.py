# SPDX-License-Identifier: AGPL-3.0-only
import errno
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from stage_serving import stage

class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'source'; self.source.mkdir()
        self.derived = self.root / 'derived'; self.derived.mkdir()
        self.output = self.root / 'serving'
        assets = {'config.json': '{}', 'tokenizer_config.json': '{}',
                  'tokenization_kimi.py': '# source', 'tiktoken.model': 'ranks',
                  'model.safetensors.index.json': json.dumps({'weight_map': {'weight': 'model-01.safetensors'}}),
                  'model-01.safetensors': 'test weight bytes'}
        self.manifest = {'schema': 1, 'repo': 'moonshotai/Kimi-K3', 'revision': 'a'*40, 'files': []}
        for name, text in assets.items():
            (self.source / name).write_text(text)
            self.manifest['files'].append({'path': name, 'size': len(text), 'algorithm': 'sha256',
                                          'digest': hashlib.sha256(text.encode()).hexdigest()})
        (self.derived / 'tokenizer.json').write_text('{}')
        self.receipt = {'schema': 1, 'sources': {name: hashlib.sha256((self.source/name).read_bytes()).hexdigest()
                        for name in ['tokenization_kimi.py', 'tiktoken.model', 'tokenizer_config.json']},
                        'derived_tokenizer_sha256': hashlib.sha256(b'{}').hexdigest()}
        (self.derived / 'derived-tokenizer.json').write_text(json.dumps(self.receipt))

    def run_stage(self):
        return stage(self.source, self.derived, self.output, self.manifest)

    def test_hardlinks_stay_inside_loader_root_and_metadata_is_independent(self):
        receipt = self.run_stage()
        self.assertEqual(receipt['hardlinked_weight_files'], 1)
        weight = self.output/'model-01.safetensors'
        self.assertFalse(weight.is_symlink())
        self.assertTrue(weight.resolve().is_relative_to(self.output.resolve()))
        self.assertEqual(weight.stat().st_ino, (self.source/weight.name).stat().st_ino)
        self.assertNotEqual((self.output/'config.json').stat().st_ino, (self.source/'config.json').stat().st_ino)
        self.assertFalse((self.source/'serving-stage.json').exists())

    def test_missing_asset_and_modified_asset_refused(self):
        (self.source/'model-01.safetensors').unlink()
        with self.assertRaises(ValueError): self.run_stage()
        self.assertFalse(self.output.exists())
        (self.source/'model-01.safetensors').write_text('corrupted')
        with self.assertRaises(ValueError): self.run_stage()

    def test_escaping_symlink_and_index_refused(self):
        original = self.source/'model-01.safetensors'
        external = self.root/'external'; original.rename(external); original.symlink_to(external)
        with self.assertRaises(ValueError): self.run_stage()
        original.unlink(); external.rename(original)
        path = self.source/'model.safetensors.index.json'
        path.write_text(json.dumps({'weight_map': {'weight': '../external'}}))
        self.refresh_entry(path.name)
        with self.assertRaisesRegex(ValueError, 'unverified or unsupported'): self.run_stage()

    def refresh_entry(self, name):
        path = self.source/name
        item = next(item for item in self.manifest['files'] if item['path'] == name)
        item['size'] = path.stat().st_size
        item['digest'] = hashlib.sha256(path.read_bytes()).hexdigest()

    def test_nested_verified_shard_layout_is_refused(self):
        folder = self.source/'nested'; folder.mkdir()
        (self.source/'model-01.safetensors').rename(folder/'model-01.safetensors')
        item = next(item for item in self.manifest['files'] if item['path'] == 'model-01.safetensors')
        item['path'] = 'nested/model-01.safetensors'
        index = self.source/'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {'weight': item['path']}}))
        self.refresh_entry(index.name)
        with self.assertRaisesRegex(ValueError, 'unverified or unsupported'): self.run_stage()

    def test_duplicate_index_keys_refused(self):
        index = self.source/'model.safetensors.index.json'
        index.write_text('{"weight_map":{"w":"model-01.safetensors","w":"model-01.safetensors"}}')
        self.refresh_entry(index.name)
        with self.assertRaisesRegex(ValueError, 'Duplicate index key'): self.run_stage()

    def test_existing_output_refused(self):
        self.output.mkdir()
        with self.assertRaises(ValueError): self.run_stage()

    def test_cross_filesystem_never_copies_weights(self):
        with patch('stage_serving.os.link', side_effect=OSError(errno.EXDEV, 'cross filesystem')):
            with self.assertRaises(ValueError): self.run_stage()
        self.assertFalse(self.output.exists())

    def test_wrong_derived_source_or_output_digest_refused(self):
        (self.derived/'tokenizer.json').write_text('different')
        with self.assertRaises(ValueError): self.run_stage()
        self.assertFalse(self.output.exists())
