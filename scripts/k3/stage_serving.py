#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verify an immutable snapshot and stage hardlinked weights with derived tokenizer."""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil

from checkpoint import check_file, safe_path, validate


def reject_symlink_parents(path):
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f'Symlink input/output ancestry is unsupported: {parent}')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate index key: {key}')
        result[key] = value
    return result


def stage(source, derived, output, manifest):
    validate(manifest)
    for path in (source, derived, output):
        reject_symlink_parents(path.absolute())
    source, derived, output = source.absolute(), derived.absolute(), output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Output must be a new directory')
    if not output.parent.is_dir() or output.is_relative_to(source) or output.is_relative_to(derived):
        raise ValueError('Output needs an existing parent outside source and derived assets')
    entries = {item['path']: item for item in manifest['files']}
    required = {'config.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    if not required <= entries.keys():
        raise ValueError('Manifest lacks required model config/tokenizer config/weight index')
    if 'tokenizer.json' in entries or 'serving-stage.json' in entries:
        raise ValueError('Snapshot conflicts with derived serving assets')
    fingerprints = {}
    for name, item in entries.items():
        if not check_file(source, item):
            raise ValueError(f'Source manifest verification failed: {name}')
        stat = safe_path(source, name).stat()
        fingerprints[name] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    index_path = safe_path(source, 'model.safetensors.index.json')
    if index_path.stat().st_size > 128*1024*1024:
        raise ValueError('Weight index exceeds 128 MiB limit')
    index = json.loads(index_path.read_text(), object_pairs_hook=unique_object)
    weight_map = index.get('weight_map')
    if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(v, str) for v in weight_map.values()):
        raise ValueError('Weight index must contain a nonempty string-valued weight_map')
    weights = set(weight_map.values())
    if any(name not in entries or len(Path(name).parts) != 1 or not name.endswith('.safetensors') for name in weights):
        raise ValueError('Weight index refers to an unverified or unsupported shard')
    derived_path = safe_path(derived, 'tokenizer.json')
    receipt_path = safe_path(derived, 'derived-tokenizer.json')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != 1 or sha256(derived_path) != receipt.get('derived_tokenizer_sha256'):
        raise ValueError('Derived tokenizer receipt/hash mismatch')
    expected_sources = {'tiktoken.model', 'tokenization_kimi.py', 'tokenizer_config.json'}
    if not isinstance(receipt.get('sources'), dict) or set(receipt['sources']) != expected_sources:
        raise ValueError('Derived tokenizer receipt lacks exact source inventory')
    for name, digest in receipt['sources'].items():
        if name not in entries or sha256(safe_path(source, name)) != digest:
            raise ValueError(f'Derived tokenizer source identity mismatch: {name}')
    device = output.parent.stat().st_dev
    for name in entries:
        if name.endswith('.safetensors') and fingerprints[name][0] != device:
            raise ValueError('Weights and output must be on the same filesystem; no weight-copy fallback')
    output.mkdir()
    linked = 0
    try:
        for name in entries:
            origin = safe_path(source, name)
            stat = origin.stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != fingerprints[name]:
                raise ValueError(f'Source changed during staging: {name}')
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith('.safetensors'):
                try:
                    os.link(origin, destination, follow_symlinks=False)
                except OSError as error:
                    if error.errno == errno.EXDEV:
                        raise ValueError('Cross-filesystem hardlink refused; no weight-copy fallback') from error
                    raise
                linked += 1
            else:
                shutil.copyfile(origin, destination)
        shutil.copyfile(derived_path, output/'tokenizer.json')
        if sha256(output/'tokenizer.json') != receipt['derived_tokenizer_sha256']:
            raise ValueError('Derived tokenizer changed during staging')
        result = {'schema': 1, 'repo': manifest['repo'], 'revision': manifest['revision'],
                  'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                  'verified_source_files': len(entries), 'hardlinked_weight_files': linked,
                  'copied_metadata_files': len(entries)-linked,
                  'derived_tokenizer': receipt,
                  'contract': 'Read-only serving weights share source inodes; never modify either copy in place'}
        (output/'serving-stage.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    except BaseException:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'derived', 'output', 'manifest'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.source, args.derived, args.output, json.loads(args.manifest.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
