# SPDX-License-Identifier: AGPL-3.0-only
"""Strict raw-completion canary payloads; token IDs are never retokenized."""
import json
from pathlib import Path


def validate_prompt(prompt):
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    if (isinstance(prompt, list) and prompt
            and all(type(token) is int and 0 <= token <= 0xffffffff for token in prompt)):
        return prompt
    raise ValueError('prompt must be nonempty text or a nonempty array of unsigned 32-bit token IDs')


def validate_payload(payload):
    required = {'model', 'prompt', 'max_tokens', 'temperature', 'stream'}
    if not isinstance(payload, dict) or not required <= payload.keys() or payload.keys()-required-{'stop'}:
        raise ValueError('prepared request must contain model, prompt, max_tokens, temperature, stream and optional stop')
    if not isinstance(payload['model'], str) or not payload['model'].strip():
        raise ValueError('prepared model must be nonempty text')
    validate_prompt(payload['prompt'])
    if type(payload['max_tokens']) is not int or payload['max_tokens'] <= 0:
        raise ValueError('prepared max_tokens must be a positive integer')
    if type(payload['temperature']) not in (int, float) or payload['temperature'] != 0:
        raise ValueError('canary temperature must be zero')
    if payload['stream'] is not False:
        raise ValueError('canary stream must be false')
    if 'stop' in payload and (not isinstance(payload['stop'], list) or not payload['stop']
                             or any(not isinstance(s, str) or not s for s in payload['stop'])):
        raise ValueError('prepared stop must be a nonempty array of nonempty strings')
    return payload


def read_payload(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate prepared request field: {key}')
            result[key] = value
        return result
    with Path(path).open('rb') as source:
        raw = source.read(2*1024*1024+1)
    if len(raw) > 2*1024*1024:
        raise ValueError('prepared request exceeds 2 MiB')
    return validate_payload(json.loads(raw, object_pairs_hook=unique))
