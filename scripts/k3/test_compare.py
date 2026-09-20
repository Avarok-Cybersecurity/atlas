# SPDX-License-Identifier: AGPL-3.0-only
import copy
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import compare as suite
from compare import compare, validate_cases

class CompareTests(unittest.TestCase):
    def test_rejects_mismatch_even_when_each_generation_passes(self):
        row={'request':{'prompt':'A','model':'fixture','max_tokens':1},'status':'passed','response':{'model':'fixture','choices':[{'text':'B','finish_reason':'stop'}], 'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}}
        compare(row,row)
        other=copy.deepcopy(row);other['response']['choices'][0]['text']='C'
        with self.assertRaisesRegex(ValueError,'text mismatch'):compare(row,other)
        other=copy.deepcopy(row);other['request']['prompt']='changed'
        with self.assertRaisesRegex(ValueError,'request differs'):compare(row,other)
        other=copy.deepcopy(row);other['status']='failed'
        with self.assertRaises(ValueError):compare(row,other)

    def test_suite_cannot_overwrite_or_escape_receipts(self):
        case={'id':'a','prompt':'A','max_tokens':1}
        validate_cases([case])
        with self.assertRaises(ValueError):validate_cases([case,case])
        case['id']='../outside'
        with self.assertRaises(ValueError):validate_cases([case])

    def test_malformed_receipt_is_a_comparison_failure(self):
        for record in (None, [], {}, {'request': {}, 'status': 'passed', 'response': {}}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                compare(record, record)

    def test_malformed_reference_retains_failure_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root/'cases.json'
            cases.write_text(json.dumps([{'id':'a', 'prompt':'A', 'max_tokens':1}]))
            reference = root/'reference'
            reference.mkdir()
            (reference/'a.json').write_text('{}')
            output = root/'output'
            def completed_probe(command, **kwargs):
                path = Path(command[command.index('--output')+1])
                path.write_text('{}')
            argv = ['compare.py', '--cases', str(cases), '--endpoint', 'http://127.0.0.1:1',
                    '--model', 'fixture', '--deadline', '1', '--output', str(output),
                    '--reference', str(reference)]
            with patch('sys.argv', argv), patch('compare.subprocess.run', completed_probe):
                self.assertEqual(suite.main(), 1)
            summary = json.loads((output/'summary.json').read_text())
            self.assertFalse(summary[0]['passed'])
            self.assertIn('malformed', summary[0]['error'])
