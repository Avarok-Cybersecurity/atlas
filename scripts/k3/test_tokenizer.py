# SPDX-License-Identifier: AGPL-3.0-only
import unittest
from tokenizer import read_pattern, read_ranks, byte_alphabet
from pathlib import Path
import tempfile

class TokenizerTests(unittest.TestCase):
    def test_pattern_is_data_not_executed(self):
        self.assertEqual(read_pattern('raise RuntimeError()\nclass T:\n pat_str="|".join([r"abc",r"def"])'), 'abc|def')
        with self.assertRaises(ValueError): read_pattern('pat_str = dangerous()')

    def test_ranks_refuse_duplicates_and_gaps(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ranks'
            for text in ['YQ== 0\nYQ== 1\n', 'YQ== 2\n', 'YQ== 0\nYg== 0\n']:
                p.write_text(text)
                with self.assertRaises(ValueError): read_ranks(p)

    def test_byte_alphabet_is_bijective(self):
        alphabet=byte_alphabet()
        self.assertEqual(len(alphabet),256)
        self.assertEqual(len(set(alphabet.values())),256)
