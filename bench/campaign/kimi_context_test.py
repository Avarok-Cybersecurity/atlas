#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""PRD section 16 oracle: every Kimi render uses the scored 49152 context."""
import unittest

import vllm_pins_test


class KimiContextTest(unittest.TestCase):
    setUp = vllm_pins_test.PinTests.setUp
    render = vllm_pins_test.PinTests.render
    def test_kimi_context(self):
        for entry in self.doc['entries']:
            if entry['model_key'] != 'kimi-k3':
                continue
            with self.subTest(sku=entry['sku']):
                rc, out, err, files = self.render(entry=entry, spec='off')
                self.assertEqual(rc, 0, out + err)
                for name, raw in files.items():
                    if name.startswith('node') and name.endswith('.argv'):
                        argv = raw.decode().strip('\0').split('\0')
                        self.assertEqual(argv[argv.index('--max-model-len') + 1], '49152')


# Run only this finding's regression; PinTests supplies the fixture methods.
if __name__ == '__main__':
    suite = unittest.TestSuite([KimiContextTest('test_kimi_context')])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
