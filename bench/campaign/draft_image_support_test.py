#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""No external-draft revision argv without proof for the exact selected image."""
import unittest
import vllm_pins_test
import vllm_render

DIGEST = 'sha256:' + 'a' * 64


class DraftImageSupportTest(unittest.TestCase):
    setUp = vllm_pins_test.PinTests.setUp
    render = vllm_pins_test.PinTests.render

    def test_unverified_external_drafts_refuse_before_staging(self):
        for entry in self.doc['entries']:
            if entry.get('draft_revision'):
                with self.subTest(model=entry['model_key'], sku=entry['sku']):
                    rc, out, err, files = self.render(entry=entry, spec='on')
                    self.assertEqual(rc, 4, out + err)
                    self.assertIn('RECIPE_GAP', err)
                    self.assertEqual(files, {})
                    self.assertEqual(self.render(entry=entry, spec='off')[0], 0)

    def test_synthetic_support_receipt_must_match_image_and_digest(self):
        entry = next(e for e in self.doc['entries'] if e.get('draft_revision'))
        # This is a synthetic CPU fixture, not an image-support claim.
        entry['draft_revision_image_refs'] = [vllm_render.image_ref(entry['image'], DIGEST)]
        self.assertEqual(self.render(entry=entry, spec='on')[0], 4, 'A tag alone is not the proven image')
        self.assertEqual(self.render(entry=entry, spec='on', digest=DIGEST)[0], 0)
        self.assertEqual(self.render(entry=entry, spec='on', digest='sha256:' + 'b' * 64)[0], 4)
        self.assertEqual(self.render(entry=entry, spec='on', digest=DIGEST, image='elsewhere/vllm:custom')[0], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
