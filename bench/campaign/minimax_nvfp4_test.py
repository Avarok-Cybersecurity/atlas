#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Official NVIDIA NVFP4/B200 recipe and PRD quant policy are the oracles."""
import unittest
import vllm_pins_test


class MiniMaxNvfp4Test(unittest.TestCase):
    setUp = vllm_pins_test.PinTests.setUp
    render = vllm_pins_test.PinTests.render

    def test_b200_uses_captured_nvfp4_recipe(self):
        entry = next(e for e in self.doc['entries'] if (e['model_key'], e['sku']) == ('minimax-m3', 'b200'))
        self.assertEqual(entry['hf_id'], 'nvidia/MiniMax-M3-NVFP4')
        self.assertEqual(entry['quant'], 'nvfp4')
        self.assertEqual(entry['revision'], '901464083161bf8612a29ff7ad29914cd4ab4a85')
        self.assertEqual(entry['topology']['tp'], 8)
        self.assertEqual(entry['env']['VLLM_FLASHINFER_ALLREDUCE_BACKEND'], 'trtllm')
        rc, out, err, files = self.render(entry=entry, spec='off')
        self.assertEqual(rc, 0, out + err)
        argv = files['head.argv'].decode().strip('\0').split('\0')
        for flag, value in (('--attention_config.backend', 'FLASHINFER'),
                            ('--attention_config.use_trtllm_attention', 'true'),
                            ('--attention_config.indexer_kv_dtype', 'fp8'),
                            ('--attention_config.minimax_m3_msa_decode_backend', 'cutlass')):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertFalse(entry['spec_args'], 'Do not transplant the BF16 recipe draft into this NVFP4 recipe')

    def test_hopper_profiles_keep_their_original_bf16_checkpoint(self):
        for entry in self.doc['entries']:
            if entry['model_key'] == 'minimax-m3' and entry['sku'] in ('h100', 'h200'):
                self.assertEqual(entry['hf_id'], 'MiniMaxAI/MiniMax-M3')
                self.assertEqual(entry['quant'], 'bf16')


if __name__ == '__main__':
    unittest.main(verbosity=2)
