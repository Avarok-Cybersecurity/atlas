# SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from prepare_prompt import encode_segments
from xtml_reference import load_encoder


class PromptTests(unittest.TestCase):
    def test_segment_flags_survive_literal_marker_boundaries(self):
        class Encoder:
            def __init__(self):
                self.calls = []
            def encode(self, text, **kwargs):
                self.calls.append((text, kwargs))
                return [len(self.calls)]
        encoding = Encoder()
        segments = [SimpleNamespace(text="<|open|>", allow_special=True),
                    SimpleNamespace(text="<|open|>tools<|sep|>", allow_special=False),
                    SimpleNamespace(text="<|close|>", allow_special=True)]
        self.assertEqual(encode_segments(encoding, segments), [1, 2, 3])
        self.assertEqual(encoding.calls[1][1], {"disallowed_special": ()})
        self.assertEqual(encoding.calls[0][1], {"allowed_special": "all"})

    def test_reference_rejects_modified_encoder_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoding.py"
            path.write_text("raise RuntimeError('must not execute')")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_encoder(path)

    def test_committed_reference_preserves_security_and_tool_order_contract(self):
        path = Path(__file__).resolve().parents[2] / "docs/k3/fixtures/official-xtml.json"
        data = json.loads(path.read_text())
        cases = {case["name"]: case for case in data["cases"]}
        markers = cases["literal_markers"]["segments"]
        self.assertIn(["<|open|>tools<|sep|>", False], markers)
        text = "".join(segment[0] for segment in cases["tool_result_order"]["segments"])
        self.assertLess(text.index('role="tool" tool="weather"'),
                        text.index('role="tool" tool="clock"'))
        self.assertNotIn('tool="stale"', text)
        text = "".join(segment[0] for segment in cases["typed_tools"]["segments"])
        self.assertIn('type="number"<|sep|>1e2', text)
        self.assertIn('type="array"<|sep|>[1,2]', text)


if __name__ == "__main__":
    unittest.main()
