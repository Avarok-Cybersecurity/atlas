# SPDX-License-Identifier: AGPL-3.0-only
import struct
import unittest
from unittest.mock import MagicMock, patch
import audit_headers


class BoundedHeaderReads(unittest.TestCase):
    def response(self, status, content_range, data=b"12345678"):
        response = MagicMock()
        response.status = status
        response.headers = {"Content-Range": content_range}
        response.read.return_value = data
        response.__enter__.return_value = response
        return response

    def test_ignored_or_wrong_range_refuses_before_body_read(self):
        for status, value in [(200, ""), (206, "bytes 8-15/100")]:
            response = self.response(status, value)
            with patch.object(audit_headers.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(RuntimeError):
                    audit_headers.read_range("model.safetensors", 0, 7)
            response.read.assert_not_called()

    def test_valid_and_overlong_range(self):
        for data in [b"12345678", b"123456789"]:
            response = self.response(206, "bytes 0-7/100", data)
            with patch.object(audit_headers.urllib.request, "urlopen", return_value=response):
                if len(data) == 8:
                    self.assertEqual(audit_headers.read_range("model.safetensors", 0, 7), data)
                else:
                    with self.assertRaises(RuntimeError):
                        audit_headers.read_range("model.safetensors", 0, 7)
            response.read.assert_called_once_with(9)

    def test_oversized_header_refuses_before_header_payload_read(self):
        with patch.object(audit_headers, "read_range", return_value=struct.pack("<Q", 8_000_000)) as read:
            with self.assertRaises(RuntimeError):
                audit_headers.read_header("model.safetensors")
        read.assert_called_once_with("model.safetensors", 0, 7)


if __name__ == "__main__":
    unittest.main()
