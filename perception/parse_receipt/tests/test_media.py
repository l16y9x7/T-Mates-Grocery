import base64
import tempfile
import unittest
from pathlib import Path

from receipt_recognizer.errors import InputFileError
from receipt_recognizer.media import (
    image_bytes_to_data_url,
    prepare_input,
)

class MediaTests(unittest.TestCase):
    def test_png_becomes_jpeg_data_url(self):
        data_url = image_bytes_to_data_url(
            _one_pixel_png(),
            max_edge=1000,
        )

        prefix, encoded = data_url.split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        decoded = base64.b64decode(encoded)
        self.assertTrue(decoded.startswith(b"\xff\xd8\xff"))

    def test_rejects_unknown_extension(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "receipt.txt"
            path.write_text("not an image")
            with self.assertRaises(InputFileError):
                prepare_input(path)

    def test_single_page_pdf_becomes_jpeg_data_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "receipt.pdf"
            path.write_bytes(_single_page_pdf())

            data_urls = prepare_input(path)

        self.assertEqual(len(data_urls), 1)
        prefix, encoded = data_urls[0].split(",", 1)
        self.assertEqual(prefix, "data:image/jpeg;base64")
        self.assertTrue(
            base64.b64decode(encoded).startswith(b"\xff\xd8\xff")
        )


def _one_pixel_png() -> bytes:
    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0"
        "lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )


def _single_page_pdf() -> bytes:
    stream = (
        b"BT /F1 16 Tf 20 120 Td (TEST RECEIPT) Tj "
        b"0 -24 Td /F1 12 Tf (CHIPS x3) Tj ET"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 300 160] "
            b"/Resources << /Font << /F1 5 0 R >> >> "
            b"/Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} "
            "/Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(output)


if __name__ == "__main__":
    unittest.main()
