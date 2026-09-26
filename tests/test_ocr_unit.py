from unittest.mock import patch

import pytest

from app.services.ocr import TesseractService

pytest.importorskip("PIL")
pytest.importorskip("pytesseract")


@patch("app.services.ocr.pytesseract.image_to_string", return_value="  extracted text\n")
@patch("app.services.ocr.Image.open")
def test_tesseract_extracts_and_strips_text(mock_open, mock_ocr):
    image = mock_open.return_value.__enter__.return_value
    result = TesseractService(language="eng").extract_text(b"image-bytes")
    assert result == "extracted text"
    mock_ocr.assert_called_once_with(image, lang="eng")
