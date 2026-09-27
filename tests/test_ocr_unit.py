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


@patch("app.services.ocr.pytesseract.get_tesseract_version", return_value="5.3.0")
def test_tesseract_health_reports_available_executable(mock_version):
    health = TesseractService(command="C:\\Programs\\Tesseract\\tesseract.exe").check_connection()
    assert health["connected"] is True
    assert health["version"] == "5.3.0"
    assert health["command"] == "C:\\Programs\\Tesseract\\tesseract.exe"
    mock_version.assert_called_once_with()


@patch("app.services.ocr.pytesseract.get_tesseract_version", side_effect=RuntimeError("not found"))
def test_tesseract_health_reports_missing_executable(mock_version):
    health = TesseractService().check_connection()
    assert health["connected"] is False
    assert health["error"] == "not found"
