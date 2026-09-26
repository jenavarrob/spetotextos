from pathlib import Path


class TesseractService:
    """Local image-to-text service backed by the Tesseract executable."""

    def __init__(self, command: str | None = None, language: str = "eng"):
        self.command = command
        self.language = language

    def extract_text(self, contents: bytes) -> str:
        from io import BytesIO
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "Local OCR dependencies are missing. Install them with "
                "'python -m pip install Pillow pytesseract'."
            ) from exc

        if self.command:
            pytesseract.pytesseract.tesseract_cmd = self.command

        with Image.open(BytesIO(contents)) as image:
            text = pytesseract.image_to_string(image, lang=self.language)
        return text.strip()
