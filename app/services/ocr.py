from pathlib import Path
from io import BytesIO

try:
    from PIL import Image
    import pytesseract
except ImportError: 
    Image = None
    pytesseract = None

class TesseractService:
    """Local image-to-text service backed by the Tesseract executable."""

    def __init__(self, command: str | None = None, language: str = "eng"):
        self.command = command
        self.language = language

    def extract_text(self, contents: bytes) -> str:
        if Image is None or pytesseract is None:
            raise RuntimeError(
                "Local OCR dependencies are missing. Install them with "
                "'python -m pip install Pillow pytesseract'."
            )

        if self.command:
            pytesseract.pytesseract.tesseract_cmd = self.command

        with Image.open(BytesIO(contents)) as image:
            text = pytesseract.image_to_string(image, lang=self.language)
        return text.strip()
